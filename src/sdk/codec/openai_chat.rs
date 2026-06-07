//! OpenAI Chat Completions (`/v1/chat/completions`) codec. Tool calls live in
//! `tool_calls` / `role:"tool"` messages with JSON-string arguments, so this is
//! the most structurally distant from the IR.

use std::collections::HashMap;

use axum::http::{header, HeaderMap, HeaderValue};
use serde_json::{json, Map, Value};

use crate::{
    errors::GatewayError,
    sdk::{
        codec::{
            anthropic::{strip_known, take_string},
            ir::{
                BlockStart, CacheMarkers, ChatRequest, ChatResponse, ContentBlock, Effort,
                ImageSource, Message, ReasoningConfig, ResponseFormat, Role, StopReason,
                StreamEvent, ToolChoice, ToolDef, Usage,
            },
            stream::{sse_frame, SseEvent, StreamParser, StreamRenderer},
            ProtocolCodec, RequestCtx,
        },
        router::Deployment,
    },
};

const KNOWN_REQUEST_KEYS: &[&str] = &[
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
    "stream_options",
];

pub struct OpenAiChatCodec;

impl ProtocolCodec for OpenAiChatCodec {
    fn parse_request(&self, body: Value) -> Result<ChatRequest, GatewayError> {
        let Value::Object(mut obj) = body else {
            return Err(GatewayError::InvalidJsonMessage(
                "request body must be a JSON object".to_owned(),
            ));
        };

        let model = take_string(&mut obj, "model").unwrap_or_default();

        // Hoist system/developer messages into IR `system`; everything else maps
        // to IR messages.
        let mut system = Vec::new();
        let mut messages = Vec::new();
        if let Some(Value::Array(arr)) = obj.remove("messages") {
            for m in &arr {
                let Some(mo) = m.as_object() else { continue };
                let role = mo.get("role").and_then(Value::as_str).unwrap_or("user");
                match role {
                    "system" | "developer" => {
                        if let Some(text) = content_to_text(mo.get("content")) {
                            system.push(ContentBlock::Text { text });
                        }
                    }
                    "assistant" => messages.push(parse_assistant(mo)),
                    "tool" => messages.push(parse_tool_message(mo)),
                    _ => messages.push(parse_user(mo)),
                }
            }
        }

        let tools = match obj.remove("tools") {
            Some(Value::Array(arr)) => arr.iter().filter_map(tool_from_openai).collect(),
            _ => Vec::new(),
        };

        let tool_choice = obj.remove("tool_choice").and_then(parse_tool_choice);

        let stop = match obj.remove("stop") {
            Some(Value::String(s)) => vec![s],
            Some(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
            _ => Vec::new(),
        };

        let max_tokens = obj
            .remove("max_tokens")
            .or_else(|| obj.remove("max_completion_tokens"))
            .and_then(|v| v.as_u64());

        let parallel_tool_calls = obj.remove("parallel_tool_calls").and_then(|v| v.as_bool());
        let response_format = obj
            .remove("response_format")
            .and_then(response_format_from_openai);
        let reasoning = obj
            .remove("reasoning_effort")
            .and_then(|v| v.as_str().and_then(Effort::parse))
            .map(|e| ReasoningConfig {
                effort: Some(e),
                budget_tokens: None,
            });

        let req = ChatRequest {
            model,
            system,
            messages,
            tools,
            // OpenAI prefix caching is automatic; nothing to carry from the wire.
            cache: CacheMarkers::default(),
            tool_choice,
            parallel_tool_calls,
            response_format,
            reasoning,
            max_tokens,
            temperature: obj.remove("temperature").and_then(|v| v.as_f64()),
            top_p: obj.remove("top_p").and_then(|v| v.as_f64()),
            stop,
            stream: obj
                .remove("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            extra: strip_known(obj, KNOWN_REQUEST_KEYS),
        };
        Ok(req)
    }

    fn render_request(&self, req: &ChatRequest) -> Result<Value, GatewayError> {
        let mut messages: Vec<Value> = Vec::new();
        if !req.system.is_empty() {
            messages.push(json!({"role": "system", "content": join_text(&req.system)}));
        }
        for msg in &req.messages {
            flatten_message(msg, &mut messages);
        }

        let mut obj = Map::new();
        obj.insert("model".to_owned(), json!(req.model));
        obj.insert("messages".to_owned(), Value::Array(messages));
        let function_tools: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| t.builtin.is_none())
            .map(tool_to_openai)
            .collect();
        let has_tools = !function_tools.is_empty();
        if has_tools {
            obj.insert("tools".to_owned(), Value::Array(function_tools));
        }
        if let Some(tc) = &req.tool_choice {
            obj.insert("tool_choice".to_owned(), tool_choice_to_openai(tc));
        }
        if let Some(parallel) = req.parallel_tool_calls {
            if has_tools {
                obj.insert("parallel_tool_calls".to_owned(), json!(parallel));
            }
        }
        if let Some(rf) = &req.response_format {
            obj.insert("response_format".to_owned(), response_format_to_openai(rf));
        }
        if let Some(r) = &req.reasoning {
            obj.insert(
                "reasoning_effort".to_owned(),
                json!(r.derived_effort().as_str()),
            );
        }
        if let Some(m) = req.max_tokens {
            obj.insert("max_tokens".to_owned(), json!(m));
        }
        if let Some(t) = req.temperature {
            obj.insert("temperature".to_owned(), json!(t));
        }
        if let Some(p) = req.top_p {
            obj.insert("top_p".to_owned(), json!(p));
        }
        if !req.stop.is_empty() {
            obj.insert("stop".to_owned(), json!(req.stop));
        }
        if req.stream {
            obj.insert("stream".to_owned(), json!(true));
            obj.insert("stream_options".to_owned(), json!({"include_usage": true}));
        }
        Ok(Value::Object(obj))
    }

    fn parse_response(&self, body: Value) -> Result<ChatResponse, GatewayError> {
        let obj = body.as_object().ok_or_else(|| {
            GatewayError::InvalidJsonMessage("response body must be a JSON object".to_owned())
        })?;
        let choice = obj
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        let message = choice.and_then(|c| c.get("message"));

        let mut content = Vec::new();
        if let Some(text) = message
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
        {
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text: text.to_owned(),
                });
            }
        }
        if let Some(tcs) = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for tc in tcs {
                if let Some(block) = tool_call_to_block(tc) {
                    content.push(block);
                }
            }
        }

        Ok(ChatResponse {
            id: obj
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            model: obj
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            content,
            stop_reason: choice
                .and_then(|c| c.get("finish_reason"))
                .and_then(Value::as_str)
                .map(StopReason::from_openai),
            usage: usage_from_openai(obj.get("usage")),
        })
    }

    fn render_response(
        &self,
        resp: &ChatResponse,
        ctx: &RequestCtx,
    ) -> Result<Value, GatewayError> {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for block in &resp.content {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::Thinking { text: t, .. } => reasoning.push_str(t),
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": value_to_args(input)},
                    }));
                }
                _ => {}
            }
        }

        let mut message = Map::new();
        message.insert("role".to_owned(), json!("assistant"));
        message.insert(
            "content".to_owned(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        if !reasoning.is_empty() {
            message.insert("reasoning_content".to_owned(), json!(reasoning));
        }
        if !tool_calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        }

        let id = if resp.id.is_empty() {
            "chatcmpl-litellm".to_owned()
        } else {
            resp.id.clone()
        };
        Ok(json!({
            "id": id,
            "object": "chat.completion",
            "created": 0,
            "model": ctx.model,
            "choices": [{
                "index": 0,
                "message": Value::Object(message),
                "finish_reason": resp
                    .stop_reason
                    .as_ref()
                    .map(StopReason::to_openai)
                    .unwrap_or_else(|| "stop".to_owned()),
            }],
            "usage": openai_usage(&resp.usage),
        }))
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(OpenAiChatStreamParser::default())
    }

    fn stream_renderer(&self, ctx: &RequestCtx) -> Box<dyn StreamRenderer> {
        Box::new(OpenAiChatStreamRenderer {
            model: ctx.model.clone(),
            id: String::new(),
            role_sent: false,
            tool_index: HashMap::new(),
            next_tool: 0,
            done_sent: false,
        })
    }

    fn outbound_headers(
        &self,
        deployment: &Deployment,
        _inbound: &HeaderMap,
    ) -> Result<HeaderMap, GatewayError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", deployment.api_key))
                .map_err(|_| GatewayError::InvalidConfig("invalid api_key".to_owned()))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(headers)
    }

    fn response_headers(&self, upstream: &HeaderMap, stream: bool) -> HeaderMap {
        openai_response_headers(upstream, stream)
    }
}

pub(crate) fn openai_response_headers(upstream: &HeaderMap, stream: bool) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let content_type = if stream {
        HeaderValue::from_static("text/event-stream")
    } else {
        upstream
            .get(header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("application/json"))
    };
    headers.insert(header::CONTENT_TYPE, content_type);
    if let Some(request_id) = upstream.get("x-request-id").cloned() {
        headers.insert("x-request-id", request_id);
    }
    headers
}

// ---- request message mapping ----------------------------------------------

fn content_to_text(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let mut text = String::new();
            for part in arr {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            Some(text)
        }
        _ => None,
    }
}

fn parse_user(mo: &Map<String, Value>) -> Message {
    let content = match mo.get("content") {
        Some(Value::String(s)) => vec![ContentBlock::Text { text: s.clone() }],
        Some(Value::Array(arr)) => arr.iter().filter_map(part_to_block).collect(),
        _ => Vec::new(),
    };
    Message {
        role: Role::User,
        content,
    }
}

fn parse_assistant(mo: &Map<String, Value>) -> Message {
    let mut content = Vec::new();
    if let Some(text) = mo.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: text.to_owned(),
            });
        }
    }
    if let Some(tcs) = mo.get("tool_calls").and_then(Value::as_array) {
        for tc in tcs {
            if let Some(block) = tool_call_to_block(tc) {
                content.push(block);
            }
        }
    }
    Message {
        role: Role::Assistant,
        content,
    }
}

fn parse_tool_message(mo: &Map<String, Value>) -> Message {
    let tool_use_id = mo
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let text = content_to_text(mo.get("content")).unwrap_or_default();
    Message {
        role: Role::Tool,
        content: vec![ContentBlock::ToolResult {
            tool_use_id,
            content: vec![ContentBlock::Text { text }],
            is_error: false,
        }],
    }
}

fn part_to_block(part: &Value) -> Option<ContentBlock> {
    let obj = part.as_object()?;
    match obj.get("type").and_then(Value::as_str) {
        Some("text") => Some(ContentBlock::Text {
            text: obj.get("text").and_then(Value::as_str)?.to_owned(),
        }),
        Some("image_url") => {
            let url = obj
                .get("image_url")
                .and_then(|iu| iu.get("url"))
                .and_then(Value::as_str)?;
            Some(ContentBlock::Image {
                source: data_url_to_source(url),
            })
        }
        _ => None,
    }
}

fn data_url_to_source(url: &str) -> ImageSource {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(",") {
            let media_type = meta.split(';').next().unwrap_or("image/png").to_owned();
            return ImageSource::Base64 {
                media_type,
                data: data.to_owned(),
            };
        }
    }
    ImageSource::Url(url.to_owned())
}

fn source_to_data_url(source: &ImageSource) -> String {
    match source {
        ImageSource::Url(url) => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    }
}

fn tool_call_to_block(tc: &Value) -> Option<ContentBlock> {
    let func = tc.get("function")?;
    let args = func
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let input = serde_json::from_str(args).unwrap_or_else(|_| json!(args));
    Some(ContentBlock::ToolUse {
        id: tc
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        name: func
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        input,
    })
}

/// Flatten one IR message into one or more OpenAI messages. Tool results become
/// separate `role:"tool"` messages; tool uses become `tool_calls` on assistant.
fn flatten_message(msg: &Message, out: &mut Vec<Value>) {
    // Tool-result blocks always become standalone tool messages, regardless of
    // the IR role they were grouped under (Anthropic nests them in user turns).
    for block in &msg.content {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            out.push(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": join_text(content),
            }));
        }
    }

    match msg.role {
        Role::Assistant => {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::ToolUse { id, name, input } => tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": value_to_args(input)},
                    })),
                    _ => {}
                }
            }
            let mut m = Map::new();
            m.insert("role".to_owned(), json!("assistant"));
            m.insert(
                "content".to_owned(),
                if text.is_empty() {
                    Value::Null
                } else {
                    json!(text)
                },
            );
            if !tool_calls.is_empty() {
                m.insert("tool_calls".to_owned(), Value::Array(tool_calls));
            }
            // Skip an entirely empty assistant turn.
            if !text.is_empty() || m.contains_key("tool_calls") {
                out.push(Value::Object(m));
            }
        }
        _ => {
            let content = render_user_content(&msg.content);
            if !is_empty_content(&content) {
                out.push(json!({"role": "user", "content": content}));
            }
        }
    }
}

fn render_user_content(blocks: &[ContentBlock]) -> Value {
    let has_image = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    if !has_image {
        return json!(join_text(blocks));
    }
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => parts.push(json!({"type": "text", "text": text})),
            ContentBlock::Image { source } => parts.push(json!({
                "type": "image_url",
                "image_url": {"url": source_to_data_url(source)},
            })),
            _ => {}
        }
    }
    Value::Array(parts)
}

fn is_empty_content(v: &Value) -> bool {
    match v {
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        _ => true,
    }
}

pub(crate) fn join_text(blocks: &[ContentBlock]) -> String {
    let mut text = String::new();
    for block in blocks {
        if let ContentBlock::Text { text: t } = block {
            text.push_str(t);
        }
    }
    text
}

pub(crate) fn value_to_args(input: &Value) -> String {
    match input {
        Value::Null => "{}".to_owned(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn tool_from_openai(v: &Value) -> Option<ToolDef> {
    let func = v.get("function")?;
    Some(ToolDef {
        name: func.get("name").and_then(Value::as_str)?.to_owned(),
        description: func
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        parameters: func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
        builtin: None,
    })
}

fn response_format_from_openai(v: Value) -> Option<ResponseFormat> {
    let obj = v.as_object()?;
    match obj.get("type").and_then(Value::as_str) {
        Some("json_object") => Some(ResponseFormat::JsonObject),
        Some("json_schema") => {
            let js = obj.get("json_schema")?;
            Some(ResponseFormat::JsonSchema {
                name: js
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("response")
                    .to_owned(),
                schema: js
                    .get("schema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
                strict: js.get("strict").and_then(Value::as_bool).unwrap_or(true),
            })
        }
        _ => None,
    }
}

fn response_format_to_openai(rf: &ResponseFormat) -> Value {
    match rf {
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => json!({
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": strict},
        }),
    }
}

fn tool_to_openai(tool: &ToolDef) -> Value {
    let mut func = json!({"name": tool.name, "parameters": tool.parameters});
    if let Some(desc) = &tool.description {
        func["description"] = json!(desc);
    }
    json!({"type": "function", "function": func})
}

fn parse_tool_choice(v: Value) -> Option<ToolChoice> {
    match v {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Value::Object(o) => o
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Tool(n.to_owned())),
        _ => None,
    }
}

fn tool_choice_to_openai(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({"type": "function", "function": {"name": name}}),
    }
}

fn usage_from_openai(v: Option<&Value>) -> Usage {
    let Some(obj) = v.and_then(Value::as_object) else {
        return Usage::default();
    };
    // OpenAI's `prompt_tokens` is already inclusive of cached tokens.
    let cached = obj
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input_tokens: obj
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: obj
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
    }
}

/// Build an OpenAI Chat `usage` object, adding `prompt_tokens_details.cached_tokens`
/// only when a cache hit occurred (keeps zero-cache output byte-identical).
fn openai_usage(u: &Usage) -> Value {
    let mut usage = json!({
        "prompt_tokens": u.input_tokens,
        "completion_tokens": u.output_tokens,
        "total_tokens": u.input_tokens + u.output_tokens,
    });
    if u.cache_read_input_tokens > 0 {
        usage["prompt_tokens_details"] = json!({"cached_tokens": u.cache_read_input_tokens});
    }
    usage
}

// ---- streaming ------------------------------------------------------------

#[derive(Default)]
struct OpenAiChatStreamParser {
    started: bool,
    text_index: Option<usize>,
    think_index: Option<usize>,
    tool_indices: HashMap<u64, usize>,
    next_index: usize,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
    blocks_closed: bool,
    delta_emitted: bool,
    message_stopped: bool,
}

impl OpenAiChatStreamParser {
    fn alloc(&mut self) -> usize {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    fn open_indices(&self) -> Vec<usize> {
        let mut idxs: Vec<usize> = self
            .text_index
            .into_iter()
            .chain(self.think_index)
            .chain(self.tool_indices.values().copied())
            .collect();
        idxs.sort_unstable();
        idxs
    }

    fn finalize(&mut self) -> Vec<StreamEvent> {
        if !self.started {
            return Vec::new();
        }
        let mut out = Vec::new();
        if !self.blocks_closed {
            self.blocks_closed = true;
            for index in self.open_indices() {
                out.push(StreamEvent::ContentBlockStop { index });
            }
        }
        if !self.delta_emitted {
            self.delta_emitted = true;
            out.push(StreamEvent::MessageDelta {
                stop_reason: self.stop_reason.take(),
                usage: self.usage.take(),
            });
        }
        if !self.message_stopped {
            self.message_stopped = true;
            out.push(StreamEvent::MessageStop);
        }
        out
    }
}

impl StreamParser for OpenAiChatStreamParser {
    fn push(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, GatewayError> {
        if event.data.trim() == "[DONE]" {
            return Ok(self.finalize());
        }
        if event.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let data: Value = serde_json::from_str(&event.data)
            .map_err(|e| GatewayError::InvalidJsonMessage(e.to_string()))?;

        let mut out = Vec::new();
        if let Some(u) = data.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(usage_from_openai(Some(u)));
        }

        if !self.started {
            self.started = true;
            out.push(StreamEvent::MessageStart {
                id: data
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                model: data
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }

        let Some(choices) = data.get("choices").and_then(Value::as_array) else {
            return Ok(out);
        };
        for choice in choices {
            let delta = choice.get("delta");
            if let Some(text) = delta.and_then(|d| d.get("content")).and_then(Value::as_str) {
                if !text.is_empty() {
                    let index = match self.text_index {
                        Some(i) => i,
                        None => {
                            let i = self.alloc();
                            self.text_index = Some(i);
                            out.push(StreamEvent::ContentBlockStart {
                                index: i,
                                block: BlockStart::Text,
                            });
                            i
                        }
                    };
                    out.push(StreamEvent::TextDelta {
                        index,
                        text: text.to_owned(),
                    });
                }
            }
            if let Some(rt) = delta
                .and_then(|d| d.get("reasoning_content"))
                .and_then(Value::as_str)
            {
                if !rt.is_empty() {
                    let index = match self.think_index {
                        Some(i) => i,
                        None => {
                            let i = self.alloc();
                            self.think_index = Some(i);
                            out.push(StreamEvent::ContentBlockStart {
                                index: i,
                                block: BlockStart::Thinking,
                            });
                            i
                        }
                    };
                    out.push(StreamEvent::ThinkingDelta {
                        index,
                        text: rt.to_owned(),
                    });
                }
            }
            if let Some(tcs) = delta
                .and_then(|d| d.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tc in tcs {
                    let j = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                    let index = match self.tool_indices.get(&j) {
                        Some(i) => *i,
                        None => {
                            let i = self.alloc();
                            self.tool_indices.insert(j, i);
                            out.push(StreamEvent::ContentBlockStart {
                                index: i,
                                block: BlockStart::ToolUse {
                                    id: tc
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .unwrap_or_default()
                                        .to_owned(),
                                    name: tc
                                        .get("function")
                                        .and_then(|f| f.get("name"))
                                        .and_then(Value::as_str)
                                        .unwrap_or_default()
                                        .to_owned(),
                                },
                            });
                            i
                        }
                    };
                    if let Some(args) = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        if !args.is_empty() {
                            out.push(StreamEvent::ToolUseInputDelta {
                                index,
                                partial_json: args.to_owned(),
                            });
                        }
                    }
                }
            }
            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                self.stop_reason = Some(StopReason::from_openai(fr));
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        self.finalize()
    }
}

struct OpenAiChatStreamRenderer {
    model: String,
    id: String,
    role_sent: bool,
    tool_index: HashMap<usize, usize>,
    next_tool: usize,
    done_sent: bool,
}

impl OpenAiChatStreamRenderer {
    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> Vec<u8> {
        let data = json!({
            "id": if self.id.is_empty() { "chatcmpl-litellm" } else { &self.id },
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        });
        sse_frame(None, &data.to_string())
    }
}

impl StreamRenderer for OpenAiChatStreamRenderer {
    fn push(&mut self, event: &StreamEvent) -> Vec<u8> {
        match event {
            StreamEvent::MessageStart { id, .. } => {
                self.id = if id.is_empty() {
                    "chatcmpl-litellm".to_owned()
                } else {
                    id.clone()
                };
                self.role_sent = true;
                self.chunk(json!({"role": "assistant", "content": ""}), None)
            }
            StreamEvent::ContentBlockStart {
                index,
                block: BlockStart::ToolUse { id, name },
            } => {
                let j = self.next_tool;
                self.next_tool += 1;
                self.tool_index.insert(*index, j);
                self.chunk(
                    json!({"tool_calls": [{
                        "index": j,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]}),
                    None,
                )
            }
            StreamEvent::ContentBlockStart { .. } => Vec::new(),
            StreamEvent::TextDelta { text, .. } => self.chunk(json!({"content": text}), None),
            StreamEvent::ThinkingDelta { text, .. } => {
                self.chunk(json!({"reasoning_content": text}), None)
            }
            StreamEvent::ToolUseInputDelta {
                index,
                partial_json,
            } => {
                let j = self.tool_index.get(index).copied().unwrap_or(0);
                self.chunk(
                    json!({"tool_calls": [{
                        "index": j,
                        "function": {"arguments": partial_json},
                    }]}),
                    None,
                )
            }
            StreamEvent::ContentBlockStop { .. } => Vec::new(),
            StreamEvent::MessageDelta { stop_reason, usage } => {
                let reason = stop_reason
                    .as_ref()
                    .map(StopReason::to_openai)
                    .unwrap_or_else(|| "stop".to_owned());
                let mut out = self.chunk(json!({}), Some(&reason));
                if let Some(u) = usage {
                    let data = json!({
                        "id": if self.id.is_empty() { "chatcmpl-litellm" } else { &self.id },
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": self.model,
                        "choices": [],
                        "usage": openai_usage(u),
                    });
                    out.extend(sse_frame(None, &data.to_string()));
                }
                out
            }
            StreamEvent::MessageStop => {
                self.done_sent = true;
                sse_frame(None, "[DONE]")
            }
        }
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.done_sent {
            Vec::new()
        } else {
            self.done_sent = true;
            sse_frame(None, "[DONE]")
        }
    }
}
