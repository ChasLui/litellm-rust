//! OpenAI Responses (`/v1/responses`) codec.
//!
//! The Responses API differs from Chat Completions: system → `instructions`,
//! `messages` → an `input` array of items, tool calls/results are top-level
//! `function_call` / `function_call_output` items, and tools are flat.

use std::collections::HashSet;

use axum::http::{header, HeaderMap, HeaderName, HeaderValue};
use serde_json::{json, Map, Value};

use crate::{
    errors::GatewayError,
    sdk::{
        codec::{
            anthropic::{strip_known, take_string},
            ir::{
                BlockStart, ChatRequest, ChatResponse, ContentBlock, Effort, Message,
                ReasoningConfig, ResponseFormat, Role, StopReason, StreamEvent, ToolChoice,
                ToolDef, Usage,
            },
            openai_chat::{join_text, openai_response_headers, value_to_args},
            stream::{sse_frame, SseEvent, StreamParser, StreamRenderer},
            ProtocolCodec, RequestCtx,
        },
        router::Deployment,
    },
};

const FORWARDED_HEADERS: &[&str] = &[
    "accept",
    "originator",
    "session-id",
    "thread-id",
    "x-client-request-id",
    "x-codex-beta-features",
    "x-codex-turn-metadata",
    "x-codex-window-id",
];

const KNOWN_REQUEST_KEYS: &[&str] = &[
    "model",
    "instructions",
    "input",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "max_output_tokens",
    "temperature",
    "top_p",
    "stream",
];

pub struct OpenAiResponsesCodec;

impl ProtocolCodec for OpenAiResponsesCodec {
    fn parse_request(&self, body: Value) -> Result<ChatRequest, GatewayError> {
        let Value::Object(mut obj) = body else {
            return Err(GatewayError::InvalidJsonMessage(
                "request body must be a JSON object".to_owned(),
            ));
        };

        let model = take_string(&mut obj, "model").unwrap_or_default();
        let system = match take_string(&mut obj, "instructions") {
            Some(s) => vec![ContentBlock::Text { text: s }],
            None => Vec::new(),
        };

        let messages = match obj.remove("input") {
            Some(Value::String(s)) => vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: s }],
            }],
            Some(Value::Array(arr)) => parse_input_items(&arr),
            _ => Vec::new(),
        };

        let tools = match obj.remove("tools") {
            Some(Value::Array(arr)) => arr.iter().filter_map(tool_from_responses).collect(),
            _ => Vec::new(),
        };

        let parallel_tool_calls = obj.remove("parallel_tool_calls").and_then(|v| v.as_bool());
        let response_format = obj
            .remove("text")
            .and_then(|t| t.get("format").cloned())
            .and_then(response_format_from_responses);
        let reasoning = obj.remove("reasoning").and_then(|r| {
            r.get("effort")
                .and_then(Value::as_str)
                .and_then(Effort::parse)
                .map(|e| ReasoningConfig {
                    effort: Some(e),
                    budget_tokens: None,
                })
        });

        let req = ChatRequest {
            model,
            system,
            messages,
            tools,
            tool_choice: obj.remove("tool_choice").and_then(parse_tool_choice),
            parallel_tool_calls,
            response_format,
            reasoning,
            max_tokens: obj.remove("max_output_tokens").and_then(|v| v.as_u64()),
            temperature: obj.remove("temperature").and_then(|v| v.as_f64()),
            top_p: obj.remove("top_p").and_then(|v| v.as_f64()),
            stop: Vec::new(),
            stream: obj
                .remove("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            extra: strip_known(obj, KNOWN_REQUEST_KEYS),
        };
        Ok(req)
    }

    fn render_request(&self, req: &ChatRequest) -> Result<Value, GatewayError> {
        let mut obj = Map::new();
        obj.insert("model".to_owned(), json!(req.model));
        if !req.system.is_empty() {
            obj.insert("instructions".to_owned(), json!(join_text(&req.system)));
        }
        let mut input: Vec<Value> = Vec::new();
        for msg in &req.messages {
            flatten_message(msg, &mut input);
        }
        obj.insert("input".to_owned(), Value::Array(input));
        let function_tools: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| t.builtin.is_none())
            .map(tool_to_responses)
            .collect();
        let has_tools = !function_tools.is_empty();
        if has_tools {
            obj.insert("tools".to_owned(), Value::Array(function_tools));
        }
        if let Some(tc) = &req.tool_choice {
            obj.insert("tool_choice".to_owned(), tool_choice_to_responses(tc));
        }
        if let Some(parallel) = req.parallel_tool_calls {
            if has_tools {
                obj.insert("parallel_tool_calls".to_owned(), json!(parallel));
            }
        }
        if let Some(rf) = &req.response_format {
            obj.insert(
                "text".to_owned(),
                json!({"format": response_format_to_responses(rf)}),
            );
        }
        if let Some(r) = &req.reasoning {
            obj.insert(
                "reasoning".to_owned(),
                json!({"effort": r.derived_effort().as_str(), "summary": "auto"}),
            );
        }
        if let Some(m) = req.max_tokens {
            obj.insert("max_output_tokens".to_owned(), json!(m));
        }
        if let Some(t) = req.temperature {
            obj.insert("temperature".to_owned(), json!(t));
        }
        if let Some(p) = req.top_p {
            obj.insert("top_p".to_owned(), json!(p));
        }
        if req.stream {
            obj.insert("stream".to_owned(), json!(true));
        }
        Ok(Value::Object(obj))
    }

    fn parse_response(&self, body: Value) -> Result<ChatResponse, GatewayError> {
        let obj = body.as_object().ok_or_else(|| {
            GatewayError::InvalidJsonMessage("response body must be a JSON object".to_owned())
        })?;

        let mut content = Vec::new();
        let mut saw_tool = false;
        if let Some(output) = obj.get("output").and_then(Value::as_array) {
            for item in output {
                match item.get("type").and_then(Value::as_str) {
                    Some("message") => {
                        if let Some(parts) = item.get("content").and_then(Value::as_array) {
                            for part in parts {
                                if let Some(text) = part
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .filter(|t| !t.is_empty())
                                {
                                    content.push(ContentBlock::Text {
                                        text: text.to_owned(),
                                    });
                                }
                            }
                        }
                    }
                    Some("function_call") => {
                        saw_tool = true;
                        content.push(function_call_to_block(item));
                    }
                    Some("reasoning") => {
                        if let Some(text) = reasoning_text(item) {
                            content.push(ContentBlock::Thinking {
                                text,
                                signature: None,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        let stop_reason = match obj.get("status").and_then(Value::as_str) {
            Some("incomplete") => Some(StopReason::MaxTokens),
            _ if saw_tool => Some(StopReason::ToolUse),
            _ => Some(StopReason::EndTurn),
        };

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
            stop_reason,
            usage: usage_from_responses(obj.get("usage")),
        })
    }

    fn render_response(
        &self,
        resp: &ChatResponse,
        ctx: &RequestCtx,
    ) -> Result<Value, GatewayError> {
        let mut output: Vec<Value> = Vec::new();
        let mut text = String::new();
        for block in &resp.content {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::ToolUse { id, name, input } => output.push(json!({
                    "type": "function_call",
                    "id": format!("fc_{id}"),
                    "call_id": id,
                    "name": name,
                    "arguments": value_to_args(input),
                })),
                ContentBlock::Thinking { text: t, .. } => output.push(json!({
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": t}],
                })),
                _ => {}
            }
        }
        if !text.is_empty() {
            // Message item first to match OpenAI ordering.
            output.insert(
                0,
                json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }),
            );
        }

        let id = if resp.id.is_empty() {
            "resp_litellm".to_owned()
        } else {
            resp.id.clone()
        };
        let status = match resp.stop_reason {
            Some(StopReason::MaxTokens) => "incomplete",
            _ => "completed",
        };
        Ok(json!({
            "id": id,
            "object": "response",
            "model": ctx.model,
            "status": status,
            "output": output,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            },
        }))
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(ResponsesStreamParser::default())
    }

    fn stream_renderer(&self, ctx: &RequestCtx) -> Box<dyn StreamRenderer> {
        Box::new(ResponsesStreamRenderer {
            model: ctx.model.clone(),
            id: String::new(),
            next_oi: 0,
            stop_reason: None,
            usage: None,
        })
    }

    fn outbound_headers(
        &self,
        deployment: &Deployment,
        inbound: &HeaderMap,
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
        for name in FORWARDED_HEADERS {
            if let Some(value) = inbound.get(*name) {
                if let Ok(header_name) = HeaderName::from_bytes(name.as_bytes()) {
                    headers.insert(header_name, value.clone());
                }
            }
        }
        Ok(headers)
    }

    fn response_headers(&self, upstream: &HeaderMap, stream: bool) -> HeaderMap {
        openai_response_headers(upstream, stream)
    }
}

// ---- request item mapping -------------------------------------------------

fn parse_input_items(arr: &[Value]) -> Vec<Message> {
    let mut messages = Vec::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        match obj.get("type").and_then(Value::as_str) {
            Some("function_call") => messages.push(Message {
                role: Role::Assistant,
                content: vec![function_call_to_block(item)],
            }),
            Some("function_call_output") => {
                let call_id = obj
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let output = match obj.get("output") {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                messages.push(Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: call_id,
                        content: vec![ContentBlock::Text { text: output }],
                        is_error: false,
                    }],
                });
            }
            // A message item (explicit "message" type or a bare {role, content}).
            _ => {
                let role = match obj.get("role").and_then(Value::as_str) {
                    Some("assistant") => Role::Assistant,
                    Some("system") | Some("developer") => Role::System,
                    _ => Role::User,
                };
                let content = match obj.get("content") {
                    Some(Value::String(s)) => vec![ContentBlock::Text { text: s.clone() }],
                    Some(Value::Array(parts)) => {
                        parts.iter().filter_map(content_part_to_block).collect()
                    }
                    _ => Vec::new(),
                };
                messages.push(Message { role, content });
            }
        }
    }
    messages
}

fn content_part_to_block(part: &Value) -> Option<ContentBlock> {
    let obj = part.as_object()?;
    match obj.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") | Some("text") => Some(ContentBlock::Text {
            text: obj.get("text").and_then(Value::as_str)?.to_owned(),
        }),
        _ => None,
    }
}

fn function_call_to_block(item: &Value) -> ContentBlock {
    let args = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let input = serde_json::from_str(args).unwrap_or_else(|_| json!(args));
    ContentBlock::ToolUse {
        id: item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        input,
    }
}

fn reasoning_text(item: &Value) -> Option<String> {
    let summary = item.get("summary").and_then(Value::as_array)?;
    let mut text = String::new();
    for part in summary {
        if let Some(t) = part.get("text").and_then(Value::as_str) {
            text.push_str(t);
        }
    }
    (!text.is_empty()).then_some(text)
}

fn flatten_message(msg: &Message, out: &mut Vec<Value>) {
    // Tool results become standalone function_call_output items.
    for block in &msg.content {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            out.push(json!({
                "type": "function_call_output",
                "call_id": tool_use_id,
                "output": join_text(content),
            }));
        }
    }

    match msg.role {
        Role::Assistant => {
            let mut text = String::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::ToolUse { id, name, input } => out.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": value_to_args(input),
                    })),
                    _ => {}
                }
            }
            if !text.is_empty() {
                out.push(json!({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }));
            }
        }
        _ => {
            let mut parts: Vec<Value> = Vec::new();
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    parts.push(json!({"type": "input_text", "text": text}));
                }
            }
            if !parts.is_empty() {
                out.push(json!({"role": "user", "content": parts}));
            }
        }
    }
}

fn tool_from_responses(v: &Value) -> Option<ToolDef> {
    let obj = v.as_object()?;
    // Built-in tools (web_search, file_search, code_interpreter, image_generation,
    // computer_use, mcp, …) carry a non-"function" type.
    if let Some(t) = obj.get("type").and_then(Value::as_str) {
        if t != "function" {
            return Some(ToolDef {
                name: obj
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(t)
                    .to_owned(),
                description: None,
                parameters: json!({"type": "object"}),
                builtin: Some(v.clone()),
            });
        }
    }
    // Function tool — flat (name at top level), but tolerate the nested Chat shape.
    let name = obj
        .get("name")
        .or_else(|| obj.get("function").and_then(|f| f.get("name")))
        .and_then(Value::as_str)?;
    let description = obj
        .get("description")
        .or_else(|| obj.get("function").and_then(|f| f.get("description")))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let parameters = obj
        .get("parameters")
        .or_else(|| obj.get("function").and_then(|f| f.get("parameters")))
        .cloned()
        .unwrap_or_else(|| json!({"type": "object"}));
    Some(ToolDef {
        name: name.to_owned(),
        description,
        parameters,
        builtin: None,
    })
}

fn response_format_from_responses(v: Value) -> Option<ResponseFormat> {
    let obj = v.as_object()?;
    match obj.get("type").and_then(Value::as_str) {
        Some("json_object") => Some(ResponseFormat::JsonObject),
        Some("json_schema") => Some(ResponseFormat::JsonSchema {
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("response")
                .to_owned(),
            schema: obj
                .get("schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"})),
            strict: obj.get("strict").and_then(Value::as_bool).unwrap_or(true),
        }),
        _ => None,
    }
}

fn response_format_to_responses(rf: &ResponseFormat) -> Value {
    match rf {
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => json!({
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": strict,
        }),
    }
}

fn tool_to_responses(tool: &ToolDef) -> Value {
    let mut o = json!({"type": "function", "name": tool.name, "parameters": tool.parameters});
    if let Some(desc) = &tool.description {
        o["description"] = json!(desc);
    }
    o
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
            .get("name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Tool(n.to_owned())),
        _ => None,
    }
}

fn tool_choice_to_responses(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({"type": "function", "name": name}),
    }
}

fn usage_from_responses(v: Option<&Value>) -> Usage {
    let Some(obj) = v.and_then(Value::as_object) else {
        return Usage::default();
    };
    Usage {
        input_tokens: obj.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
        output_tokens: obj
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

// ---- streaming ------------------------------------------------------------

#[derive(Default)]
struct ResponsesStreamParser {
    started: bool,
    opened: HashSet<usize>,
    saw_tool: bool,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
    message_stopped: bool,
}

impl ResponsesStreamParser {
    fn finalize(&mut self) -> Vec<StreamEvent> {
        if !self.started || self.message_stopped {
            return Vec::new();
        }
        self.message_stopped = true;
        let mut out = Vec::new();
        let mut open: Vec<usize> = self.opened.drain().collect();
        open.sort_unstable();
        for index in open {
            out.push(StreamEvent::ContentBlockStop { index });
        }
        let stop = self.stop_reason.take().or(if self.saw_tool {
            Some(StopReason::ToolUse)
        } else {
            Some(StopReason::EndTurn)
        });
        out.push(StreamEvent::MessageDelta {
            stop_reason: stop,
            usage: self.usage.take(),
        });
        out.push(StreamEvent::MessageStop);
        out
    }
}

fn output_index(data: &Value) -> usize {
    data.get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

impl StreamParser for ResponsesStreamParser {
    fn push(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, GatewayError> {
        if event.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let data: Value = serde_json::from_str(&event.data)
            .map_err(|e| GatewayError::InvalidJsonMessage(e.to_string()))?;
        let t = data.get("type").and_then(Value::as_str).unwrap_or_default();

        Ok(match t {
            "response.created" => {
                self.started = true;
                let resp = data.get("response");
                vec![StreamEvent::MessageStart {
                    id: resp
                        .and_then(|r| r.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    model: resp
                        .and_then(|r| r.get("model"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }]
            }
            "response.output_item.added" => {
                let oi = output_index(&data);
                let item = data.get("item");
                match item.and_then(|i| i.get("type")).and_then(Value::as_str) {
                    Some("function_call") => {
                        self.saw_tool = true;
                        self.opened.insert(oi);
                        vec![StreamEvent::ContentBlockStart {
                            index: oi,
                            block: BlockStart::ToolUse {
                                id: item
                                    .and_then(|i| i.get("call_id"))
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned(),
                                name: item
                                    .and_then(|i| i.get("name"))
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned(),
                            },
                        }]
                    }
                    Some("reasoning") => {
                        self.opened.insert(oi);
                        vec![StreamEvent::ContentBlockStart {
                            index: oi,
                            block: BlockStart::Thinking,
                        }]
                    }
                    _ => Vec::new(),
                }
            }
            "response.content_part.added" => {
                let oi = output_index(&data);
                let is_text = data
                    .get("part")
                    .and_then(|p| p.get("type"))
                    .and_then(Value::as_str)
                    == Some("output_text");
                if is_text && self.opened.insert(oi) {
                    vec![StreamEvent::ContentBlockStart {
                        index: oi,
                        block: BlockStart::Text,
                    }]
                } else {
                    Vec::new()
                }
            }
            "response.output_text.delta" => vec![StreamEvent::TextDelta {
                index: output_index(&data),
                text: data
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }],
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                vec![StreamEvent::ThinkingDelta {
                    index: output_index(&data),
                    text: data
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }]
            }
            "response.function_call_arguments.delta" => vec![StreamEvent::ToolUseInputDelta {
                index: output_index(&data),
                partial_json: data
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }],
            "response.output_item.done" => {
                let oi = output_index(&data);
                if self.opened.remove(&oi) {
                    vec![StreamEvent::ContentBlockStop { index: oi }]
                } else {
                    Vec::new()
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                if t == "response.incomplete" {
                    self.stop_reason = Some(StopReason::MaxTokens);
                }
                self.usage = Some(usage_from_responses(
                    data.get("response").and_then(|r| r.get("usage")),
                ));
                self.finalize()
            }
            _ => Vec::new(),
        })
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        self.finalize()
    }
}

struct ResponsesStreamRenderer {
    model: String,
    id: String,
    next_oi: usize,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
}

impl ResponsesStreamRenderer {
    fn item_id(oi: usize) -> String {
        format!("item_{oi}")
    }

    fn frame(t: &str, data: Value) -> Vec<u8> {
        sse_frame(Some(t), &data.to_string())
    }
}

impl StreamRenderer for ResponsesStreamRenderer {
    fn push(&mut self, event: &StreamEvent) -> Vec<u8> {
        match event {
            StreamEvent::MessageStart { id, .. } => {
                self.id = if id.is_empty() {
                    "resp_litellm".to_owned()
                } else {
                    id.clone()
                };
                Self::frame(
                    "response.created",
                    json!({
                        "type": "response.created",
                        "response": {"id": self.id, "object": "response", "model": self.model, "status": "in_progress"},
                    }),
                )
            }
            StreamEvent::ContentBlockStart { index, block } => {
                let oi = *index;
                self.next_oi = self.next_oi.max(oi + 1);
                let item_id = Self::item_id(oi);
                match block {
                    BlockStart::Text => {
                        let mut out = Self::frame(
                            "response.output_item.added",
                            json!({
                                "type": "response.output_item.added",
                                "output_index": oi,
                                "item": {"type": "message", "id": item_id, "role": "assistant", "content": []},
                            }),
                        );
                        out.extend(Self::frame(
                            "response.content_part.added",
                            json!({
                                "type": "response.content_part.added",
                                "item_id": Self::item_id(oi),
                                "output_index": oi,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": ""},
                            }),
                        ));
                        out
                    }
                    BlockStart::Thinking => Self::frame(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "output_index": oi,
                            "item": {"type": "reasoning", "id": item_id, "summary": []},
                        }),
                    ),
                    BlockStart::ToolUse { id, name } => Self::frame(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "output_index": oi,
                            "item": {"type": "function_call", "id": item_id, "call_id": id, "name": name, "arguments": ""},
                        }),
                    ),
                }
            }
            StreamEvent::TextDelta { index, text } => Self::frame(
                "response.output_text.delta",
                json!({
                    "type": "response.output_text.delta",
                    "item_id": Self::item_id(*index),
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                }),
            ),
            StreamEvent::ThinkingDelta { index, text } => Self::frame(
                "response.reasoning_summary_text.delta",
                json!({
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": Self::item_id(*index),
                    "output_index": index,
                    "delta": text,
                }),
            ),
            StreamEvent::ToolUseInputDelta {
                index,
                partial_json,
            } => Self::frame(
                "response.function_call_arguments.delta",
                json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": Self::item_id(*index),
                    "output_index": index,
                    "delta": partial_json,
                }),
            ),
            StreamEvent::ContentBlockStop { index } => Self::frame(
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "output_index": index,
                }),
            ),
            StreamEvent::MessageDelta { stop_reason, usage } => {
                self.stop_reason = stop_reason.clone();
                self.usage = usage.clone();
                Vec::new()
            }
            StreamEvent::MessageStop => {
                let status = match self.stop_reason {
                    Some(StopReason::MaxTokens) => "incomplete",
                    _ => "completed",
                };
                let usage = self.usage.clone().unwrap_or_default();
                Self::frame(
                    "response.completed",
                    json!({
                        "type": "response.completed",
                        "response": {
                            "id": self.id,
                            "object": "response",
                            "model": self.model,
                            "status": status,
                            "usage": {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "total_tokens": usage.input_tokens + usage.output_tokens,
                            },
                        },
                    }),
                )
            }
        }
    }
}
