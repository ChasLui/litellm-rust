//! Gemini (`generateContent` / `streamGenerateContent`) codec.
//!
//! Gemini differs sharply: roles are `user`/`model`, system goes in
//! `systemInstruction`, tool calls/results are `functionCall`/`functionResponse`
//! parts with *object* (not string) arguments, and function calls carry no id —
//! we key tool results back to calls by function name.

use std::collections::HashMap;

use axum::http::{header, HeaderMap, HeaderValue};
use serde_json::{json, Map, Value};

use crate::{
    errors::GatewayError,
    sdk::{
        codec::{
            ir::{
                BlockStart, CacheMarkers, ChatRequest, ChatResponse, ContentBlock, ImageSource,
                Message, ReasoningConfig, ResponseFormat, Role, StopReason, StreamEvent,
                ToolChoice, ToolDef, Usage,
            },
            stream::{sse_frame, SseEvent, StreamParser, StreamRenderer},
            ProtocolCodec, RequestCtx,
        },
        router::Deployment,
    },
};

pub struct GeminiCodec;

/// Read a field by camelCase or snake_case name (Gemini REST uses camelCase but
/// the proto-JSON form accepts snake_case).
fn field<'a>(obj: &'a Map<String, Value>, camel: &str, snake: &str) -> Option<&'a Value> {
    obj.get(camel).or_else(|| obj.get(snake))
}

impl ProtocolCodec for GeminiCodec {
    fn parse_request(&self, body: Value) -> Result<ChatRequest, GatewayError> {
        let obj = body.as_object().ok_or_else(|| {
            GatewayError::InvalidJsonMessage("request body must be a JSON object".to_owned())
        })?;

        let system = field(obj, "systemInstruction", "system_instruction")
            .map(parse_parts_as_text)
            .filter(|t| !t.is_empty())
            .map(|t| vec![ContentBlock::Text { text: t }])
            .unwrap_or_default();

        let messages = obj
            .get("contents")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(content_to_message).collect())
            .unwrap_or_default();

        let tools = obj
            .get("tools")
            .and_then(Value::as_array)
            .map(|arr| tools_from_gemini(arr))
            .unwrap_or_default();

        let tool_choice = field(obj, "toolConfig", "tool_config")
            .and_then(|tc| {
                field(
                    tc.as_object()?,
                    "functionCallingConfig",
                    "function_calling_config",
                )
            })
            .and_then(parse_tool_choice);

        let gen = field(obj, "generationConfig", "generation_config").and_then(Value::as_object);
        let max_tokens = gen
            .and_then(|g| field(g, "maxOutputTokens", "max_output_tokens"))
            .and_then(Value::as_u64);
        let temperature = gen
            .and_then(|g| g.get("temperature"))
            .and_then(Value::as_f64);
        let top_p = gen
            .and_then(|g| field(g, "topP", "top_p"))
            .and_then(Value::as_f64);
        let stop = gen
            .and_then(|g| field(g, "stopSequences", "stop_sequences"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();

        let response_format = gen.and_then(|g| {
            let mime = field(g, "responseMimeType", "response_mime_type").and_then(Value::as_str);
            if mime != Some("application/json") {
                return None;
            }
            match field(g, "responseJsonSchema", "response_json_schema")
                .or_else(|| field(g, "responseSchema", "response_schema"))
                .cloned()
            {
                Some(schema) => Some(ResponseFormat::JsonSchema {
                    name: "response".to_owned(),
                    schema,
                    strict: true,
                }),
                None => Some(ResponseFormat::JsonObject),
            }
        });

        let reasoning = gen
            .and_then(|g| field(g, "thinkingConfig", "thinking_config"))
            .and_then(Value::as_object)
            .and_then(|tc| {
                let budget = field(tc, "thinkingBudget", "thinking_budget").and_then(Value::as_u64);
                budget.map(|b| ReasoningConfig {
                    effort: None,
                    budget_tokens: Some(b),
                })
            });

        Ok(ChatRequest {
            model: String::new(),
            system,
            messages,
            tools,
            // Gemini implicit caching is automatic; nothing to carry from the wire.
            cache: CacheMarkers::default(),
            tool_choice,
            parallel_tool_calls: None,
            response_format,
            reasoning,
            max_tokens,
            temperature,
            top_p,
            stop,
            stream: false,
            extra: Map::new(),
        })
    }

    fn render_request(&self, req: &ChatRequest) -> Result<Value, GatewayError> {
        let mut obj = Map::new();
        if !req.system.is_empty() {
            let text = join_text(&req.system);
            obj.insert(
                "systemInstruction".to_owned(),
                json!({"parts": [{"text": text}]}),
            );
        }

        let names = tool_name_map(&req.messages);
        // Coalesce consecutive same-role contents: Gemini expects user/model to
        // alternate, but parallel tool results arrive as separate IR `Role::Tool`
        // messages that all map to the "user" role.
        let mut contents: Vec<Value> = Vec::new();
        for msg in &req.messages {
            let Some(content) = message_to_content(msg, &names) else {
                continue;
            };
            match contents.last_mut() {
                Some(last) if last.get("role") == content.get("role") => {
                    if let (Some(Value::Array(dst)), Some(Value::Array(src))) =
                        (last.get_mut("parts"), content.get("parts"))
                    {
                        dst.extend(src.iter().cloned());
                    }
                }
                _ => contents.push(content),
            }
        }
        obj.insert("contents".to_owned(), Value::Array(contents));

        let decls: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| t.builtin.is_none())
            .map(tool_to_gemini)
            .collect();
        if !decls.is_empty() {
            obj.insert(
                "tools".to_owned(),
                json!([{ "functionDeclarations": decls }]),
            );
        }
        if let Some(tc) = &req.tool_choice {
            obj.insert("toolConfig".to_owned(), tool_choice_to_gemini(tc));
        }

        let mut gen = Map::new();
        if let Some(m) = req.max_tokens {
            gen.insert("maxOutputTokens".to_owned(), json!(m));
        }
        if let Some(t) = req.temperature {
            gen.insert("temperature".to_owned(), json!(t));
        }
        if let Some(p) = req.top_p {
            gen.insert("topP".to_owned(), json!(p));
        }
        if !req.stop.is_empty() {
            gen.insert("stopSequences".to_owned(), json!(req.stop));
        }
        if let Some(rf) = &req.response_format {
            gen.insert("responseMimeType".to_owned(), json!("application/json"));
            if let ResponseFormat::JsonSchema { schema, .. } = rf {
                gen.insert("responseJsonSchema".to_owned(), schema.clone());
            }
        }
        if let Some(r) = &req.reasoning {
            gen.insert(
                "thinkingConfig".to_owned(),
                json!({"thinkingBudget": r.derived_budget()}),
            );
        }
        if !gen.is_empty() {
            obj.insert("generationConfig".to_owned(), Value::Object(gen));
        }
        Ok(Value::Object(obj))
    }

    fn parse_response(&self, body: Value) -> Result<ChatResponse, GatewayError> {
        let obj = body.as_object().ok_or_else(|| {
            GatewayError::InvalidJsonMessage("response body must be a JSON object".to_owned())
        })?;
        let candidate = obj
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|a| a.first());

        let mut content = Vec::new();
        let mut saw_tool = false;
        if let Some(parts) = candidate
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(block) = part_to_block(part) {
                    if matches!(block, ContentBlock::ToolUse { .. }) {
                        saw_tool = true;
                    }
                    content.push(block);
                }
            }
        }

        let finish = candidate
            .and_then(|c| field(c.as_object()?, "finishReason", "finish_reason"))
            .and_then(Value::as_str);
        let stop_reason = if saw_tool {
            Some(StopReason::ToolUse)
        } else {
            finish.map(StopReason::from_gemini)
        };

        Ok(ChatResponse {
            id: String::new(),
            model: obj
                .get("modelVersion")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            content,
            stop_reason,
            usage: usage_from_gemini(field(obj, "usageMetadata", "usage_metadata")),
        })
    }

    fn render_response(
        &self,
        resp: &ChatResponse,
        _ctx: &RequestCtx,
    ) -> Result<Value, GatewayError> {
        let parts: Vec<Value> = resp.content.iter().filter_map(block_to_part).collect();
        let finish = resp
            .stop_reason
            .as_ref()
            .map(StopReason::to_gemini)
            .unwrap_or_else(|| "STOP".to_owned());
        Ok(json!({
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": finish,
                "index": 0,
            }],
            "usageMetadata": gemini_usage(&resp.usage),
        }))
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(GeminiStreamParser::default())
    }

    fn stream_renderer(&self, _ctx: &RequestCtx) -> Box<dyn StreamRenderer> {
        Box::new(GeminiStreamRenderer::default())
    }

    fn outbound_headers(
        &self,
        deployment: &Deployment,
        _inbound: &HeaderMap,
    ) -> Result<HeaderMap, GatewayError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-goog-api-key",
            HeaderValue::from_str(&deployment.api_key)
                .map_err(|_| GatewayError::InvalidConfig("invalid api_key".to_owned()))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(headers)
    }

    fn response_headers(&self, upstream: &HeaderMap, stream: bool) -> HeaderMap {
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
        headers
    }
}

// ---- content/part mapping -------------------------------------------------

fn parse_parts_as_text(v: &Value) -> String {
    let parts = v.get("parts").and_then(Value::as_array);
    let mut text = String::new();
    if let Some(parts) = parts {
        for p in parts {
            if let Some(t) = p.get("text").and_then(Value::as_str) {
                text.push_str(t);
            }
        }
    }
    text
}

fn content_to_message(v: &Value) -> Option<Message> {
    let obj = v.as_object()?;
    let role = match obj.get("role").and_then(Value::as_str) {
        Some("model") => Role::Assistant,
        _ => Role::User,
    };
    let content = obj
        .get("parts")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(part_to_block).collect())
        .unwrap_or_default();
    Some(Message { role, content })
}

fn part_to_block(part: &Value) -> Option<ContentBlock> {
    let obj = part.as_object()?;
    if let Some(fc) = obj.get("functionCall").or_else(|| obj.get("function_call")) {
        let name = fc.get("name").and_then(Value::as_str).unwrap_or_default();
        return Some(ContentBlock::ToolUse {
            id: fc
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(name)
                .to_owned(),
            name: name.to_owned(),
            input: fc.get("args").cloned().unwrap_or_else(|| json!({})),
        });
    }
    if let Some(fr) = obj
        .get("functionResponse")
        .or_else(|| obj.get("function_response"))
    {
        let name = fr.get("name").and_then(Value::as_str).unwrap_or_default();
        let response = fr.get("response").cloned().unwrap_or(Value::Null);
        let text = match &response {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        return Some(ContentBlock::ToolResult {
            tool_use_id: name.to_owned(),
            content: vec![ContentBlock::Text { text }],
            is_error: false,
        });
    }
    if let Some(inline) = obj.get("inlineData").or_else(|| obj.get("inline_data")) {
        return Some(ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: field(inline.as_object()?, "mimeType", "mime_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png")
                    .to_owned(),
                data: inline
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            },
        });
    }
    // A thought part is text flagged with `thought: true`.
    if obj.get("thought").and_then(Value::as_bool) == Some(true) {
        return Some(ContentBlock::Thinking {
            text: obj
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            signature: None,
        });
    }
    if let Some(text) = obj.get("text").and_then(Value::as_str) {
        return Some(ContentBlock::Text {
            text: text.to_owned(),
        });
    }
    None
}

fn block_to_part(block: &ContentBlock) -> Option<Value> {
    match block {
        ContentBlock::Text { text } => Some(json!({"text": text})),
        ContentBlock::Thinking { text, .. } => Some(json!({"text": text, "thought": true})),
        ContentBlock::ToolUse { name, input, .. } => Some(json!({
            "functionCall": {"name": name, "args": normalize_args(input)}
        })),
        ContentBlock::Image { source } => match source {
            ImageSource::Base64 { media_type, data } => {
                Some(json!({"inlineData": {"mimeType": media_type, "data": data}}))
            }
            ImageSource::Url(url) => Some(json!({"fileData": {"fileUri": url}})),
        },
        ContentBlock::ToolResult { .. } => None, // handled at the content level
    }
}

/// Build one Gemini content per IR message. Tool results map to `functionResponse`
/// parts in a user-role content, keyed by name via `names`.
fn message_to_content(msg: &Message, names: &HashMap<String, String>) -> Option<Value> {
    let role = if msg.role == Role::Assistant {
        "model"
    } else {
        "user"
    };
    let mut parts: Vec<Value> = Vec::new();
    for block in &msg.content {
        match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                let name = names
                    .get(tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| tool_use_id.clone());
                parts.push(json!({
                    "functionResponse": {"name": name, "response": response_object(content)},
                }));
            }
            other => {
                if let Some(part) = block_to_part(other) {
                    parts.push(part);
                }
            }
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(json!({"role": role, "parts": parts}))
}

fn response_object(content: &[ContentBlock]) -> Value {
    let text = join_text(content);
    match serde_json::from_str::<Value>(&text) {
        Ok(v @ Value::Object(_)) => v,
        _ => json!({"result": text}),
    }
}

fn normalize_args(input: &Value) -> Value {
    match input {
        Value::Object(_) => input.clone(),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        _ => json!({}),
    }
}

/// Map tool-use ids → function names so tool results can be rendered with the
/// name Gemini matches on.
fn tool_name_map(messages: &[Message]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                map.insert(id.clone(), name.clone());
            }
        }
    }
    map
}

fn join_text(blocks: &[ContentBlock]) -> String {
    let mut text = String::new();
    for block in blocks {
        if let ContentBlock::Text { text: t } = block {
            text.push_str(t);
        }
    }
    text
}

const GEMINI_BUILTIN_TOOL_KEYS: &[&str] = &[
    "google_search",
    "googleSearch",
    "code_execution",
    "codeExecution",
    "url_context",
    "urlContext",
    "google_search_retrieval",
    "googleSearchRetrieval",
];

fn tools_from_gemini(arr: &[Value]) -> Vec<ToolDef> {
    let mut tools = Vec::new();
    for entry in arr {
        let decls = entry
            .get("functionDeclarations")
            .or_else(|| entry.get("function_declarations"))
            .and_then(Value::as_array);
        if let Some(decls) = decls {
            for d in decls {
                if let Some(name) = d.get("name").and_then(Value::as_str) {
                    tools.push(ToolDef {
                        name: name.to_owned(),
                        description: d
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        parameters: d
                            .get("parameters")
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object"})),
                        builtin: None,
                    });
                }
            }
        }
        // Built-in / grounding tools (google_search, code_execution, …).
        for key in GEMINI_BUILTIN_TOOL_KEYS {
            if entry.get(*key).is_some() {
                tools.push(ToolDef {
                    name: (*key).to_owned(),
                    description: None,
                    parameters: json!({"type": "object"}),
                    builtin: Some(entry.clone()),
                });
            }
        }
    }
    tools
}

fn tool_to_gemini(tool: &ToolDef) -> Value {
    let mut o = json!({"name": tool.name, "parameters": tool.parameters});
    if let Some(desc) = &tool.description {
        o["description"] = json!(desc);
    }
    o
}

fn parse_tool_choice(cfg: &Value) -> Option<ToolChoice> {
    let mode = cfg.get("mode").and_then(Value::as_str)?;
    match mode.to_ascii_uppercase().as_str() {
        "AUTO" => Some(ToolChoice::Auto),
        "NONE" => Some(ToolChoice::None),
        "ANY" => cfg
            .get("allowedFunctionNames")
            .or_else(|| cfg.get("allowed_function_names"))
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Tool(n.to_owned()))
            .or(Some(ToolChoice::Required)),
        _ => None,
    }
}

fn tool_choice_to_gemini(tc: &ToolChoice) -> Value {
    let cfg = match tc {
        ToolChoice::Auto => json!({"mode": "AUTO"}),
        ToolChoice::None => json!({"mode": "NONE"}),
        ToolChoice::Required => json!({"mode": "ANY"}),
        ToolChoice::Tool(name) => json!({"mode": "ANY", "allowedFunctionNames": [name]}),
    };
    json!({"functionCallingConfig": cfg})
}

fn usage_from_gemini(v: Option<&Value>) -> Usage {
    let Some(obj) = v.and_then(Value::as_object) else {
        return Usage::default();
    };
    // Gemini's `promptTokenCount` is already inclusive of cached tokens.
    let cached = field(obj, "cachedContentTokenCount", "cached_content_token_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input_tokens: field(obj, "promptTokenCount", "prompt_token_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: field(obj, "candidatesTokenCount", "candidates_token_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
    }
}

/// Build a Gemini `usageMetadata` object, adding `cachedContentTokenCount` only
/// on a cache hit (keeps zero-cache output byte-identical).
fn gemini_usage(u: &Usage) -> Value {
    let mut usage = json!({
        "promptTokenCount": u.input_tokens,
        "candidatesTokenCount": u.output_tokens,
        "totalTokenCount": u.input_tokens + u.output_tokens,
    });
    if u.cache_read_input_tokens > 0 {
        usage["cachedContentTokenCount"] = json!(u.cache_read_input_tokens);
    }
    usage
}

// ---- streaming ------------------------------------------------------------

#[derive(Default)]
struct GeminiStreamParser {
    started: bool,
    text_index: Option<usize>,
    think_index: Option<usize>,
    next_index: usize,
    saw_tool: bool,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
    message_stopped: bool,
}

impl GeminiStreamParser {
    fn alloc(&mut self) -> usize {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    fn finalize(&mut self) -> Vec<StreamEvent> {
        if !self.started || self.message_stopped {
            return Vec::new();
        }
        self.message_stopped = true;
        let mut out = Vec::new();
        let mut open: Vec<usize> = self
            .text_index
            .take()
            .into_iter()
            .chain(self.think_index.take())
            .collect();
        open.sort_unstable();
        for index in open {
            out.push(StreamEvent::ContentBlockStop { index });
        }
        // Gemini reports finishReason STOP even when it emitted a functionCall, so
        // prefer ToolUse whenever a tool call was seen (mirrors parse_response).
        let stop_reason = if self.saw_tool {
            Some(StopReason::ToolUse)
        } else {
            self.stop_reason.take().or(Some(StopReason::EndTurn))
        };
        out.push(StreamEvent::MessageDelta {
            stop_reason,
            usage: self.usage.take(),
        });
        out.push(StreamEvent::MessageStop);
        out
    }
}

impl StreamParser for GeminiStreamParser {
    fn push(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, GatewayError> {
        if event.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let data: Value = serde_json::from_str(&event.data)
            .map_err(|e| GatewayError::InvalidJsonMessage(e.to_string()))?;

        let mut out = Vec::new();
        if !self.started {
            self.started = true;
            out.push(StreamEvent::MessageStart {
                id: String::new(),
                model: data
                    .get("modelVersion")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }

        if let Some(u) = data
            .get("usageMetadata")
            .or_else(|| data.get("usage_metadata"))
        {
            self.usage = Some(usage_from_gemini(Some(u)));
        }

        let candidate = data
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(parts) = candidate
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(fc) = part
                    .get("functionCall")
                    .or_else(|| part.get("function_call"))
                {
                    self.saw_tool = true;
                    let index = self.alloc();
                    let name = fc.get("name").and_then(Value::as_str).unwrap_or_default();
                    out.push(StreamEvent::ContentBlockStart {
                        index,
                        block: BlockStart::ToolUse {
                            id: name.to_owned(),
                            name: name.to_owned(),
                        },
                    });
                    let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    out.push(StreamEvent::ToolUseInputDelta {
                        index,
                        partial_json: args.to_string(),
                    });
                    out.push(StreamEvent::ContentBlockStop { index });
                } else if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
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
                            text: text.to_owned(),
                        });
                    }
                } else if let Some(text) = part.get("text").and_then(Value::as_str) {
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
        }

        if let Some(fr) = candidate
            .and_then(|c| c.get("finishReason").or_else(|| c.get("finish_reason")))
            .and_then(Value::as_str)
        {
            self.stop_reason = Some(StopReason::from_gemini(fr));
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        self.finalize()
    }
}

#[derive(Default)]
struct GeminiStreamRenderer {
    /// Buffered partial-json args for in-flight tool-use blocks, by index.
    tool_args: HashMap<usize, (String, String)>, // index -> (name, buffered json)
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
    sent_finish: bool,
}

impl GeminiStreamRenderer {
    fn chunk(parts: Vec<Value>, finish: Option<&str>, usage: Option<&Usage>) -> Vec<u8> {
        let mut candidate = json!({
            "content": {"role": "model", "parts": parts},
            "index": 0,
        });
        if let Some(f) = finish {
            candidate["finishReason"] = json!(f);
        }
        let mut data = json!({"candidates": [candidate]});
        if let Some(u) = usage {
            data["usageMetadata"] = gemini_usage(u);
        }
        sse_frame(None, &data.to_string())
    }
}

impl StreamRenderer for GeminiStreamRenderer {
    fn push(&mut self, event: &StreamEvent) -> Vec<u8> {
        match event {
            StreamEvent::MessageStart { .. }
            | StreamEvent::ContentBlockStart {
                block: BlockStart::Text,
                ..
            } => Vec::new(),
            StreamEvent::ContentBlockStart {
                index,
                block: BlockStart::ToolUse { name, .. },
            } => {
                self.tool_args.insert(*index, (name.clone(), String::new()));
                Vec::new()
            }
            StreamEvent::ContentBlockStart {
                block: BlockStart::Thinking,
                ..
            } => Vec::new(),
            StreamEvent::TextDelta { text, .. } => {
                Self::chunk(vec![json!({"text": text})], None, None)
            }
            StreamEvent::ThinkingDelta { text, .. } => {
                Self::chunk(vec![json!({"text": text, "thought": true})], None, None)
            }
            StreamEvent::ToolUseInputDelta {
                index,
                partial_json,
            } => {
                if let Some((_, buf)) = self.tool_args.get_mut(index) {
                    buf.push_str(partial_json);
                }
                Vec::new()
            }
            StreamEvent::ContentBlockStop { index } => {
                if let Some((name, buf)) = self.tool_args.remove(index) {
                    let args: Value = serde_json::from_str(&buf).unwrap_or_else(|_| json!({}));
                    Self::chunk(
                        vec![json!({"functionCall": {"name": name, "args": args}})],
                        None,
                        None,
                    )
                } else {
                    Vec::new()
                }
            }
            StreamEvent::MessageDelta { stop_reason, usage } => {
                self.stop_reason = stop_reason.clone();
                self.usage = usage.clone();
                Vec::new()
            }
            StreamEvent::MessageStop => {
                self.sent_finish = true;
                let finish = self
                    .stop_reason
                    .as_ref()
                    .map(StopReason::to_gemini)
                    .unwrap_or_else(|| "STOP".to_owned());
                Self::chunk(Vec::new(), Some(&finish), self.usage.clone().as_ref())
            }
        }
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.sent_finish {
            Vec::new()
        } else {
            self.sent_finish = true;
            Self::chunk(Vec::new(), Some("STOP"), None)
        }
    }
}
