//! Anthropic Messages (`/v1/messages`) codec. Closest to the IR shape since the
//! IR mirrors Anthropic content blocks.

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

const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u64 = 4096;

const KNOWN_REQUEST_KEYS: &[&str] = &[
    "model",
    "system",
    "messages",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
    "output_format",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "stream",
];

pub struct AnthropicCodec;

impl ProtocolCodec for AnthropicCodec {
    fn parse_request(&self, body: Value) -> Result<ChatRequest, GatewayError> {
        let Value::Object(mut obj) = body else {
            return Err(GatewayError::InvalidJsonMessage(
                "request body must be a JSON object".to_owned(),
            ));
        };

        let model = take_string(&mut obj, "model").unwrap_or_default();
        let system_raw = obj.remove("system");
        let system_cached =
            matches!(&system_raw, Some(Value::Array(arr)) if array_has_cache_control(arr));
        let system = match system_raw {
            Some(Value::String(s)) => vec![ContentBlock::Text { text: s }],
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(block_from_anthropic)
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };

        let mut message_cache_idx = Vec::new();
        let messages = match obj.remove("messages") {
            Some(Value::Array(arr)) => {
                let mut out = Vec::new();
                for raw in &arr {
                    if let Some(msg) = message_from_anthropic(raw) {
                        if message_has_cache_control(raw) {
                            message_cache_idx.push(out.len());
                        }
                        out.push(msg);
                    }
                }
                out
            }
            _ => Vec::new(),
        };

        let tools_raw = obj.remove("tools");
        let tools_cached =
            matches!(&tools_raw, Some(Value::Array(arr)) if array_has_cache_control(arr));
        let tools = match tools_raw {
            Some(Value::Array(arr)) => arr.iter().filter_map(tool_from_anthropic).collect(),
            _ => Vec::new(),
        };

        let tc_raw = obj.remove("tool_choice");
        let parallel_tool_calls = tc_raw
            .as_ref()
            .and_then(|tc| tc.get("disable_parallel_tool_use"))
            .and_then(Value::as_bool)
            .map(|disabled| !disabled);
        let tool_choice = tc_raw.and_then(|tc| match tc {
            Value::Object(o) => match o.get("type").and_then(Value::as_str) {
                Some("auto") => Some(ToolChoice::Auto),
                Some("any") => Some(ToolChoice::Required),
                Some("none") => Some(ToolChoice::None),
                Some("tool") => o
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|n| ToolChoice::Tool(n.to_owned())),
                _ => None,
            },
            _ => None,
        });

        let reasoning = obj.remove("thinking").and_then(|t| {
            let enabled = t.get("type").and_then(Value::as_str) == Some("enabled");
            let budget = t.get("budget_tokens").and_then(Value::as_u64);
            (enabled || budget.is_some()).then_some(ReasoningConfig {
                effort: None,
                budget_tokens: budget,
            })
        });

        let response_format = obj
            .remove("output_format")
            .or_else(|| {
                obj.remove("output_config")
                    .and_then(|oc| oc.get("format").cloned())
            })
            .and_then(response_format_from_anthropic);

        let stop = match obj.remove("stop_sequences") {
            Some(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
            _ => Vec::new(),
        };

        let req = ChatRequest {
            model,
            system,
            messages,
            tools,
            cache: CacheMarkers {
                tools: tools_cached,
                system: system_cached,
                messages: message_cache_idx,
            },
            tool_choice,
            parallel_tool_calls,
            response_format,
            reasoning,
            max_tokens: obj.remove("max_tokens").and_then(|v| v.as_u64()),
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
        let mut obj = Map::new();
        obj.insert("model".to_owned(), json!(req.model));
        obj.insert(
            "max_tokens".to_owned(),
            json!(req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        );
        if !req.system.is_empty() {
            let mut system: Vec<Value> = req.system.iter().map(block_to_anthropic).collect();
            if req.cache.system {
                if let Some(last) = system.last_mut() {
                    set_cache_control(last);
                }
            }
            obj.insert("system".to_owned(), Value::Array(system));
        }
        obj.insert(
            "messages".to_owned(),
            Value::Array(render_messages(&req.messages, &req.cache.messages)),
        );
        // Built-in/server tools are origin-specific; drop them rather than
        // render a bogus function tool the other side can't satisfy.
        let mut function_tools: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| t.builtin.is_none())
            .map(tool_to_anthropic)
            .collect();
        if !function_tools.is_empty() {
            if req.cache.tools {
                if let Some(last) = function_tools.last_mut() {
                    set_cache_control(last);
                }
            }
            obj.insert("tools".to_owned(), Value::Array(function_tools));
        }
        if let Some(tc) = &req.tool_choice {
            let mut choice = tool_choice_to_anthropic(tc);
            if req.parallel_tool_calls == Some(false) {
                if let Some(o) = choice.as_object_mut() {
                    o.insert("disable_parallel_tool_use".to_owned(), json!(true));
                }
            }
            obj.insert("tool_choice".to_owned(), choice);
        }

        // Extended thinking: Anthropic requires 1024 <= budget_tokens < max_tokens
        // and the default sampling temperature, so gate temperature/top_p on it.
        let mt = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let thinking_on = match &req.reasoning {
            Some(r) => {
                let budget = r
                    .budget_tokens
                    .or_else(|| r.effort.map(|e| e.to_budget()))
                    .unwrap_or(0);
                if budget >= 1024 && mt > 1024 {
                    obj.insert(
                        "thinking".to_owned(),
                        json!({"type": "enabled", "budget_tokens": budget.min(mt - 1)}),
                    );
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !thinking_on {
            if let Some(t) = req.temperature {
                obj.insert("temperature".to_owned(), json!(t));
            }
            if let Some(p) = req.top_p {
                obj.insert("top_p".to_owned(), json!(p));
            }
        }
        if !req.stop.is_empty() {
            obj.insert("stop_sequences".to_owned(), json!(req.stop));
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
        let content = obj
            .get("content")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(block_from_anthropic).collect())
            .unwrap_or_default();
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
            stop_reason: obj
                .get("stop_reason")
                .and_then(Value::as_str)
                .map(StopReason::from_anthropic),
            usage: usage_from_anthropic(obj.get("usage")),
        })
    }

    fn render_response(
        &self,
        resp: &ChatResponse,
        ctx: &RequestCtx,
    ) -> Result<Value, GatewayError> {
        let id = if resp.id.is_empty() {
            "msg_litellm".to_owned()
        } else {
            resp.id.clone()
        };
        Ok(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": ctx.model,
            "content": Value::Array(resp.content.iter().map(block_to_anthropic).collect()),
            "stop_reason": resp.stop_reason.as_ref().map(StopReason::to_anthropic),
            "stop_sequence": Value::Null,
            "usage": {
                // Anthropic reports `input_tokens` as the post-breakpoint remainder.
                "input_tokens": resp.usage.non_cached_input_tokens(),
                "output_tokens": resp.usage.output_tokens,
                "cache_creation_input_tokens": resp.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": resp.usage.cache_read_input_tokens,
            },
        }))
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(AnthropicStreamParser::default())
    }

    fn stream_renderer(&self, ctx: &RequestCtx) -> Box<dyn StreamRenderer> {
        Box::new(AnthropicStreamRenderer {
            model: ctx.model.clone(),
        })
    }

    fn outbound_headers(
        &self,
        deployment: &Deployment,
        inbound: &HeaderMap,
    ) -> Result<HeaderMap, GatewayError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&deployment.api_key)
                .map_err(|_| GatewayError::InvalidConfig("invalid api_key".to_owned()))?,
        );
        headers.insert(
            "anthropic-version",
            inbound
                .get("anthropic-version")
                .cloned()
                .unwrap_or_else(|| HeaderValue::from_static(DEFAULT_ANTHROPIC_VERSION)),
        );
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        if let Some(beta) = inbound.get("anthropic-beta") {
            headers.insert("anthropic-beta", beta.clone());
        }
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
        if let Some(request_id) = upstream.get("request-id").cloned() {
            headers.insert("request-id", request_id);
        }
        headers
    }
}

// ---- content block mapping ------------------------------------------------

fn block_from_anthropic(v: &Value) -> Option<ContentBlock> {
    let obj = v.as_object()?;
    match obj.get("type").and_then(Value::as_str)? {
        "text" => Some(ContentBlock::Text {
            text: obj.get("text").and_then(Value::as_str)?.to_owned(),
        }),
        "thinking" => Some(ContentBlock::Thinking {
            text: obj
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            signature: obj
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }),
        "tool_use" => Some(ContentBlock::ToolUse {
            id: obj
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            input: obj.get("input").cloned().unwrap_or(Value::Null),
        }),
        "tool_result" => {
            let content = match obj.get("content") {
                Some(Value::String(s)) => vec![ContentBlock::Text { text: s.clone() }],
                Some(Value::Array(arr)) => arr.iter().filter_map(block_from_anthropic).collect(),
                _ => Vec::new(),
            };
            Some(ContentBlock::ToolResult {
                tool_use_id: obj
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                content,
                is_error: obj
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        }
        "image" => {
            let src = obj.get("source")?.as_object()?;
            let source = match src.get("type").and_then(Value::as_str) {
                Some("base64") => ImageSource::Base64 {
                    media_type: src
                        .get("media_type")
                        .and_then(Value::as_str)
                        .unwrap_or("image/png")
                        .to_owned(),
                    data: src
                        .get("data")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                },
                Some("url") => ImageSource::Url(src.get("url").and_then(Value::as_str)?.to_owned()),
                _ => return None,
            };
            Some(ContentBlock::Image { source })
        }
        _ => None,
    }
}

fn block_to_anthropic(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text { text } => json!({"type": "text", "text": text}),
        ContentBlock::Thinking { text, signature } => {
            let mut o = json!({"type": "thinking", "thinking": text});
            if let Some(sig) = signature {
                o["signature"] = json!(sig);
            }
            o
        }
        ContentBlock::ToolUse { id, name, input } => {
            json!({"type": "tool_use", "id": id, "name": name, "input": input})
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let rendered = if let [ContentBlock::Text { text }] = content.as_slice() {
                json!(text)
            } else {
                Value::Array(content.iter().map(block_to_anthropic).collect())
            };
            json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": rendered,
                "is_error": is_error,
            })
        }
        ContentBlock::Image { source } => {
            let src = match source {
                ImageSource::Base64 { media_type, data } => {
                    json!({"type": "base64", "media_type": media_type, "data": data})
                }
                ImageSource::Url(url) => json!({"type": "url", "url": url}),
            };
            json!({"type": "image", "source": src})
        }
    }
}

fn message_from_anthropic(v: &Value) -> Option<Message> {
    let obj = v.as_object()?;
    let role = match obj.get("role").and_then(Value::as_str)? {
        "assistant" => Role::Assistant,
        _ => Role::User,
    };
    let content = match obj.get("content") {
        Some(Value::String(s)) => vec![ContentBlock::Text { text: s.clone() }],
        Some(Value::Array(arr)) => arr.iter().filter_map(block_from_anthropic).collect(),
        _ => Vec::new(),
    };
    Some(Message { role, content })
}

/// Render IR messages to Anthropic, coalescing consecutive same-role turns into
/// one. Anthropic requires user/assistant to alternate, but a single IR turn can
/// span several messages (e.g. parallel tool results arrive as separate
/// `Role::Tool` messages that all map to the "user" wire role). Empty-content
/// messages are dropped, since Anthropic rejects an empty `content` array.
fn render_messages(messages: &[Message], cache_idx: &[usize]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        let role = msg.role.as_anthropic();
        let blocks: Vec<Value> = msg.content.iter().map(block_to_anthropic).collect();
        if blocks.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.get("role").and_then(Value::as_str) == Some(role) => {
                if let Some(Value::Array(arr)) = last.get_mut("content") {
                    arr.extend(blocks);
                }
            }
            _ => out.push(json!({"role": role, "content": Value::Array(blocks)})),
        }
        // A cache breakpoint on this message lands on the last block we just
        // appended (its tail, even after coalescing into the previous turn).
        if cache_idx.contains(&i) {
            if let Some(Value::Array(arr)) = out.last_mut().and_then(|m| m.get_mut("content")) {
                if let Some(last_block) = arr.last_mut() {
                    set_cache_control(last_block);
                }
            }
        }
    }
    out
}

fn tool_from_anthropic(v: &Value) -> Option<ToolDef> {
    let obj = v.as_object()?;
    // Server-side tools carry a versioned `type` (e.g. web_search_20250305);
    // function tools have no `type` or `type:"custom"`.
    if let Some(t) = obj.get("type").and_then(Value::as_str) {
        if t != "custom" {
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
    Some(ToolDef {
        name: obj.get("name").and_then(Value::as_str)?.to_owned(),
        description: obj
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        parameters: obj
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
        builtin: None,
    })
}

fn response_format_from_anthropic(v: Value) -> Option<ResponseFormat> {
    let obj = v.as_object()?;
    match obj.get("type").and_then(Value::as_str) {
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
            strict: true,
        }),
        Some("json_object") | Some("json") => Some(ResponseFormat::JsonObject),
        _ => None,
    }
}

fn tool_to_anthropic(tool: &ToolDef) -> Value {
    let mut o = json!({"name": tool.name, "input_schema": tool.parameters});
    if let Some(desc) = &tool.description {
        o["description"] = json!(desc);
    }
    o
}

fn tool_choice_to_anthropic(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::Required => json!({"type": "any"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Tool(name) => json!({"type": "tool", "name": name}),
    }
}

fn usage_from_anthropic(v: Option<&Value>) -> Usage {
    let Some(obj) = v.and_then(Value::as_object) else {
        return Usage::default();
    };
    let u = |key: &str| obj.get(key).and_then(Value::as_u64).unwrap_or(0);
    let cache_read = u("cache_read_input_tokens");
    let cache_creation = u("cache_creation_input_tokens");
    Usage {
        // Anthropic's `input_tokens` excludes the cached/created portions; add
        // them back so the IR field is the inclusive total (see `Usage` docs).
        input_tokens: u("input_tokens") + cache_read + cache_creation,
        output_tokens: u("output_tokens"),
        cache_creation_input_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
    }
}

// ---- prompt-cache helpers -------------------------------------------------

/// Add a 5-minute `ephemeral` cache breakpoint to a rendered content block.
fn set_cache_control(block: &mut Value) {
    if let Some(o) = block.as_object_mut() {
        o.insert("cache_control".to_owned(), json!({"type": "ephemeral"}));
    }
}

/// Whether any element of a raw block/tool array carries a `cache_control` key.
fn array_has_cache_control(arr: &[Value]) -> bool {
    arr.iter().any(|b| b.get("cache_control").is_some())
}

/// Whether a raw message has a cache breakpoint, either at the message level or
/// on any of its content blocks.
fn message_has_cache_control(msg: &Value) -> bool {
    if msg.get("cache_control").is_some() {
        return true;
    }
    msg.get("content")
        .and_then(Value::as_array)
        .is_some_and(|arr| array_has_cache_control(arr))
}

// ---- shared small helpers -------------------------------------------------

pub(crate) fn take_string(obj: &mut Map<String, Value>, key: &str) -> Option<String> {
    match obj.remove(key) {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}

pub(crate) fn strip_known(mut obj: Map<String, Value>, known: &[&str]) -> Map<String, Value> {
    for k in known {
        obj.remove(*k);
    }
    obj
}

// ---- streaming ------------------------------------------------------------

#[derive(Default)]
struct AnthropicStreamParser {
    /// Usage from `message_start` (Anthropic reports cache tokens there). Folded
    /// into the final `message_delta` so downstream renderers see complete usage.
    start_usage: Usage,
}

impl StreamParser for AnthropicStreamParser {
    fn push(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>, GatewayError> {
        let kind = event.event.as_deref().unwrap_or_default();
        if event.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let data: Value = serde_json::from_str(&event.data)
            .map_err(|e| GatewayError::InvalidJsonMessage(e.to_string()))?;
        Ok(match kind {
            "message_start" => {
                let msg = data.get("message");
                self.start_usage = usage_from_anthropic(msg.and_then(|m| m.get("usage")));
                vec![StreamEvent::MessageStart {
                    id: msg
                        .and_then(|m| m.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    model: msg
                        .and_then(|m| m.get("model"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }]
            }
            "content_block_start" => {
                let index = data.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let cb = data.get("content_block");
                let block = match cb.and_then(|c| c.get("type")).and_then(Value::as_str) {
                    Some("tool_use") => BlockStart::ToolUse {
                        id: cb
                            .and_then(|c| c.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        name: cb
                            .and_then(|c| c.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    },
                    Some("thinking") => BlockStart::Thinking,
                    _ => BlockStart::Text,
                };
                vec![StreamEvent::ContentBlockStart { index, block }]
            }
            "content_block_delta" => {
                let index = data.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let delta = data.get("delta");
                match delta.and_then(|d| d.get("type")).and_then(Value::as_str) {
                    Some("text_delta") => vec![StreamEvent::TextDelta {
                        index,
                        text: delta
                            .and_then(|d| d.get("text"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    }],
                    Some("thinking_delta") => vec![StreamEvent::ThinkingDelta {
                        index,
                        text: delta
                            .and_then(|d| d.get("thinking"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    }],
                    Some("input_json_delta") => vec![StreamEvent::ToolUseInputDelta {
                        index,
                        partial_json: delta
                            .and_then(|d| d.get("partial_json"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    }],
                    _ => Vec::new(),
                }
            }
            "content_block_stop" => {
                let index = data.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                vec![StreamEvent::ContentBlockStop { index }]
            }
            "message_delta" => {
                let stop_reason = data
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                    .map(StopReason::from_anthropic);
                // The delta carries output_tokens; fold in input + cache tokens
                // captured from message_start so the emitted usage is complete.
                let usage = data.get("usage").map(|u| {
                    let mut u = usage_from_anthropic(Some(u));
                    u.input_tokens = u.input_tokens.max(self.start_usage.input_tokens);
                    u.cache_read_input_tokens = self.start_usage.cache_read_input_tokens;
                    u.cache_creation_input_tokens = self.start_usage.cache_creation_input_tokens;
                    u
                });
                vec![StreamEvent::MessageDelta { stop_reason, usage }]
            }
            "message_stop" => vec![StreamEvent::MessageStop],
            _ => Vec::new(),
        })
    }
}

struct AnthropicStreamRenderer {
    model: String,
}

impl StreamRenderer for AnthropicStreamRenderer {
    fn push(&mut self, event: &StreamEvent) -> Vec<u8> {
        match event {
            StreamEvent::MessageStart { id, .. } => {
                let id = if id.is_empty() { "msg_litellm" } else { id };
                let data = json!({
                    "type": "message_start",
                    "message": {
                        "id": id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                });
                sse_frame(Some("message_start"), &data.to_string())
            }
            StreamEvent::ContentBlockStart { index, block } => {
                let content_block = match block {
                    BlockStart::Text => json!({"type": "text", "text": ""}),
                    BlockStart::Thinking => json!({"type": "thinking", "thinking": ""}),
                    BlockStart::ToolUse { id, name } => {
                        json!({"type": "tool_use", "id": id, "name": name, "input": {}})
                    }
                };
                let data = json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": content_block,
                });
                sse_frame(Some("content_block_start"), &data.to_string())
            }
            StreamEvent::TextDelta { index, text } => {
                let data = json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                });
                sse_frame(Some("content_block_delta"), &data.to_string())
            }
            StreamEvent::ThinkingDelta { index, text } => {
                let data = json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                });
                sse_frame(Some("content_block_delta"), &data.to_string())
            }
            StreamEvent::ToolUseInputDelta {
                index,
                partial_json,
            } => {
                let data = json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                });
                sse_frame(Some("content_block_delta"), &data.to_string())
            }
            StreamEvent::ContentBlockStop { index } => {
                let data = json!({"type": "content_block_stop", "index": index});
                sse_frame(Some("content_block_stop"), &data.to_string())
            }
            StreamEvent::MessageDelta { stop_reason, usage } => {
                let mut usage_obj =
                    json!({"output_tokens": usage.as_ref().map(|u| u.output_tokens).unwrap_or(0)});
                // Surface cache tokens when present (e.g. a non-Anthropic upstream
                // reported them at end-of-stream). Omitted when zero so the common
                // case stays byte-identical.
                if let Some(u) = usage {
                    if u.cache_read_input_tokens > 0 || u.cache_creation_input_tokens > 0 {
                        usage_obj["input_tokens"] = json!(u.non_cached_input_tokens());
                        usage_obj["cache_read_input_tokens"] = json!(u.cache_read_input_tokens);
                        usage_obj["cache_creation_input_tokens"] =
                            json!(u.cache_creation_input_tokens);
                    }
                }
                let data = json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason.as_ref().map(StopReason::to_anthropic)},
                    "usage": usage_obj,
                });
                sse_frame(Some("message_delta"), &data.to_string())
            }
            StreamEvent::MessageStop => sse_frame(
                Some("message_stop"),
                &json!({"type": "message_stop"}).to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn parses_and_renders_cache_control_breakpoints() {
        let body = json!({
            "model": "claude",
            "system": [
                {"type": "text", "text": "you are helpful"},
                {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [
                {"name": "a", "input_schema": {"type": "object"}},
                {"name": "b", "input_schema": {"type": "object"},
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]}
            ],
            "max_tokens": 100
        });
        let req = AnthropicCodec.parse_request(body).unwrap();
        assert!(req.cache.tools);
        assert!(req.cache.system);
        assert_eq!(req.cache.messages, vec![0]);

        let out = AnthropicCodec.render_request(&req).unwrap();
        let sys = out["system"].as_array().unwrap();
        assert!(sys[0].get("cache_control").is_none());
        assert!(sys[1].get("cache_control").is_some());
        let tools = out["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
        assert!(tools[1].get("cache_control").is_some());
        let content = out["messages"][0]["content"].as_array().unwrap();
        assert!(content.last().unwrap().get("cache_control").is_some());
    }

    #[test]
    fn no_breakpoints_when_client_sets_none() {
        let body = json!({
            "model": "claude",
            "system": [{"type": "text", "text": "hi"}],
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100
        });
        let req = AnthropicCodec.parse_request(body).unwrap();
        assert!(req.cache.is_empty());
        let out = AnthropicCodec.render_request(&req).unwrap();
        assert!(out["system"][0].get("cache_control").is_none());
    }

    #[test]
    fn stream_parser_folds_cache_usage_from_message_start() {
        use crate::sdk::codec::stream::SseEvent;
        let mut p = AnthropicStreamParser::default();
        p.push(&SseEvent {
            event: Some("message_start".to_owned()),
            data: r#"{"type":"message_start","message":{"id":"m","model":"claude","usage":{"input_tokens":5,"cache_read_input_tokens":1000,"cache_creation_input_tokens":200,"output_tokens":0}}}"#.to_owned(),
        })
        .unwrap();
        let evs = p
            .push(&SseEvent {
                event: Some("message_delta".to_owned()),
                data: r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}"#.to_owned(),
            })
            .unwrap();
        match &evs[0] {
            StreamEvent::MessageDelta {
                usage: Some(u), ..
            } => {
                assert_eq!(u.input_tokens, 1205);
                assert_eq!(u.output_tokens, 42);
                assert_eq!(u.cache_read_input_tokens, 1000);
                assert_eq!(u.cache_creation_input_tokens, 200);
            }
            other => panic!("expected MessageDelta with usage, got {other:?}"),
        }
    }

    #[test]
    fn usage_makes_input_inclusive_and_round_trips() {
        let usage = usage_from_anthropic(Some(&json!({
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 1000
        })));
        assert_eq!(usage.input_tokens, 1250);
        assert_eq!(usage.cache_read_input_tokens, 1000);
        assert_eq!(usage.cache_creation_input_tokens, 200);
        assert_eq!(usage.non_cached_input_tokens(), 50);

        let resp = ChatResponse {
            usage,
            ..Default::default()
        };
        let ctx = RequestCtx {
            model: "claude".to_owned(),
            stream: false,
        };
        let out = AnthropicCodec.render_response(&resp, &ctx).unwrap();
        assert_eq!(out["usage"]["input_tokens"], 50);
        assert_eq!(out["usage"]["cache_read_input_tokens"], 1000);
        assert_eq!(out["usage"]["cache_creation_input_tokens"], 200);
    }
}
