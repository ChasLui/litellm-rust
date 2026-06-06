//! Advanced cross-protocol feature mapping: built-in tools, structured outputs,
//! reasoning effort/budget, parallel-tool-call control. These assert the shape
//! of the OUTBOUND (upstream) request the gateway produces.

use std::{collections::HashMap, sync::Arc};

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use litellm_rust::{
    http::routes::router,
    proxy::{
        config::{GatewayConfig, GeneralSettings, LiteLlmParams, ModelEntry},
        state::AppState,
    },
    sdk::{
        providers::{self, transform::ProviderRegistry},
        router::Router as ModelRouter,
    },
};
use serde_json::{json, Value};
use tower::util::ServiceExt;
use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

fn model_entry(model_name: &str, model: &str, api_base: &str) -> ModelEntry {
    ModelEntry {
        model_name: model_name.to_owned(),
        litellm_params: LiteLlmParams {
            model: model.to_owned(),
            api_key: Some("sk-upstream".to_owned()),
            api_base: Some(api_base.to_owned()),
            wire_api: None,
            extra: Default::default(),
        },
    }
}

fn build_state(entries: Vec<ModelEntry>) -> Arc<AppState> {
    let config = GatewayConfig {
        model_list: entries,
        mcp_servers: HashMap::new(),
        general_settings: GeneralSettings {
            master_key: Some("sk-local".to_owned()),
            ..Default::default()
        },
        agents: Vec::new(),
    };
    let mut providers = ProviderRegistry::new();
    providers::register_all(&mut providers);
    let model_router = ModelRouter::from_config(&config, &providers).unwrap();
    let http = AppState::build_http_client().unwrap();
    Arc::new(AppState::new(config, model_router, http, HashMap::new(), None).unwrap())
}

fn anthropic_ok() -> Value {
    json!({
        "id": "m", "type": "message", "role": "assistant", "model": "x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

fn responses_ok() -> Value {
    json!({
        "id": "r", "object": "response", "model": "x", "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

fn gemini_ok() -> Value {
    json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}
    })
}

/// POST an inbound request and return the body the gateway sent upstream.
async fn capture_outbound(
    entries: Vec<ModelEntry>,
    upstream: &MockServer,
    uri: &str,
    inbound_body: Value,
) -> Value {
    let app = router(build_state(entries));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(inbound_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let reqs = upstream.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1, "expected exactly one upstream request");
    serde_json::from_slice(&reqs[0].body).unwrap()
}

// ---- #4 built-in / server tools: dropped cross-protocol, not mangled --------

#[tokio::test]
async fn builtin_web_search_dropped_anthropic_to_chat() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "model": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        })))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())],
        &upstream,
        "/v1/messages",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        }),
    )
    .await;

    // The built-in must NOT appear as a bogus client function tool.
    let has_web_search = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools.iter().any(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    == Some("web_search")
            })
        })
        .unwrap_or(false);
    assert!(
        !has_web_search,
        "built-in web_search leaked as function tool: {body}"
    );
}

#[tokio::test]
async fn function_tool_preserved_anthropic_to_chat() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "model": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        })))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())],
        &upstream,
        "/v1/messages",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
        }),
    )
    .await;

    assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
}

// ---- #3 structured outputs --------------------------------------------------

#[tokio::test]
async fn response_format_chat_to_responses() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "openai/gpt-5", &upstream.uri())],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {"type": "object"}, "strict": true}}
        }),
    )
    .await;

    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["name"], "person");
}

#[tokio::test]
async fn response_format_chat_to_gemini() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "gemini/gemini-2.5-pro", &upstream.uri())],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {"type": "object", "properties": {"n": {"type": "string"}}}}}
        }),
    )
    .await;

    assert_eq!(
        body["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert!(body["generationConfig"]["responseJsonSchema"].is_object());
}

// ---- #5 reasoning effort / budget ------------------------------------------

#[tokio::test]
async fn reasoning_chat_to_anthropic_clamps_and_drops_temperature() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry(
            "gw",
            "anthropic/claude-sonnet-4-5",
            &upstream.uri(),
        )],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "max_tokens": 20000,
            "temperature": 0.5,
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(body["thinking"]["type"], "enabled");
    let budget = body["thinking"]["budget_tokens"].as_u64().unwrap();
    assert!(
        (1024..20000).contains(&budget),
        "budget out of range: {budget}"
    );
    // Extended thinking forbids a custom temperature.
    assert!(
        body.get("temperature").is_none(),
        "temperature must be dropped: {body}"
    );
}

#[tokio::test]
async fn reasoning_chat_to_gemini_budget() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "gemini/gemini-2.5-pro", &upstream.uri())],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"]
            .as_u64()
            .unwrap()
            > 0
    );
}

// ---- #2 parallel tool calls -------------------------------------------------

#[tokio::test]
async fn parallel_disabled_chat_to_anthropic() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "anthropic/claude-sonnet-4-5", &upstream.uri())],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "tool_choice": "required",
            "parallel_tool_calls": false,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(body["tool_choice"]["type"], "any");
    assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
}

#[tokio::test]
async fn parallel_disabled_anthropic_to_chat() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "c", "object": "chat.completion", "model": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        })))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())],
        &upstream,
        "/v1/messages",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(body["parallel_tool_calls"], false);
}

// ---- consecutive same-role coalescing (Anthropic/Gemini alternation) --------

/// Parallel tool results from an OpenAI client become separate `role:tool`
/// messages; rendered to Anthropic they must collapse into a single user turn,
/// or Anthropic 400s on consecutive user roles.
#[tokio::test]
async fn parallel_tool_results_coalesced_chat_to_anthropic() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry(
            "gw",
            "anthropic/claude-sonnet-4-5",
            &upstream.uri(),
        )],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "weather in SF and NYC?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_a", "content": "sunny"},
                {"role": "tool", "tool_call_id": "call_b", "content": "cold"}
            ]
        }),
    )
    .await;

    let messages = body["messages"].as_array().unwrap();
    // user → assistant → user (both tool results merged), no consecutive users.
    assert_eq!(messages.len(), 3, "expected 3 alternating turns: {body}");
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    let results = messages[2]["content"].as_array().unwrap();
    assert_eq!(results.len(), 2, "both tool_results in one turn: {body}");
    assert_eq!(results[0]["type"], "tool_result");
    assert_eq!(results[0]["tool_use_id"], "call_a");
    assert_eq!(results[1]["tool_use_id"], "call_b");
    // No two adjacent messages share a role.
    for pair in messages.windows(2) {
        assert_ne!(
            pair[0]["role"], pair[1]["role"],
            "roles must alternate: {body}"
        );
    }
}

/// Same scenario rendered to Gemini: the two functionResponse parts must share a
/// single user content rather than two consecutive user turns.
#[tokio::test]
async fn parallel_tool_results_coalesced_chat_to_gemini() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry("gw", "gemini/gemini-2.5-pro", &upstream.uri())],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_a", "content": "sunny"},
                {"role": "tool", "tool_call_id": "call_b", "content": "cold"}
            ]
        }),
    )
    .await;

    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3, "expected 3 alternating contents: {body}");
    assert_eq!(contents[2]["role"], "user");
    let parts = contents[2]["parts"].as_array().unwrap();
    assert_eq!(
        parts.len(),
        2,
        "both functionResponses in one content: {body}"
    );
    assert!(parts[0]["functionResponse"].is_object());
    assert!(parts[1]["functionResponse"].is_object());
}

// ---- multimodal / image content --------------------------------------------

/// An OpenAI `image_url` data URL must render as an Anthropic base64 image block,
/// preserving the media type and base64 payload.
#[tokio::test]
async fn image_chat_to_anthropic() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry(
            "gw",
            "anthropic/claude-sonnet-4-5",
            &upstream.uri(),
        )],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ]}]
        }),
    )
    .await;

    let parts = body["messages"][0]["content"].as_array().unwrap();
    let image = parts
        .iter()
        .find(|b| b["type"] == "image")
        .unwrap_or_else(|| panic!("no image block: {body}"));
    assert_eq!(image["source"]["type"], "base64");
    assert_eq!(image["source"]["media_type"], "image/png");
    assert_eq!(image["source"]["data"], "iVBORw0KGgo=");
}

// ---- system message hoisting -----------------------------------------------

/// An OpenAI `role:"system"` message is hoisted to Anthropic's top-level
/// `system` field (rendered as a text-block array) and must not appear as a chat
/// message role.
#[tokio::test]
async fn system_chat_to_anthropic() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry(
            "gw",
            "anthropic/claude-sonnet-4-5",
            &upstream.uri(),
        )],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]
        }),
    )
    .await;

    // Anthropic renders system as an array of text blocks at the top level.
    assert_eq!(body["system"][0]["type"], "text");
    assert_eq!(body["system"][0]["text"], "be terse");
    // No message carries the system role.
    let messages = body["messages"].as_array().unwrap();
    assert!(
        messages.iter().all(|m| m["role"] != "system"),
        "system role leaked into messages: {body}"
    );
}

// ---- parallel_tool_calls without tool_choice (accepted degradation #1) ------

/// Anthropic only expresses "disable parallel tool use" as a field nested inside
/// `tool_choice`. With `parallel_tool_calls:false` but no `tool_choice`, there is
/// no carrier object, so the intent is dropped. This is an accepted semantic
/// degradation (option #1): rather than synthesize a `tool_choice`, we let the
/// upstream default (parallel allowed) stand.
#[tokio::test]
async fn parallel_disabled_without_tool_choice_chat_to_anthropic() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok()))
        .mount(&upstream)
        .await;

    let body = capture_outbound(
        vec![model_entry(
            "gw",
            "anthropic/claude-sonnet-4-5",
            &upstream.uri(),
        )],
        &upstream,
        "/v1/chat/completions",
        json!({
            "model": "gw",
            "max_tokens": 16,
            "parallel_tool_calls": false,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert!(
        !body.to_string().contains("disable_parallel_tool_use"),
        "intent must be dropped without tool_choice carrier: {body}"
    );
}
