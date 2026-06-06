//! Cross-protocol conversion: inbound endpoint protocol != outbound provider
//! protocol, exercising the IR pivot with tool calling and streaming.

use std::{collections::HashMap, sync::Arc};

use axum::{
    body::{to_bytes, Body},
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
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

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

fn build_state(config: GatewayConfig) -> Arc<AppState> {
    let mut providers = ProviderRegistry::new();
    providers::register_all(&mut providers);
    let model_router = ModelRouter::from_config(&config, &providers).unwrap();
    let http = AppState::build_http_client().unwrap();
    Arc::new(AppState::new(config, model_router, http, HashMap::new(), None).unwrap())
}

fn config_with(entries: Vec<ModelEntry>) -> GatewayConfig {
    GatewayConfig {
        model_list: entries,
        mcp_servers: HashMap::new(),
        general_settings: GeneralSettings {
            master_key: Some("sk-local".to_owned()),
            ..Default::default()
        },
        agents: Vec::new(),
    }
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ---- openai_chat inbound → anthropic outbound -----------------------------

#[tokio::test]
async fn chat_in_anthropic_out_tool_call() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-claude",
        "anthropic/claude-sonnet-4-5",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-claude",
                        "messages": [{"role": "user", "content": "weather in SF?"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                            }
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["object"], "chat.completion");
    let tc = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tc["function"]["name"], "get_weather");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "SF");
}

#[tokio::test]
async fn chat_in_anthropic_out_streaming_tool_call() {
    let upstream = MockServer::start().await;
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"SF\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-claude",
        "anthropic/claude-sonnet-4-5",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-claude",
                        "stream": true,
                        "messages": [{"role": "user", "content": "weather?"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    let text = body_text(response).await;
    assert!(text.contains("chat.completion.chunk"), "body: {text}");
    assert!(text.contains("get_weather"));
    assert!(text.contains("\\\"city\\\":"));
    assert!(text.contains("\"finish_reason\":\"tool_calls\""));
    assert!(text.contains("[DONE]"));
}

// ---- anthropic inbound → openai_chat outbound -----------------------------

#[tokio::test]
async fn anthropic_in_chat_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gpt",
        "openai_chat/gpt-4o",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-gpt",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "Hello there");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["model"], "gw-gpt");
}

#[tokio::test]
async fn anthropic_in_chat_out_streaming_text() {
    let upstream = MockServer::start().await;
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gpt",
        "openai_chat/gpt-4o",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-gpt",
                        "max_tokens": 16,
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("event: message_start"), "body: {text}");
    assert!(text.contains("content_block_delta"));
    assert!(text.contains("Hello"));
    assert!(text.contains("event: message_stop"));
}

// ---- openai_responses ↔ anthropic -----------------------------------------

const ANTHROPIC_TEXT_SSE: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

const RESPONSES_TEXT_SSE: &str = concat!(
    "event: response.created\n",
    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"gpt-5\",\"status\":\"in_progress\"}}\n\n",
    "event: response.output_item.added\n",
    "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\n\n",
    "event: response.content_part.added\n",
    "data: {\"type\":\"response.content_part.added\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n",
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\"}}\n\n",
    "event: response.completed\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}}\n\n",
);

#[tokio::test]
async fn responses_in_anthropic_out_tool_call() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-claude",
        "anthropic/claude-sonnet-4-5",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-claude",
                        "input": [{"role": "user", "content": [{"type": "input_text", "text": "weather in SF?"}]}],
                        "tools": [{
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["object"], "response");
    let fc = &body["output"][0];
    assert_eq!(fc["type"], "function_call");
    assert_eq!(fc["name"], "get_weather");
    let args: Value = serde_json::from_str(fc["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "SF");
}

#[tokio::test]
async fn anthropic_in_responses_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw-oai", "openai/gpt-5", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-oai",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["text"], "Hi there");
    assert_eq!(body["stop_reason"], "end_turn");
}

#[tokio::test]
async fn responses_in_anthropic_out_streaming_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_TEXT_SSE),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-claude",
        "anthropic/claude-sonnet-4-5",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "gw-claude", "stream": true, "input": "hi"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("response.output_text.delta"), "body: {text}");
    assert!(text.contains("Hello"));
    assert!(text.contains("response.completed"));
}

#[tokio::test]
async fn anthropic_in_responses_out_streaming_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(RESPONSES_TEXT_SSE),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw-oai", "openai/gpt-5", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-oai",
                        "max_tokens": 16,
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("event: message_start"), "body: {text}");
    assert!(text.contains("Hello"));
    assert!(text.contains("event: message_stop"));
}

// ---- gemini ↔ anthropic ---------------------------------------------------

#[tokio::test]
async fn gemini_in_anthropic_out_tool_call() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gemini",
        "anthropic/claude-sonnet-4-5",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1beta/models/gw-gemini:generateContent")
                .header("x-goog-api-key", "sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "contents": [{"role": "user", "parts": [{"text": "weather in SF?"}]}],
                        "tools": [{"functionDeclarations": [{
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                        }]}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    let fc = &body["candidates"][0]["content"]["parts"][0]["functionCall"];
    assert_eq!(fc["name"], "get_weather");
    assert_eq!(fc["args"]["city"], "SF");
}

#[tokio::test]
async fn anthropic_in_gemini_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hi there"}]},
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gem",
        "gemini/gemini-2.5-pro",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-gem",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["text"], "Hi there");
    assert_eq!(body["stop_reason"], "end_turn");
}

#[tokio::test]
async fn gemini_in_anthropic_out_streaming_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(ANTHROPIC_TEXT_SSE),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gemini",
        "anthropic/claude-sonnet-4-5",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1beta/models/gw-gemini:streamGenerateContent?alt=sse")
                .header("x-goog-api-key", "sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("candidates"), "body: {text}");
    assert!(text.contains("Hello"));
    assert!(text.contains("finishReason"));
}

#[tokio::test]
async fn anthropic_in_gemini_out_streaming_text() {
    let upstream = MockServer::start().await;
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]},\"index\":0}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]},\"index\":0}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gem",
        "gemini/gemini-2.5-pro",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-gem",
                        "max_tokens": 16,
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("event: message_start"), "body: {text}");
    assert!(text.contains("content_block_delta"));
    assert!(text.contains("Hel"));
    assert!(text.contains("lo"));
    assert!(text.contains("event: message_stop"));
}

/// Gemini reports finishReason STOP even when it streamed a functionCall. The
/// stop reason must still surface as a tool call to the client (regression).
#[tokio::test]
async fn chat_in_gemini_out_streaming_tool_call_stop_reason() {
    let upstream = MockServer::start().await;
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}}]},\"index\":0}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gem",
        "gemini/gemini-2.5-pro",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-gem",
                        "stream": true,
                        "messages": [{"role": "user", "content": "weather?"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("get_weather"), "body: {text}");
    assert!(
        text.contains("\"finish_reason\":\"tool_calls\""),
        "stop reason must be tool_calls, not stop: {text}"
    );
    assert!(text.contains("[DONE]"));
}

/// Gemini streams `thought` parts before the answer text. The two must land on
/// separate content blocks (thinking then text), not share one index, or the
/// answer text gets mislabelled as thinking (regression).
#[tokio::test]
async fn anthropic_in_gemini_out_streaming_thinking_then_text() {
    let upstream = MockServer::start().await;
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"pondering\",\"thought\":true}]},\"index\":0}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Answer\"}]},\"index\":0}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw-gem",
        "gemini/gemini-2.5-pro",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw-gem",
                        "max_tokens": 16,
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    // A thinking block carrying the thought, and a *separate* text block for the
    // answer — proving the answer wasn't merged onto the thinking index. (serde
    // serializes object keys alphabetically, hence text-before-type below.)
    assert!(text.contains("thinking_delta"), "body: {text}");
    assert!(text.contains("pondering"), "body: {text}");
    assert!(
        text.contains("\"text\":\"Answer\",\"type\":\"text_delta\""),
        "answer must be a text_delta, not thinking: {text}"
    );
    assert!(
        text.contains("{\"text\":\"\",\"type\":\"text\"}"),
        "a dedicated text content_block_start must open: {text}"
    );
}

// ---- openai_chat ↔ openai_responses ---------------------------------------

#[tokio::test]
async fn chat_in_responses_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "Hi there");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn chat_in_responses_out_streaming_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(RESPONSES_TEXT_SSE),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "gw",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("chat.completion.chunk"), "body: {text}");
    assert!(text.contains("Hello"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn responses_in_chat_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "gw", "input": "hi"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["output"][0]["content"][0]["text"], "Hello there");
    assert_eq!(body["status"], "completed");
}

#[tokio::test]
async fn responses_in_chat_out_streaming_text() {
    let upstream = MockServer::start().await;
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "gw", "stream": true, "input": "hi"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await;
    assert!(text.contains("response.output_text.delta"), "body: {text}");
    assert!(text.contains("Hello"));
    assert!(text.contains("response.completed"));
}

// ---- gemini ↔ openai_chat / openai_responses ------------------------------

#[tokio::test]
async fn gemini_in_chat_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1beta/models/gw:generateContent")
                .header("x-goog-api-key", "sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["candidates"][0]["content"]["parts"][0]["text"], "Hi there");
    assert!(body["candidates"][0]["finishReason"].is_string());
}

#[tokio::test]
async fn gemini_in_chat_out_tool_call() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai_chat/gpt-4o", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1beta/models/gw:generateContent")
                .header("x-goog-api-key", "sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "contents": [{"role": "user", "parts": [{"text": "weather in SF?"}]}],
                        "tools": [{"functionDeclarations": [{
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                        }]}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    let fc = &body["candidates"][0]["content"]["parts"][0]["functionCall"];
    assert_eq!(fc["name"], "get_weather");
    assert_eq!(fc["args"]["city"], "SF");
}

#[tokio::test]
async fn responses_in_gemini_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hi there"}]},
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry(
        "gw",
        "gemini/gemini-2.5-pro",
        &upstream.uri(),
    )]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::AUTHORIZATION, "Bearer sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "gw", "input": "hi"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["output"][0]["content"][0]["text"], "Hi there");
}

#[tokio::test]
async fn gemini_in_responses_out_text() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "object": "response",
            "model": "gpt-5",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .mount(&upstream)
        .await;

    let config = config_with(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let app = router(build_state(config));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1beta/models/gw:generateContent")
                .header("x-goog-api-key", "sk-local")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await).unwrap();
    assert_eq!(body["candidates"][0]["content"]["parts"][0]["text"], "Hi there");
}
