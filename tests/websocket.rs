use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use futures_util::{SinkExt, StreamExt};
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
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, Message},
};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[path = "conversion_support/mod.rs"]
mod support;

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

async fn serve(state: Arc<AppState>) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router(state)).await.unwrap();
    });
    addr
}

async fn websocket_roundtrip(
    addr: SocketAddr,
    endpoint: &str,
    event: Value,
) -> (Vec<Value>, Value) {
    let url = format!("ws://{addr}{endpoint}");
    let mut request = url.into_client_request().unwrap();
    request
        .headers_mut()
        .insert("Authorization", HeaderValue::from_static("Bearer sk-local"));
    let (mut socket, _) = connect_async(request).await.unwrap();
    socket
        .send(Message::Text(event.to_string().into()))
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(message) = socket.next().await {
        let message = message.unwrap();
        let Message::Text(text) = message else {
            continue;
        };
        let event: Value = serde_json::from_str(&text).unwrap();
        let done = matches!(
            event.get("type").and_then(Value::as_str),
            Some("response.completed")
                | Some("message_stop")
                | Some("done")
                | Some("response.failed")
                | Some("response.incomplete")
        ) || event
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("finishReason"))
            .is_some();
        events.push(event);
        if done {
            break;
        }
    }
    socket.close(None).await.unwrap();
    let last = events.last().cloned().unwrap();
    (events, last)
}

async fn mount_responses_sse(upstream: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(support::RESPONSES_TEXT_SSE.as_bytes(), "text/event-stream"),
        )
        .mount(upstream)
        .await;
}

#[tokio::test]
async fn codex_responses_websocket_forwards_responses_events() {
    let upstream = MockServer::start().await;
    mount_responses_sse(&upstream).await;
    let state = build_state(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let addr = serve(state).await;

    let (events, last) = websocket_roundtrip(
        addr,
        "/v1/responses",
        json!({
            "type": "response.create",
            "model": "gw",
            "input": "hi"
        }),
    )
    .await;

    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(last["type"], "response.completed");
    let reqs = upstream.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1);
    let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(body["model"], "gpt-5");
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn websocket_accepts_chat_protocol() {
    let upstream = MockServer::start().await;
    mount_responses_sse(&upstream).await;
    let state = build_state(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let addr = serve(state).await;

    let (_, last) = websocket_roundtrip(
        addr,
        "/v1/responses",
        json!({
            "type": "response.create",
            "protocol": "chat",
            "model": "gw",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(last["type"], "done");
    let reqs = upstream.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(body["input"][0]["role"], "user");
}

#[tokio::test]
async fn websocket_accepts_anthropic_protocol() {
    let upstream = MockServer::start().await;
    mount_responses_sse(&upstream).await;
    let state = build_state(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let addr = serve(state).await;

    let (_, last) = websocket_roundtrip(
        addr,
        "/v1/responses",
        json!({
            "type": "response.create",
            "protocol": "anthropic",
            "model": "gw",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(last["type"], "message_stop");
    let reqs = upstream.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(body["input"][0]["role"], "user");
}

#[tokio::test]
async fn websocket_accepts_gemini_protocol() {
    let upstream = MockServer::start().await;
    mount_responses_sse(&upstream).await;
    let state = build_state(vec![model_entry("gw", "openai/gpt-5", &upstream.uri())]);
    let addr = serve(state).await;

    let (_, last) = websocket_roundtrip(
        addr,
        "/v1/responses",
        json!({
            "type": "response.create",
            "protocol": "gemini",
            "model": "gw",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
        }),
    )
    .await;

    assert_eq!(last["candidates"][0]["finishReason"], "STOP");
    let reqs = upstream.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(body["input"][0]["content"][0]["text"], "hi");
}
