//! Exact-match response cache: hits skip the upstream entirely.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{header, Request, StatusCode},
    response::Response,
};
use litellm_rust::{
    http::routes::router,
    proxy::{
        config::{
            CacheBackendKind, CacheSettings, GatewayConfig, GeneralSettings, LiteLlmParams,
            ModelEntry,
        },
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

fn cache_config(api_base: String, master_key: Option<&str>) -> GatewayConfig {
    cache_config_with(
        api_base,
        master_key,
        CacheSettings {
            enabled: true,
            ..Default::default()
        },
    )
}

fn cache_config_with(
    api_base: String,
    master_key: Option<&str>,
    cache: CacheSettings,
) -> GatewayConfig {
    GatewayConfig {
        model_list: vec![ModelEntry {
            model_name: "claude".to_owned(),
            litellm_params: LiteLlmParams {
                model: "anthropic/claude-sonnet-4-5".to_owned(),
                api_key: Some("sk-ant-test".to_owned()),
                api_base: Some(api_base),
                wire_api: None,
                extra: Default::default(),
            },
        }],
        mcp_servers: HashMap::new(),
        general_settings: GeneralSettings {
            master_key: master_key.map(str::to_owned),
            cache,
            ..Default::default()
        },
        agents: Vec::new(),
    }
}

fn build_state(config: &GatewayConfig) -> Arc<AppState> {
    let mut providers = ProviderRegistry::new();
    providers::register_all(&mut providers);
    let model_router = ModelRouter::from_config(config, &providers).unwrap();
    let http = AppState::build_http_client().unwrap();
    Arc::new(AppState::new(config.clone(), model_router, http, HashMap::new(), None).unwrap())
}

async fn send(
    state: &Arc<AppState>,
    auth: Option<&str>,
    no_cache: bool,
    body: &Value,
) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(a) = auth {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {a}"));
    }
    if no_cache {
        builder = builder.header("cache-control", "no-cache");
    }
    let req = builder.body(Body::from(body.to_string())).unwrap();
    router(state.clone()).oneshot(req).await.unwrap()
}

async fn body_bytes(resp: Response) -> Vec<u8> {
    to_bytes(resp.into_body(), 1 << 20).await.unwrap().to_vec()
}

fn json_mock() -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
}

fn body() -> Value {
    json!({
        "model": "claude",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    })
}

#[tokio::test]
async fn serves_identical_request_from_cache() {
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    let state = build_state(&cache_config(upstream.uri(), Some("sk-local")));
    let b = body();

    let first = body_bytes(send(&state, Some("sk-local"), false, &b).await).await;

    let r2 = send(&state, Some("sk-local"), false, &b).await;
    assert_eq!(r2.status(), StatusCode::OK);
    assert_eq!(r2.headers().get("x-litellm-cache").unwrap(), "hit");
    let second = body_bytes(r2).await;

    assert_eq!(first, second);
    assert_eq!(upstream.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn redb_served_from_cache() {
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    let dir = tempfile::TempDir::new().unwrap();
    let redb_path = dir.path().join("cache.redb");
    let state = build_state(&cache_config_with(
        upstream.uri(),
        Some("sk-local"),
        CacheSettings {
            enabled: true,
            backend: CacheBackendKind::Redb,
            redb_path: Some(redb_path.to_str().unwrap().to_owned()),
            ..Default::default()
        },
    ));
    let b = body();

    let first = body_bytes(send(&state, Some("sk-local"), false, &b).await).await;

    let r2 = send(&state, Some("sk-local"), false, &b).await;
    assert_eq!(r2.status(), StatusCode::OK);
    assert_eq!(r2.headers().get("x-litellm-cache").unwrap(), "hit");
    let second = body_bytes(r2).await;

    assert_eq!(first, second);
    assert_eq!(upstream.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn isolates_cache_by_api_key() {
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    // No master key configured → any bearer token is accepted, but each is a
    // distinct cache tenant.
    let state = build_state(&cache_config(upstream.uri(), None));
    let b = body();

    let _ = send(&state, Some("tenant-a"), false, &b).await;
    let r2 = send(&state, Some("tenant-b"), false, &b).await;
    // Tenant B must NOT see tenant A's cached response.
    assert!(r2.headers().get("x-litellm-cache").is_none());

    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn no_cache_directive_bypasses() {
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    let state = build_state(&cache_config(upstream.uri(), Some("sk-local")));
    let b = body();

    let _ = send(&state, Some("sk-local"), false, &b).await;
    let _ = send(&state, Some("sk-local"), true, &b).await; // cache-control: no-cache

    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn body_cache_no_cache_param_bypasses() {
    // Upstream litellm honours a request-body `cache: {no-cache: true}`; so must we.
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    let state = build_state(&cache_config(upstream.uri(), Some("sk-local")));

    let _ = send(&state, Some("sk-local"), false, &body()).await;
    let mut with_param = body();
    with_param["cache"] = json!({"no-cache": true});
    let _ = send(&state, Some("sk-local"), false, &with_param).await;

    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn body_cache_param_stripped_so_no_store_reads_hit() {
    // The proprietary `cache` control field must be stripped before keying, so a
    // request carrying it still hits the entry stored by a plain request (and
    // no-store permits reads) — proving no key fragmentation.
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    let state = build_state(&cache_config(upstream.uri(), Some("sk-local")));

    let _ = send(&state, Some("sk-local"), false, &body()).await;
    let mut with_param = body();
    with_param["cache"] = json!({"no-store": true});
    let r2 = send(&state, Some("sk-local"), false, &with_param).await;
    assert_eq!(r2.headers().get("x-litellm-cache").unwrap(), "hit");
    assert_eq!(upstream.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn skips_non_deterministic_requests() {
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    let state = build_state(&cache_config(upstream.uri(), Some("sk-local")));
    let b = json!({
        "model": "claude",
        "max_tokens": 16,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let _ = send(&state, Some("sk-local"), false, &b).await;
    let _ = send(&state, Some("sk-local"), false, &b).await;

    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn unauthenticated_requests_not_cached_when_scoped() {
    let upstream = MockServer::start().await;
    json_mock().mount(&upstream).await;
    // No master key (auth optional) + scope_by_api_key (default on): a request
    // with no API key can't be isolated, so it must not be cached.
    let state = build_state(&cache_config(upstream.uri(), None));
    let b = body();

    let _ = send(&state, None, false, &b).await;
    let r2 = send(&state, None, false, &b).await;
    assert!(r2.headers().get("x-litellm-cache").is_none());

    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn oversized_stream_not_cached() {
    let upstream = MockServer::start().await;
    let sse = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;
    let state = build_state(&cache_config_with(
        upstream.uri(),
        Some("sk-local"),
        CacheSettings {
            enabled: true,
            max_stream_bytes: 8, // far below the SSE body → buffering aborts
            ..Default::default()
        },
    ));
    let b = json!({
        "model": "claude",
        "max_tokens": 16,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let _ = body_bytes(send(&state, Some("sk-local"), false, &b).await).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let r2 = send(&state, Some("sk-local"), false, &b).await;
    // Over-cap stream was forwarded but not stored → second request hits upstream.
    assert!(r2.headers().get("x-litellm-cache").is_none());

    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn caches_and_replays_streaming() {
    let upstream = MockServer::start().await;
    let sse = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&upstream)
        .await;
    let state = build_state(&cache_config(upstream.uri(), Some("sk-local")));
    let b = json!({
        "model": "claude",
        "max_tokens": 16,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let first = body_bytes(send(&state, Some("sk-local"), false, &b).await).await;
    // The streaming store is spawned on clean completion; give it a moment.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let r2 = send(&state, Some("sk-local"), false, &b).await;
    assert_eq!(r2.status(), StatusCode::OK);
    assert_eq!(r2.headers().get("x-litellm-cache").unwrap(), "hit");
    assert_eq!(
        r2.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    let second = body_bytes(r2).await;

    assert_eq!(first, second);
    assert_eq!(upstream.received_requests().await.unwrap().len(), 1);
}
