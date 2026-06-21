use std::{collections::HashMap, fs, sync::Arc};

use axum::{
    body::{to_bytes, Body},
    http::{header, Request, StatusCode},
};
use litellm_rust::{
    http::routes::router,
    proxy::{
        config::{
            ComplexityRoutingConfig, ComplexityScorerKind, ComplexityTierConfig, GatewayConfig,
            GeneralSettings, LiteLlmParams, ModelEntry,
        },
        state::AppState,
    },
    sdk::{
        providers::{self, transform::ProviderRegistry},
        router::Router as ModelRouter,
    },
};
use tempfile::TempDir;
use tower::util::ServiceExt;

#[tokio::test]
async fn serves_static_ui() {
    let ui_dir = write_ui_fixture();
    std::env::set_var("LITELLM_UI_DIR", ui_dir.path());
    let app = router(build_state(&test_config()));

    assert_redirects_to_sessions(app.clone()).await;
    assert_serves_sessions_html(app).await;
}

#[tokio::test]
async fn lists_models_in_openai_shape() {
    let app = router(build_state(&test_config()));

    let response = get_authed(app, "/v1/models").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;

    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "claude");
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["owned_by"], "anthropic");
    assert_eq!(body["data"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn lists_models_in_anthropic_shape() {
    let app = router(build_state(&test_config()));

    let response = get_authed(app, "/v1/messages/models").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;

    assert_eq!(body["data"][0]["type"], "model");
    assert_eq!(body["data"][0]["id"], "claude");
    assert_eq!(body["has_more"], false);
    assert_eq!(body["first_id"], "claude");
    assert_eq!(body["last_id"], "gemini/*");
}

#[tokio::test]
async fn lists_models_in_gemini_shape() {
    let app = router(build_state(&test_config()));

    let response = get_authed(app, "/v1beta/models").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;

    assert_eq!(body["models"][0]["name"], "models/claude");
    assert_eq!(body["models"][0]["baseModelId"], "claude-sonnet-4-5");
    assert_eq!(
        body["models"][0]["supportedGenerationMethods"][0],
        "generateContent"
    );
}

#[tokio::test]
async fn v1_models_detects_shape_from_headers_and_user_agent() {
    for (headers, assertion) in [
        (
            vec![("anthropic-version", "2023-06-01")],
            assert_anthropic_shape as fn(&serde_json::Value),
        ),
        (
            vec![(header::USER_AGENT.as_str(), "anthropic-python/1.2.3")],
            assert_anthropic_shape,
        ),
        (
            vec![("x-goog-api-client", "genai-js/0.24.1")],
            assert_gemini_shape,
        ),
        (
            vec![(header::USER_AGENT.as_str(), "google-genai-sdk/1.0")],
            assert_gemini_shape,
        ),
        (vec![("openai-project", "proj_test")], assert_openai_shape),
    ] {
        let app = router(build_state(&test_config()));
        let response = get_authed_with_headers(app, "/v1/models", &headers).await;
        assert_eq!(response.status(), StatusCode::OK);
        assertion(&json_body(response).await);
    }
}

#[tokio::test]
async fn v1_models_can_render_protocol_shape_by_format() {
    let app = router(build_state(&test_config()));

    let anthropic = json_body(get_authed(app.clone(), "/v1/models?format=anthropic").await).await;
    let gemini = json_body(get_authed(app, "/v1/models?format=gemini").await).await;

    assert_eq!(anthropic["data"][0]["type"], "model");
    assert_eq!(gemini["models"][0]["name"], "models/claude");
}

#[tokio::test]
async fn v1_models_query_format_overrides_detected_headers() {
    let app = router(build_state(&test_config()));

    let response = get_authed_with_headers(
        app,
        "/v1/models?format=gemini",
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;

    assert_eq!(body["models"][0]["name"], "models/claude");
}

fn assert_anthropic_shape(body: &serde_json::Value) {
    assert_eq!(body["data"][0]["type"], "model");
    assert_eq!(body["data"][0]["id"], "claude");
}

fn assert_gemini_shape(body: &serde_json::Value) {
    assert_eq!(body["models"][0]["name"], "models/claude");
}

fn assert_openai_shape(body: &serde_json::Value) {
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "model");
}

fn write_ui_fixture() -> TempDir {
    let ui_dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(ui_dir.path().join("sessions")).unwrap();
    fs::write(
        ui_dir.path().join("sessions/index.html"),
        "<html>sessions</html>",
    )
    .unwrap();
    fs::write(ui_dir.path().join("404.html"), "<html>not found</html>").unwrap();
    ui_dir
}

async fn assert_redirects_to_sessions(app: axum::Router) {
    let response = get(app, "/").await;
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(
        response.headers().get(header::LOCATION).unwrap(),
        "/sessions/"
    );
}

async fn assert_serves_sessions_html(app: axum::Router) {
    let response = get(app, "/sessions/").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    assert!(std::str::from_utf8(&body).unwrap().contains("sessions"));
}

async fn get(app: axum::Router, uri: &str) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn get_authed(app: axum::Router, uri: &str) -> axum::response::Response {
    get_authed_with_headers(app, uri, &[]).await
}

async fn get_authed_with_headers(
    app: axum::Router,
    uri: &str,
    headers: &[(&str, &str)],
) -> axum::response::Response {
    let mut request = Request::builder()
        .method("GET")
        .uri(uri)
        .header(header::AUTHORIZATION, "Bearer sk-local");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    app.oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> serde_json::Value {
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn test_config() -> GatewayConfig {
    GatewayConfig {
        model_list: vec![
            model_entry("claude", "anthropic/claude-sonnet-4-5"),
            model_entry("gpt", "openai/gpt-5"),
            model_entry("gemini/*", "gemini/*"),
            complexity_entry("auto"),
        ],
        mcp_servers: HashMap::new(),
        general_settings: GeneralSettings {
            master_key: Some("sk-local".to_owned()),
            ..Default::default()
        },
        agents: Vec::new(),
    }
}

fn model_entry(model_name: &str, model: &str) -> ModelEntry {
    ModelEntry {
        model_name: model_name.to_owned(),
        litellm_params: LiteLlmParams {
            model: model.to_owned(),
            api_key: Some("sk-test".to_owned()),
            api_base: Some("http://127.0.0.1:1".to_owned()),
            wire_api: None,
            complexity_routing: None,
            extra: Default::default(),
        },
    }
}

fn complexity_entry(model_name: &str) -> ModelEntry {
    let mut entry = model_entry(model_name, "");
    entry.litellm_params.complexity_routing = Some(ComplexityRoutingConfig {
        scorer: ComplexityScorerKind::Heuristic,
        tiers: vec![ComplexityTierConfig {
            max_score: None,
            model: "claude".to_owned(),
        }],
    });
    entry
}

fn build_router(config: &GatewayConfig) -> ModelRouter {
    let mut providers = ProviderRegistry::new();
    providers::register_all(&mut providers);
    ModelRouter::from_config(config, &providers).unwrap()
}

fn build_state(config: &GatewayConfig) -> Arc<AppState> {
    let http = AppState::build_http_client().unwrap();
    Arc::new(
        AppState::new(
            config.clone(),
            build_router(config),
            http,
            HashMap::new(),
            None,
        )
        .unwrap(),
    )
}
