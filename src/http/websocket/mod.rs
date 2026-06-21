mod handler;
mod stream;

use std::sync::Arc;

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::HeaderMap,
    response::IntoResponse,
};

use crate::{
    errors::GatewayError,
    http::gemini,
    proxy::{auth::master_key::require_any_gateway_key, state::AppState},
    sdk::codec::WireFormat,
};

pub async fn responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Result<impl IntoResponse, GatewayError> {
    require_any_gateway_key(&headers, &state)?;
    Ok(
        ws.on_upgrade(move |socket| {
            run_socket(state, headers, WireFormat::OpenAiResponses, socket)
        }),
    )
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Result<impl IntoResponse, GatewayError> {
    require_any_gateway_key(&headers, &state)?;
    Ok(ws.on_upgrade(move |socket| {
        run_socket(state, headers, WireFormat::AnthropicMessages, socket)
    }))
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Result<impl IntoResponse, GatewayError> {
    require_any_gateway_key(&headers, &state)?;
    Ok(ws.on_upgrade(move |socket| run_socket(state, headers, WireFormat::OpenAiChat, socket)))
}

pub async fn gemini(
    State(state): State<Arc<AppState>>,
    Path(model_method): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    mut headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Result<impl IntoResponse, GatewayError> {
    let scope_key = gemini::authorize(&state, &headers, params.get("key").map(String::as_str))?;
    if let Some(key) = scope_key {
        if let Ok(value) = axum::http::HeaderValue::from_str(&format!("Bearer {key}")) {
            headers.insert(axum::http::header::AUTHORIZATION, value);
        }
    }
    let Some((model, method)) = model_method.split_once(':') else {
        return Err(GatewayError::InvalidJsonMessage(
            "gemini path must be models/{model}:{method}".to_owned(),
        ));
    };
    if !matches!(method, "generateContent" | "streamGenerateContent") {
        return Err(GatewayError::InvalidJsonMessage(format!(
            "unsupported gemini method: {method}"
        )));
    }
    let model = model.to_owned();
    Ok(ws.on_upgrade(move |socket| {
        run_socket_with_model(state, headers, WireFormat::Gemini, model, socket)
    }))
}

async fn run_socket(
    state: Arc<AppState>,
    headers: HeaderMap,
    default_wire: WireFormat,
    socket: WebSocket,
) {
    handler::run(state, headers, default_wire, None, socket).await;
}

async fn run_socket_with_model(
    state: Arc<AppState>,
    headers: HeaderMap,
    default_wire: WireFormat,
    model: String,
    socket: WebSocket,
) {
    handler::run(state, headers, default_wire, Some(model), socket).await;
}
