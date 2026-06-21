use std::sync::Arc;

use axum::{
    extract::ws::{Message, WebSocket},
    http::HeaderMap,
};
use serde_json::{json, Value};

use crate::{
    errors::GatewayError,
    http::{pipeline, websocket::stream},
    proxy::state::AppState,
    sdk::codec::WireFormat,
};

pub async fn run(
    state: Arc<AppState>,
    headers: HeaderMap,
    default_wire: WireFormat,
    model_from_path: Option<String>,
    mut socket: WebSocket,
) {
    while let Some(message) = socket.recv().await {
        let result = match message {
            Ok(Message::Text(text)) => {
                handle_text_event(
                    &state,
                    &headers,
                    default_wire,
                    model_from_path.as_deref(),
                    &mut socket,
                    &text,
                )
                .await
            }
            Ok(Message::Binary(bytes)) => {
                let text = String::from_utf8_lossy(&bytes);
                handle_text_event(
                    &state,
                    &headers,
                    default_wire,
                    model_from_path.as_deref(),
                    &mut socket,
                    &text,
                )
                .await
            }
            Ok(Message::Ping(bytes)) => socket.send(Message::Pong(bytes)).await.map_err(ws_error),
            Ok(Message::Pong(_)) => Ok(()),
            Ok(Message::Close(frame)) => {
                let _ = socket.send(Message::Close(frame)).await;
                break;
            }
            Err(err) => Err(GatewayError::WebSocket(err.to_string())),
        };

        if let Err(err) = result {
            let _ = send_error(&mut socket, err.to_string()).await;
        }
    }
}

async fn handle_text_event(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    default_wire: WireFormat,
    model_from_path: Option<&str>,
    socket: &mut WebSocket,
    text: &str,
) -> Result<(), GatewayError> {
    let event: Value = serde_json::from_str(text).map_err(GatewayError::InvalidJson)?;
    let t = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("response.create");
    if t != "response.create" {
        return send_ack(socket, t).await;
    }
    let inbound_wire = event
        .get("protocol")
        .or_else(|| event.get("wire_api"))
        .and_then(Value::as_str)
        .and_then(WireFormat::parse)
        .unwrap_or(default_wire);
    let mut body = request_body(event)?;
    if let Some(model) = model_from_path {
        body["model"] = json!(model);
    }
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or(GatewayError::MissingModel)?
        .to_owned();
    body["stream"] = json!(true);

    let prepared =
        pipeline::prepare_upstream(state, inbound_wire, model, true, body, headers).await?;
    stream::proxy_upstream(socket, state, prepared).await
}

fn request_body(event: Value) -> Result<Value, GatewayError> {
    let body = event
        .get("response")
        .or_else(|| event.get("request"))
        .cloned()
        .unwrap_or_else(|| request_from_create_event(event.clone()));
    if body.is_object() {
        Ok(body)
    } else {
        Err(GatewayError::InvalidJsonMessage(
            "websocket response.create must contain a JSON object response".to_owned(),
        ))
    }
}

fn request_from_create_event(mut event: Value) -> Value {
    if let Some(obj) = event.as_object_mut() {
        obj.remove("type");
        obj.remove("event_id");
        obj.remove("protocol");
        obj.remove("wire_api");
    }
    event
}

async fn send_ack(socket: &mut WebSocket, event_type: &str) -> Result<(), GatewayError> {
    socket
        .send(Message::Text(
            json!({"type": "ack", "event": event_type})
                .to_string()
                .into(),
        ))
        .await
        .map_err(ws_error)
}

async fn send_error(socket: &mut WebSocket, message: String) -> Result<(), GatewayError> {
    socket
        .send(Message::Text(
            json!({"type": "error", "error": {"message": message}})
                .to_string()
                .into(),
        ))
        .await
        .map_err(ws_error)
}

fn ws_error(err: axum::Error) -> GatewayError {
    GatewayError::WebSocket(err.to_string())
}
