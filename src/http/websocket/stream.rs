use std::sync::Arc;

use axum::{
    extract::ws::{Message, WebSocket},
    http::{header, HeaderMap},
};
use futures_util::StreamExt;
use serde_json::json;

use crate::{
    errors::GatewayError,
    http::{
        llm,
        pipeline::{self, PreparedUpstreamRequest},
    },
    proxy::state::AppState,
    sdk::codec::{codec_for, stream::SseDecoder},
};

pub async fn proxy_upstream(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    prepared: PreparedUpstreamRequest,
) -> Result<(), GatewayError> {
    let ctx = pipeline::websocket_response_context(&prepared);
    let upstream = llm::send_request(
        &state.http,
        prepared.url,
        serde_json::to_vec(&prepared.body)?,
        prepared.headers,
    )
    .await?;
    if !upstream.status().is_success() {
        let status = upstream.status();
        let bytes = upstream.bytes().await.map_err(GatewayError::Upstream)?;
        return Err(GatewayError::WebSocket(format!(
            "upstream returned {status}: {}",
            String::from_utf8_lossy(&bytes)
        )));
    }
    if !is_event_stream(upstream.headers()) {
        let bytes = upstream.bytes().await.map_err(GatewayError::Upstream)?;
        return Err(GatewayError::WebSocket(format!(
            "upstream returned non-SSE response: {}",
            String::from_utf8_lossy(&bytes)
        )));
    }

    let mut stream = upstream.bytes_stream();
    if prepared.inbound_wire == prepared.outbound_wire {
        let mut decoder = SseDecoder::default();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(GatewayError::Upstream)?;
            for event in decoder.push(&bytes) {
                send_sse_event(socket, event.data).await?;
            }
        }
        return Ok(());
    }

    let mut parser = codec_for(prepared.outbound_wire).stream_parser();
    let mut renderer = codec_for(prepared.inbound_wire).stream_renderer(&ctx);
    let mut decoder = SseDecoder::default();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(GatewayError::Upstream)?;
        for event in decoder.push(&bytes) {
            for ir_event in parser.push(&event)? {
                send_rendered(socket, renderer.push(&ir_event)).await?;
            }
        }
    }
    for ir_event in parser.finish() {
        send_rendered(socket, renderer.push(&ir_event)).await?;
    }
    send_rendered(socket, renderer.finish()).await
}

async fn send_rendered(socket: &mut WebSocket, bytes: Vec<u8>) -> Result<(), GatewayError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let text = String::from_utf8(bytes).map_err(|err| GatewayError::WebSocket(err.to_string()))?;
    let mut decoder = SseDecoder::default();
    for event in decoder.push(text.as_bytes()) {
        send_sse_event(socket, event.data).await?;
    }
    Ok(())
}

async fn send_sse_event(socket: &mut WebSocket, data: String) -> Result<(), GatewayError> {
    let data = if data == "[DONE]" {
        json!({"type": "done"}).to_string()
    } else {
        data
    };
    socket
        .send(Message::Text(data.into()))
        .await
        .map_err(|err| GatewayError::WebSocket(err.to_string()))
}

fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
}
