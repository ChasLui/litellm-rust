//! Request pipeline: translate inbound → IR → outbound, send upstream, then
//! translate the response back to the inbound protocol. When inbound and
//! outbound wire formats match, the body is passed through untouched (fast
//! path), preserving the original low-overhead behaviour.

use std::{pin::Pin, sync::Arc};

use axum::{
    http::{HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

use crate::{
    errors::GatewayError,
    http::{credential_overrides, llm},
    proxy::state::AppState,
    sdk::codec::{
        codec_for,
        stream::{SseDecoder, StreamParser, StreamRenderer},
        RequestCtx, WireFormat,
    },
};

/// Drive one request through the gateway. `model` is the public model name (from
/// the body or, for Gemini, the URL path); `stream` is the resolved streaming
/// flag for the request.
pub async fn handle(
    state: &Arc<AppState>,
    inbound_wire: WireFormat,
    model: String,
    stream: bool,
    body: Value,
    inbound_headers: &HeaderMap,
) -> Result<Response, GatewayError> {
    let route = credential_overrides::apply(state, state.router.resolve(&model)?).await?;
    let deployment = route.deployment;
    let out_wire = deployment.wire;
    let url = deployment.upstream_url(stream);

    // Fast path: same protocol both sides — rewrite the model and pass through.
    if inbound_wire == out_wire {
        let out_codec = codec_for(out_wire);
        let mut body = body;
        // Gemini carries the model in the URL, so its body has none to rewrite.
        if out_wire != WireFormat::Gemini
            && body.get("model").and_then(Value::as_str) != Some(deployment.upstream_model.as_str())
        {
            body["model"] = json!(deployment.upstream_model);
        }
        let headers = out_codec.outbound_headers(&deployment, inbound_headers)?;
        let upstream =
            llm::send_request(&state.http, url, serde_json::to_vec(&body)?, headers).await?;
        let resp_headers = out_codec.response_headers(upstream.headers(), stream);
        return Ok(llm::build_response(upstream, resp_headers).await);
    }

    // Cross-protocol: parse to IR, render to the outbound wire.
    let in_codec = codec_for(inbound_wire);
    let out_codec = codec_for(out_wire);
    let ctx = RequestCtx {
        model: model.clone(),
        stream,
    };

    let mut ir_req = in_codec.parse_request(body)?;
    ir_req.model = deployment.upstream_model.clone();
    ir_req.stream = stream;
    let out_body = out_codec.render_request(&ir_req)?;
    let headers = out_codec.outbound_headers(&deployment, inbound_headers)?;
    let upstream =
        llm::send_request(&state.http, url, serde_json::to_vec(&out_body)?, headers).await?;

    let status = upstream.status();
    if !status.is_success() {
        // Provider errors are passed through as-is, not translated.
        let err_headers = in_codec.response_headers(upstream.headers(), false);
        return Ok(llm::build_response(upstream, err_headers).await);
    }

    if stream {
        let resp_headers = in_codec.response_headers(upstream.headers(), true);
        let parser = out_codec.stream_parser();
        let renderer = in_codec.stream_renderer(&ctx);
        let body_stream = transform_stream(upstream, parser, renderer);
        return Ok(llm::build_stream_response(
            StatusCode::OK,
            resp_headers,
            body_stream,
        ));
    }

    let resp_headers = in_codec.response_headers(upstream.headers(), false);
    let bytes = upstream.bytes().await.map_err(GatewayError::Upstream)?;
    let upstream_json: Value = serde_json::from_slice(&bytes).map_err(GatewayError::InvalidJson)?;
    let ir_resp = out_codec.parse_response(upstream_json)?;
    let client_value = in_codec.render_response(&ir_resp, &ctx)?;
    Ok(llm::build_bytes_response(
        status,
        resp_headers,
        serde_json::to_vec(&client_value)?,
    ))
}

struct StreamState {
    upstream: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    decoder: SseDecoder,
    parser: Box<dyn StreamParser>,
    renderer: Box<dyn StreamRenderer>,
    finished: bool,
}

/// Bridge the upstream SSE byte stream through the outbound parser and inbound
/// renderer, re-emitting the client protocol's bytes.
fn transform_stream(
    upstream: reqwest::Response,
    parser: Box<dyn StreamParser>,
    renderer: Box<dyn StreamRenderer>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
    let state = StreamState {
        upstream: Box::pin(upstream.bytes_stream()),
        decoder: SseDecoder::new(),
        parser,
        renderer,
        finished: false,
    };

    futures_util::stream::unfold(state, |mut state| async move {
        loop {
            if state.finished {
                return None;
            }
            match state.upstream.next().await {
                Some(Ok(chunk)) => {
                    let mut out = Vec::new();
                    for sse in state.decoder.push(&chunk) {
                        match state.parser.push(&sse) {
                            Ok(events) => {
                                for ev in events {
                                    out.extend(state.renderer.push(&ev));
                                }
                            }
                            Err(e) => tracing::warn!(error = %e, "stream parse error"),
                        }
                    }
                    if out.is_empty() {
                        continue;
                    }
                    return Some((Ok(Bytes::from(out)), state));
                }
                Some(Err(e)) => {
                    state.finished = true;
                    return Some((Err(std::io::Error::other(e)), state));
                }
                None => {
                    let mut out = Vec::new();
                    for ev in state.parser.finish() {
                        out.extend(state.renderer.push(&ev));
                    }
                    out.extend(state.renderer.finish());
                    state.finished = true;
                    if out.is_empty() {
                        return None;
                    }
                    return Some((Ok(Bytes::from(out)), state));
                }
            }
        }
    })
}
