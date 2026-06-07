//! Request pipeline: translate inbound → IR → outbound, send upstream, then
//! translate the response back to the inbound protocol. When inbound and
//! outbound wire formats match, the body is passed through untouched (fast
//! path), preserving the original low-overhead behaviour.

use std::{pin::Pin, sync::Arc};

use axum::{
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::{
    errors::GatewayError,
    http::{credential_overrides, llm},
    proxy::{
        auth::master_key::presented_key,
        cache::{key as cache_key, semantic, CachedResponse},
        state::AppState,
    },
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
    mut body: Value,
    inbound_headers: &HeaderMap,
) -> Result<Response, GatewayError> {
    let route = credential_overrides::apply(state, state.router.resolve(&model)?).await?;
    let deployment = route.deployment;
    let out_wire = deployment.wire;
    let url = deployment.upstream_url(stream);

    // Response cache (exact-match) + optional semantic cache. Both try a read and
    // remember what to store on a miss. Skipped entirely when disabled, leaving
    // the request path unchanged.
    let cache_settings = &state.config.general_settings.cache;
    let any_cache = state.cache.is_enabled() || state.semantic.is_enabled();
    let directive = if any_cache {
        let directive = cache_key::read_directive(inbound_headers, &body);
        // Strip the litellm-proprietary `cache` control field after reading it so
        // it neither fragments the cache key nor reaches the upstream provider
        // (which would reject the unknown body param) on the same-protocol fast path.
        if let Some(obj) = body.as_object_mut() {
            obj.remove("cache");
        }
        directive
    } else {
        cache_key::CacheDirective {
            read: false,
            store: false,
        }
    };
    let scope = if any_cache && cache_settings.scope_by_api_key {
        presented_key(inbound_headers).map(cache_key::hash_scope)
    } else {
        None
    };
    let scope_str = scope.as_deref().unwrap_or("");
    // With per-tenant scoping on, a request that presents no API key cannot be
    // safely isolated — caching it would let unauthenticated callers share (and
    // leak) each other's responses. So such requests neither read nor write.
    let scope_ok = !cache_settings.scope_by_api_key || scope.is_some();

    let mut store_key: Option<String> = None;
    if state.cache.is_enabled()
        && scope_ok
        && (directive.read || directive.store)
        && (!stream || cache_settings.cache_streaming)
        && cache_key::is_deterministic(&body, cache_settings)
    {
        let key = cache_key::build_key(
            scope.as_deref(),
            inbound_wire,
            &deployment.provider_id,
            &deployment.api_base,
            &deployment.upstream_model,
            stream,
            &body,
        );
        if directive.read {
            if let Some(hit) = state.cache.get(&key).await {
                return Ok(replay_cached(hit, "hit"));
            }
        }
        if directive.store {
            store_key = Some(key);
        }
    }

    // Semantic cache: deterministic, tool-free, non-streaming requests only.
    let mut semantic_text: Option<String> = None;
    if state.semantic.is_enabled()
        && scope_ok
        && !stream
        && (directive.read || directive.store)
        && cache_key::is_deterministic(&body, cache_settings)
        && semantic::eligible(&body, &cache_settings.semantic)
    {
        let text = semantic::query_text(&body);
        if directive.read {
            if let Some(hit) = state.semantic.lookup(scope_str, &text).await {
                return Ok(replay_cached(hit, "semantic"));
            }
        }
        if directive.store {
            semantic_text = Some(text);
        }
    }

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
        let status = upstream.status();
        // Only cache successful responses; errors pass through unstored.
        let want_store = status.is_success() && (store_key.is_some() || semantic_text.is_some());
        if want_store {
            let ct = content_type_of(&resp_headers);
            if stream {
                // Semantic caching is non-streaming; only the exact key applies.
                if let Some(key) = store_key {
                    let inner = upstream.bytes_stream().map_err(std::io::Error::other);
                    return Ok(llm::build_stream_response(
                        StatusCode::OK,
                        resp_headers,
                        tee_and_store(
                            state.clone(),
                            key,
                            status.as_u16(),
                            ct,
                            cache_settings.max_stream_bytes,
                            Box::pin(inner),
                        ),
                    ));
                }
                return Ok(llm::build_response(upstream, resp_headers).await);
            }
            let bytes = upstream.bytes().await.map_err(GatewayError::Upstream)?;
            store_response(
                state,
                store_key,
                semantic_text.as_deref().map(|t| (scope_str, t)),
                status.as_u16(),
                ct,
                bytes.to_vec(),
            )
            .await;
            return Ok(llm::build_bytes_response(status, resp_headers, bytes.to_vec()));
        }
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
    // Auto-inject Anthropic cache breakpoints for clients that can't express
    // them, when routed to an Anthropic upstream and the operator opted in.
    if out_wire == WireFormat::AnthropicMessages {
        let pc = &state.config.general_settings.prompt_caching;
        if pc.enabled && pc.auto_inject {
            crate::sdk::codec::cache_inject::auto_inject_anthropic_breakpoints(
                &mut ir_req,
                pc.max_breakpoints as usize,
                pc.min_tokens,
                pc.chars_per_token,
            );
        }
    }
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
        if let Some(key) = store_key {
            let ct = content_type_of(&resp_headers);
            // Cross-protocol streaming always responds 200 (the live path does
            // too, above), so the cached status matches what a fresh call returns.
            return Ok(llm::build_stream_response(
                StatusCode::OK,
                resp_headers,
                tee_and_store(
                    state.clone(),
                    key,
                    StatusCode::OK.as_u16(),
                    ct,
                    cache_settings.max_stream_bytes,
                    Box::pin(body_stream),
                ),
            ));
        }
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
    let usage = &ir_resp.usage;
    let cost = state
        .model_cost_map
        .get(&deployment.upstream_model)
        .and_then(|info| info.compute_cost(usage));
    tracing::info!(
        model = %deployment.upstream_model,
        input_tokens = usage.input_tokens,
        output_tokens = usage.output_tokens,
        cache_read_tokens = usage.cache_read_input_tokens,
        cache_creation_tokens = usage.cache_creation_input_tokens,
        cost_usd = ?cost,
        "request usage"
    );
    let client_value = in_codec.render_response(&ir_resp, &ctx)?;
    let out_bytes = serde_json::to_vec(&client_value)?;
    if store_key.is_some() || semantic_text.is_some() {
        let ct = content_type_of(&resp_headers);
        store_response(
            state,
            store_key,
            semantic_text.as_deref().map(|t| (scope_str, t)),
            status.as_u16(),
            ct,
            out_bytes.clone(),
        )
        .await;
    }
    Ok(llm::build_bytes_response(status, resp_headers, out_bytes))
}

/// Content-type to record for a cached response, defaulting to JSON.
fn content_type_of(headers: &HeaderMap) -> String {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_owned()
}

/// Reconstruct an HTTP response from a cache hit (never calls the upstream).
/// `tag` is echoed in `x-litellm-cache` (`hit` for exact-match, `semantic`).
fn replay_cached(cached: CachedResponse, tag: &'static str) -> Response {
    let mut headers = HeaderMap::new();
    if let Ok(ct) = HeaderValue::from_str(&cached.content_type) {
        headers.insert(CONTENT_TYPE, ct);
    }
    headers.insert("x-litellm-cache", HeaderValue::from_static(tag));
    let status = StatusCode::from_u16(cached.status).unwrap_or(StatusCode::OK);
    if cached.is_stream {
        let body = cached.body;
        let stream = futures_util::stream::once(async move { Ok::<_, std::io::Error>(Bytes::from(body)) });
        llm::build_stream_response(status, headers, stream)
    } else {
        llm::build_bytes_response(status, headers, cached.body)
    }
}

/// Store a fully-buffered (non-streaming) response into the exact-match cache
/// and/or record it in the semantic cache (`(scope, query_text)`).
async fn store_response(
    state: &Arc<AppState>,
    exact_key: Option<String>,
    semantic: Option<(&str, &str)>,
    status: u16,
    content_type: String,
    body: Vec<u8>,
) {
    let cached = CachedResponse {
        status,
        content_type,
        body,
        is_stream: false,
    };
    if let Some((scope, text)) = semantic {
        state.semantic.record(scope, text, cached.clone()).await;
    }
    if let Some(key) = exact_key {
        state.cache.set(key, cached).await;
    }
}

/// Wrap a byte stream so it forwards each chunk to the client while buffering the
/// full body; on clean completion the buffer is stored (spawned, never blocking
/// the client). A stream that errors mid-flight, or whose body exceeds
/// `max_bytes`, is forwarded but never stored (bounds memory).
fn tee_and_store(
    state: Arc<AppState>,
    key: String,
    status: u16,
    content_type: String,
    max_bytes: u64,
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
    struct St {
        inner: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
        acc: Vec<u8>,
        /// Set when the stream errored or outgrew `max_bytes`; suppresses storing.
        abort_store: bool,
        done: bool,
        max_bytes: usize,
        state: Arc<AppState>,
        key: String,
        status: u16,
        content_type: String,
    }
    let st = St {
        inner,
        acc: Vec::new(),
        abort_store: false,
        done: false,
        max_bytes: max_bytes.try_into().unwrap_or(usize::MAX),
        state,
        key,
        status,
        content_type,
    };
    futures_util::stream::unfold(st, |mut st| async move {
        if st.done {
            return None;
        }
        match st.inner.next().await {
            Some(Ok(chunk)) => {
                // Keep forwarding, but stop buffering once we'd exceed the cap.
                if !st.abort_store {
                    if st.acc.len() + chunk.len() > st.max_bytes {
                        st.abort_store = true;
                        st.acc = Vec::new();
                    } else {
                        st.acc.extend_from_slice(&chunk);
                    }
                }
                Some((Ok(chunk), st))
            }
            Some(Err(e)) => {
                st.abort_store = true;
                Some((Err(e), st))
            }
            None => {
                st.done = true;
                if !st.abort_store && !st.acc.is_empty() {
                    let body = std::mem::take(&mut st.acc);
                    let state = st.state.clone();
                    let key = std::mem::take(&mut st.key);
                    let content_type = std::mem::take(&mut st.content_type);
                    let status = st.status;
                    let bytes = body.len();
                    tokio::spawn(async move {
                        state
                            .cache
                            .set(
                                key,
                                CachedResponse {
                                    status,
                                    content_type,
                                    body,
                                    is_stream: true,
                                },
                            )
                            .await;
                        tracing::trace!(bytes, "stored streaming response in cache");
                    });
                }
                None
            }
        }
    })
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
