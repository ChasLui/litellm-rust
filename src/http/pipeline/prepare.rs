use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::Value;

use crate::{
    errors::GatewayError,
    http::{credential_overrides, pipeline::dispatch},
    proxy::state::AppState,
    sdk::{
        codec::{codec_for, RequestCtx, WireFormat},
        router::Route,
    },
};

pub struct PreparedUpstreamRequest {
    pub url: String,
    pub stream: bool,
    pub inbound_wire: WireFormat,
    pub outbound_wire: WireFormat,
    pub inbound_model: String,
    pub body: Value,
    pub headers: HeaderMap,
}

pub async fn prepare_upstream(
    state: &Arc<AppState>,
    inbound_wire: WireFormat,
    model: String,
    stream: bool,
    body: Value,
    inbound_headers: &HeaderMap,
) -> Result<PreparedUpstreamRequest, GatewayError> {
    let route =
        credential_overrides::apply(state, state.router.resolve_wire(inbound_wire, &model)?)
            .await?;
    prepare_upstream_with_route(
        state,
        route,
        inbound_wire,
        model,
        stream,
        body,
        inbound_headers,
    )
    .await
}

async fn prepare_upstream_with_route(
    state: &Arc<AppState>,
    route: Route,
    inbound_wire: WireFormat,
    model: String,
    stream: bool,
    mut body: Value,
    inbound_headers: &HeaderMap,
) -> Result<PreparedUpstreamRequest, GatewayError> {
    let deployment = &route.deployment;
    let outbound_wire = deployment.wire;
    let url = deployment.upstream_url(stream);
    let out_codec = codec_for(outbound_wire);
    let headers = out_codec.outbound_headers(deployment, inbound_headers)?;

    if inbound_wire == outbound_wire {
        dispatch::rewrite_for_fast_path(&mut body, deployment);
    } else {
        let in_codec = codec_for(inbound_wire);
        let mut ir_req = in_codec.parse_request(body)?;
        ir_req.model = deployment.upstream_model.clone();
        ir_req.stream = stream;
        dispatch::inject_request_breakpoints(state, outbound_wire, &mut ir_req);
        body = out_codec.render_request(&ir_req)?;
    }

    Ok(PreparedUpstreamRequest {
        url,
        stream,
        inbound_wire,
        outbound_wire,
        inbound_model: model,
        body,
        headers,
    })
}

pub fn websocket_response_context(prepared: &PreparedUpstreamRequest) -> RequestCtx {
    RequestCtx {
        model: prepared.inbound_model.clone(),
        stream: true,
    }
}
