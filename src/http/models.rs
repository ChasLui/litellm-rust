use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::{header, HeaderMap},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    errors::GatewayError,
    proxy::{auth::master_key::require_any_gateway_key, config::GatewayConfig, state::AppState},
};

pub async fn models(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ModelListParams>,
    headers: HeaderMap,
) -> Result<Json<Value>, GatewayError> {
    require_any_gateway_key(&headers, &state)?;

    Ok(Json(model_list_for_request(
        &state.config,
        &headers,
        params,
    )))
}

pub async fn anthropic_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, GatewayError> {
    require_any_gateway_key(&headers, &state)?;

    Ok(Json(anthropic_model_list(&state.config)))
}

pub async fn gemini_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, GatewayError> {
    require_any_gateway_key(&headers, &state)?;

    Ok(Json(gemini_model_list(&state.config)))
}

fn openai_models(config: &GatewayConfig) -> Value {
    let data: Vec<Value> = configured_models(config)
        .into_iter()
        .map(|model| {
            json!({
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": model.provider_id,
            })
        })
        .collect();

    json!({ "object": "list", "data": data })
}

fn model_list_for_request(
    config: &GatewayConfig,
    headers: &HeaderMap,
    params: ModelListParams,
) -> Value {
    match requested_model_list_format(headers, &params) {
        ModelListFormat::Anthropic => anthropic_model_list(config),
        ModelListFormat::Gemini => gemini_model_list(config),
        ModelListFormat::OpenAi => openai_models(config),
    }
}

fn requested_model_list_format(headers: &HeaderMap, params: &ModelListParams) -> ModelListFormat {
    if let Some(format) = params.format.as_deref().and_then(parse_model_list_format) {
        return format;
    }

    if headers.contains_key("anthropic-version") || headers.contains_key("anthropic-beta") {
        return ModelListFormat::Anthropic;
    }
    if headers.contains_key("x-goog-api-key") || headers.contains_key("x-goog-api-client") {
        return ModelListFormat::Gemini;
    }
    if headers.contains_key("openai-organization")
        || headers.contains_key("openai-project")
        || headers.contains_key("openai-beta")
    {
        return ModelListFormat::OpenAi;
    }

    let user_agent = headers
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ["anthropic", "claude-code"]
        .iter()
        .any(|needle| user_agent.contains(needle))
    {
        return ModelListFormat::Anthropic;
    }
    if [
        "gemini",
        "google-genai",
        "google-generative-ai",
        "google-generativeai",
        "generativelanguage",
    ]
    .iter()
    .any(|needle| user_agent.contains(needle))
    {
        return ModelListFormat::Gemini;
    }

    ModelListFormat::OpenAi
}

fn parse_model_list_format(format: &str) -> Option<ModelListFormat> {
    match format.trim().to_ascii_lowercase().as_str() {
        "anthropic" | "messages" | "anthropic_messages" => Some(ModelListFormat::Anthropic),
        "gemini" | "google" => Some(ModelListFormat::Gemini),
        "openai" | "chat" | "responses" | "openai_chat" | "openai_responses" => {
            Some(ModelListFormat::OpenAi)
        }
        _ => None,
    }
}

fn anthropic_model_list(config: &GatewayConfig) -> Value {
    let data: Vec<Value> = configured_models(config)
        .into_iter()
        .map(|model| {
            json!({
                "type": "model",
                "id": model.id,
                "display_name": model.display_name,
                "created_at": "1970-01-01T00:00:00Z",
            })
        })
        .collect();
    let first_id = data.first().and_then(|model| model.get("id")).cloned();
    let last_id = data.last().and_then(|model| model.get("id")).cloned();

    json!({
        "data": data,
        "has_more": false,
        "first_id": first_id,
        "last_id": last_id,
    })
}

fn gemini_model_list(config: &GatewayConfig) -> Value {
    let models: Vec<Value> = configured_models(config)
        .into_iter()
        .map(|model| {
            json!({
                "name": format!("models/{}", model.id),
                "baseModelId": model.upstream_model,
                "version": "001",
                "displayName": model.display_name,
                "description": format!("Configured gateway model {}", model.id),
                "inputTokenLimit": 0,
                "outputTokenLimit": 0,
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            })
        })
        .collect();

    json!({ "models": models })
}

fn configured_models(config: &GatewayConfig) -> Vec<ModelCatalogItem<'_>> {
    config
        .model_list
        .iter()
        .filter(|entry| entry.litellm_params.complexity_routing.is_none())
        .map(|entry| {
            let (provider_id, upstream_model) = entry
                .litellm_params
                .model
                .split_once('/')
                .unwrap_or(("litellm", entry.litellm_params.model.as_str()));
            ModelCatalogItem {
                id: entry.model_name.as_str(),
                provider_id,
                upstream_model,
                display_name: display_name(entry.model_name.as_str()),
            }
        })
        .collect()
}

fn display_name(id: &str) -> String {
    id.trim_end_matches("/*").replace(['_', '-', '/'], " ")
}

struct ModelCatalogItem<'a> {
    id: &'a str,
    provider_id: &'a str,
    upstream_model: &'a str,
    display_name: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelListParams {
    #[serde(default)]
    format: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelListFormat {
    OpenAi,
    Anthropic,
    Gemini,
}
