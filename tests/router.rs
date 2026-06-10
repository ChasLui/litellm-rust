use std::collections::HashMap;

use litellm_rust::{
    proxy::config::{GatewayConfig, LiteLlmParams, ModelEntry},
    sdk::{
        providers::{self, transform::ProviderRegistry},
        router::Router,
    },
};

fn providers() -> ProviderRegistry {
    let mut providers = ProviderRegistry::new();
    providers::register_all(&mut providers);
    providers
}

fn config(entries: Vec<ModelEntry>) -> GatewayConfig {
    GatewayConfig {
        model_list: entries,
        mcp_servers: HashMap::new(),
        general_settings: Default::default(),
        agents: Vec::new(),
    }
}

fn model_entry(model_name: &str, model: &str) -> ModelEntry {
    ModelEntry {
        model_name: model_name.to_owned(),
        litellm_params: LiteLlmParams {
            model: model.to_owned(),
            api_key: Some("sk".to_owned()),
            api_base: None,
            wire_api: None,
            complexity_routing: None,
            extra: Default::default(),
        },
    }
}

#[test]
fn resolves_model_to_upstream() {
    let config = config(vec![model_entry("claude", "anthropic/claude-sonnet-4-5")]);
    let router = Router::from_config(&config, &providers()).unwrap();
    let route = router.resolve("claude").unwrap();
    assert_eq!(route.deployment.upstream_model, "claude-sonnet-4-5");
    assert_eq!(route.deployment.provider_id, "anthropic");
}

#[test]
fn resolves_wildcard_model_to_anthropic_passthrough() {
    let config = config(vec![model_entry("anthropic/*", "anthropic/*")]);
    let router = Router::from_config(&config, &providers()).unwrap();
    let route = router.resolve("anthropic/claude-opus-4-8").unwrap();
    assert_eq!(route.deployment.provider_id, "anthropic");
    assert_eq!(route.deployment.upstream_model, "claude-opus-4-8");
}

#[test]
fn strips_provider_prefix_from_wildcard_model() {
    let config = config(vec![model_entry("anthropic/*", "anthropic/*")]);
    let router = Router::from_config(&config, &providers()).unwrap();
    let route = router.resolve("anthropic/claude-opus-4-8").unwrap();
    assert_eq!(route.deployment.upstream_model, "claude-opus-4-8");
}

#[test]
fn exact_route_takes_precedence_over_wildcard() {
    let config = config(vec![
        model_entry("claude", "anthropic/claude-sonnet-4-5"),
        model_entry("anthropic/*", "anthropic/*"),
    ]);
    let router = Router::from_config(&config, &providers()).unwrap();
    let route = router.resolve("claude").unwrap();
    assert_eq!(route.deployment.upstream_model, "claude-sonnet-4-5");
}
