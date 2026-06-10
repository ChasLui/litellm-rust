use std::{collections::HashMap, io::Write};

use litellm_rust::{
    proxy::{
        config::{load_config, GatewayConfig},
        state::AppState,
    },
    sdk::{
        codec::WireFormat,
        complexity::{
            features::extract_features,
            scorer::{ComplexityScorer, HeuristicScorer},
            ComplexityRouter, ComplexityTier, RoutingPlan,
        },
        providers::{self, transform::ProviderRegistry},
        router::Router as ModelRouter,
    },
};
use serde_json::json;
use tempfile::NamedTempFile;

fn config_with_alias(tiers: &str) -> GatewayConfig {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        r#"
model_list:
  - model_name: auto
    litellm_params:
      complexity_routing:
        scorer: heuristic
        tiers:
{tiers}
  - model_name: claude-haiku
    litellm_params:
      model: anthropic/claude-haiku-4-5
      api_key: sk-test
  - model_name: claude-opus
    litellm_params:
      model: anthropic/claude-opus-4-5
      api_key: sk-test
"#
    )
    .unwrap();
    load_config(file.path()).unwrap()
}

fn boot(config: GatewayConfig) -> Result<(), String> {
    let mut providers = ProviderRegistry::new();
    providers::register_all(&mut providers);
    let router =
        ModelRouter::from_config(&config, &providers).map_err(|error| error.to_string())?;
    let http = AppState::build_http_client().map_err(|error| error.to_string())?;
    AppState::new(config, router, http, HashMap::new(), None)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[test]
fn loads_complexity_routing_config() {
    let config = config_with_alias(
        r#"
          - max_score: 0.3
            model: claude-haiku
          - model: claude-opus"#,
    );
    let routing = config.model_list[0]
        .litellm_params
        .complexity_routing
        .as_ref()
        .unwrap();
    assert_eq!(routing.tiers.len(), 2);
    assert_eq!(routing.tiers[0].max_score, Some(0.3));
    assert_eq!(routing.tiers[1].model, "claude-opus");
    assert!(boot(config).is_ok());
}

#[test]
fn rejects_unknown_tier_model() {
    let config = config_with_alias(
        r#"
          - max_score: 0.3
            model: missing
          - model: claude-opus"#,
    );
    let error = boot(config).unwrap_err();
    assert!(error.contains("unknown model missing"));
}

#[test]
fn rejects_missing_catch_all() {
    let config = config_with_alias(
        r#"
          - max_score: 0.3
            model: claude-haiku
          - max_score: 0.7
            model: claude-opus"#,
    );
    let error = boot(config).unwrap_err();
    assert!(error.contains("catch-all tier"));
}

#[test]
fn rejects_non_monotonic_max_score() {
    let config = config_with_alias(
        r#"
          - max_score: 0.7
            model: claude-haiku
          - max_score: 0.3
            model: claude-opus
          - model: claude-opus"#,
    );
    let error = boot(config).unwrap_err();
    assert!(error.contains("max_score values must increase"));
}

#[test]
fn heuristic_scores_openai_chat_features() {
    let body = json!({
        "model": "auto",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "please analyze this architecture\n```rust\nfn main() {}\n```"}
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    });
    let features = extract_features(WireFormat::OpenAiChat, &body);
    let score = HeuristicScorer.score(&features);
    assert!(features.has_code);
    assert!(features.has_tools);
    assert!(score > 0.4);
}

#[test]
fn routes_simple_anthropic_to_low_tier() {
    let router = test_router();
    let selected = router.select(
        WireFormat::AnthropicMessages,
        "auto".to_owned(),
        &json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    );
    assert_eq!(selected, "claude-haiku");
}

#[test]
fn routes_complex_openai_chat_to_high_tier() {
    let router = test_router();
    let selected = router.select(
        WireFormat::OpenAiChat,
        "auto".to_owned(),
        &json!({
            "model": "auto",
            "messages": [
                {"role": "user", "content": "analyze plan debug optimize architecture"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "```python\nclass Example:\n    def run(self):\n        return 'x'\n```"}
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    assert_eq!(selected, "claude-opus");
}

fn test_router() -> ComplexityRouter {
    let mut routes = HashMap::new();
    routes.insert(
        "auto".to_owned(),
        RoutingPlan::heuristic(vec![
            ComplexityTier {
                max_score: Some(0.3),
                model: "claude-haiku".to_owned(),
            },
            ComplexityTier {
                max_score: None,
                model: "claude-opus".to_owned(),
            },
        ]),
    );
    ComplexityRouter::new(routes)
}
