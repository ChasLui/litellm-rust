pub mod features;
pub mod scorer;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde_json::Value;

use crate::{
    errors::GatewayError,
    proxy::config::{ComplexityScorerKind, ComplexityTierConfig, GatewayConfig},
    sdk::{
        codec::WireFormat,
        complexity::{
            features::extract_features,
            scorer::{ComplexityScorer, HeuristicScorer},
        },
    },
};

#[derive(Clone)]
pub struct ComplexityRouter {
    routes: HashMap<String, RoutingPlan>,
}

impl ComplexityRouter {
    pub fn new(routes: HashMap<String, RoutingPlan>) -> Self {
        Self { routes }
    }

    pub fn empty() -> Self {
        Self::new(HashMap::new())
    }

    pub fn from_config(config: &GatewayConfig) -> Result<Self, GatewayError> {
        let concrete = concrete_models(config);
        let mut routes = HashMap::new();
        for entry in &config.model_list {
            let Some(routing) = &entry.litellm_params.complexity_routing else {
                continue;
            };
            let tiers = validate_tiers(&entry.model_name, &routing.tiers, &concrete)?;
            let plan = match routing.scorer {
                ComplexityScorerKind::Heuristic => RoutingPlan::heuristic(tiers),
            };
            routes.insert(entry.model_name.clone(), plan);
        }
        Ok(Self::new(routes))
    }

    pub fn select(&self, inbound_wire: WireFormat, model: String, body: &Value) -> String {
        let Some(plan) = self.routes.get(&model) else {
            return model;
        };
        let features = extract_features(inbound_wire, body);
        let score = plan.scorer.score(&features);
        let selected = plan.select(score);
        tracing::info!(
            alias = model,
            score,
            selected_model = selected,
            "complexity routing selected tier"
        );
        selected.to_owned()
    }
}

impl std::fmt::Debug for ComplexityRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplexityRouter")
            .field("aliases", &self.routes.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Clone)]
pub struct RoutingPlan {
    tiers: Vec<ComplexityTier>,
    scorer: Arc<dyn ComplexityScorer>,
}

impl RoutingPlan {
    pub fn heuristic(tiers: Vec<ComplexityTier>) -> Self {
        Self {
            tiers,
            scorer: Arc::new(HeuristicScorer),
        }
    }

    fn select(&self, score: f64) -> &str {
        self.tiers
            .iter()
            .find(|tier| tier.max_score.is_none_or(|max| score <= max))
            .map(|tier| tier.model.as_str())
            .unwrap_or_else(|| {
                self.tiers
                    .last()
                    .map(|tier| tier.model.as_str())
                    .unwrap_or("")
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityTier {
    pub max_score: Option<f64>,
    pub model: String,
}

fn concrete_models(config: &GatewayConfig) -> HashSet<&str> {
    config
        .model_list
        .iter()
        .filter(|entry| entry.litellm_params.complexity_routing.is_none())
        .map(|entry| entry.model_name.as_str())
        .collect()
}

fn validate_tiers(
    alias: &str,
    tiers: &[ComplexityTierConfig],
    concrete: &HashSet<&str>,
) -> Result<Vec<ComplexityTier>, GatewayError> {
    let mut last = None;
    let mut has_catch_all = false;
    let mut validated = Vec::with_capacity(tiers.len());
    for (index, tier) in tiers.iter().enumerate() {
        validate_tier(
            alias,
            tier,
            index == tiers.len().saturating_sub(1),
            concrete,
            &mut last,
            &mut has_catch_all,
        )?;
        validated.push(ComplexityTier {
            max_score: tier.max_score,
            model: tier.model.clone(),
        });
    }
    if !has_catch_all {
        return Err(GatewayError::InvalidConfig(format!(
            "{alias} complexity_routing requires a catch-all tier"
        )));
    }
    Ok(validated)
}

fn validate_tier(
    alias: &str,
    tier: &ComplexityTierConfig,
    is_last: bool,
    concrete: &HashSet<&str>,
    last: &mut Option<f64>,
    has_catch_all: &mut bool,
) -> Result<(), GatewayError> {
    if !concrete.contains(tier.model.as_str()) {
        return Err(GatewayError::InvalidConfig(format!(
            "{alias} complexity_routing tier references unknown model {}",
            tier.model
        )));
    }
    match tier.max_score {
        Some(score) if !(0.0..=1.0).contains(&score) => Err(GatewayError::InvalidConfig(format!(
            "{alias} complexity_routing max_score must be between 0 and 1"
        ))),
        Some(score) if last.is_some_and(|prev| score <= prev) => Err(GatewayError::InvalidConfig(
            format!("{alias} complexity_routing max_score values must increase"),
        )),
        Some(score) => {
            *last = Some(score);
            Ok(())
        }
        None if is_last => {
            *has_catch_all = true;
            Ok(())
        }
        None => Err(GatewayError::InvalidConfig(format!(
            "{alias} complexity_routing catch-all tier must be last"
        ))),
    }
}
