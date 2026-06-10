use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ComplexityRoutingConfig {
    #[serde(default)]
    pub scorer: ComplexityScorerKind,
    pub tiers: Vec<ComplexityTierConfig>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplexityScorerKind {
    #[default]
    Heuristic,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComplexityTierConfig {
    pub max_score: Option<f64>,
    pub model: String,
}
