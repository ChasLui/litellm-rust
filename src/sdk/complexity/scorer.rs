use crate::sdk::complexity::features::RequestFeatures;

const TOKEN_WEIGHT: f64 = 0.28;
const TURN_WEIGHT: f64 = 0.14;
const TOOLS_WEIGHT: f64 = 0.18;
const CODE_WEIGHT: f64 = 0.16;
const REASONING_WEIGHT: f64 = 0.14;
const MAX_MESSAGE_WEIGHT: f64 = 0.10;

const TOKEN_SCALE: f64 = 4_000.0;
const TURN_SCALE: f64 = 12.0;
const REASONING_SCALE: f64 = 5.0;
const MAX_MESSAGE_SCALE: f64 = 8_000.0;

pub trait ComplexityScorer: Send + Sync {
    fn score(&self, features: &RequestFeatures) -> f64;
}

#[derive(Debug, Clone, Default)]
pub struct HeuristicScorer;

impl ComplexityScorer for HeuristicScorer {
    fn score(&self, features: &RequestFeatures) -> f64 {
        let score = norm(features.estimated_tokens, TOKEN_SCALE) * TOKEN_WEIGHT
            + norm(features.turns, TURN_SCALE) * TURN_WEIGHT
            + flag(features.has_tools) * TOOLS_WEIGHT
            + flag(features.has_code) * CODE_WEIGHT
            + norm(features.reasoning_keyword_hits, REASONING_SCALE) * REASONING_WEIGHT
            + norm(features.max_message_chars, MAX_MESSAGE_SCALE) * MAX_MESSAGE_WEIGHT;
        score.clamp(0.0, 1.0)
    }
}

fn norm(value: f64, scale: f64) -> f64 {
    (value / scale).clamp(0.0, 1.0)
}

fn flag(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}
