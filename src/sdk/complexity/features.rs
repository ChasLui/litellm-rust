use serde_json::Value;

use crate::sdk::codec::WireFormat;

const CHARS_PER_TOKEN: f64 = 4.0;
const REASONING_KEYWORDS: [&str; 10] = [
    "analyze",
    "architecture",
    "debug",
    "derive",
    "explain",
    "optimize",
    "plan",
    "prove",
    "reason",
    "tradeoff",
];

#[derive(Debug, Clone, Default)]
pub struct RequestFeatures {
    pub estimated_tokens: f64,
    pub turns: f64,
    pub has_tools: bool,
    pub has_code: bool,
    pub reasoning_keyword_hits: f64,
    pub max_message_chars: f64,
}

pub fn extract_features(inbound_wire: WireFormat, body: &Value) -> RequestFeatures {
    let mut collector = FeatureCollector::default();
    match inbound_wire {
        WireFormat::AnthropicMessages => collect_anthropic(body, &mut collector),
        WireFormat::OpenAiChat => collect_openai_chat(body, &mut collector),
        WireFormat::OpenAiResponses => collect_openai_responses(body, &mut collector),
        WireFormat::Gemini => collect_gemini(body, &mut collector),
    }
    collector.finish(body)
}

#[derive(Default)]
struct FeatureCollector {
    chars: usize,
    turns: usize,
    has_code: bool,
    reasoning_keyword_hits: usize,
    max_message_chars: usize,
}

impl FeatureCollector {
    fn push_text(&mut self, text: &str) {
        let len = text.chars().count();
        self.chars += len;
        self.max_message_chars = self.max_message_chars.max(len);
        self.has_code |= looks_like_code(text);
        self.reasoning_keyword_hits += reasoning_hits(text);
    }

    fn finish(self, body: &Value) -> RequestFeatures {
        RequestFeatures {
            estimated_tokens: self.chars as f64 / CHARS_PER_TOKEN,
            turns: self.turns as f64,
            has_tools: has_tools(body),
            has_code: self.has_code,
            reasoning_keyword_hits: self.reasoning_keyword_hits as f64,
            max_message_chars: self.max_message_chars as f64,
        }
    }
}

fn collect_anthropic(body: &Value, collector: &mut FeatureCollector) {
    collect_string_field(body, "system", collector);
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        collector.turns += messages.len();
        for message in messages {
            collect_content(message.get("content"), collector);
        }
    }
}

fn collect_openai_chat(body: &Value, collector: &mut FeatureCollector) {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        collector.turns += messages.len();
        for message in messages {
            collect_content(message.get("content"), collector);
        }
    }
}

fn collect_openai_responses(body: &Value, collector: &mut FeatureCollector) {
    match body.get("input") {
        Some(Value::String(text)) => {
            collector.turns += 1;
            collector.push_text(text);
        }
        Some(Value::Array(items)) => {
            collector.turns += items.len();
            for item in items {
                collect_content(item.get("content"), collector);
                collect_string_field(item, "text", collector);
            }
        }
        _ => {}
    }
}

fn collect_gemini(body: &Value, collector: &mut FeatureCollector) {
    if let Some(contents) = body.get("contents").and_then(Value::as_array) {
        collector.turns += contents.len();
        for content in contents {
            if let Some(parts) = content.get("parts").and_then(Value::as_array) {
                for part in parts {
                    collect_string_field(part, "text", collector);
                }
            }
        }
    }
}

fn collect_content(value: Option<&Value>, collector: &mut FeatureCollector) {
    match value {
        Some(Value::String(text)) => collector.push_text(text),
        Some(Value::Array(parts)) => {
            for part in parts {
                collect_string_field(part, "text", collector);
            }
        }
        _ => {}
    }
}

fn collect_string_field(value: &Value, field: &str, collector: &mut FeatureCollector) {
    if let Some(text) = value.get(field).and_then(Value::as_str) {
        collector.push_text(text);
    }
}

fn has_tools(body: &Value) -> bool {
    ["tools", "functions"].iter().any(|key| {
        body.get(key)
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
    }) || body
        .get("toolConfig")
        .or_else(|| body.get("tool_config"))
        .is_some()
}

fn looks_like_code(text: &str) -> bool {
    text.contains("```")
        || text.contains("fn ")
        || text.contains("class ")
        || text.contains("function ")
        || text.contains("=>")
        || text.contains("SELECT ")
}

fn reasoning_hits(text: &str) -> usize {
    let lower = text.to_ascii_lowercase();
    REASONING_KEYWORDS
        .iter()
        .filter(|word| lower.contains(*word))
        .count()
}
