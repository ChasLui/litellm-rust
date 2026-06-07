//! Canonical intermediate representation (IR). Every wire protocol parses into
//! and renders out of these types, so converting between N protocols needs N
//! codecs instead of N×N point-to-point translators. The shape mirrors
//! Anthropic content blocks because they are the most expressive of the four
//! protocols (text / tool calls / tool results / thinking are all blocks).

use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Image reference. Protocols disagree on inline-base64 vs URL, so we keep both.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    /// Assistant asking to call a tool.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// User-supplied result of a previous tool call.
    ToolResult {
        tool_use_id: String,
        content: Vec<ContentBlock>,
        is_error: bool,
    },
    /// Extended-thinking / reasoning text.
    Thinking {
        text: String,
        signature: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema object for the tool's parameters.
    pub parameters: Value,
    /// For provider built-in / server-side tools (web search, code execution,
    /// …), the verbatim native tool entry. `None` for ordinary function tools.
    /// Built-ins are dropped on cross-protocol render rather than mangled into a
    /// bogus function tool.
    pub builtin: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
}

/// Requested structured-output format.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Any valid JSON object.
    JsonObject,
    /// JSON constrained to a schema.
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

/// Reasoning / extended-thinking request knob. Both forms are carried when known
/// so each codec can pick the closest native shape without double-converting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReasoningConfig {
    pub effort: Option<Effort>,
    pub budget_tokens: Option<u64>,
}

impl ReasoningConfig {
    /// The effort tier, deriving it from a token budget when only that is known.
    pub fn derived_effort(&self) -> Effort {
        self.effort
            .unwrap_or_else(|| Effort::from_budget(self.budget_tokens.unwrap_or(0)))
    }

    /// A token budget, deriving it from the effort tier when only that is known.
    pub fn derived_budget(&self) -> u64 {
        self.budget_tokens
            .or_else(|| self.effort.map(|e| e.to_budget()))
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
}

impl Effort {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "minimal" | "none" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Heuristic token budget for protocols that take a number instead of a tier.
    pub fn to_budget(&self) -> u64 {
        match self {
            Self::Minimal => 1024,
            Self::Low => 4096,
            Self::Medium => 8192,
            Self::High => 16384,
        }
    }

    pub fn from_budget(budget: u64) -> Self {
        match budget {
            0..=1024 => Self::Minimal,
            1025..=4096 => Self::Low,
            4097..=8192 => Self::Medium,
            _ => Self::High,
        }
    }
}

/// Prompt-cache breakpoints for a request, kept out-of-band so the content-block
/// types stay untouched. Anthropic breakpoints always sit at a prefix boundary
/// (end of tools / end of system / the tail block of a message), so marking the
/// carrier rather than an individual block covers real usage; a breakpoint in the
/// middle of a message collapses to that message's last block on render.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheMarkers {
    /// Breakpoint on the last tool definition (caches the whole `tools` prefix).
    pub tools: bool,
    /// Breakpoint on the last system block (caches `tools` + `system`).
    pub system: bool,
    /// Indices into `messages` whose tail block carries a breakpoint.
    pub messages: Vec<usize>,
}

impl CacheMarkers {
    pub fn is_empty(&self) -> bool {
        !self.tools && !self.system && self.messages.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub model: String,
    /// System / developer instructions, usually a single `Text` block.
    pub system: Vec<ContentBlock>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: Option<ToolChoice>,
    /// Prompt-cache breakpoints. Empty unless the client set `cache_control` or
    /// the gateway auto-injected them. Only honoured when rendering to Anthropic.
    pub cache: CacheMarkers,
    /// `Some(false)` forbids parallel tool calls; `None` leaves it unspecified.
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<ResponseFormat>,
    pub reasoning: Option<ReasoningConfig>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Vec<String>,
    pub stream: bool,
    /// Params we do not model explicitly, carried through to the outbound body
    /// when the target protocol is shape-compatible (best effort).
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
    ContentFilter,
    Other(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    /// TOTAL input tokens processed, INCLUDING `cache_read_input_tokens` and
    /// `cache_creation_input_tokens`. OpenAI/Gemini already report inclusive
    /// prompt counts; Anthropic reports only the post-breakpoint remainder, so
    /// its codec adds the cache counts back to keep this field inclusive.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Subset of `input_tokens` written to the prompt cache this turn (Anthropic,
    /// billed ~1.25x). OpenAI/Gemini have no creation concept, so 0 there.
    pub cache_creation_input_tokens: u64,
    /// Subset of `input_tokens` served from the prompt cache (billed ~0.1x).
    pub cache_read_input_tokens: u64,
}

impl Usage {
    /// Input tokens billed at the full rate (total minus the cached/created
    /// portions). Saturates so a malformed upstream count can't underflow.
    pub fn non_cached_input_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_sub(self.cache_read_input_tokens)
            .saturating_sub(self.cache_creation_input_tokens)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<StopReason>,
    pub usage: Usage,
}

/// What kind of content block a stream is opening.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockStart {
    Text,
    Thinking,
    ToolUse { id: String, name: String },
}

/// Normalized streaming events. Mirrors Anthropic's SSE shape (the richest of
/// the four) so any protocol's stream can be reconstructed from this sequence.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    MessageStart {
        id: String,
        model: String,
    },
    ContentBlockStart {
        index: usize,
        block: BlockStart,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    ThinkingDelta {
        index: usize,
        text: String,
    },
    /// Partial JSON for a `ToolUse` block's `input`.
    ToolUseInputDelta {
        index: usize,
        partial_json: String,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        stop_reason: Option<StopReason>,
        usage: Option<Usage>,
    },
    MessageStop,
}

impl StopReason {
    pub fn from_anthropic(s: &str) -> Self {
        match s {
            "end_turn" => Self::EndTurn,
            "max_tokens" => Self::MaxTokens,
            "tool_use" => Self::ToolUse,
            "stop_sequence" => Self::StopSequence,
            "refusal" => Self::ContentFilter,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn to_anthropic(&self) -> String {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::ToolUse => "tool_use",
            Self::StopSequence => "stop_sequence",
            Self::ContentFilter => "refusal",
            Self::Other(s) => s,
        }
        .to_owned()
    }

    /// OpenAI `finish_reason` value.
    pub fn from_openai(s: &str) -> Self {
        match s {
            "stop" => Self::EndTurn,
            "length" => Self::MaxTokens,
            "tool_calls" | "function_call" => Self::ToolUse,
            "content_filter" => Self::ContentFilter,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn to_openai(&self) -> String {
        match self {
            Self::EndTurn => "stop",
            Self::MaxTokens => "length",
            Self::ToolUse => "tool_calls",
            Self::StopSequence => "stop",
            Self::ContentFilter => "content_filter",
            Self::Other(s) => s,
        }
        .to_owned()
    }

    /// Gemini `finishReason` value.
    pub fn from_gemini(s: &str) -> Self {
        match s {
            "STOP" => Self::EndTurn,
            "MAX_TOKENS" => Self::MaxTokens,
            "SAFETY" | "PROHIBITED_CONTENT" => Self::ContentFilter,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn to_gemini(&self) -> String {
        match self {
            Self::EndTurn | Self::ToolUse => "STOP",
            Self::MaxTokens => "MAX_TOKENS",
            Self::StopSequence => "STOP",
            Self::ContentFilter => "SAFETY",
            Self::Other(_) => "STOP",
        }
        .to_owned()
    }
}

impl Role {
    pub fn as_anthropic(&self) -> &'static str {
        match self {
            // Anthropic has no top-level system role inside `messages`; callers
            // hoist system blocks out. User/Tool both map to "user" turns.
            Self::System | Self::User | Self::Tool => "user",
            Self::Assistant => "assistant",
        }
    }
}
