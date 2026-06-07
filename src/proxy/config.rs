use std::{collections::HashMap, fs, path::Path};

use percent_encoding::{percent_encode, AsciiSet, CONTROLS};
use serde::Deserialize;

use crate::{
    agents::config::{validate_agents, AgentDefinition, E2bSandboxParams},
    errors::GatewayError,
    proxy::mcp_config::{is_mcp_sequence_error, validate_mcp_servers},
};

pub use crate::proxy::mcp_config::{McpAuthType, McpServerEntry, McpTransport};

#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    #[serde(default)]
    pub model_list: Vec<ModelEntry>,

    #[serde(default)]
    pub mcp_servers: HashMap<String, McpServerEntry>,

    #[serde(default)]
    pub general_settings: GeneralSettings,

    #[serde(default)]
    pub agents: Vec<AgentDefinition>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GeneralSettings {
    pub master_key: Option<String>,
    pub database_url: Option<String>,
    pub sandbox_choice: Option<String>,
    #[serde(default)]
    pub e2b_sandbox_params: E2bSandboxParams,
    #[serde(default)]
    pub prompt_caching: PromptCachingSettings,
    #[serde(default)]
    pub cache: CacheSettings,
}

/// Exact-match response cache. Disabled by default; when on, an identical request
/// returns the stored response without calling the upstream (0 tokens).
#[derive(Debug, Clone, Deserialize)]
pub struct CacheSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub backend: CacheBackendKind,
    /// Required when `backend = redis`; env-expandable.
    pub redis_url: Option<String>,
    /// File path for the `redb` backend; env-expandable. Defaults to
    /// `litellm-cache.redb` when `backend = redb` and this is unset.
    #[serde(default)]
    pub redb_path: Option<String>,
    #[serde(default = "default_cache_ttl")]
    pub ttl_secs: u64,
    /// Entry-count cap (memory: immediate; redb: soft, reconciled by a periodic
    /// sweep; Redis: ignored, TTL only).
    #[serde(default = "default_cache_max_entries")]
    pub max_entries: u64,
    /// Cache requests with `temperature > 0` (non-deterministic) too. Off by
    /// default so only deterministic requests are cached.
    #[serde(default)]
    pub cache_non_deterministic: bool,
    /// Buffer and replay streaming (SSE) responses. On by default.
    #[serde(default = "default_true")]
    pub cache_streaming: bool,
    /// Max bytes buffered for a single streaming response; if a stream exceeds
    /// this it is forwarded to the client but not cached (bounds memory).
    #[serde(default = "default_max_stream_bytes")]
    pub max_stream_bytes: u64,
    /// Include a hash of the caller's API key in the cache key so tenants never
    /// see each other's cached responses. On by default.
    #[serde(default = "default_true")]
    pub scope_by_api_key: bool,
    #[serde(default)]
    pub semantic: SemanticCacheSettings,
}

/// Embedding-based semantic cache (feature `semantic-cache`). ⚠️ Off by default
/// and not recommended for coding-agent workloads (low hit rate, wrong-answer
/// risk). Restricted to deterministic, tool-free, non-streaming requests.
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticCacheSettings {
    #[serde(default)]
    pub enabled: bool,
    /// OpenAI-compatible embeddings endpoint base (env-expandable).
    pub embedding_api_base: Option<String>,
    /// API key for the embeddings endpoint (env-expandable).
    pub embedding_api_key: Option<String>,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    /// Cosine similarity above which a cached response is reused (0..1).
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    /// Max in-process embedding entries (LRU).
    #[serde(default = "default_semantic_max_entries")]
    pub max_entries: u64,
    /// Skip prompts longer than this (bounds embedding cost).
    #[serde(default = "default_semantic_max_chars")]
    pub max_chars: u64,
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_owned()
}
fn default_similarity_threshold() -> f32 {
    0.92
}
fn default_semantic_max_entries() -> u64 {
    1000
}
fn default_semantic_max_chars() -> u64 {
    8000
}

impl Default for SemanticCacheSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            embedding_api_base: None,
            embedding_api_key: None,
            embedding_model: default_embedding_model(),
            similarity_threshold: default_similarity_threshold(),
            max_entries: default_semantic_max_entries(),
            max_chars: default_semantic_max_chars(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheBackendKind {
    #[default]
    Memory,
    Redb,
    Redis,
}

fn default_cache_ttl() -> u64 {
    300
}
fn default_cache_max_entries() -> u64 {
    10_000
}
fn default_max_stream_bytes() -> u64 {
    8 * 1024 * 1024
}
fn default_true() -> bool {
    true
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: CacheBackendKind::Memory,
            redis_url: None,
            redb_path: None,
            ttl_secs: default_cache_ttl(),
            max_entries: default_cache_max_entries(),
            cache_non_deterministic: false,
            cache_streaming: true,
            max_stream_bytes: default_max_stream_bytes(),
            scope_by_api_key: true,
            semantic: SemanticCacheSettings::default(),
        }
    }
}

/// Subset of an upstream litellm `litellm_settings` block we know how to honour.
/// Only the response-cache stanza is translated; every other key is ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct LitellmSettingsCompat {
    /// Upstream master switch (`litellm_settings.cache: true`).
    #[serde(default)]
    pub cache: Option<bool>,
    /// Upstream `litellm_settings.cache_params`.
    #[serde(default)]
    pub cache_params: Option<LitellmCacheParams>,
}

/// The upstream `cache_params` keys litellm-rust can map onto its native cache.
/// Only `type`/`disk_cache_dir` are typed; `ttl`, `host`/`port`/`password`/
/// `username`/`ssl` and any unsupported keys stay in `extra` and are read
/// leniently (string or number) so an odd-but-valid value never fails the load.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LitellmCacheParams {
    #[serde(rename = "type")]
    pub cache_type: Option<String>,
    pub disk_cache_dir: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

/// Gateway-side prompt-cache (Anthropic breakpoint) policy. Disabled by default
/// so the request path is unchanged unless an operator opts in.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptCachingSettings {
    /// Master switch for any gateway-side prompt-cache handling.
    #[serde(default)]
    pub enabled: bool,
    /// Auto-inject breakpoints for clients that didn't set `cache_control` when
    /// the request is routed to an Anthropic upstream. Off by default because it
    /// assumes a stable system/tools prefix (true for agent loops); on a volatile
    /// prefix it can cost more than it saves.
    #[serde(default)]
    pub auto_inject: bool,
    /// Max breakpoints to inject (clamped to Anthropic's hard cap of 4).
    #[serde(default = "default_max_breakpoints")]
    pub max_breakpoints: u8,
    /// Minimum estimated tokens a cached prefix must reach to be worth a
    /// breakpoint (Anthropic ignores prefixes below ~1024 tokens).
    #[serde(default = "default_min_tokens")]
    pub min_tokens: u64,
    /// Chars-per-token divisor for the size estimate (no tokenizer is run).
    #[serde(default = "default_chars_per_token")]
    pub chars_per_token: u64,
}

fn default_max_breakpoints() -> u8 {
    4
}
fn default_min_tokens() -> u64 {
    1024
}
fn default_chars_per_token() -> u64 {
    4
}

impl Default for PromptCachingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_inject: false,
            max_breakpoints: default_max_breakpoints(),
            min_tokens: default_min_tokens(),
            chars_per_token: default_chars_per_token(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub model_name: String,
    pub litellm_params: LiteLlmParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LiteLlmParams {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    /// Override the provider's default wire format: `chat` | `responses` |
    /// `gemini` | `anthropic`. When absent, the provider id's default is used.
    #[serde(default)]
    pub wire_api: Option<String>,

    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

pub fn load_config(path: &Path) -> Result<GatewayConfig, GatewayError> {
    let raw = fs::read_to_string(path)?;
    let mut config: GatewayConfig = serde_yaml::from_str(&raw).map_err(|error| {
        // `mcp_servers` changed from a list to a dict keyed by server name.
        // serde reports this as an "invalid type: sequence" error; translate it
        // into actionable guidance for anyone upgrading an old config.
        if is_mcp_sequence_error(&raw, &error) {
            GatewayError::InvalidConfig(
                "mcp_servers is now a dict keyed by server name (was a list). \
                 See docs/mcp.md for the new format."
                    .to_owned(),
            )
        } else {
            GatewayError::from(error)
        }
    })?;
    expand_env(&mut config)?;
    if let Some(ls) = litellm_settings_from_raw(&raw)? {
        apply_cache_compat(&mut config.general_settings.cache, &ls)?;
    }
    validate(&config)?;
    Ok(config)
}

pub fn expand_env_value(value: &str) -> Result<String, GatewayError> {
    let Some(name) = value.strip_prefix("os.environ/") else {
        return Ok(value.to_owned());
    };

    std::env::var(name).map_err(|_| {
        GatewayError::InvalidConfig(format!("environment variable {name} is required"))
    })
}

fn expand_env(config: &mut GatewayConfig) -> Result<(), GatewayError> {
    if let Some(master_key) = config.general_settings.master_key.as_deref() {
        config.general_settings.master_key = Some(expand_env_value(master_key)?);
    }
    if let Some(database_url) = config.general_settings.database_url.as_deref() {
        config.general_settings.database_url = Some(expand_env_value(database_url)?);
    }
    if let Some(redis_url) = config.general_settings.cache.redis_url.as_deref() {
        config.general_settings.cache.redis_url = Some(expand_env_value(redis_url)?);
    }
    if let Some(redb_path) = config.general_settings.cache.redb_path.as_deref() {
        config.general_settings.cache.redb_path = Some(expand_env_value(redb_path)?);
    }
    {
        let semantic = &mut config.general_settings.cache.semantic;
        if let Some(base) = semantic.embedding_api_base.as_deref() {
            semantic.embedding_api_base = Some(expand_env_value(base)?);
        }
        if let Some(key) = semantic.embedding_api_key.as_deref() {
            semantic.embedding_api_key = Some(expand_env_value(key)?);
        }
    }

    for entry in &mut config.model_list {
        if let Some(api_key) = entry.litellm_params.api_key.as_deref() {
            entry.litellm_params.api_key = Some(expand_env_value(api_key)?);
        }
        if let Some(api_base) = entry.litellm_params.api_base.as_deref() {
            entry.litellm_params.api_base = Some(expand_env_value(api_base)?);
        }
    }

    for server in config.mcp_servers.values_mut() {
        server.url = expand_env_value(&server.url)?;
        if let Some(auth_value) = server.auth_value.as_deref() {
            server.auth_value = Some(expand_env_value(auth_value)?);
        }
        for value in server.static_headers.values_mut() {
            *value = expand_env_value(value)?;
        }
    }

    if let Some(api_key) = config
        .general_settings
        .e2b_sandbox_params
        .e2b_api_key
        .as_deref()
    {
        config.general_settings.e2b_sandbox_params.e2b_api_key = Some(expand_env_value(api_key)?);
    }
    for value in config.general_settings.e2b_sandbox_params.envs.values_mut() {
        *value = expand_env_value(value)?;
    }

    Ok(())
}

/// Pull an upstream `litellm_settings` block straight out of the raw YAML. It is
/// not a field on [`GatewayConfig`] (litellm-rust configures everything under
/// `general_settings`); this read-only shim only exists to translate the cache
/// stanza, so we parse it on the side rather than widen the typed config.
fn litellm_settings_from_raw(raw: &str) -> Result<Option<LitellmSettingsCompat>, GatewayError> {
    let doc: serde_yaml::Value = serde_yaml::from_str(raw)?;
    let Some(node) = doc.get("litellm_settings") else {
        return Ok(None);
    };
    // Best-effort: a malformed litellm_settings block must not abort startup for an
    // otherwise-valid config (litellm-rust configures caching under general_settings).
    match serde_yaml::from_value::<LitellmSettingsCompat>(node.clone()) {
        Ok(settings) => Ok(Some(settings)),
        Err(e) => {
            tracing::warn!(
                "litellm_settings could not be read ({e}); ignoring it — configure \
                 response caching under general_settings.cache"
            );
            Ok(None)
        }
    }
}

/// Translate an upstream litellm `litellm_settings.cache` block into the native
/// `general_settings.cache` so an existing litellm config.yaml keeps caching after
/// a drop-in migration. Native `general_settings.cache` wins when caching is
/// already enabled there; otherwise an upstream `cache: true` block is honoured.
/// Runs after `expand_env`, so it expands `os.environ/…` in the values it reads.
/// Unsupported `type`s / keys are warned about, never silently dropped.
fn apply_cache_compat(
    cache: &mut CacheSettings,
    ls: &LitellmSettingsCompat,
) -> Result<(), GatewayError> {
    // The native config takes precedence once the operator explicitly enabled it.
    if cache.enabled {
        return Ok(());
    }
    if ls.cache != Some(true) {
        return Ok(());
    }
    let params = ls.cache_params.clone().unwrap_or_default();

    // Upstream defaults an absent `type` to "redis" (matching the upstream proxy).
    let backend = match params.cache_type.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("local") => CacheBackendKind::Memory,
        Some("disk") => CacheBackendKind::Redb,
        Some("redis") | None => CacheBackendKind::Redis,
        Some(other) => {
            tracing::warn!(
                "litellm_settings.cache_params.type = {other} is not supported by litellm-rust \
                 (supported: local→memory, disk→redb, redis); response caching stays off — \
                 configure general_settings.cache instead"
            );
            return Ok(());
        }
    };
    warn_unsupported_cache_params(&params);

    let synthesized_redis_url = if backend == CacheBackendKind::Redis {
        synth_redis_url(&params)?
    } else {
        None
    };
    cache.enabled = true;
    cache.backend = backend;
    // `ttl` is read leniently (string or number, env-expandable) and applied only
    // at whole-second granularity; a sub-second value would truncate to 0 (a
    // useless permanent miss), so it falls back to the default instead.
    let ttl = params
        .extra
        .get("ttl")
        .and_then(yaml_scalar)
        .map(|s| expand_env_value(&s))
        .transpose()?
        .and_then(|s| s.parse::<f64>().ok());
    if let Some(ttl) = ttl {
        if ttl >= 1.0 {
            cache.ttl_secs = ttl as u64;
        }
    }
    match backend {
        CacheBackendKind::Redis if cache.redis_url.is_none() => cache.redis_url = synthesized_redis_url,
        CacheBackendKind::Redb if cache.redb_path.is_none() => {
            if let Some(dir) = params.disk_cache_dir.as_deref() {
                // Upstream `disk_cache_dir` names a directory; redb is a single
                // file, so place the db inside it rather than colliding with it.
                let path = Path::new(&expand_env_value(dir)?).join("litellm-cache.redb");
                cache.redb_path = Some(path.to_string_lossy().into_owned());
            }
        }
        _ => {}
    }
    Ok(())
}

/// Coerce a YAML scalar (string/number/bool) to a string; `None` for collections.
fn yaml_scalar(value: &serde_yaml::Value) -> Option<String> {
    match value {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// RFC 3986 userinfo percent-encode set (matches the `url` crate), so a Redis
/// `username`/`password` with reserved chars (`/`, `:`, `@`, …) yields a URL the
/// redis client can parse instead of silently failing to connect.
const USERINFO: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'=')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'|')
    .add(b'%');

/// Build `redis(s)://[user]:[password]@host:port` from upstream
/// `host`/`port`/`username`/`password`/`ssl`. With no host, fall back to a
/// `REDIS_URL` env var (the common Docker/k8s pattern upstream also supports);
/// other `REDIS_*` vars are out of scope. `None` only when nothing is configured.
fn synth_redis_url(params: &LitellmCacheParams) -> Result<Option<String>, GatewayError> {
    let Some(host) = params.extra.get("host").and_then(yaml_scalar) else {
        return Ok(std::env::var("REDIS_URL").ok());
    };
    let host = expand_env_value(&host)?;
    let port = match params.extra.get("port").and_then(yaml_scalar) {
        Some(p) => expand_env_value(&p)?,
        None => "6379".to_owned(),
    };
    let scheme = if params.extra.get("ssl").is_some_and(yaml_truthy) {
        "rediss"
    } else {
        "redis"
    };
    let userinfo = redis_userinfo(params)?;
    Ok(Some(format!("{scheme}://{userinfo}{host}:{port}")))
}

/// `user:password@` with each segment percent-encoded, or empty when neither set.
fn redis_userinfo(params: &LitellmCacheParams) -> Result<String, GatewayError> {
    let user = match params.extra.get("username").and_then(yaml_scalar) {
        Some(u) => expand_env_value(&u)?,
        None => String::new(),
    };
    let pass = match params.extra.get("password").and_then(yaml_scalar) {
        Some(p) => expand_env_value(&p)?,
        None => String::new(),
    };
    if user.is_empty() && pass.is_empty() {
        return Ok(String::new());
    }
    let enc = |s: &str| percent_encode(s.as_bytes(), USERINFO).to_string();
    Ok(format!("{}:{}@", enc(&user), enc(&pass)))
}

/// Interpret a YAML scalar as a boolean flag (`true`/`1`/`yes`, or non-zero).
fn yaml_truthy(value: &serde_yaml::Value) -> bool {
    match value {
        serde_yaml::Value::Bool(b) => *b,
        serde_yaml::Value::String(s) => {
            matches!(s.to_ascii_lowercase().as_str(), "true" | "1" | "yes")
        }
        serde_yaml::Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        _ => false,
    }
}

/// Warn (once) about upstream `cache_params` keys litellm-rust cannot honour, so
/// they fail loud in the log instead of vanishing silently.
fn warn_unsupported_cache_params(params: &LitellmCacheParams) {
    const UNSUPPORTED: &[&str] = &[
        "mode",
        "namespace",
        "default_in_memory_ttl",
        "default_in_redis_ttl",
        "supported_call_types",
        "similarity_threshold",
    ];
    let present: Vec<&str> = UNSUPPORTED
        .iter()
        .copied()
        .filter(|k| params.extra.contains_key(*k))
        .collect();
    if !present.is_empty() {
        tracing::warn!(
            "litellm_settings.cache_params keys ignored by litellm-rust: {}; \
             see docs/protocols.md for the supported general_settings.cache options",
            present.join(", ")
        );
    }
}

fn validate(config: &GatewayConfig) -> Result<(), GatewayError> {
    validate_required_surface(config)?;
    validate_model_entries(
        &config.model_list,
        config.general_settings.database_url.is_some(),
    )?;
    validate_mcp_servers(&config.mcp_servers)?;
    validate_agents(
        &config.agents,
        config.general_settings.sandbox_choice.as_deref(),
        &config.general_settings.e2b_sandbox_params,
    )?;
    Ok(())
}

fn validate_required_surface(config: &GatewayConfig) -> Result<(), GatewayError> {
    if config.model_list.is_empty()
        && config.mcp_servers.is_empty()
        && config.agents.is_empty()
        && config.general_settings.database_url.is_none()
    {
        return Err(GatewayError::InvalidConfig(
            "model_list, mcp_servers, agents, or general_settings.database_url must contain at least one entry".to_owned(),
        ));
    }
    Ok(())
}

fn validate_model_entries(
    entries: &[ModelEntry],
    has_database_url: bool,
) -> Result<(), GatewayError> {
    for entry in entries {
        if entry.model_name.trim().is_empty() {
            return Err(GatewayError::InvalidConfig(
                "model_name cannot be empty".to_owned(),
            ));
        }

        if !entry.litellm_params.model.contains('/') {
            return Err(GatewayError::InvalidConfig(format!(
                "model must include provider prefix (e.g. anthropic/...), got {}",
                entry.litellm_params.model
            )));
        }

        if entry
            .litellm_params
            .api_key
            .as_deref()
            .unwrap_or("")
            .is_empty()
            && !has_database_url
        {
            return Err(GatewayError::InvalidConfig(format!(
                "{} is missing litellm_params.api_key",
                entry.model_name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod cache_compat_tests {
    use super::*;

    fn ls(yaml: &str) -> LitellmSettingsCompat {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn maps_upstream_redis_cache_params() {
        let mut cache = CacheSettings::default();
        apply_cache_compat(
            &mut cache,
            &ls(r#"
cache: true
cache_params:
  type: redis
  host: localhost
  port: 6379
  password: secret
  ttl: 600
"#),
        )
        .unwrap();
        assert!(cache.enabled);
        assert_eq!(cache.backend, CacheBackendKind::Redis);
        assert_eq!(
            cache.redis_url.as_deref(),
            Some("redis://:secret@localhost:6379")
        );
        assert_eq!(cache.ttl_secs, 600);
    }

    #[test]
    fn maps_upstream_disk_to_redb() {
        let mut cache = CacheSettings::default();
        apply_cache_compat(
            &mut cache,
            &ls(r#"
cache: true
cache_params:
  type: disk
  disk_cache_dir: /var/cache/litellm
  ttl: 120
"#),
        )
        .unwrap();
        assert!(cache.enabled);
        assert_eq!(cache.backend, CacheBackendKind::Redb);
        assert_eq!(
            cache.redb_path.as_deref(),
            Some("/var/cache/litellm/litellm-cache.redb")
        );
        assert_eq!(cache.ttl_secs, 120);
    }

    #[test]
    fn absent_type_defaults_to_redis() {
        // Upstream proxy defaults an unset cache type to redis.
        let mut cache = CacheSettings::default();
        apply_cache_compat(&mut cache, &ls("cache: true\ncache_params:\n  host: r\n")).unwrap();
        assert_eq!(cache.backend, CacheBackendKind::Redis);
        assert_eq!(cache.redis_url.as_deref(), Some("redis://r:6379"));
    }

    #[test]
    fn native_cache_takes_precedence() {
        let mut cache = CacheSettings {
            enabled: true,
            backend: CacheBackendKind::Memory,
            ..Default::default()
        };
        apply_cache_compat(
            &mut cache,
            &ls("cache: true\ncache_params:\n  type: redis\n  host: x\n"),
        )
        .unwrap();
        assert_eq!(cache.backend, CacheBackendKind::Memory);
        assert!(cache.redis_url.is_none());
    }

    #[test]
    fn unsupported_type_leaves_cache_off() {
        let mut cache = CacheSettings::default();
        apply_cache_compat(
            &mut cache,
            &ls("cache: true\ncache_params:\n  type: s3\n  s3_bucket_name: b\n"),
        )
        .unwrap();
        assert!(!cache.enabled);
    }

    #[test]
    fn cache_not_enabled_is_ignored() {
        let mut cache = CacheSettings::default();
        apply_cache_compat(
            &mut cache,
            &ls("cache: false\ncache_params:\n  type: redis\n  host: x\n"),
        )
        .unwrap();
        assert!(!cache.enabled);
    }

    #[test]
    fn from_raw_extracts_only_litellm_settings() {
        let raw = "model_list: []\nlitellm_settings:\n  cache: true\n  cache_params:\n    type: disk\n";
        let parsed = litellm_settings_from_raw(raw).unwrap().unwrap();
        assert_eq!(parsed.cache, Some(true));
        assert!(litellm_settings_from_raw("model_list: []\n")
            .unwrap()
            .is_none());
    }

    #[test]
    fn malformed_litellm_settings_is_ignored_not_fatal() {
        // A litellm_settings block we can't shape (cache_params as a scalar) must
        // degrade to "ignored", never abort the whole config load.
        let raw = "model_list: []\nlitellm_settings:\n  cache: true\n  cache_params: not-a-map\n";
        assert!(litellm_settings_from_raw(raw).unwrap().is_none());
    }

    #[test]
    fn tolerates_quoted_ttl() {
        // A quoted ttl ("600") would fail a strict f64 field; it must still apply.
        let mut cache = CacheSettings::default();
        apply_cache_compat(
            &mut cache,
            &ls("cache: true\ncache_params:\n  type: local\n  ttl: \"600\"\n"),
        )
        .unwrap();
        assert_eq!(cache.backend, CacheBackendKind::Memory);
        assert_eq!(cache.ttl_secs, 600);
    }

    #[test]
    fn synthesizes_rediss_url_with_username_and_encoded_password() {
        let mut cache = CacheSettings::default();
        apply_cache_compat(
            &mut cache,
            &ls(r#"
cache: true
cache_params:
  type: redis
  host: redis.example.com
  port: 6380
  username: admin
  password: "p@ss/w:rd"
  ssl: true
"#),
        )
        .unwrap();
        assert_eq!(
            cache.redis_url.as_deref(),
            Some("rediss://admin:p%40ss%2Fw%3Ard@redis.example.com:6380")
        );
    }

    #[test]
    fn redis_url_env_fallback_when_no_host() {
        std::env::set_var("REDIS_URL", "redis://from-env:6379");
        let mut cache = CacheSettings::default();
        apply_cache_compat(&mut cache, &ls("cache: true\ncache_params:\n  type: redis\n")).unwrap();
        assert_eq!(cache.redis_url.as_deref(), Some("redis://from-env:6379"));
        std::env::remove_var("REDIS_URL");
    }
}
