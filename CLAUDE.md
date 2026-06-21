# litellm-rust — agent guide

**Read first:** [`AGENTS.md`](./AGENTS.md) (canonical agent guide — toolchain, task-completion gate, commit style) and [`CODING_STANDARDS.md`](./CODING_STANDARDS.md) (authoritative code conventions).

This is a low-overhead, LiteLLM-compatible AI gateway (`lite`) for coding agents. v0 supports Anthropic only; unsupported providers must fail at boot. Add a provider by adding a provider module — never by branching through the app. No `unwrap()`/`expect()` in request paths.

**Toolchain (details in AGENTS.md):** GitNexus = blast radius / call chains (block below) · Serena = semantic nav + `.serena/memories/` · Understand-Anything = `.understand-anything/knowledge-graph.json` + `/understand-dashboard` · Paseo = worktrees + `paseo.json` scripts. Prefer Serena/GitNexus over raw grep.

Reply in Chinese; commit messages in Chinese (conventional-commits prefix).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **litellm-rust** (2536 symbols, 5526 relationships, 211 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/litellm-rust/context` | Codebase overview, check index freshness |
| `gitnexus://repo/litellm-rust/clusters` | All functional areas |
| `gitnexus://repo/litellm-rust/processes` | All execution flows |
| `gitnexus://repo/litellm-rust/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->