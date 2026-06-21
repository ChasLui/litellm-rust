# AGENTS.md

Before making implementation changes, read and follow the repo-wide
[`CODING_STANDARDS.md`](./CODING_STANDARDS.md).

## First-time setup

Run once after cloning to activate the committed git hooks:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook keeps `model_prices_backup.json` in sync with the
upstream litellm JSON on every commit. It warns and skips silently if
the network is unavailable — it never blocks a commit.

## Source of truth

- **Code conventions / architecture rules:** [`CODING_STANDARDS.md`](./CODING_STANDARDS.md) is authoritative — read it before any implementation change.
- **Layered request flow & module map:** [`docs/architecture.md`](./docs/architecture.md).

## Agent toolchain — which tool, when

Four code-intelligence layers are wired into this repo. They have distinct jobs; do not mix them. All four local dirs (`.gitnexus/`, `.serena/`, `.understand-anything/`, `.claude/`) are gitignored — only this file, `CLAUDE.md`, and `paseo.json` are committed.

| Need | Tool | Entry point |
|------|------|-------------|
| **Blast radius before editing a symbol; call chains; execution flows; safe rename** | GitNexus | `gitnexus_*` MCP tools — see the GitNexus block below |
| **Semantic symbol nav & precise edits; persisted project memories** | Serena | `serena_*` MCP tools; 5 onboarding memories in `.serena/memories/` (project_overview, architecture, suggested_commands, code_style_conventions, task_completion_checklist) |
| **Architecture overview, guided onboarding tour, domain map** | Understand-Anything | `.understand-anything/knowledge-graph.json` (596 nodes / 9 layers / 13-step zh tour); run `/understand-dashboard` to explore; `/understand` to refresh (config: `autoUpdate:false`, `outputLanguage:zh`) |
| **Run/iterate in isolated worktrees; multi-agent orchestration; long-running gateway service** | Paseo | [`paseo.json`](./paseo.json) (worktree setup + `ci`/`test`/`serve` scripts); orchestration provider prefs in `~/.paseo/orchestration-preferences.json` |

Rule of thumb: **GitNexus answers "what breaks if I change X"**, **Serena answers "where is X and edit it"**, **Understand-Anything answers "how is this project shaped"**, **Paseo runs the work**. Default to Serena/GitNexus over raw grep for code navigation and impact.

### In a Paseo worktree

The four agent dirs are gitignored, so a fresh worktree does not contain them. `paseo.json`'s `setup` symlinks the main checkout's `.serena/memories` + `project.yml` and `.claude/skills/gitnexus` into the worktree, so **Serena memories and the GitNexus skill files work there automatically**. For the **GitNexus MCP tools**, the worktree path is not indexed — always pass the repo name explicitly, e.g. `gitnexus_impact({repo: "litellm-rust", target: "...", direction: "upstream"})`; it reads the main checkout's index (which reflects the last committed state, so re-run `npx gitnexus analyze` in the main checkout if the worktree branch changed core symbols). Understand-Anything auto-redirects graph reads/writes to the main repo, so `/understand-dashboard` and `/understand` work from a worktree unchanged.

## Task completion

Before declaring a change done, run the CI gate (also available as `paseo run ... ci`):

```bash
cargo fmt --all --check
cargo check --all-targets --locked
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
python3 scripts/check_code_size.py   # ≤300 lines/file, ≤50 LOC/fn
```

Commit messages: Chinese, human-friendly, conventional-commits prefix (feat/fix/chore/docs).

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
