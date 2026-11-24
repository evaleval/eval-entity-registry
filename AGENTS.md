# AGENTS.md — eval-entity-registry

Entry point for coding agents in this repo. (Open standard — see https://agents.md.
Claude Code also reads this file and any `.claude/skills/`.)

This repo resolves raw model/benchmark/metric/harness strings (from the EEE
datastore) to stable **canonical ids**, and stores resolved results.

## Skills (agent-invoked, loaded on demand)
| Skill | Use when |
|---|---|
| [`registry-entity-aliases`](.claude/skills/registry-entity-aliases/SKILL.md) | A slug lands on `no_match`/`draft`; adding aliases or a new canonical to the seed; an EEE adapter's ids won't resolve |
| [`registry-refresh-triage`](.agents/skills/registry-refresh-triage/SKILL.md) | A scheduled refresh cron (`refresh-models`, `refresh-hub-stats`, `seed + publish`) fails, opens a stall beacon, or breaks a gate/fixpoint check |

## Layout
- `seed/` — the seed data. Curated layers: `models/core.yaml`,
  `models/enrichments/` (e.g. `aliases.yaml`), and the flat
  `benchmarks.yaml` / `metrics.yaml` / `harnesses.yaml` / `orgs.yaml`.
  `models/sources/*.generated.yaml` are machine-generated (daily cron) —
  never hand-edit them.
- `packages/eval-entity-resolver/` — the standalone `Resolver` (importable elsewhere).
- `README.md` → `## ID conventions` / `## How it works` — the id standards & resolution strategy.
- `CONTRIBUTING.md` — the seed-change + verify workflow.

## Conventions
- **Search first; alias to the existing canonical; new canonical only if genuinely absent.**
- **Don't add mechanical variants** — the `normalized` matcher already collapses
  case + separators + dots. See `README.md` → `## ID conventions` for the id standards
  and `CONTRIBUTING.md` for the seed/verify workflow.
- Verify with `find fixtures -name '*.parquet' -delete 2>/dev/null || true; uv run eval-entity-registry seed --local`
  (prune stale fixtures first) → `Resolver.from_parquet("fixtures/")` → `pytest`.
  A PR states which slugs were `no_match` before and their new canonical.

## Human docs
`README.md` (how it works) and `CONTRIBUTING.md` (how to contribute) are for people;
the skill points agents at them. Keep agent-only instructions in `.claude/skills/`.
