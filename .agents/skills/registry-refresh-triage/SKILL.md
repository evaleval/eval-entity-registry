---
name: registry-refresh-triage
description: >-
  Diagnose and fix failed eval-entity-registry refresh cron runs
  (refresh-models, refresh-hub-stats, seed + publish), stall beacon issues,
  fixpoint mismatches, and regression gate failures. Use when a scheduled
  registry workflow fails or blocks on test_gate_invariants.py.
metadata:
  version: 0.1.0
---

# Registry refresh triage — agent procedure

Use this runbook when a scheduled `eval-entity-registry` workflow (`Refresh models.dev source`, `Refresh hub-stats source`, or `Seed + publish`) fails or opens a `refresh-*-stalled` issue.

## 1. Inspect the failure and classify the tier

```bash
gh run list -R evaleval/eval-entity-registry -L 10
gh issue list -R evaleval/eval-entity-registry --state open --label refresh-models-stalled
gh issue list -R evaleval/eval-entity-registry --state open --label refresh-hub-stats-stalled
gh run view <run_id> -R evaleval/eval-entity-registry --log-failed
```

Classify immediately into one of three tiers:
1. **Infra / CI dispatch**: `ci_dispatch_seed.py`, `seed.yml` superseded/diverged, HF Hub download error, or missing `hub_stats_index`.
2. **Fixpoint check** (`sha256sum --check` failed in `refresh-models.yml`): non-idempotent carry-forward vs reconciliation loop across consecutive runs.
3. **Regression gate** (`tests/test_gate_invariants.py` or `tests/test_refresh_modelsdev_dedup.py`): upstream catalog data caused a resolution, duplicate, or org-integrity break.

## 2. Reproduce in an isolated worktree off `origin/main`

Never mutate a dirty `main` checkout. Set `REPO` to the local `eval-entity-registry` clone (harnesses launched from the parent workspace start one level above it), fetch `origin`, and create a named branch in a clean worktree if you plan to make edits (or a detached worktree for read-only inspection only):

```bash
REPO=/path/to/eval-entity-registry
WT=/tmp/registry-repro

git -C "$REPO" fetch origin
# Use `-b fix/<short-name>` when implementing a fix; `--detach` for read-only triage.
git -C "$REPO" worktree add --detach "$WT" origin/main
```

Reuse the main clone's `.venv` when available (avoids rebuilding dependencies offline); otherwise run `uv sync` inside `$WT`. **Always export `PYTHONPATH` and `FIXTURES_PATH` pointing at `$WT`** before invoking `$PY`, `$CLI`, or `$PT` so editable installs from `$REPO/.venv` never shadow the worktree's code or fixtures:

```bash
if [ -x "$REPO/.venv/bin/python" ]; then
  PY="$REPO/.venv/bin/python"
  CLI="$REPO/.venv/bin/eval-entity-registry"
  PT="$REPO/.venv/bin/pytest"
else
  (cd "$WT" && uv sync)
  PY="$WT/.venv/bin/python"
  CLI="$WT/.venv/bin/eval-entity-registry"
  PT="$WT/.venv/bin/pytest"
fi
export PYTHONPATH="$WT/src:$WT/packages/eval-entity-resolver/src"
export FIXTURES_PATH="$WT/fixtures"
```

### Branch by failed workflow

- **`Refresh models.dev source` (`refresh-models.yml`)**:
  ```bash
  curl -fsSL https://models.dev/api.json -o /tmp/modelsdev_api.json
  mkdir -p "$WT/fixtures"
  curl -fsSL https://huggingface.co/datasets/evaleval/entity-registry-data/resolve/main/hub_stats_index/part-0.parquet \
    -o "$WT/fixtures/hub_stats_index.parquet"

  (
    cd "$WT"
    LOCAL_MODE=true bash -euo pipefail -c '
      "$0" scripts/refresh_from_modelsdev.py --no-fetch
      "$0" scripts/refresh_from_modelsdev.py --catalog --no-fetch
      "$0" scripts/refresh_from_modelsdev.py --reconcile-orgs
      find fixtures -name "*.parquet" ! -name "hub_stats_index.parquet" -delete
      "$1" seed --local --seed-dir seed
      "$2" tests/test_gate_invariants.py tests/test_refresh_modelsdev_dedup.py -q
    ' "$PY" "$CLI" "$PT"
  )
  ```
- **`Refresh hub-stats source` (`refresh-hub-stats.yml`)**:
  From `$WT` (with `PYTHONPATH` and `FIXTURES_PATH` exported above), run `HF_TOKEN="$HF_TOKEN" "$PY" scripts/refresh_from_hub_stats.py --live`, prune `fixtures/*.parquet` (keeping `hub_stats_index.parquet`), run `LOCAL_MODE=true "$CLI" seed --local --seed-dir seed`, and run `LOCAL_MODE=true "$PT" tests/test_gate_invariants.py tests/test_refresh_modelsdev_dedup.py -q`.
- **`Seed + publish` (`seed.yml`) / `Publish registry data` (`publish-registry-data.yml`)**:
  From `$WT` (with `PYTHONPATH` and `FIXTURES_PATH` exported above), inspect `scripts/publish_registry_data.py` (`--freshness-check-only` / `--dry-run --skip-seed`) and `scripts/ci_dispatch_seed.py`; run `LOCAL_MODE=true "$PT" tests/test_ci_dispatch_seed.py -q` and the full `LOCAL_MODE=true "$PT" tests/ -q` suite against freshly seeded `$WT/fixtures/`.

## 3. Run the diff & side-effect audit (`models.dev`)

Before choosing a fix for a `models.dev` failure, run the advisory side-effect summary (note: it exits 0 even when inputs are missing and only checks bare slash-free author IDs for `no_match`, so treat it as diagnostic input, not a gate):

```bash
PYTHONPATH="$WT/src:$WT/packages/eval-entity-resolver/src" \
  "$PY" "$WT/.agents/skills/registry-refresh-triage/scripts/triage_diff.py" "$WT" --api-json /tmp/modelsdev_api.json
```

Review:
- Added / removed canonical IDs in `models_dev.generated.yaml`.
- Every `review_status` flip (`draft -> reviewed` or `reviewed -> draft`) — critical because `cli.py` grants bare-name ownership (`lab_leaf_owner`) to `reviewed` lab entries.
- First-party author-lab IDs in `models.dev` that resolve to `no_match`.

## 4. Gate-failure decision table

| Failing check | First place to look | Standard fix |
|---|---|---|
| `test_openrouter_adoption_no_resolution_regression` | Check `triage_diff.py` for `draft -> reviewed` flips (triggering `cli.py`'s `lab_leaf_owner` redirect) and upstream display-name / `-latest` pointer rotations. | Fix the generator rule that minted or promoted the conflicting canonical. Do **not** add to `_ADOPTION_REGRESSION_EXEMPT` unless the old target was demonstrably wrong. |
| `test_no_two_full_entries_share_folded_identity` | Check whether one member is a stale `tier3_inferred.generated.yaml` draft under a non-canonical prefix (`mistral/`, `together/`, `fireworks/`) superseded by a new first-party `models.dev` canonical. | Drop the stale tier-3 draft in `seed/models/core.yaml` (`skip_source_ids`) and alias the raw form onto the canonical in `seed/models/enrichments/aliases.yaml`. |
| `test_bare_canonical_org_ambiguity_is_explicit` (`_BARE_ID_CEILING`) | Inspect new org-less mints in `models_dev.generated.yaml`. | Curate known developers in `core.yaml` (or alias product wrappers in `enrichments/aliases.yaml` + `skip_source_ids`). Only raise `_BARE_ID_CEILING` for genuinely ownerless stealth models. |
| Fixpoint (`sha256sum --check`) | Diff pass-1 vs pass-2 output directly. | Resolve the carry-forward vs `reconcile_generated_against_existing` disagreement and add a 2-run fixpoint unit test in `tests/test_refresh_modelsdev_dedup.py`. |
| `test_alias_enrichments_never_mint_canonicals` | An `enrichments/aliases.yaml` target only exists after regenerating `sources/*.generated.yaml`. | Land the regenerated `sources/*.generated.yaml` in the same PR (or second commit on the branch) so PR dry-run CI sees the target. |

## 5. Verification & safe cleanup

1. **Branch before editing**: if you started in a detached worktree, create a named branch (`git switch -c fix/<short-name>`) before making edits so nothing can be orphaned.
2. **Code hygiene**: stage hand-edited files and run `"$PY" scripts/check_code_hygiene.py`.
3. **Regression test**: add a test that fails on the bad input before the fix and passes after.
4. **3-pass fixpoint (`models.dev`)**: run refresh → catalog → reconcile-orgs twice, hash all 6 tracked YAML files, run a third time, verify `shasum -a 256 -c` (or `sha256sum --check`) passes.
5. **Fresh seed + gates** (run from `$WT` with `PYTHONPATH="$WT/src:$WT/packages/eval-entity-resolver/src"` and `FIXTURES_PATH="$WT/fixtures"` exported): prune `fixtures/*.parquet` (keep `hub_stats_index.parquet`), run `LOCAL_MODE=true "$CLI" seed --local --seed-dir seed`, then run `LOCAL_MODE=true "$PT" tests/test_gate_invariants.py tests/test_refresh_modelsdev_dedup.py tests/test_refresh_modelsdev.py -q`.
6. **Owner approval & cleanup**: never push, open a PR, or trigger `seed.yml` without explicit owner confirmation. Remove a temporary read-only worktree only once clean with plain `git -C "$REPO" worktree remove "$WT"` (never `--force` on a worktree holding work in progress).
