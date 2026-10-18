#!/usr/bin/env python3
"""
Scan all EEE configs for raw `model_info.id` values, with frequency and
per-config breakdown. Optionally check coverage against the current local
registry — emits the unresolved-by-frequency list as a YAML stub for
seed/models/core.yaml.

Usage:
    uv run python scripts/scan_eee_models.py                      # scan + frequency table
    uv run python scripts/scan_eee_models.py --check-coverage     # also resolve each
    uv run python scripts/scan_eee_models.py --emit-overrides-stub  # emit YAML stub for misses
    uv run python scripts/scan_eee_models.py --top 50             # limit output

Output is text/YAML to stdout; progress to stderr.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import datasets


def _scan_local_cache_for_configs() -> list[str]:
    """EEE configs in the local HF datasets cache, minus the -samples auxiliary
    configs (different schema) and the synthetic 'default-*' fallback."""
    from pathlib import Path
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "evaleval___eee_datastore"
    if not cache_root.exists():
        return []
    configs = sorted({p.name for p in cache_root.iterdir() if p.is_dir()})
    return [c for c in configs
            if not c.endswith("_samples")
            and not c.startswith("default-")
            and c != "default"]


def _config_names_offline_fallback() -> list[str]:
    """EEE config names; offline, get_dataset_config_names returns ['default'],
    so fall back to the local HF datasets cache."""
    try:
        names = datasets.get_dataset_config_names("evaleval/EEE_datastore")
        if names == ["default"]:
            cache_names = _scan_local_cache_for_configs()
            if cache_names:
                print(f"[scan] HF Hub unreachable; using {len(cache_names)} configs "
                      f"from local cache.", file=sys.stderr)
                return cache_names
        return names
    except Exception as e:
        print(f"[scan] get_dataset_config_names failed ({e}); scanning local cache.",
              file=sys.stderr)
        return _scan_local_cache_for_configs()


def scan_models() -> dict:
    """Returns frequency counter + per-config map of model_info.id values."""
    config_names = _config_names_offline_fallback()
    models: Counter = Counter()
    model_configs: dict[str, set] = defaultdict(set)

    for cfg in config_names:
        print(f"  scanning {cfg}...", file=sys.stderr)
        try:
            ds = datasets.load_dataset(
                "evaleval/EEE_datastore", cfg,
                download_mode="reuse_dataset_if_exists",
            )["train"]
        except Exception as e:
            print(f"    [error] {cfg}: {e}", file=sys.stderr)
            continue

        for i in range(len(ds)):
            record = dict(ds[i])
            mi = record.get("model_info") or {}
            raw_id = mi.get("id") or mi.get("model_name") or mi.get("name")
            if raw_id and isinstance(raw_id, str) and raw_id.strip():
                key = raw_id.strip()
                models[key] += 1
                model_configs[key].add(cfg)

    return {"models": models, "configs": model_configs}


def check_coverage(models: Counter) -> tuple[dict[str, dict], dict]:
    """Resolve each unique raw_id against the local registry. Returns
    (per-id results, summary stats)."""
    os.environ["LOCAL_MODE"] = "true"
    from eval_entity_registry.store.hf_store import get_store
    from eval_entity_registry.services.resolution_service import ResolutionService

    store = get_store()
    store.load()
    svc = ResolutionService(store)

    per_id: dict[str, dict] = {}
    n_distinct_resolved = 0
    n_occurrences_resolved = 0
    n_distinct_total = len(models)
    n_occurrences_total = sum(models.values())

    for raw_id, count in models.items():
        r = svc.resolve(raw_id, "model", None, None)
        # An auto-drafted result is no canonical match; count it as unresolved.
        resolved = (r["canonical_id"] is not None) and (not r["created_new"]) and (r["strategy"] != "auto_draft")
        per_id[raw_id] = {
            "count": count,
            "canonical_id": r["canonical_id"],
            "strategy": r["strategy"],
            "resolved": resolved,
        }
        if resolved:
            n_distinct_resolved += 1
            n_occurrences_resolved += count

    summary = {
        "distinct_total": n_distinct_total,
        "distinct_resolved": n_distinct_resolved,
        "distinct_pct": (n_distinct_resolved / n_distinct_total * 100) if n_distinct_total else 0,
        "occurrences_total": n_occurrences_total,
        "occurrences_resolved": n_occurrences_resolved,
        "occurrences_pct": (n_occurrences_resolved / n_occurrences_total * 100) if n_occurrences_total else 0,
    }
    return per_id, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check-coverage", action="store_true",
                   help="Resolve each raw_id against the local registry; report coverage")
    p.add_argument("--emit-overrides-stub", action="store_true",
                   help="Print a YAML stub for the top unresolved entries")
    p.add_argument("--top", type=int, default=100,
                   help="Limit number of entries displayed (default 100)")
    args = p.parse_args()

    data = scan_models()
    counter: Counter = data["models"]
    configs = data["configs"]

    if not args.check_coverage and not args.emit_overrides_stub:
        print(f"\n{'='*70}")
        print(f"  EEE model_info.id frequency ({len(counter)} distinct, "
              f"{sum(counter.values())} total occurrences)")
        print(f"{'='*70}")
        for name, count in counter.most_common(args.top):
            n_configs = len(configs.get(name, set()))
            print(f"  {count:5d}× ({n_configs:2d} configs)  {name}")
        return 0

    per_id, summary = check_coverage(counter)

    print(f"\n{'='*70}")
    print(f"  EEE model coverage vs local registry")
    print(f"{'='*70}")
    print(f"  distinct ids:    {summary['distinct_resolved']:5d} / {summary['distinct_total']:5d}  "
          f"({summary['distinct_pct']:5.1f}%)")
    print(f"  occurrences:     {summary['occurrences_resolved']:5d} / {summary['occurrences_total']:5d}  "
          f"({summary['occurrences_pct']:5.1f}%)")
    print()

    # EEE carries many one-off synthetic model ids; coverage on ids present in
    # N+ configs is the better proxy for real-model coverage.
    print("  Coverage by config-presence threshold:")
    print(f"  {'min_configs':>12s}  {'distinct':>14s}  {'occurrences':>14s}")
    for threshold in (1, 2, 3, 5, 7):
        d_total = d_resolved = o_total = o_resolved = 0
        for raw_id, info in per_id.items():
            if len(configs.get(raw_id, set())) >= threshold:
                d_total += 1
                o_total += info["count"]
                if info["resolved"]:
                    d_resolved += 1
                    o_resolved += info["count"]
        d_pct = (d_resolved / d_total * 100) if d_total else 0
        o_pct = (o_resolved / o_total * 100) if o_total else 0
        print(f"  {'≥'+str(threshold):>12s}  "
              f"{d_resolved:5d}/{d_total:5d} ({d_pct:4.1f}%)  "
              f"{o_resolved:5d}/{o_total:5d} ({o_pct:4.1f}%)")
    print()

    unresolved = [
        (raw_id, info["count"]) for raw_id, info in per_id.items() if not info["resolved"]
    ]
    unresolved.sort(key=lambda x: -x[1])

    if args.emit_overrides_stub:
        print("# Unresolved EEE model_info.id values (top by occurrence count).")
        print("# Add curated entries to seed/models/core.yaml.")
        print(f"# Coverage: {summary['occurrences_pct']:.1f}% by occurrence, "
              f"{summary['distinct_pct']:.1f}% by distinct id.")
        print()
        for raw_id, count in unresolved[:args.top]:
            n_configs = len(configs.get(raw_id, set()))
            print(f"# {count:5d}× ({n_configs} configs) — {raw_id}")
    else:
        print(f"  Top {args.top} unresolved by occurrence:")
        print(f"  {'-'*60}")
        for raw_id, count in unresolved[:args.top]:
            n_configs = len(configs.get(raw_id, set()))
            print(f"  {count:5d}× ({n_configs:2d} cfg)  {raw_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
