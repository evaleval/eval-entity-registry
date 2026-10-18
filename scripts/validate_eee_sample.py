#!/usr/bin/env python3
"""Validate the resolver against a real sample of the EEE datastore.

Unlike the frozen-oracle gate suite, this measures coverage on live rows from
`evaleval/EEE_datastore`. For each sampled record it extracts the raw model /
harness / benchmark / metric strings the same way services/ingestion.py does,
resolves each (resolve-only, no auto-create) against the committed fixtures,
and reports per entity type: coverage, strategy breakdown, the most-frequent
no_match strings, and a random sample of resolutions.

Needs network (pulls EEE configs from HF) + the seeded fixtures
(`LOCAL_MODE=true uv run eval-entity-registry seed --local` first).

Usage:
    LOCAL_MODE=true uv run python scripts/validate_eee_sample.py
    LOCAL_MODE=true uv run python scripts/validate_eee_sample.py --configs 20 --rows 500
    LOCAL_MODE=true uv run python scripts/validate_eee_sample.py --config hfopenllm_v2  # one named config
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REGISTRY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REGISTRY_ROOT / "fixtures"

# Raws must be extracted exactly as services/ingestion.py does.
from eval_entity_registry.services.ingestion import _detect_sub_categories
from eval_entity_resolver.eee import clean_eval_name, extract_metric
from eval_entity_resolver.resolver import Resolver


def _extract_raws(record: dict) -> dict[str, list[tuple[str, str | None]]]:
    """Return {entity_type -> [(raw_value, source_config_or_None), ...]} for one
    EEE record — mirrors services/ingestion._process_record's field selection."""
    out: dict[str, list[tuple[str, str | None]]] = defaultdict(list)

    model_info = record.get("model_info") or {}
    model_raw = model_info.get("id") or model_info.get("model_name") or model_info.get("name")
    if model_raw:
        out["model"].append((model_raw, None))

    eval_lib = record.get("eval_library") or {}
    harness_raw = eval_lib.get("name") or eval_lib.get("library_name")
    if harness_raw:
        out["harness"].append((harness_raw, None))

    evaluation_results = record.get("evaluation_results") or []
    if not isinstance(evaluation_results, list):
        evaluation_results = []
    eval_to_parent = _detect_sub_categories(evaluation_results)
    return _extract_bench_metric(out, evaluation_results, eval_to_parent)


def _extract_bench_metric(out, evaluation_results, eval_to_parent):
    for parent in {v for v in eval_to_parent.values() if v}:
        out["benchmark"].append((parent, None))
    for er in evaluation_results:
        if not isinstance(er, dict):
            continue
        eval_name = er.get("evaluation_name")
        if eval_name:
            bench = clean_eval_name(eval_name)
            if bench:
                out["benchmark"].append((bench, None))
        mc = er.get("metric_config") or {}
        metric_raw = extract_metric(
            mc.get("metric_name") or mc.get("metric_id")
            or mc.get("evaluation_description") or ""
        )
        if metric_raw:
            out["metric"].append((metric_raw, None))
    return out


def _config_names(datasets) -> list[str]:
    """EEE config names; offline, get_dataset_config_names returns ['default'],
    so fall back to the local HF datasets cache (minus -samples auxiliary
    configs and the synthetic 'default-*' fallback)."""
    def _from_cache() -> list[str]:
        root = Path.home() / ".cache" / "huggingface" / "datasets" / "evaleval___eee_datastore"
        if not root.exists():
            return []
        names = sorted({p.name for p in root.iterdir() if p.is_dir()})
        return [c for c in names
                if not c.endswith("_samples") and not c.startswith("default-") and c != "default"]

    try:
        names = datasets.get_dataset_config_names("evaleval/EEE_datastore")
        if names == ["default"]:
            cached = _from_cache()
            if cached:
                print(f"[validate] HF Hub unreachable; using {len(cached)} configs from "
                      f"local cache.", file=sys.stderr)
                return cached
        return names
    except Exception as e:  # noqa: BLE001
        print(f"[validate] config listing failed ({e}); using local cache.", file=sys.stderr)
        return _from_cache()


def _sample_configs(all_configs: list[str], n: int, rng: random.Random) -> list[str]:
    if n >= len(all_configs):
        return sorted(all_configs)
    # Deterministic spread across the sorted config list (variety, not the first n).
    ordered = sorted(all_configs)
    step = len(ordered) / n
    return [ordered[int(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=12, help="number of EEE configs to sample")
    ap.add_argument("--rows", type=int, default=300, help="max rows per config")
    ap.add_argument("--config", action="append", help="explicit config name(s); repeatable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-size", type=int, default=15, help="resolutions to print per entity type")
    args = ap.parse_args()

    if not FIXTURES.exists() or not (FIXTURES / "aliases.parquet").exists():
        print("[validate] missing fixtures — run `LOCAL_MODE=true uv run eval-entity-registry "
              "seed --local` first.", file=sys.stderr)
        return 1

    import datasets

    rng = random.Random(args.seed)
    if args.config:
        configs = args.config
    else:
        print("[validate] listing EEE configs…", file=sys.stderr)
        configs = _sample_configs(_config_names(datasets), args.configs, rng)
    print(f"[validate] sampling {len(configs)} config(s): {configs}", file=sys.stderr)

    resolver = Resolver.from_parquet(str(FIXTURES))

    raw_counts: dict[str, Counter] = defaultdict(Counter)
    raw_sc: dict[str, dict[str, str | None]] = defaultdict(dict)
    for cfg in configs:
        try:
            ds = datasets.load_dataset(
                "evaleval/EEE_datastore", cfg, split="train",
                download_mode="reuse_dataset_if_exists",
            )
        except Exception as e:  # noqa: BLE001 — one bad config shouldn't abort the sweep
            print(f"[validate]   skip {cfg}: {e}", file=sys.stderr)
            continue
        n = min(args.rows, len(ds))
        for i in range(n):
            for etype, items in _extract_raws(ds[i]).items():
                for raw, sc in items:
                    raw_counts[etype][raw] += 1
                    raw_sc[etype].setdefault(raw, sc)
        print(f"[validate]   {cfg}: {n} rows", file=sys.stderr)

    print("\n" + "=" * 72)
    print("RESOLVER VALIDATION — real EEE sample (resolve-only, vs committed fixtures)")
    print("=" * 72)
    for etype in ("model", "benchmark", "metric", "harness"):
        raws = raw_counts[etype]
        if not raws:
            continue
        strat = Counter()
        occ_total = occ_resolved = 0
        no_match: Counter = Counter()
        resolved_samples: list[tuple[str, str, str]] = []
        for raw, cnt in raws.items():
            res = resolver.resolve(raw, etype, raw_sc[etype].get(raw))
            cid = res.canonical_id
            st = getattr(res, "strategy", None) or ("no_match" if cid is None else "?")
            strat[st] += 1
            occ_total += cnt
            if cid is None:
                no_match[raw] += cnt
            else:
                occ_resolved += cnt
                resolved_samples.append((raw, cid, st))
        distinct = len(raws)
        cov_d = 100 * (distinct - len(no_match)) / distinct
        cov_o = 100 * occ_resolved / occ_total if occ_total else 0
        print(f"\n### {etype.upper()}  — {distinct} distinct raw / {occ_total} occurrences")
        print(f"  coverage: {cov_d:.1f}% distinct  |  {cov_o:.1f}% occurrence-weighted")
        print(f"  strategy: {dict(strat.most_common())}")
        if no_match:
            print(f"  TOP no_match ({len(no_match)} distinct):")
            for raw, c in no_match.most_common(args.sample_size):
                print(f"     ({c:>5}x) {raw!r}")
        if resolved_samples:
            print(f"  sample resolutions (eyeball correctness):")
            for raw, cid, st in rng.sample(resolved_samples, min(args.sample_size, len(resolved_samples))):
                print(f"     {raw!r:42} -> {cid!r}  [{st}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
