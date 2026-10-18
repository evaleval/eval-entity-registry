#!/usr/bin/env python3
"""
Verify sync results: check resolution quality after seed+sync.

Usage:
    uv run python scripts/verify_sync.py
"""
import os
os.environ["LOCAL_MODE"] = "true"

from eval_entity_registry.store.hf_store import get_store

store = get_store()
store.load()

for table in ["canonical_models", "canonical_benchmarks", "canonical_metrics", "eval_harnesses"]:
    df = store.table(table)
    total = len(df)
    reviewed = int((df["review_status"] == "reviewed").sum()) if "review_status" in df.columns else 0
    draft = int((df["review_status"] == "draft").sum()) if "review_status" in df.columns else 0
    print(f"  {table:30s}  total={total:4d}  reviewed={reviewed:4d}  draft={draft:4d}")

eval_df = store.table("eval_results")
print(f"\n  eval_results: {len(eval_df)} rows")

aliases_df = store.table("aliases")
confirmed = int((aliases_df["status"] == "confirmed").sum())
auto = int((aliases_df["status"] == "auto").sum())
uncertain = int((aliases_df["status"] == "uncertain").sum())
print(f"  aliases: total={len(aliases_df)}  confirmed={confirmed}  auto={auto}  uncertain={uncertain}")

if len(eval_df) > 0:
    bench_df = store.table("canonical_benchmarks")
    reviewed_ids = set(bench_df[bench_df["review_status"] == "reviewed"]["id"])
    matched = eval_df["benchmark_id"].isin(reviewed_ids).sum()
    total = len(eval_df)
    pct = matched / total * 100
    print(f"\n  Benchmark resolution quality:")
    print(f"    {matched}/{total} results ({pct:.1f}%) resolved to seeded (reviewed) benchmarks")
    print(f"    {total - matched}/{total} results ({100-pct:.1f}%) resolved to auto-drafted benchmarks")

    metric_df = store.table("canonical_metrics")
    reviewed_metric_ids = set(metric_df[metric_df["review_status"] == "reviewed"]["id"])
    metric_matched = eval_df["metric_id"].dropna().isin(reviewed_metric_ids).sum()
    metric_total = eval_df["metric_id"].notna().sum()
    if metric_total > 0:
        mpct = metric_matched / metric_total * 100
        print(f"\n  Metric resolution quality:")
        print(f"    {metric_matched}/{metric_total} results ({mpct:.1f}%) resolved to seeded (reviewed) metrics")
        print(f"    {metric_total - metric_matched}/{metric_total} results ({100-mpct:.1f}%) resolved to auto-drafted metrics")

    draft_benchmarks = bench_df[bench_df["review_status"] == "draft"]["id"].tolist()
    if draft_benchmarks:
        print(f"\n  Auto-drafted benchmarks ({len(draft_benchmarks)}):")
        for b in sorted(draft_benchmarks):
            count = (eval_df["benchmark_id"] == b).sum()
            print(f"    {b:40s} ({count} results)")

    draft_metrics = metric_df[metric_df["review_status"] == "draft"]["id"].tolist()
    if draft_metrics:
        print(f"\n  Auto-drafted metrics ({len(draft_metrics)}):")
        for m in sorted(draft_metrics):
            count = (eval_df["metric_id"] == m).sum()
            print(f"    {m:40s} ({count} results)")
