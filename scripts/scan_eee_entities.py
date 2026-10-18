#!/usr/bin/env python3
"""
Scan all EEE configs and extract distinct entity names.

Usage:
    uv run python scripts/scan_eee_entities.py
    uv run python scripts/scan_eee_entities.py --output-yaml   # emit draft seed YAML to stdout

Outputs a frequency table of benchmarks, metrics, and harnesses found across
all EEE configs. With --output-yaml, emits draft YAML suitable for pasting
into seed/*.yaml files (needs human review before committing).
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

import datasets


def _strip_benchmark_qualifier(desc: str) -> str:
    return re.sub(r"\s+on\s+\S+.*$", "", desc).strip()


def _slugify(value: str) -> str:
    slug = value.lower().strip()
    slug = re.sub(r"[^\w\s\-/]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-") or "unknown"


def scan_all_configs() -> dict:
    """Scan all EEE configs. Returns {benchmarks, metrics, harnesses} counters."""
    config_names = datasets.get_dataset_config_names("evaleval/EEE_datastore")

    benchmarks: Counter = Counter()
    metrics: Counter = Counter()
    harnesses: Counter = Counter()
    bench_configs: dict[str, set] = defaultdict(set)
    metric_configs: dict[str, set] = defaultdict(set)
    harness_configs: dict[str, set] = defaultdict(set)

    for cfg in config_names:
        print(f"  scanning {cfg}...", file=sys.stderr)
        try:
            ds = datasets.load_dataset("evaleval/EEE_datastore", cfg)["train"]
        except Exception as e:
            print(f"    [error] {cfg}: {e}", file=sys.stderr)
            continue

        for i in range(len(ds)):
            record = dict(ds[i])

            eval_lib = record.get("eval_library") or {}
            harness_raw = eval_lib.get("name") or eval_lib.get("library_name")
            if harness_raw and harness_raw.strip():
                harnesses[harness_raw.strip()] += 1
                harness_configs[harness_raw.strip()].add(cfg)

            eval_results = record.get("evaluation_results") or []
            if not isinstance(eval_results, list):
                continue
            for er in eval_results:
                eval_name = er.get("evaluation_name")
                if eval_name and eval_name.strip():
                    benchmarks[eval_name.strip()] += 1
                    bench_configs[eval_name.strip()].add(cfg)

                metric_config = er.get("metric_config") or {}
                metric_raw = (
                    metric_config.get("metric_id")
                    or metric_config.get("metric_name")
                    or _strip_benchmark_qualifier(metric_config.get("evaluation_description", ""))
                )
                if metric_raw and metric_raw.strip():
                    metrics[metric_raw.strip()] += 1
                    metric_configs[metric_raw.strip()].add(cfg)

    return {
        "benchmarks": benchmarks,
        "metrics": metrics,
        "harnesses": harnesses,
        "bench_configs": bench_configs,
        "metric_configs": metric_configs,
        "harness_configs": harness_configs,
    }


def print_frequency_table(data: dict) -> None:
    for entity_type in ["benchmarks", "metrics", "harnesses"]:
        counter = data[entity_type]
        config_key = {"benchmarks": "bench_configs", "metrics": "metric_configs", "harnesses": "harness_configs"}
        configs_map = data[config_key[entity_type]]

        print(f"\n{'='*60}")
        print(f"  {entity_type.upper()} ({len(counter)} distinct)")
        print(f"{'='*60}")
        for name, count in counter.most_common():
            n_configs = len(configs_map.get(name, set()))
            print(f"  {count:5d} occurrences  {n_configs:2d} configs  {name}")


def emit_yaml(data: dict) -> None:
    """Emit draft seed YAML for entities appearing in 2+ configs."""
    print("# --- DRAFT benchmarks (review before adding to seed/benchmarks.yaml) ---")
    for name, _count in data["benchmarks"].most_common():
        n_configs = len(data["bench_configs"].get(name, set()))
        if n_configs < 2:
            continue
        slug = _slugify(name)
        print(f"""
- id: {slug}
  display_name: "{name}"
  description: null  # TODO
  dataset_repo: null
  parent_benchmark_id: null
  tags: '[]'
  metadata: '{{}}'
  review_status: reviewed""")

    print("\n\n# --- DRAFT metrics (review before adding to seed/metrics.yaml) ---")
    for name, _count in data["metrics"].most_common():
        n_configs = len(data["metric_configs"].get(name, set()))
        if n_configs < 2:
            continue
        slug = _slugify(name)
        print(f"""
- id: {slug}
  display_name: "{name}"
  score_type: continuous
  lower_is_better: false
  min_score: null
  max_score: null
  metadata: '{{}}'
  review_status: reviewed""")

    print("\n\n# --- DRAFT harnesses (review before adding to seed/harnesses.yaml) ---")
    for name, _count in data["harnesses"].most_common():
        n_configs = len(data["harness_configs"].get(name, set()))
        if n_configs < 2:
            continue
        slug = _slugify(name)
        print(f"""
- id: {slug}
  display_name: "{name}"
  version: null
  fork_url: null
  metadata: '{{}}'
  review_status: reviewed""")


if __name__ == "__main__":
    data = scan_all_configs()
    print_frequency_table(data)
    if "--output-yaml" in sys.argv:
        print("\n\n")
        emit_yaml(data)
