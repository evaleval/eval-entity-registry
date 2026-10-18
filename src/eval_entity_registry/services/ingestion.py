"""EEE sync pipeline: per-record sub-category detection, resolution, and
eval_results rows. Upstream quirk: some configs put a metric keyword in
``evaluation_name`` for aggregate rows (``reward-bench`` "Score", ``ace``
"Overall", ``helm_capabilities`` "Mean score", ``helm_instruct`` "Mean win
rate"), which would mint bogus benchmark entities."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Iterator, Optional

from eval_entity_resolver.eee import clean_eval_name, extract_metric

from eval_entity_registry.config import settings
from eval_entity_registry.store.hf_store import RegistryStore
from eval_entity_registry.store import queries
from eval_entity_registry.services.resolution_service import ResolutionService


def _iter_eee_config(source_config: str) -> Iterator[dict]:
    """Load EEE datastore config and yield rows as dicts."""
    import datasets

    ds = datasets.load_dataset("evaleval/EEE_datastore", source_config)["train"]
    for i in range(len(ds)):
        yield dict(ds[i])


def _detect_sub_categories(
    evaluation_results: list[dict],
) -> dict[str, Optional[str]]:
    """{evaluation_name: parent dataset_name or None}. Several evaluation_names
    sharing one dataset_name is the sub-category pattern: the dataset is the
    parent benchmark, each evaluation_name a child."""
    dataset_to_eval_names: dict[str, list[str]] = defaultdict(list)
    for result in evaluation_results:
        eval_name = result.get("evaluation_name", "")
        dataset_name = (result.get("source_data") or {}).get("dataset_name", "")
        if eval_name and dataset_name:
            dataset_to_eval_names[dataset_name].append(eval_name)

    eval_to_parent: dict[str, Optional[str]] = {}
    for result in evaluation_results:
        eval_name = result.get("evaluation_name", "")
        dataset_name = (result.get("source_data") or {}).get("dataset_name", "")
        if not eval_name:
            continue
        if dataset_name and len(dataset_to_eval_names.get(dataset_name, [])) > 1:
            eval_to_parent[eval_name] = dataset_name
        else:
            eval_to_parent[eval_name] = None
    return eval_to_parent


def process_record(
    record: dict,
    source_config: str,
    svc: ResolutionService,
    sync_run_id: str,
    rerun: bool,
) -> Optional[list[dict]]:
    """Resolve all entities in one EEE record into result rows (one per
    evaluation result); None when the record has no model info."""
    # --- Model ---
    model_info = record.get("model_info") or {}
    model_raw = model_info.get("id") or model_info.get("model_name") or model_info.get("name")
    if not model_raw:
        return None

    model_res = svc.resolve(
        raw_value=model_raw,
        entity_type="model",
        source_config=source_config,
        source_field="model_info.id",
        sync_run_id=sync_run_id,
        rerun=rerun,
    )

    # --- Harness ---
    eval_lib = record.get("eval_library") or {}
    harness_raw = eval_lib.get("name") or eval_lib.get("library_name")
    harness_res = None
    if harness_raw:
        harness_res = svc.resolve(
            raw_value=harness_raw,
            entity_type="harness",
            source_config=source_config,
            source_field="eval_library.name",
            sync_run_id=sync_run_id,
            rerun=rerun,
        )

    # --- Benchmarks & Metrics ---
    evaluation_results = record.get("evaluation_results") or []
    if not isinstance(evaluation_results, list):
        evaluation_results = []

    eval_to_parent = _detect_sub_categories(evaluation_results)

    parent_cache: dict[str, str] = {}
    for dataset_name in set(v for v in eval_to_parent.values() if v is not None):
        parent_res = svc.resolve(
            raw_value=dataset_name,
            entity_type="benchmark",
            source_config=source_config,
            source_field="source_data.dataset_name",
            sync_run_id=sync_run_id,
            rerun=rerun,
        )
        parent_cache[dataset_name] = parent_res["canonical_id"]

    # evaluation_id keys eval_results rows, so the fallback for a missing one
    # must be deterministic.
    source_meta = record.get("source_metadata") or {}
    eval_id = record.get("evaluation_id") or source_meta.get("evaluation_id") or source_meta.get("id")
    if not eval_id:
        import hashlib
        fallback_key = f"{source_config}:{model_raw}"
        eval_id = f"{source_config}/auto-{hashlib.sha256(fallback_key.encode()).hexdigest()[:12]}"

    result_rows = []
    for idx, er in enumerate(evaluation_results):
        eval_name = er.get("evaluation_name")
        if not eval_name:
            continue

        parent_dataset = eval_to_parent.get(eval_name)
        parent_benchmark_id = parent_cache.get(parent_dataset) if parent_dataset else None

        bench_name = clean_eval_name(eval_name)
        bench_res = svc.resolve(
            raw_value=bench_name,
            entity_type="benchmark",
            source_config=source_config,
            source_field="evaluation_results[].evaluation_name",
            sync_run_id=sync_run_id,
            rerun=rerun,
        )

        # Human-readable metric_name first, then dot-notation metric_id, then
        # the verbose evaluation_description; extract_metric normalises all three.
        metric_config = er.get("metric_config") or {}
        metric_raw = extract_metric(
            metric_config.get("metric_name")
            or metric_config.get("metric_id")
            or metric_config.get("evaluation_description")
            or ""
        )
        metric_res = None
        if metric_raw:
            metric_res = svc.resolve(
                raw_value=metric_raw,
                entity_type="metric",
                source_config=source_config,
                source_field="metric_config",
                sync_run_id=sync_run_id,
                rerun=rerun,
            )

        # `is not None` checks keep valid 0 / 0.0 scores.
        score_details_raw = er.get("score_details") or er.get("details") or {}
        score = None
        if isinstance(score_details_raw, dict):
            score = score_details_raw.get("score")
        if score is None:
            for key in ("score", "value", "result"):
                val = er.get(key)
                if val is not None:
                    score = val
                    break

        result_rows.append(
            {
                "evaluation_id": eval_id,
                "result_index": idx,
                "source_config": source_config,
                "model_id": model_res["canonical_id"],
                "harness_id": harness_res["canonical_id"] if harness_res else None,
                "benchmark_id": bench_res["canonical_id"],
                "parent_benchmark_id": parent_benchmark_id,
                "metric_id": metric_res["canonical_id"] if metric_res else None,
                "benchmark_card_id": None,
                "score": score,
                "score_details": json.dumps(score_details_raw) if score_details_raw else None,
                # Serving platform captured from the model spelling (e.g. `-bedrock`).
                "inference_platform": model_res.get("inference_platform"),
            }
        )

    return result_rows


def run_sync(
    source_config: str,
    registry_store: RegistryStore,
    rerun: bool = False,
) -> dict:
    """Sync one EEE config and return a counts dict. Does not push to the Hub;
    the caller calls push_to_hub() once after all configs."""
    svc = ResolutionService(registry_store)

    # A prior sync may have crashed with these populated.
    queries._alias_index.clear()
    queries._pending_result_ids.clear()

    queries._rebuild_alias_index(registry_store)

    run_id = queries.start_sync_run(registry_store, source_config, rerun)

    counts = {
        "entities_created": 0,
        "entities_updated": 0,
        "aliases_created": 0,
        "aliases_updated": 0,
    }
    errors = []

    aliases_before = len(registry_store.table("aliases")) + len(queries._get_pending(registry_store, "aliases"))
    models_before = len(registry_store.table("canonical_models"))
    benchmarks_before = len(registry_store.table("canonical_benchmarks"))
    metrics_before = len(registry_store.table("canonical_metrics"))
    harnesses_before = len(registry_store.table("eval_harnesses"))

    record_count = 0
    try:
        for record in _iter_eee_config(source_config):
            try:
                result_rows = process_record(record, source_config, svc, run_id, rerun)
                if result_rows:
                    for row in result_rows:
                        queries.upsert_eval_result(registry_store, row)
            except Exception as e:
                errors.append(str(e))

            record_count += 1
            if record_count % 500 == 0:
                print(f"  [{source_config}] {record_count} records processed...", file=sys.stderr)
    finally:
        # Flush even on crash so successfully processed records survive.
        queries.flush_pending(registry_store)

        queries._alias_index.clear()
        queries._pending_result_ids.clear()

    print(f"  [{source_config}] {record_count} records total, flushed.", file=sys.stderr)

    entities_after = (
        len(registry_store.table("canonical_models"))
        + len(registry_store.table("canonical_benchmarks"))
        + len(registry_store.table("canonical_metrics"))
        + len(registry_store.table("eval_harnesses"))
    )
    entities_before = models_before + benchmarks_before + metrics_before + harnesses_before
    aliases_after = len(registry_store.table("aliases"))

    counts["entities_created"] = max(0, entities_after - entities_before)
    counts["aliases_created"] = max(0, aliases_after - aliases_before) if not rerun else 0
    counts["aliases_updated"] = max(0, aliases_after - aliases_before) if rerun else 0

    queries.finish_sync_run(registry_store, run_id, counts, errors)

    return counts
