"""Empty DataFrame schemas for each table."""
import pandas as pd


_SCHEMAS: dict[str, dict] = {
    "canonical_orgs": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "parent_org_id": pd.StringDtype(),
        "website": pd.StringDtype(),
        # Frontend-relative path (e.g. "/org-logos/huggingface.svg") or absolute
        # URL; the frontend serves the bytes. Null → monogram fallback.
        "logo_url": pd.StringDtype(),
        "hf_org": pd.StringDtype(),
        # `lab` (curated first-party) | `community` (multi-model HF org) |
        # `individual` | `unknown` (auto-created, undecided).
        "kind": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_models": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "developer": pd.StringDtype(),
        "org_id": pd.StringDtype(),
        "family": pd.StringDtype(),
        "architecture": pd.StringDtype(),
        "params_billions": "float64",
        # JSON-encoded list of parent edges:
        #   [{"id": "...", "relationship": "variant|finetune|quantized|merge|adapter",
        #     "axis": "size|mode|modality|domain|version|training_stage|tier"}]
        # `quantized` is a relationship, not a `variant` axis, matching
        # hub-stats's baseModels.relation. `axis` applies to `variant` only.
        # Multiple edges express merges or a variant that is also a finetune.
        "parents": pd.StringDtype(),
        # Identity-group root via identity-preserving `parents` edges; NULL
        # when self is the root. Derived by `derive_model_lineage_fields`.
        "model_group_id": pd.StringDtype(),
        # Family-release root: folds the release line, stopping at finetune/
        # merge and major-version boundaries. NULL when self is the root.
        "model_family_id": pd.StringDtype(),
        # Deepest non-`variant` ancestor; NULL when self is the origin.
        "lineage_origin_model_id": pd.StringDtype(),
        # `org_id` of the deepest non-`variant` ancestor; recomputed on every
        # refresh, treat as a cache.
        "lineage_origin_model_org_id": pd.StringDtype(),
        # Provenance enum {hf|models_dev|curated|inferred|none}; null by default.
        "resolution_source": pd.StringDtype(),
        # Enum {variant|group|family}; null by default.
        "resolution_granularity": pd.StringDtype(),
        # NULL when unknown. models.dev supplies it directly; hub-stats infers
        # True iff safetensors/gguf data is present; core.yaml may set it.
        "open_weights": pd.BooleanDtype(),
        # ISO date (YYYY-MM-DD or YYYY-MM) of the earliest known snapshot;
        # per-snapshot dates stay in metadata.release_dates.
        "release_date": pd.StringDtype(),
        # JSON-encoded lists (e.g. ["text", "image"]); NULL when unknown.
        "input_modalities": pd.StringDtype(),
        "output_modalities": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_benchmarks": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "description": pd.StringDtype(),
        "dataset_repo": pd.StringDtype(),
        "parent_benchmark_id": pd.StringDtype(),
        # Default metric for the merged benchmark view; null → consumer fallback.
        "preferred_metric_id": pd.StringDtype(),
        # Is `preferred_metric_id` LLM-judged? null = unstated; non-null only
        # when `preferred_metric_id` is set. Says nothing about other metrics.
        "preferred_metric_llm_judged": pd.BooleanDtype(),
        "tags": pd.StringDtype(),
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Curated groupings of benchmarks sharing an object of measurement
    # (e.g. mmlu + mmlu-pro). Singletons are absent: the producer defaults
    # family.id == benchmark.id.
    "canonical_families": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        # Single value from the curated enum; null defaults to "other" downstream.
        "category": pd.StringDtype(),
        # JSON-encoded list of member benchmark ids; seed validation allows a
        # benchmark in at most one family.
        "benchmark_ids": pd.StringDtype(),
        # Member benchmark whose primary metric is the family-level rollup;
        # null when none is designated.
        "primary_benchmark_key": pd.StringDtype(),
        # JSON-encoded list of EEE source_config values that resolve to this
        # family; the producer joins on it when classifying raw folders.
        "folder_aliases": pd.StringDtype(),
        # JSON-encoded list of composite slugs nested under this family.
        "composite_keys": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Named leaderboard surfaces aggregating one or more EEE source_configs
    # (e.g. HELM Classic, Open LLM Leaderboard v2). A one-to-one slug needs
    # no entry; a curated entry overrides display name or groups configs.
    "canonical_composites": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "category": pd.StringDtype(),
        # JSON-encoded list of member EEE source_config values.
        "source_configs": pd.StringDtype(),
        # Family this composite nests under; null when it is top-level.
        "family_id": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_metrics": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "score_type": pd.StringDtype(),
        "lower_is_better": "bool",
        "min_score": "float64",
        "max_score": "float64",
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Per-benchmark metric folds: `from_metric_id` on `benchmark_id` is the
    # same measurement as `to_metric_id`. Naming folds only; protocol variants
    # (cot-correct vs accuracy) never fold. `source_config` null = benchmark-
    # wide; set = a per-source scale conversion, which may keep the metric
    # unchanged when only the scale differs.
    "benchmark_metric_folds": {
        "benchmark_id": pd.StringDtype(),
        "from_metric_id": pd.StringDtype(),
        "to_metric_id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        # canonical = published * scale_factor + scale_offset (e.g. 1/9 and
        # -1/9 for a 1-10 score onto [0,1]). Scoped rows only; null = none.
        "scale_factor": "float64",
        "scale_offset": "float64",
        "note": pd.StringDtype(),
    },
    "eval_harnesses": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "version": pd.StringDtype(),
        "fork_url": pd.StringDtype(),
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Host-token spellings (`fireworks/`, `-bedrock`, …) live in `aliases`
    # and are inverted into the platform map by lib/inference_platforms_map.py.
    "canonical_inference_platforms": {
        "id": pd.StringDtype(),            # PK; = models.dev provider slug where one exists
        "display_name": pd.StringDtype(),
        # Closed enum: author_lab | inference_platform |
        # aggregator_gateway | regional_variant | coding_plan
        "kind": pd.StringDtype(),
        "aliases": pd.StringDtype(),       # JSON-encoded list of host-token spellings
        "canonical_org": pd.StringDtype(), # FK→canonical_orgs.id when kind=author_lab; nullable
        "variant_of": pd.StringDtype(),    # base platform id for regional_variant/coding_plan; nullable
        "homepage": pd.StringDtype(),      # provenance (models.dev / provider doc URL); nullable
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "aliases": {
        "id": pd.StringDtype(),
        "raw_value": pd.StringDtype(),
        "entity_type": pd.StringDtype(),
        "canonical_id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "source_field": pd.StringDtype(),
        "status": pd.StringDtype(),
        "strategy": pd.StringDtype(),
        "confidence": "float64",
        "notes": pd.StringDtype(),
        # FK→canonical_inference_platforms.id; serving platform of this spelling.
        "inference_platform": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "resolution_log": {
        "id": pd.StringDtype(),
        "sync_run_id": pd.StringDtype(),
        "raw_value": pd.StringDtype(),
        "entity_type": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "strategy": pd.StringDtype(),
        "confidence": "float64",
        "canonical_id": pd.StringDtype(),
        "created_new": "bool",
        "timestamp": pd.StringDtype(),
    },
    "eval_results": {
        "id": pd.StringDtype(),            # hash(evaluation_id + result_index)
        "evaluation_id": pd.StringDtype(), # from EEE record
        "result_index": "Int64",           # position in evaluation_results array
        "source_config": pd.StringDtype(),
        "model_id": pd.StringDtype(),      # resolved canonical ID
        "harness_id": pd.StringDtype(),    # resolved canonical ID (nullable)
        "benchmark_id": pd.StringDtype(),  # resolved canonical ID
        "parent_benchmark_id": pd.StringDtype(),  # nullable
        "metric_id": pd.StringDtype(),     # resolved canonical ID (nullable)
        "benchmark_card_id": pd.StringDtype(),  # FK to auto-benchmarkcard output (nullable)
        "score": "float64",
        "score_details": pd.StringDtype(), # JSON-encoded dict
        # FK→canonical_inference_platforms.id: matched provider spelling, else
        # a raw-id host token, else null.
        "inference_platform": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "sync_runs": {
        "id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "started_at": pd.StringDtype(),
        "completed_at": pd.StringDtype(),
        "status": pd.StringDtype(),
        "rerun": "bool",
        "entities_created": "Int64",
        "entities_updated": "Int64",
        "aliases_created": "Int64",
        "aliases_updated": "Int64",
        "errors": pd.StringDtype(),  # JSON-encoded list
    },
    # Cron-built index of HF model ids with downloadable weights. The resolve
    # path uses it to confirm an exact HF id never minted into the registry;
    # never written by seed/sync.
    "hub_stats_index": {
        # HF-true repo id (e.g. `meta-llama/Llama-3.1-8B`).
        "id": pd.StringDtype(),
        # services.hub_stats.normalize form, so separator variants hit.
        "id_norm": pd.StringDtype(),
        "release_date": pd.StringDtype(),
        "pipeline_tag": pd.StringDtype(),
        "params_billions": "float64",
        "downloads": "Int64",
        # Always True: the index is filtered to safetensors/gguf repos.
        "open_weights": pd.BooleanDtype(),
    },
}


def empty(table: str) -> pd.DataFrame:
    schema = _SCHEMAS[table]
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
