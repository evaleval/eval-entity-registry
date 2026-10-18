"""ResolutionService: hub-stats index priority, draft lifecycle, rerun,
leaf preservation through both resolve paths, and tier-3 inference."""
import json
import pytest

from eval_entity_registry.store.hf_store import RegistryStore
from eval_entity_registry.store import schemas
from eval_entity_registry.services.resolution_service import ResolutionService


def _fresh_store() -> RegistryStore:
    store = RegistryStore()
    from eval_entity_registry.store import schemas as s
    store._tables = {name: s.empty(name) for name in [
        "canonical_orgs",
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


def _seed_benchmark(store: RegistryStore, id: str, display_name: str):
    from eval_entity_registry.store import queries
    import json
    queries.upsert_entity(store, "canonical_benchmarks", {
        "id": id,
        "display_name": display_name,
        "description": None,
        "dataset_repo": None,
        "parent_benchmark_id": None,
        "tags": "[]",
        "metadata": "{}",
        "review_status": "reviewed",
    })
    queries.add_alias(store, {
        "raw_value": display_name,
        "entity_type": "benchmark",
        "canonical_id": id,
        "source_config": None,
        "source_field": None,
        "status": "confirmed",
        "strategy": "exact",
        "confidence": 1.0,
        "notes": None,
    })


class TestHubStatsIndexPriority:
    """The live hub-stats index registers canonical ids before aliases, as the
    generator does; otherwise a baseModels edge resolves to a different
    parent live than at generate time."""

    def test_canonical_id_wins_over_a_colliding_alias(self):
        import pandas as pd

        store = _fresh_store()
        # B claims A's id as an alias.
        store._tables["canonical_models"] = pd.DataFrame(
            [{"id": "deepseek-ai/DeepSeek-V3.1"}, {"id": "deepseek-ai/deepseek-v3.1-collapsed"}]
        )
        store._tables["aliases"] = pd.DataFrame(
            [{
                "entity_type": "model",
                "raw_value": "deepseek-ai/DeepSeek-V3.1",
                "canonical_id": "deepseek-ai/deepseek-v3.1-collapsed",
            }]
        )
        svc = ResolutionService(store)
        a2c, _ = svc._build_hub_stats_indices()
        from eval_entity_registry.services.hub_stats import normalize as _hsnorm

        assert a2c[_hsnorm("deepseek-ai/DeepSeek-V3.1")] == "deepseek-ai/DeepSeek-V3.1"

    @pytest.mark.parametrize(
        ("hf_org", "canonical_org"),
        [("CohereLabs", "cohere"), ("ibm-granite", "ibm")],
    )
    def test_live_hub_stats_index_uses_complete_curated_org_map(
        self, hf_org, canonical_org
    ):
        """Runtime enrichment and the bulk cron share one org map."""
        import pandas as pd

        store = _fresh_store()
        store._tables["canonical_orgs"] = pd.DataFrame(
            [
                {"id": "cohere", "hf_org": "CohereForAI"},
                {"id": "ibm", "hf_org": "ibm"},
            ]
        )
        svc = ResolutionService(store)

        _, org_map = svc._build_hub_stats_indices()

        assert org_map[hf_org.lower()] == canonical_org


class TestResolutionService:
    def test_resolves_known_entity(self):
        store = _fresh_store()
        _seed_benchmark(store, "ifeval", "IFEval")
        svc = ResolutionService(store)
        result = svc.resolve("IFEval", "benchmark", None, None)
        assert result["canonical_id"] == "ifeval"
        assert result["created_new"] is False

    def test_auto_creates_draft_on_no_match(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        result = svc.resolve("Some Unknown Benchmark", "benchmark", "test_config", None)
        assert result["canonical_id"] is not None
        assert result["created_new"] is True
        assert result["review_status"] == "draft"

    def test_idempotent_on_second_call(self):
        from eval_entity_registry.store import queries
        store = _fresh_store()
        svc = ResolutionService(store)
        r1 = svc.resolve("Novel Benchmark X", "benchmark", "cfg", None)
        r2 = svc.resolve("Novel Benchmark X", "benchmark", "cfg", None)
        assert r1["canonical_id"] == r2["canonical_id"]
        queries.flush_pending(store)
        df = store.table("canonical_benchmarks")
        assert len(df) == 1

    def test_rerun_bypasses_cache(self):
        store = _fresh_store()
        _seed_benchmark(store, "real-benchmark-y", "Novel Benchmark Y")
        svc = ResolutionService(store)

        r1 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None)
        assert r1["canonical_id"] == "real-benchmark-y"

        r2 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None, rerun=False)
        assert r2["canonical_id"] == "real-benchmark-y"
        assert r2["created_new"] is False

        svc.invalidate_resolver()
        r3 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None, rerun=True)
        assert r3["canonical_id"] == "real-benchmark-y"

    def test_subset_alias_resolves_to_parent(self):
        """A subset aliased to a parent resolves to the parent, not a new entity."""
        store = _fresh_store()
        _seed_benchmark(store, "parent-bench", "Parent Bench")
        from eval_entity_registry.store import queries
        queries.add_alias(store, {
            "raw_value": "Parent Bench Subset X",
            "entity_type": "benchmark",
            "canonical_id": "parent-bench",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        svc = ResolutionService(store)
        result = svc.resolve("Parent Bench Subset X", "benchmark", "some_config", None)
        assert result["canonical_id"] == "parent-bench"
        assert result["created_new"] is False

    def test_empty_raw_value_returns_none(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        result = svc.resolve("", "benchmark", "cfg", None)
        assert result["canonical_id"] is None
        assert result["strategy"] == "no_match"

    def test_resolve_preserves_resolved_leaf_id_for_version_chain(self):
        """On the existing-alias path, a leaf with a version-axis chain comes
        back as canonical_id with the group root in model_group_id; the
        leaf must not be clobbered by re-enrichment."""
        from eval_entity_registry.store import queries
        store = _fresh_store()
        queries.upsert_entity(store, "canonical_orgs", {
            "id": "allenai", "display_name": "Allen AI", "parent_org_id": None,
            "website": None, "hf_org": "allenai", "kind": "lab",
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-32b", "display_name": "OLMo-3 32B",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": "[]", "model_group_id": None,
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        # model_group_id set as derive_model_lineage_fields would after seed.
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-1125-32b", "display_name": "OLMo-3 32B (1125)",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": json.dumps([{
                "id": "allenai/olmo-3-32b",
                "relationship": "variant",
                "axis": "version",
            }]),
            "model_group_id": "allenai/olmo-3-32b",
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        queries.add_alias(store, {
            "raw_value": "allenai/Olmo-3-1125-32B",
            "entity_type": "model",
            "canonical_id": "allenai/olmo-3-1125-32b",
            "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "seed",
            "confidence": 1.0, "notes": None,
        })

        svc = ResolutionService(store)
        r1 = svc.resolve("allenai/Olmo-3-1125-32B", "model", None, None)
        assert r1["canonical_id"] == "allenai/olmo-3-1125-32b"
        assert r1["resolved_leaf_id"] == "allenai/olmo-3-1125-32b"
        assert r1["model_group_id"] == "allenai/olmo-3-32b"

    def test_resolve_preserves_leaf_via_fresh_resolve_path(self):
        """On the fresh-resolve path (no alias row for the exact-cased raw), a
        normalized match still returns the leaf, not the group root."""
        from eval_entity_registry.store import queries
        store = _fresh_store()
        queries.upsert_entity(store, "canonical_orgs", {
            "id": "allenai", "display_name": "Allen AI", "parent_org_id": None,
            "website": None, "hf_org": "allenai", "kind": "lab",
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-32b", "display_name": "OLMo-3 32B",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": "[]", "model_group_id": None,
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-1125-32b", "display_name": "OLMo-3 32B (1125)",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": json.dumps([{
                "id": "allenai/olmo-3-32b",
                "relationship": "variant",
                "axis": "version",
            }]),
            "model_group_id": "allenai/olmo-3-32b",
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        # Only the lowercase alias exists, so the mixed-case raw misses
        # get_alias and takes the fresh-resolve path.
        queries.add_alias(store, {
            "raw_value": "allenai/olmo-3-1125-32b",
            "entity_type": "model",
            "canonical_id": "allenai/olmo-3-1125-32b",
            "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "seed",
            "confidence": 1.0, "notes": None,
        })

        svc = ResolutionService(store)
        r = svc.resolve("allenai/Olmo-3-1125-32B", "model", None, None)
        assert r["canonical_id"] == "allenai/olmo-3-1125-32b"
        assert r["resolved_leaf_id"] == "allenai/olmo-3-1125-32b"
        assert r["model_group_id"] == "allenai/olmo-3-32b"
        assert r["strategy"] == "normalized"
        assert r["created_new"] is False


def _seed_org(store, org_id="meta"):
    from eval_entity_registry.store import queries
    queries.upsert_entity(store, "canonical_orgs", {
        "id": org_id, "display_name": org_id, "parent_org_id": None,
        "website": None, "hf_org": org_id, "kind": "lab",
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })


def _seed_model(store, mid, org_id, *, aliases=None):
    from eval_entity_registry.store import queries
    queries.upsert_entity(store, "canonical_models", {
        "id": mid, "display_name": mid,
        "developer": None, "org_id": org_id, "family": None,
        "architecture": None, "params_billions": None,
        "parents": "[]", "model_group_id": None,
        "lineage_origin_model_org_id": org_id, "open_weights": True,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        "resolution_source": "curated", "resolution_granularity": "variant",
    })
    for a in [mid] + list(aliases or []):
        queries.add_alias(store, {
            "raw_value": a, "entity_type": "model", "canonical_id": mid,
            "source_config": None, "source_field": None, "status": "confirmed",
            "strategy": "seed", "confidence": 1.0, "notes": None,
        })


class TestTier3Inference:
    def test_inferred_source_and_granularity_on_auto_create(self):
        """A non-HF-confirmed model draft is resolution_source=inferred,
        granularity=variant."""
        store = _fresh_store()
        _seed_org(store, "someorg")
        svc = ResolutionService(store)
        r = svc.resolve("someorg/Totally-Vanity-Name-3B", "model", None, None)
        assert r["created_new"] is True
        assert r["resolution_source"] == "inferred"
        assert r["resolution_granularity"] == "variant"
        assert r["review_status"] == "draft"

    def test_org_less_no_slash_flagged_org_unknown(self):
        """A bare label mints org-less with the org-unknown tag, never a guessed org."""
        store = _fresh_store()
        svc = ResolutionService(store)
        r = svc.resolve("Cohere May 2024", "model", None, None)
        assert r["created_new"] is True
        assert r["resolution_source"] == "inferred"
        from eval_entity_registry.services.resolution_service import _table_with_pending
        mid = r["canonical_id"]
        row = _table_with_pending(store, "canonical_models")
        row = row[row["id"] == mid].iloc[0]
        assert row["org_id"] is None or (isinstance(row["org_id"], float))
        assert "org-unknown" in json.loads(row["tags"])

    def test_unknown_placeholder_org_is_org_less(self):
        """The `unknown/` placeholder is org-less, not an org named 'unknown'."""
        store = _fresh_store()
        svc = ResolutionService(store)
        r = svc.resolve("unknown/iSWE_Agent", "model", None, None)
        from eval_entity_registry.services.resolution_service import _table_with_pending
        mid = r["canonical_id"]
        row = _table_with_pending(store, "canonical_models")
        row = row[row["id"] == mid].iloc[0]
        assert row["org_id"] is None or isinstance(row["org_id"], float)
        assert "org-unknown" in json.loads(row["tags"])

    def test_alias_confirmed_base_edge_emitted(self):
        """An alias-confirmed base yields one finetune edge."""
        store = _fresh_store()
        _seed_org(store, "meta")
        _seed_model(store, "meta/llama-3.1-8b", "meta",
                    aliases=["meta-llama/Llama-3.1-8B", "llama-3.1-8b"])
        svc = ResolutionService(store)
        r = svc.resolve("3rd-Degree-Burn/Llama-3.1-8B-Squareroot", "model", None, None)
        assert r["created_new"] is True
        parents = r["parents"] or []
        assert any(
            p["id"] == "meta/llama-3.1-8b" and p["relationship"] == "finetune"
            for p in parents
        ), parents

    def test_no_invented_edge_for_opaque_name(self):
        """No alias-confirmable base, no invented edge."""
        store = _fresh_store()
        _seed_org(store, "auraindustries")
        svc = ResolutionService(store)
        r = svc.resolve("AuraIndustries/Aura-MoE-2x4B", "model", None, None)
        assert r["created_new"] is True
        assert (r["parents"] or []) == []

    def test_no_self_edge_when_base_is_same_identity(self):
        """An org-less twin of the same identity is not a base: no self-edge."""
        store = _fresh_store()
        _seed_model(store, "yi-lightning", None)
        svc = ResolutionService(store)
        r = svc.resolve("01-ai/yi-lightning", "model", None, None)
        assert r["created_new"] is True
        assert (r["parents"] or []) == []
