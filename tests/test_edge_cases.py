"""Degenerate inputs, alias uniqueness, normalization, and API response consistency."""
import pytest
from fastapi.testclient import TestClient

from eval_entity_registry.main import app
from eval_entity_registry.store import hf_store, schemas, queries
from eval_entity_registry.services.resolution_service import ResolutionService, _slugify
from eval_entity_resolver.normalization import normalize


# ------------------------------------------------------------------
# Shared fixture
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    store = hf_store.RegistryStore()
    store._tables = {name: schemas.empty(name) for name in [
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    return store


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def store(fresh_store):
    return fresh_store


# ------------------------------------------------------------------
# Normalization edge cases
# ------------------------------------------------------------------

class TestNormalization:
    def test_empty_string(self):
        assert normalize("") == ""

    def test_whitespace_only(self):
        assert normalize("   ") == ""

    def test_all_punctuation(self):
        assert normalize("!!!") == ""

    def test_collapse_internal_spaces(self):
        assert normalize("math  level  5") == "math level 5"

    def test_punctuation_then_space_collapse(self):
        # Punctuation removal can leave adjacent spaces; they must collapse.
        assert normalize("hello!  world") == "hello world"

    def test_hyphen_collapsed_to_space(self):
        assert normalize("SWE-Bench") == "swe bench"

    def test_underscore_collapsed_to_space(self):
        assert normalize("mmlu_pro") == "mmlu pro"

    def test_hyphen_space_underscore_equivalent(self):
        assert normalize("lm-evaluation-harness") == normalize("lm evaluation harness")
        assert normalize("mmlu_pro") == normalize("MMLU-PRO")

    def test_slash_collapsed_like_other_separators(self):
        # '/' collapses like the other separators.
        assert normalize("meta-llama/Llama-3") == "meta llama llama 3"
        assert normalize("tau-bench-2/airline") == normalize("tau-bench-2_airline")

    def test_consistent_casing(self):
        assert normalize("MATH Level 5") == normalize("math level 5")

    def test_leading_trailing_stripped(self):
        assert normalize("  accuracy  ") == "accuracy"

    def test_unicode_passthrough(self):
        result = normalize("résumé")
        assert "résumé" in result


# ------------------------------------------------------------------
# _slugify edge cases
# ------------------------------------------------------------------

class TestSlugify:
    def test_all_punctuation_produces_auto_id(self):
        slug = _slugify("!!!")
        assert slug.startswith("auto-")
        assert len(slug) > 5

    def test_whitespace_only_produces_auto_id(self):
        slug = _slugify("   ")
        assert slug.startswith("auto-")

    def test_empty_produces_auto_id(self):
        slug = _slugify("")
        assert slug.startswith("auto-")

    def test_dashes_only_produces_auto_id(self):
        # No word chars survive the strip.
        slug = _slugify("---")
        assert slug.startswith("auto-")

    def test_normal_input(self):
        assert _slugify("MATH Level 5") == "math-level-5"

    def test_hf_model_id_preserved(self):
        slug = _slugify("meta-llama/Llama-3.1-8B")
        assert "/" in slug
        assert "meta-llama" in slug

    def test_two_degenerate_inputs_produce_different_ids(self):
        assert _slugify("!!!") != _slugify("!!!")


# ------------------------------------------------------------------
# Degenerate raw_value resolution
# ------------------------------------------------------------------

class TestDegenerateResolution:
    def test_all_punctuation_raw_value_gets_valid_id(self, store):
        svc = ResolutionService(store)
        result = svc.resolve("!!!", "benchmark", "cfg", None)
        assert result["canonical_id"] is not None
        assert result["canonical_id"] != ""
        assert result["canonical_id"].startswith("auto-")

    def test_whitespace_raw_value_returns_no_match(self, store):
        svc = ResolutionService(store)
        result = svc.resolve("   ", "benchmark", "cfg", None)
        assert result["canonical_id"] is None
        assert result["strategy"] == "no_match"

    def test_degenerate_input_idempotent(self, store):
        svc = ResolutionService(store)
        r1 = svc.resolve("!!!", "benchmark", "cfg", None)
        r2 = svc.resolve("!!!", "benchmark", "cfg", None)
        assert r1["canonical_id"] == r2["canonical_id"]
        queries.flush_pending(store)
        assert len(store.table("canonical_benchmarks")) == 1

    def test_very_long_raw_value(self, store):
        svc = ResolutionService(store)
        long_value = "A" * 500
        result = svc.resolve(long_value, "benchmark", "cfg", None)
        assert result["canonical_id"] is not None


# ------------------------------------------------------------------
# Alias uniqueness enforcement
# ------------------------------------------------------------------

class TestAliasUniqueness:
    def test_add_alias_twice_raises(self, store):
        alias_data = {
            "raw_value": "IFEval",
            "entity_type": "benchmark",
            "canonical_id": "ifeval",
            "source_config": None,
            "source_field": None,
            "status": "auto",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        }
        queries.add_alias(store, alias_data)
        assert len(store.table("aliases")) == 1

        with pytest.raises(ValueError, match="Alias already exists"):
            queries.add_alias(store, alias_data)

        assert len(store.table("aliases")) == 1

    def test_add_alias_first_succeeds(self, store):
        queries.add_alias(store, {
            "raw_value": "IFEval",
            "entity_type": "benchmark",
            "canonical_id": "ifeval",
            "source_config": None,
            "source_field": None,
            "status": "auto",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        assert len(store.table("aliases")) == 1

    def test_rejected_alias_allows_new_insert(self, store):
        """A rejected alias does not block a new alias for the same key."""
        queries.add_alias(store, {
            "raw_value": "IFEval",
            "entity_type": "benchmark",
            "canonical_id": "ifeval-wrong",
            "source_config": "cfg",
            "source_field": None,
            "status": "rejected",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        queries.add_alias(store, {
            "raw_value": "IFEval",
            "entity_type": "benchmark",
            "canonical_id": "ifeval",
            "source_config": "cfg",
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        assert len(store.table("aliases")) == 2

    def test_resolution_service_idempotent_no_duplicate(self, store):
        """Resolving the same value twice writes exactly one alias."""
        svc = ResolutionService(store)
        svc.resolve("IFEval", "benchmark", "cfg", None)
        svc.resolve("IFEval", "benchmark", "cfg", None)
        # Aliases are buffered during resolution.
        queries.flush_pending(store)
        assert len(store.table("aliases")) == 1

    def test_scoped_and_global_aliases_coexist(self, store):
        """A global and a config-scoped alias may share a raw_value."""
        queries.add_alias(store, {
            "raw_value": "MATH",
            "entity_type": "benchmark",
            "canonical_id": "math-global",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        queries.add_alias(store, {
            "raw_value": "MATH",
            "entity_type": "benchmark",
            "canonical_id": "math-helm",
            "source_config": "helm_lite",
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        assert len(store.table("aliases")) == 2

    def test_rebuilt_alias_index_preserves_global_aliases(self, store):
        """A global alias stays lookupable after its source_config reloads as pd.NA."""
        import pandas as pd

        queries.add_alias(store, {
            "raw_value": "MATH",
            "entity_type": "benchmark",
            "canonical_id": "math",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "seed",
            "confidence": 1.0,
            "notes": None,
        })
        aliases = store.table("aliases").copy()
        aliases["source_config"] = aliases["source_config"].astype("string")
        aliases.loc[0, "source_config"] = pd.NA
        store.set_table("aliases", aliases)

        queries._alias_index.clear()
        queries._rebuild_alias_index(store)

        alias = queries.get_alias(store, "MATH", "benchmark", None)
        assert alias is not None
        assert alias["canonical_id"] == "math"
        assert alias["source_config"] is None


# ------------------------------------------------------------------
# API response consistency (GET vs POST vs PATCH)
# ------------------------------------------------------------------

class TestApiResponseConsistency:
    def test_post_returns_decoded_tags(self, client):
        r = client.post("/api/v1/benchmarks", json={
            "id": "my-bench",
            "display_name": "My Bench",
            "tags": ["reasoning", "math"],
        })
        assert r.status_code == 201
        assert r.json()["tags"] == ["reasoning", "math"], "POST should return decoded list"

    def test_patch_returns_decoded_metadata(self, client):
        client.post("/api/v1/benchmarks", json={"id": "my-bench", "display_name": "My Bench"})
        r = client.patch("/api/v1/benchmarks/my-bench", json={"metadata": {"paper_url": "https://example.com"}})
        assert r.status_code == 200
        assert isinstance(r.json()["metadata"], dict), "PATCH should return decoded dict"
        assert r.json()["metadata"]["paper_url"] == "https://example.com"

    def test_get_list_returns_decoded_tags(self, client):
        client.post("/api/v1/models", json={"id": "org/model", "display_name": "Model", "tags": ["instruct"]})
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        assert r.json()[0]["tags"] == ["instruct"], "GET list should return decoded list"

    def test_post_get_round_trip_consistent(self, client):
        payload = {"id": "test-harness", "display_name": "Test", "metadata": {"key": "value"}}
        post_r = client.post("/api/v1/harnesses", json=payload)
        get_r = client.get("/api/v1/harnesses/test-harness")
        assert post_r.json()["metadata"] == get_r.json()["metadata"]

    def test_explicit_false_boolean_patch(self, client):
        """An explicit False in a PATCH persists; it is not treated as unset."""
        client.post("/api/v1/metrics", json={"id": "perplexity", "display_name": "PPL", "lower_is_better": True})
        r = client.patch("/api/v1/metrics/perplexity", json={"lower_is_better": False})
        assert r.status_code == 200
        assert r.json()["lower_is_better"] is False
        get_r = client.get("/api/v1/metrics/perplexity")
        assert get_r.json()["lower_is_better"] is False

    def test_empty_tags_default(self, client):
        """Tags default to [] rather than null."""
        client.post("/api/v1/benchmarks", json={"id": "no-tags-bench", "display_name": "No Tags"})
        r = client.get("/api/v1/benchmarks/no-tags-bench")
        assert r.status_code == 200
        assert r.json()["tags"] == []


# ------------------------------------------------------------------
# API — model IDs with slashes
# ------------------------------------------------------------------

class TestModelPathParam:
    def test_hf_model_id_with_slash(self, client):
        r = client.post("/api/v1/models", json={
            "id": "meta-llama/Llama-3.1-8B",
            "display_name": "Llama 3.1 8B",
        })
        assert r.status_code == 201

        r2 = client.get("/api/v1/models/meta-llama/Llama-3.1-8B")
        assert r2.status_code == 200
        assert r2.json()["id"] == "meta-llama/Llama-3.1-8B"

    def test_patch_hf_model_id_with_slash(self, client):
        client.post("/api/v1/models", json={"id": "org/model-v2", "display_name": "Model V2"})
        r = client.patch("/api/v1/models/org/model-v2", json={"developer": "Org"})
        assert r.status_code == 200
        assert r.json()["developer"] == "Org"


# ------------------------------------------------------------------
# Resolver — source_config scoping edge cases
# ------------------------------------------------------------------

class TestSourceConfigScoping:
    def test_same_raw_value_different_configs_normalizes_to_same_entity(self, store):
        """Identical strings from different configs map to one canonical (one
        entity, one alias per config) unless a scoped alias overrides."""
        svc = ResolutionService(store)
        r1 = svc.resolve("MATH", "benchmark", "helm_lite", None)
        r2 = svc.resolve("MATH", "benchmark", "hfopenllm_v2", None)
        assert r1["canonical_id"] == r2["canonical_id"]
        queries.flush_pending(store)
        assert len(store.table("aliases")) == 2
        assert len(store.table("canonical_benchmarks")) == 1

    def test_config_scoped_alias_overrides_global_normalization(self, store):
        """A config-scoped alias overrides the global match for its config only."""
        queries.add_alias(store, {
            "raw_value": "MATH",
            "entity_type": "benchmark",
            "canonical_id": "math",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        queries.add_alias(store, {
            "raw_value": "MATH",
            "entity_type": "benchmark",
            "canonical_id": "math-other-variant",
            "source_config": "other_config",
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        svc = ResolutionService(store)
        r1 = svc.resolve("MATH", "benchmark", "helm_lite", None)
        assert r1["canonical_id"] == "math"
        r2 = svc.resolve("MATH", "benchmark", "other_config", None)
        assert r2["canonical_id"] == "math-other-variant"

    def test_global_alias_resolves_across_configs(self, store):
        """A global alias (source_config=None) serves every config."""
        queries.add_alias(store, {
            "raw_value": "IFEval",
            "entity_type": "benchmark",
            "canonical_id": "ifeval",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        svc = ResolutionService(store)
        r1 = svc.resolve("IFEval", "benchmark", "hfopenllm_v2", None)
        r2 = svc.resolve("IFEval", "benchmark", "some_other_config", None)
        assert r1["canonical_id"] == "ifeval"
        assert r2["canonical_id"] == "ifeval"
