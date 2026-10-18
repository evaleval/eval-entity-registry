"""Live hub-stats lookup at draft creation (`_auto_create_entity`): HF-true
id minting, enrichment from a hub-stats row, graceful fallback, and the
family-version parent inference."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from eval_entity_registry.config import settings
from eval_entity_registry.services import hub_stats as _hs
from eval_entity_registry.services.resolution_service import ResolutionService
from eval_entity_registry.store import hf_store, queries, schemas


@pytest.fixture
def fresh_store():
    store = hf_store.RegistryStore()
    store._tables = {n: schemas.empty(n) for n in [
        "canonical_orgs", "canonical_models", "canonical_benchmarks",
        "canonical_metrics", "eval_harnesses", "aliases",
        "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


@pytest.fixture
def enable_lookup():
    settings.hub_stats_lookup_enabled = True
    yield
    settings.hub_stats_lookup_enabled = False


# ---------- _looks_like_hf_id heuristic ----------

@pytest.mark.parametrize("value,expected", [
    ("meta-llama/Llama-3.1-8B", True),
    ("meta-llama/Llama-3.1-8B-Instruct", True),
    ("Just-A-Model-Name", False),       # no slash
    ("multi/segment/path", False),      # multiple slashes — not a clean HF id
    ("/leading-slash", False),
    ("trailing-slash/", False),
    ("", False),
])
def test_looks_like_hf_id(value, expected):
    assert ResolutionService._looks_like_hf_id(value) is expected


# ---------- Lookup gated by config flag ----------

def test_lookup_disabled_returns_none(fresh_store):
    """With the flag off, the lookup is skipped entirely."""
    svc = ResolutionService(fresh_store)
    assert settings.hub_stats_lookup_enabled is False
    assert svc._lookup_hub_stats("meta-llama/Llama-3.1-8B") is None


# ---------- Auto-create with mocked hub-stats client ----------

def _seed_org(store, oid, hf_org=None):
    queries.upsert_entity(store, "canonical_orgs", {
        "id": oid, "display_name": oid, "parent_org_id": None,
        "website": None, "hf_org": hf_org, "kind": "lab",
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })
    queries.add_alias(store, {
        "raw_value": oid, "entity_type": "org", "canonical_id": oid,
        "source_config": None, "source_field": "test",
        "status": "confirmed", "strategy": "seed",
        "confidence": 1.0, "notes": None,
    })


def _seed_model(store, mid, org_id):
    queries.upsert_entity(store, "canonical_models", {
        "id": mid, "display_name": mid, "developer": None,
        "org_id": org_id, "family": None, "architecture": None,
        "params_billions": None, "parents": "[]",
        "model_group_id": None, "lineage_origin_model_org_id": org_id,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })
    queries.add_alias(store, {
        "raw_value": mid, "entity_type": "model", "canonical_id": mid,
        "source_config": None, "source_field": "test",
        "status": "confirmed", "strategy": "seed",
        "confidence": 1.0, "notes": None,
    })


def test_auto_create_enriches_with_hub_stats_metadata(fresh_store, enable_lookup):
    """A hub-stats row pre-populates release_date, params_billions, parents
    and lineage_origin_model_org_id on the draft."""
    _seed_org(fresh_store, "meta", hf_org="meta-llama")
    _seed_model(fresh_store, "meta/llama-3.1-8b", "meta")

    fake_row = {
        "id": "casperhansen/llama-3.1-8b-instruct-awq",
        "author": "casperhansen",
        "createdAt": datetime(2024, 12, 6),
        "tags": ["safetensors", "license:llama3.1"],
        "cardData": {"license": "llama3.1"},
        "safetensors": {"total": 8_000_000_000},
        "baseModels": {
            "relation": "quantized",
            "models": [{"id": "meta-llama/Llama-3.1-8B"}],
        },
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloads": 100, "downloadsAllTime": 5000,
        "likes": 10, "trendingScore": 0, "lastModified": None,
    }
    # baseModels resolution goes through the alias index.
    queries.add_alias(fresh_store, {
        "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model",
        "canonical_id": "meta/llama-3.1-8b", "source_config": None,
        "source_field": "test", "status": "confirmed",
        "strategy": "seed", "confidence": 1.0, "notes": None,
    })

    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=fake_row):
        cid = svc._auto_create_entity("model", "casperhansen/Llama-3.1-8B-Instruct-AWQ")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]

    assert new_row["release_date"] == "2024-12-06"
    assert new_row["params_billions"] == 8.0
    parents = json.loads(new_row["parents"])
    assert parents == [{"id": "meta/llama-3.1-8b", "relationship": "quantized"}]
    assert new_row["lineage_origin_model_org_id"] == "meta"
    assert new_row["model_group_id"] is None or queries._is_na(new_row["model_group_id"])
    md = json.loads(new_row["metadata"])
    assert md["source"] == "hub_stats"
    assert md["license"] == "llama3.1"


def test_auto_create_mints_hf_cased_id_and_marks_source_hf(fresh_store, enable_lookup):
    """When hub-stats confirms the repo, the canonical id and display_name
    take the HF-true casing (the org is remapped in org_id only), with
    resolution_source='hf' and review_status='reviewed'."""
    _seed_org(fresh_store, "meta", hf_org="meta-llama")

    fake_row = {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "author": "meta-llama",
        "createdAt": datetime(2024, 7, 23),
        "tags": ["safetensors"],
        "cardData": {"license": "llama3.1"},
        "safetensors": {"total": 16_000_000_000},
        "baseModels": None,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloads": 100, "downloadsAllTime": 5000,
        "likes": 10, "trendingScore": 0, "lastModified": None,
    }

    svc = ResolutionService(fresh_store)
    # The raw id arrives lowercased; the HF-true id drives the casing.
    with patch.object(_hs.HubStatsClient, "lookup", return_value=fake_row):
        cid = svc._auto_create_entity("model", "meta-llama/llama-3.1-8b-instruct")

    assert cid == "meta-llama/Llama-3.1-8B-Instruct"

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    assert new_row["display_name"] == "Llama-3.1-8B-Instruct"
    assert new_row["org_id"] == "meta"
    assert new_row["resolution_source"] == "hf"
    assert new_row["review_status"] == "reviewed"


def test_auto_create_falls_back_when_lookup_returns_none(fresh_store, enable_lookup):
    """No hub-stats row: the draft lands with plain defaults."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=None):
        cid = svc._auto_create_entity("model", "totally-new-author/Some-Model-7B")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    assert new_row["params_billions"] is None or queries._is_na(new_row["params_billions"])
    assert new_row["release_date"] is None or queries._is_na(new_row["release_date"])
    assert new_row["parents"] == "[]"
    assert new_row["review_status"] == "draft"


def test_auto_create_falls_back_when_lookup_raises(fresh_store, enable_lookup):
    """A lookup exception never blocks draft creation."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup",
                      side_effect=RuntimeError("network unreachable")):
        cid = svc._auto_create_entity("model", "some-org/some-model")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    assert new_row["review_status"] == "draft"
    assert new_row["parents"] == "[]"


def test_auto_create_infers_family_parent_when_hub_stats_misses(fresh_store, enable_lookup):
    """With hub-stats empty, a snapshot id that strips to an existing
    canonical still gets a version-axis parent edge; otherwise same-day
    releases orphan until the next hub-stats refresh."""
    _seed_org(fresh_store, "allenai")
    _seed_model(fresh_store, "allenai/olmo-3-32b", "allenai")
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=None):
        cid = svc._auto_create_entity("model", "allenai/Olmo-3-1125-32B")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    parents = json.loads(new_row["parents"])
    assert {
        "id": "allenai/olmo-3-32b",
        "relationship": "variant",
        "axis": "version",
    } in parents
    assert new_row["review_status"] == "draft"


def test_auto_create_handles_anthropic_yyyymmdd_snapshot(fresh_store, enable_lookup):
    """An Anthropic YYYYMMDD snapshot with the family aliased and hub-stats
    empty still gets a version-axis parent edge."""
    _seed_org(fresh_store, "anthropic")
    _seed_model(fresh_store, "anthropic/claude-opus-4-5", "anthropic")
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=None):
        cid = svc._auto_create_entity("model", "anthropic/claude-opus-4-5-20251101")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    parents = json.loads(new_row["parents"])
    assert {
        "id": "anthropic/claude-opus-4-5",
        "relationship": "variant",
        "axis": "version",
    } in parents


def test_auto_create_skips_lookup_for_non_hf_shaped_value(fresh_store, enable_lookup):
    """Non-HF-shaped values never trigger a lookup."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup") as mock_lookup:
        svc._auto_create_entity("model", "Just-A-Model-Name")
        svc._auto_create_entity("model", "multi/path/segments")
    mock_lookup.assert_not_called()


def test_auto_create_skips_lookup_for_non_model_entity_types(fresh_store, enable_lookup):
    """The lookup is model-only, even for an HF-shaped benchmark value."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup") as mock_lookup:
        svc._auto_create_entity("benchmark", "some/weird-benchmark-name")
    mock_lookup.assert_not_called()


# ---------- enrich_draft_from_row helper ----------

def test_enrich_draft_resolves_known_base_to_canonical():
    """A baseModels HF id resolves to our canonical via the alias index."""
    aliases_to_canonical = {
        "meta-llama-llama-3-1-8b": "meta/llama-3.1-8b",
    }
    org_alias_map = {"meta-llama": "meta", "meta": "meta"}
    row = {
        "id": "casperhansen/llama-3.1-8b-fp8",
        "createdAt": None,
        "tags": [],
        "cardData": None,
        "safetensors": None,
        "baseModels": {
            "relation": "quantized",
            "models": [{"id": "meta-llama/Llama-3.1-8B"}],
        },
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row, aliases_to_canonical, org_alias_map)
    parents = json.loads(out["parents"])
    assert parents == [{"id": "meta/llama-3.1-8b", "relationship": "quantized"}]
    # quantized is a non-variant edge, so lineage origin is set.
    assert out["lineage_origin_model_org_id"] == "meta"


def test_enrich_draft_sets_open_weights_for_hf_artifact_rows():
    """safetensors/gguf present means open_weights=True; absent means the
    field is omitted (NULL), never a closed-weight claim."""
    row_with = {
        "id": "x/y", "createdAt": None, "tags": [], "cardData": None,
        "safetensors": {"total": 14_000_000_000},
        "baseModels": None,
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row_with, {}, {})
    assert out.get("open_weights") is True

    row_without = {**row_with, "safetensors": None}
    out2 = _hs.enrich_draft_from_row(row_without, {}, {})
    assert "open_weights" not in out2


def test_enrich_draft_drops_unresolved_base_edges():
    """An untracked base drops the edge rather than dangling; lineage
    origin stays empty."""
    row = {
        "id": "x/y",
        "createdAt": None, "tags": [], "cardData": None,
        "safetensors": None,
        "baseModels": {
            "relation": "finetune",
            "models": [{"id": "unknown-org/totally-unknown"}],
        },
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row, {}, {})
    assert "parents" not in out
    assert "lineage_origin_model_org_id" not in out


# ---------- Family-version parent inference ----------

def test_infer_family_parent_internal_yymm():
    """An internal MMDD token (`Olmo-3-1125-32B`) strips to the aliased
    family with a version-axis edge."""
    aliases = {"allenai-olmo-3-32b": "allenai/olmo-3-32b"}
    edge = _hs.infer_family_parent_edge("allenai/Olmo-3-1125-32B", aliases)
    assert edge == {
        "id": "allenai/olmo-3-32b",
        "relationship": "variant",
        "axis": "version",
    }


def test_infer_family_parent_internal_yymm_with_mode_suffix():
    """The strip keeps the mode suffix, so an `-Instruct` snapshot points
    at the instruct family pointer."""
    aliases = {"allenai-olmo-3-7b-instruct": "allenai/olmo-3-7b-instruct"}
    edge = _hs.infer_family_parent_edge(
        "allenai/Olmo-3-1125-7B-Instruct", aliases
    )
    assert edge["id"] == "allenai/olmo-3-7b-instruct"
    assert edge["axis"] == "version"


def test_infer_family_parent_trailing_mmdd():
    """Trailing MMDD (`kimi-k2-0905`)."""
    aliases = {"moonshotai-kimi-k2": "moonshotai/kimi-k2"}
    edge = _hs.infer_family_parent_edge("moonshotai/kimi-k2-0905", aliases)
    assert edge["id"] == "moonshotai/kimi-k2"


def test_infer_family_parent_trailing_yyyymmdd():
    """Trailing YYYYMMDD (`claude-haiku-4-5-20251001`)."""
    aliases = {"anthropic-claude-haiku-4-5": "anthropic/claude-haiku-4-5"}
    edge = _hs.infer_family_parent_edge(
        "anthropic/claude-haiku-4-5-20251001", aliases
    )
    assert edge["id"] == "anthropic/claude-haiku-4-5"


def test_infer_family_parent_trailing_yyyymm():
    """Trailing YYYYMM (`step-2-16k-202411`)."""
    aliases = {"stepfun-step-2-16k": "stepfun/step-2-16k"}
    edge = _hs.infer_family_parent_edge("stepfun/step-2-16k-202411", aliases)
    assert edge["id"] == "stepfun/step-2-16k"


def test_infer_family_parent_trailing_yyyymm_rejects_invalid_month():
    """An invalid month (`-202413`) is not a date and must not strip."""
    aliases = {"foo-bar": "foo/bar"}
    assert _hs.infer_family_parent_edge("foo/bar-202413", aliases) is None


def test_infer_family_parent_iso_full_date():
    """The ISO ladder (`gpt-5-2025-08-07`) reaches the bare family when only
    the family is aliased."""
    aliases = {"openai-gpt-5": "openai/gpt-5"}
    edge = _hs.infer_family_parent_edge("openai/gpt-5-2025-08-07", aliases)
    assert edge["id"] == "openai/gpt-5"


def test_infer_family_parent_iso_full_date_prefers_intermediate_snapshot():
    """An aliased intermediate snapshot (`gpt-5-2025-08`) is preferred over
    the family root, preserving snapshot-of-snapshot lineage."""
    aliases = {
        "openai-gpt-5": "openai/gpt-5",
        "openai-gpt-5-2025-08": "openai/gpt-5-2025-08",
    }
    edge = _hs.infer_family_parent_edge("openai/gpt-5-2025-08-07", aliases)
    assert edge["id"] == "openai/gpt-5-2025-08"


def test_infer_family_parent_iso_year_only():
    """A bare trailing year strips to the family."""
    aliases = {"openai-gpt-5": "openai/gpt-5"}
    edge = _hs.infer_family_parent_edge("openai/gpt-5-2024", aliases)
    assert edge["id"] == "openai/gpt-5"


def test_infer_family_parent_user_examples():
    """MMDD, YYYYMM and YYYYMMDD shapes each get a version-axis edge to the
    aliased family."""
    aliases = {
        "google-gemini-exp": "google/gemini-exp",
        "stepfun-step-2-16k": "stepfun/step-2-16k",
        "tencent-hunyuan-turbos": "tencent/hunyuan-turbos",
    }
    cases = [
        ("google/gemini-exp-1114", "google/gemini-exp"),
        ("stepfun/step-2-16k-202411", "stepfun/step-2-16k"),
        ("tencent/hunyuan-turbos-20250313", "tencent/hunyuan-turbos"),
    ]
    for hf_id, expected_family in cases:
        edge = _hs.infer_family_parent_edge(hf_id, aliases)
        assert edge is not None, f"no edge inferred for {hf_id!r}"
        assert edge["id"] == expected_family
        assert edge["axis"] == "version"


def test_infer_family_parent_returns_none_when_no_match():
    """No alias hit means no edge; a 4-digit token only strips when it is a
    valid MMDD."""
    aliases = {"meta-llama-3-1-8b": "meta/llama-3.1-8b"}
    assert _hs.infer_family_parent_edge("foo/bar-1125-7b", aliases) is None
    assert _hs.infer_family_parent_edge("foo/bar-8000-7b", aliases) is None


def test_infer_family_parent_ignores_non_date_4digit_token():
    """A non-MMDD 4-digit token (context length, version) is not stripped
    even when the stripped form would alias something."""
    aliases = {"foo-bar-7b": "foo/bar-7b"}
    assert _hs.infer_family_parent_edge("foo/bar-1399-7b", aliases) is None
    assert _hs.infer_family_parent_edge("foo/bar-0099-7b", aliases) is None


def test_enrich_draft_adds_family_version_edge_when_basemodels_misses_it():
    """The snapshot-to-family-pointer edge is registry-only (not on HF), so
    inference adds it on top of whatever baseModels provides."""
    aliases_to_canonical = {
        "allenai-olmo-3-32b": "allenai/olmo-3-32b",
    }
    row = {
        "id": "allenai/Olmo-3-1125-32B",
        "createdAt": "2025-11-25T00:00:00",
        "tags": [],
        "cardData": None,
        "safetensors": {"total": 64_000_000_000},
        "baseModels": None,
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row, aliases_to_canonical, {})
    parents = json.loads(out["parents"])
    assert {
        "id": "allenai/olmo-3-32b",
        "relationship": "variant",
        "axis": "version",
    } in parents
    assert out["release_date"] == "2025-11-25"


def test_infer_family_parent_suppresses_self_edge():
    """When the row merges INTO the family pointer, inference must skip or
    the pointer gains a self-edge and the lineage walker breaks."""
    aliases = {"allenai-olmo-3-32b": "allenai/olmo-3-32b"}
    edge = _hs.infer_family_parent_edge(
        "allenai/Olmo-3-1125-32B",
        aliases,
        target_canonical="allenai/olmo-3-32b",
    )
    assert edge is None


def test_enrich_draft_suppresses_self_edge_via_target_canonical():
    """With target_canonical set, enrich_draft_from_row writes no self-edge."""
    aliases_to_canonical = {"allenai-olmo-3-32b": "allenai/olmo-3-32b"}
    row = {
        "id": "allenai/Olmo-3-1125-32B",
        "createdAt": "2025-11-25T00:00:00",
        "tags": [], "cardData": None, "safetensors": None,
        "baseModels": None,
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(
        row, aliases_to_canonical, {},
        target_canonical="allenai/olmo-3-32b",
    )
    assert "parents" not in out
    assert out.get("release_date") == "2025-11-25"


def test_enrich_draft_does_not_double_add_version_edge():
    """Inference adds at most one version-axis edge."""
    aliases_to_canonical = {
        "allenai-olmo-3-32b": "allenai/olmo-3-32b",
    }
    # Synthetic: HF does not emit `variant`; only the shape matters.
    row = {
        "id": "allenai/Olmo-3-1125-32B",
        "createdAt": None, "tags": [], "cardData": None, "safetensors": None,
        "baseModels": {
            "relation": "variant",
            "models": [{"id": "allenai/Olmo-3-32B"}],
        },
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    # The extracted `variant` edge carries no axis, so inference still adds
    # one version-axis edge; the guard is the alias-existence check.
    out = _hs.enrich_draft_from_row(row, aliases_to_canonical, {})
    parents = json.loads(out["parents"])
    version_edges = [
        p for p in parents
        if p.get("relationship") == "variant" and p.get("axis") == "version"
    ]
    assert len(version_edges) == 1


# ---------- HubStatsClient cache ----------

def test_hub_stats_client_caches_lookups():
    """A repeated lookup is served from cache, not the backend."""
    client = _hs.HubStatsClient()
    fake = {"id": "x/y", "createdAt": None}
    call_count = 0

    def _fake_ensure_con():
        nonlocal call_count
        call_count += 1

        class _C:
            def execute(self, sql):
                class _Cur:
                    description = [("id",), ("createdAt",)]
                    def fetchone(self):
                        return ("x/y", None)
                return _Cur()
        return _C()

    with patch.object(client, "_ensure_con", side_effect=_fake_ensure_con):
        r1 = client.lookup("x/y")
        r2 = client.lookup("x/y")
    assert r1 == r2
    assert call_count == 1, "second lookup must hit cache, not backend"
