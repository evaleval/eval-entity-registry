"""`queries.derive_model_lineage_fields`: the post-seed walk over `parents`
that fills group/family/lineage columns, inherits open_weights, and derives
release_date from an id's date suffix."""
from __future__ import annotations

import json

import pandas as pd
import pytest

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


def _add_model(store, cid, org_id, parents, release_date=None):
    queries.upsert_entity(store, "canonical_models", {
        "id": cid, "display_name": cid, "developer": None,
        "org_id": org_id, "family": None, "architecture": None,
        "params_billions": None,
        "parents": json.dumps(parents) if parents else "[]",
        "model_group_id": None, "lineage_origin_model_org_id": None,
        "release_date": release_date,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })


def test_derive_handles_cycle_without_infinite_loop(fresh_store):
    """A `parents` cycle (data corruption) must terminate, not hang."""
    _add_model(fresh_store, "lab/a", "lab", [{"id": "lab/b", "relationship": "quantized"}])
    _add_model(fresh_store, "lab/b", "lab", [{"id": "lab/a", "relationship": "quantized"}])

    counts = queries.derive_model_lineage_fields(fresh_store)
    assert counts["group_set"] >= 0

    df = fresh_store.table("canonical_models")
    assert len(df) == 2


def test_derive_lineage_origin_walks_finetune_and_quantized(fresh_store):
    """Lineage origin walks every non-variant edge to the deepest ancestor;
    the group walk stops at a finetune edge."""
    _add_model(fresh_store, "meta/llama-3.1-70b", "meta", [])
    _add_model(fresh_store, "nous/hermes-3-llama-70b", "nous-research", [
        {"id": "meta/llama-3.1-70b", "relationship": "finetune"},
    ])
    _add_model(fresh_store, "nous/hermes-3-llama-70b-fp8", "nous-research", [
        {"id": "nous/hermes-3-llama-70b", "relationship": "quantized"},
    ])

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}

    # Origin: group and family are self (membership is total); the org id
    # self-falls-back but lineage_origin_model_id stays NULL.
    assert by_id["meta/llama-3.1-70b"]["lineage_origin_model_org_id"] == "meta"
    assert by_id["meta/llama-3.1-70b"]["model_group_id"] == "meta/llama-3.1-70b"
    assert by_id["meta/llama-3.1-70b"]["model_family_id"] == "meta/llama-3.1-70b"
    assert pd.isna(by_id["meta/llama-3.1-70b"]["lineage_origin_model_id"])

    # Finetune is not identity-preserving: group is self; lineage is Meta.
    assert by_id["nous/hermes-3-llama-70b"]["lineage_origin_model_org_id"] == "meta"
    assert by_id["nous/hermes-3-llama-70b"]["model_group_id"] == "nous/hermes-3-llama-70b"
    assert by_id["nous/hermes-3-llama-70b"]["model_family_id"] == "nous/hermes-3-llama-70b"
    assert by_id["nous/hermes-3-llama-70b"]["lineage_origin_model_id"] == "meta/llama-3.1-70b"

    # Quant of the finetune: group stops at the finetune edge; lineage walks both.
    assert by_id["nous/hermes-3-llama-70b-fp8"]["lineage_origin_model_org_id"] == "meta"
    assert by_id["nous/hermes-3-llama-70b-fp8"]["model_group_id"] == "nous/hermes-3-llama-70b"
    assert by_id["nous/hermes-3-llama-70b-fp8"]["lineage_origin_model_id"] == "meta/llama-3.1-70b"


def test_derive_lineage_origin_falls_back_to_self_org(fresh_store):
    """A true root's lineage_origin_model_org_id is its own org_id."""
    _add_model(fresh_store, "meta/llama-3", "meta", [])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "meta/llama-3"].iloc[0]
    assert row["lineage_origin_model_org_id"] == "meta"


def test_open_weights_inherits_from_parent_via_variant_edges(fresh_store):
    """A mode variant with no value of its own inherits open_weights from
    its base."""
    _add_model(fresh_store, "meta/llama-3-8b", "meta", [])
    _add_model(fresh_store, "meta/llama-3-8b-instruct", "meta", [
        {"id": "meta/llama-3-8b", "relationship": "variant", "axis": "mode"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "meta/llama-3-8b", "open_weights"] = True
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    # pandas BooleanDtype returns numpy booleans, so use `==` not `is`.
    assert df[df["id"] == "meta/llama-3-8b"].iloc[0]["open_weights"] == True
    assert df[df["id"] == "meta/llama-3-8b-instruct"].iloc[0]["open_weights"] == True


def test_open_weights_inherits_through_quantized_chain(fresh_store):
    """Inheritance walks variant and quantized edges alike."""
    _add_model(fresh_store, "meta/llama-3-8b", "meta", [])
    _add_model(fresh_store, "meta/llama-3-8b-instruct", "meta", [
        {"id": "meta/llama-3-8b", "relationship": "variant", "axis": "mode"},
    ])
    _add_model(fresh_store, "meta/llama-3-8b-instruct-fp8", "meta", [
        {"id": "meta/llama-3-8b-instruct", "relationship": "quantized"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "meta/llama-3-8b", "open_weights"] = True
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    assert df[df["id"] == "meta/llama-3-8b-instruct-fp8"].iloc[0]["open_weights"] == True


def test_open_weights_does_not_inherit_through_finetune(fresh_store):
    """A finetune is its own release; it never inherits open_weights."""
    _add_model(fresh_store, "meta/llama-3.1-70b", "meta", [])
    _add_model(fresh_store, "nous/hermes-3-llama-70b", "nous-research", [
        {"id": "meta/llama-3.1-70b", "relationship": "finetune"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "meta/llama-3.1-70b", "open_weights"] = True
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    val = df[df["id"] == "nous/hermes-3-llama-70b"].iloc[0]["open_weights"]
    assert val is None or queries._is_na(val), \
        "finetune must not auto-inherit open_weights from its base"


def test_open_weights_explicit_value_never_overwritten(fresh_store):
    """An explicit open_weights on the child always beats inheritance."""
    _add_model(fresh_store, "lab/parent-open", "lab", [])
    _add_model(fresh_store, "lab/child-explicitly-closed", "lab", [
        {"id": "lab/parent-open", "relationship": "variant", "axis": "mode"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "lab/parent-open", "open_weights"] = True
    df.loc[df["id"] == "lab/child-explicitly-closed", "open_weights"] = False
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    assert df[df["id"] == "lab/child-explicitly-closed"].iloc[0]["open_weights"] == False


def test_root_collapses_through_variant_version_edge(fresh_store):
    """A dated snapshot (`variant axis=version`) is the same API identity
    and collapses to the group root like a quantized edge."""
    _add_model(fresh_store, "xai/grok-4", "xai", [])
    _add_model(fresh_store, "xai/grok-4-0709", "xai", [
        {"id": "xai/grok-4", "relationship": "variant", "axis": "version"},
    ])
    _add_model(fresh_store, "xai/grok-4-0709-fp8", "xai", [
        {"id": "xai/grok-4-0709", "relationship": "quantized"},
    ])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}

    assert by_id["xai/grok-4"]["model_group_id"] == "xai/grok-4"
    assert by_id["xai/grok-4-0709"]["model_group_id"] == "xai/grok-4"
    assert by_id["xai/grok-4-0709-fp8"]["model_group_id"] == "xai/grok-4"


def test_root_does_not_collapse_through_non_version_variant(fresh_store):
    """Non-version variant axes keep their own group: `gpt-4o-mini` scores
    differently from `gpt-4o`."""
    _add_model(fresh_store, "openai/gpt-4o", "openai", [])
    _add_model(fresh_store, "openai/gpt-4o-mini", "openai", [
        {"id": "openai/gpt-4o", "relationship": "variant", "axis": "size"},
    ])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert by_id["openai/gpt-4o-mini"]["model_group_id"] == "openai/gpt-4o-mini"


def test_derive_variant_edges_do_not_set_lineage_origin_to_parent(fresh_store):
    """A variant edge folds into the family but not the group, and never
    counts toward lineage origin."""
    _add_model(fresh_store, "meta/llama-3", "meta", [])
    _add_model(fresh_store, "meta/llama-3-8b", "meta", [
        {"id": "meta/llama-3", "relationship": "variant", "axis": "size"},
    ])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert by_id["meta/llama-3-8b"]["lineage_origin_model_org_id"] == "meta"
    assert by_id["meta/llama-3-8b"]["model_group_id"] == "meta/llama-3-8b"
    assert by_id["meta/llama-3-8b"]["model_family_id"] == "meta/llama-3"
    assert pd.isna(by_id["meta/llama-3-8b"]["lineage_origin_model_id"])


# ---------------------------------------------------------------------------
# release_date derivation from the id's date suffix; explicit values win.
# ---------------------------------------------------------------------------


def test_release_date_derived_from_iso_full_suffix(fresh_store):
    """`<prefix>-YYYY-MM-DD` fills a NULL release_date."""
    _add_model(fresh_store, "openai/gpt-4.1-mini-2025-04-14", "openai", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "openai/gpt-4.1-mini-2025-04-14"].iloc[0]
    assert row["release_date"] == "2025-04-14"


def test_release_date_derived_from_packed_suffix(fresh_store):
    """`<prefix>-YYYYMMDD` parses."""
    _add_model(fresh_store, "anthropic/claude-haiku-4.5-20251001", "anthropic", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "anthropic/claude-haiku-4.5-20251001"].iloc[0]
    assert row["release_date"] == "2025-10-01"


def test_release_date_derived_from_month_only_suffix(fresh_store):
    """`<prefix>-YYYY-MM` parses with the day defaulted to 01."""
    _add_model(fresh_store, "openai/gpt-5-2025-08", "openai", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "openai/gpt-5-2025-08"].iloc[0]
    assert row["release_date"] == "2025-08-01"


def test_explicit_release_date_wins_over_derivation(fresh_store):
    """A curated release_date beats the id's date: a `-YYYYMMDD` snapshot
    tag can differ from the actual public release."""
    _add_model(
        fresh_store, "anthropic/claude-sonnet-4-20250514", "anthropic", [],
        release_date="2025-05-22",
    )
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "anthropic/claude-sonnet-4-20250514"].iloc[0]
    assert row["release_date"] == "2025-05-22"


def test_release_date_year_range_guard(fresh_store):
    """A 4-digit tail outside the year range is not a release date."""
    _add_model(fresh_store, "lab/model-1024", "lab", [], release_date=None)
    _add_model(fresh_store, "lab/model-9999-12-31", "lab", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert queries._is_na(by_id["lab/model-1024"]["release_date"])
    assert queries._is_na(by_id["lab/model-9999-12-31"]["release_date"])


def test_release_date_no_derivation_when_no_date_suffix(fresh_store):
    """No date suffix, no manufactured value."""
    _add_model(fresh_store, "openai/gpt-5", "openai", [], release_date=None)
    _add_model(fresh_store, "alibaba/qwen2-72b", "alibaba", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert queries._is_na(by_id["openai/gpt-5"]["release_date"])
    assert queries._is_na(by_id["alibaba/qwen2-72b"]["release_date"])


def test_release_date_invalid_month_or_day_rejected(fresh_store):
    """An out-of-range month or day yields no date, never an invalid one."""
    _add_model(fresh_store, "lab/model-2025-13-01", "lab", [], release_date=None)  # month 13
    _add_model(fresh_store, "lab/model-2025-02-32", "lab", [], release_date=None)  # day 32
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert queries._is_na(by_id["lab/model-2025-13-01"]["release_date"])
    assert queries._is_na(by_id["lab/model-2025-02-32"]["release_date"])
