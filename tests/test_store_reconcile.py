"""`hf_store._reconcile_schema`: the legacy `parent_model_id` -> `parents`
migration, missing-column NA-fill, and unknown-column drop."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from eval_entity_registry.store.hf_store import _reconcile_schema
from eval_entity_registry.store import schemas


def test_reconcile_legacy_parent_model_id_migrates_to_parents():
    """A legacy scalar `parent_model_id` becomes one `variant/size` edge in
    `parents` and the column is dropped; null/empty yields no edge."""
    legacy = pd.DataFrame({
        "id": ["meta/llama-3-8b", "meta/llama-3", "rooted/no-parent"],
        "display_name": ["Llama 3 8B", "Llama 3", "Rooted"],
        "parent_model_id": ["meta/llama-3", None, ""],
        "developer": [None, None, None],
        "org_id": ["meta", "meta", "rooted"],
        "family": [None, None, None],
        "architecture": [None, None, None],
        "params_billions": [8.0, None, None],
        "release_date": [None, None, None],
        "tags": ["[]", "[]", "[]"],
        "metadata": ["{}", "{}", "{}"],
        "review_status": ["reviewed", "reviewed", "reviewed"],
        "created_at": [None, None, None],
        "updated_at": [None, None, None],
    })
    out = _reconcile_schema("canonical_models", legacy)

    assert "parent_model_id" not in out.columns, "legacy column must be dropped"
    assert "parents" in out.columns, "new column must be present"

    by_id = {row["id"]: row for _, row in out.iterrows()}
    parents_8b = json.loads(by_id["meta/llama-3-8b"]["parents"])
    assert parents_8b == [{"id": "meta/llama-3", "relationship": "variant", "axis": "size"}]
    assert pd.isna(by_id["meta/llama-3"]["parents"])
    assert pd.isna(by_id["rooted/no-parent"]["parents"])


def test_reconcile_adds_missing_columns_with_pd_na():
    """Columns the loaded parquet predates are NA-filled at the schema dtype."""
    minimal = pd.DataFrame({
        "id": ["meta/llama-3-8b"],
        "display_name": ["Llama 3 8B"],
    })
    out = _reconcile_schema("canonical_models", minimal)

    expected_cols = set(schemas._SCHEMAS["canonical_models"].keys())
    assert set(out.columns) == expected_cols
    assert pd.isna(out.iloc[0]["model_group_id"])
    assert pd.isna(out.iloc[0]["lineage_origin_model_org_id"])


def test_reconcile_drops_columns_not_in_schema():
    """Columns not in the schema (and without a migration) are dropped."""
    stray = pd.DataFrame({
        "id": ["test-org"],
        "display_name": ["Test"],
        "some_legacy_field": ["junk"],
        "another_dropped_one": [42],
    })
    out = _reconcile_schema("canonical_orgs", stray)
    assert "some_legacy_field" not in out.columns
    assert "another_dropped_one" not in out.columns
    assert "kind" in out.columns
    assert pd.isna(out.iloc[0]["kind"])
