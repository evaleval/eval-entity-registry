"""The frozen hub-stats parquet (curation/hub_stats_frozen.parquet) is
committed, carries `baseModels` lineage, is the generator's default source,
and reproduces a parent edge offline. No model ids are hardcoded, so a cache
refresh does not break these.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from conftest import load_script_module

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "curation" / "hub_stats_frozen.parquet"


@pytest.fixture(scope="module")
def mod():
    return load_script_module("refresh_from_hub_stats")


def test_frozen_cache_is_committed_and_nonempty():
    assert CACHE.exists(), f"durable offline cache missing at {CACHE}"
    con = duckdb.connect()
    n = con.execute(f"SELECT count(*) FROM read_parquet('{CACHE}')").fetchone()[0]
    assert n >= 3000, f"cache unexpectedly small ({n} rows) — re-freeze it"


def test_cache_carries_basemodels_lineage():
    """`baseModels` must be frozen, not just scalar enrichment, or parents
    cannot reproduce offline."""
    con = duckdb.connect()
    with_bm = con.execute(
        f"SELECT count(*) FROM read_parquet('{CACHE}') "
        f"WHERE baseModels.models IS NOT NULL AND len(baseModels.models) > 0"
    ).fetchone()[0]
    assert with_bm >= 1000, (
        f"only {with_bm} rows carry baseModels lineage — the cache is missing the "
        f"lineage column needed for reproducible parents"
    )


def test_generator_defaults_to_the_offline_cache(mod):
    assert mod.FROZEN_CACHE_PATH == CACHE
    assert mod.FROZEN_CACHE_PATH.exists()


def test_enrichment_reproduces_a_basemodels_parent_offline(mod):
    """enrich_draft_from_row on a real frozen row yields its baseModels parent
    edge; a minimal alias map keeps this independent of core.yaml."""
    import json

    from eval_entity_registry.services.hub_stats import (
        QUERY_COLUMNS,
        enrich_draft_from_row,
        normalize as _nz,
    )

    con = duckdb.connect()
    # One base so the expected edge is unambiguous. Read as query_hub_stats
    # does (fetchall + dict(zip)): numpy arrays would break tag filtering.
    cur = con.execute(
        f"SELECT {QUERY_COLUMNS} FROM read_parquet('{CACHE}') "
        f"WHERE baseModels.relation = 'finetune' AND len(baseModels.models) = 1 "
        f"ORDER BY id LIMIT 1"
    )
    cols = [d[0] for d in cur.description]
    row = dict(zip(cols, cur.fetchall()[0]))

    base_id = row["baseModels"]["models"][0]["id"]
    child_id = row["id"]
    aliases_to_canonical = {_nz(base_id): base_id, _nz(child_id): child_id}
    enrichment = enrich_draft_from_row(
        row, aliases_to_canonical, {}, target_canonical=child_id
    )
    assert "parents" in enrichment, f"no parents derived for {child_id}"
    parents = json.loads(enrichment["parents"])
    assert any(p.get("id") == base_id for p in parents), (
        f"expected a parent edge to {base_id} for {child_id}, got {parents}"
    )
