"""Read-only HF id confirmation via the hub_stats_index: an unminted real HF
id confirms (`resolution_source=hub_stats_index`, nothing persisted), a
registry hit takes precedence, and a missing/empty index degrades to no_match.
Write-mode minting lives in test_model_id_policy.py. The index is synthetic.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eval_entity_registry.config import settings
from eval_entity_registry.main import app
from eval_entity_registry.store import hf_store, schemas as s
from eval_entity_registry.services.resolution_service import ResolutionService
from eval_entity_registry.services.log_writer import ResolveLogWriter
from eval_entity_registry.services.hub_stats import normalize


def _row(table: str, **vals) -> dict:
    base = {col: None for col in s._SCHEMAS[table]}
    base.update({"review_status": "reviewed",
                 "created_at": "2026-01-01T00:00:00+00:00",
                 "updated_at": "2026-01-01T00:00:00+00:00"})
    for k, v in vals.items():
        base[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return base


def _alias(raw, etype, cid):
    return {"id": f"{etype}:{raw}", "raw_value": raw, "entity_type": etype,
            "canonical_id": cid, "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "exact", "confidence": 1.0,
            "notes": None, "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00"}


def _index_row(hf_id: str, *, with_norm: bool = True) -> dict:
    base = {col: None for col in s._SCHEMAS["hub_stats_index"]}
    base.update({
        "id": hf_id,
        "id_norm": normalize(hf_id) if with_norm else None,
        "release_date": "2024-07-23",
        "pipeline_tag": "text-generation",
        "params_billions": 8.03,
        "downloads": 12345,
        "open_weights": True,
    })
    return base


_ALL_TABLES = [
    "canonical_models", "canonical_benchmarks", "canonical_metrics",
    "eval_harnesses", "canonical_orgs", "canonical_families",
    "canonical_composites", "aliases", "resolution_log", "eval_results",
    "sync_runs",
]


def _base_store(monkeypatch, *, index_rows=None, include_index=True):
    """One HF-backed registry model plus an optional index of ids that are
    NOT in the registry."""
    store = hf_store.RegistryStore()
    tables = {name: s.empty(name) for name in _ALL_TABLES}

    tables["canonical_models"] = pd.DataFrame([
        _row("canonical_models", id="meta-llama/Llama-3.1-8B",
             display_name="Llama 3.1 8B", org_id="meta",
             resolution_source="hf", parents="[]"),
    ])
    tables["aliases"] = pd.DataFrame([
        _alias("meta-llama/Llama-3.1-8B", "model", "meta-llama/Llama-3.1-8B"),
    ])

    if include_index:
        rows = index_rows if index_rows is not None else [
            _index_row("Qwen/Qwen2.5-7B-Instruct"),
            _index_row("mistralai/Mistral-Nemo-Instruct-2407"),
        ]
        tables["hub_stats_index"] = (
            pd.DataFrame(rows) if rows else s.empty("hub_stats_index"))

    store._tables = tables
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


@pytest.fixture
def read_only(monkeypatch):
    original = settings.read_only
    settings.read_only = True
    yield
    settings.read_only = original


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


# --- index-only HF id confirms ---------------------------------------------

def test_index_only_id_confirms(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"})
    assert r.status_code == 200
    d = r.json()
    assert d["canonical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["resolution_source"] == "hub_stats_index"
    assert d["strategy"] == "exact"
    assert d["confidence"] == 1.0
    assert d["created_new"] is False
    assert d["review_status"] is None
    assert d["ancestry"] == []
    assert d["resolution_detail"] == {
        "granularity": None, "hf_repo_id": "Qwen/Qwen2.5-7B-Instruct"}


# --- registry precedence ---------------------------------------------------

def test_registry_entity_wins(monkeypatch, read_only, client):
    # The same id sits in the index under different casing, so the answer's
    # casing proves which source produced it.
    _base_store(monkeypatch, index_rows=[_index_row("META-LLAMA/llama-3.1-8b")])
    r = client.post("/api/v1/resolve", json={
        "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model"})
    assert r.status_code == 200
    d = r.json()
    assert d["canonical_id"] == "meta-llama/Llama-3.1-8B"
    assert d["resolution_source"] != "hub_stats_index"
    assert d["resolution_detail"]["hf_repo_id"] == "meta-llama/Llama-3.1-8B"


# --- misses stay no_match --------------------------------------------------

def test_non_hf_shaped_miss_no_match(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "just-a-name", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"
    assert d["resolution_source"] is None


def test_absent_from_both_no_match(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "someorg/totally-unknown-model", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"
    assert d["resolution_source"] is None


# --- missing / empty index degrades to no_match ----------------------------

def test_missing_index_table_degrades(monkeypatch, read_only, client):
    _base_store(monkeypatch, include_index=False)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"


def test_empty_index_table_degrades(monkeypatch, read_only, client):
    _base_store(monkeypatch, index_rows=[])
    r = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"


# --- case/separator variants hit via id_norm -------------------------------

def test_case_separator_variant_hits(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "qwen/qwen2.5_7b_instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["resolution_source"] == "hub_stats_index"
    assert d["resolution_detail"]["hf_repo_id"] == "Qwen/Qwen2.5-7B-Instruct"
    # A normalized hit is labelled as such, not "exact".
    assert d["strategy"] == "normalized"
    assert d["confidence"] == 0.95


def test_variant_hits_even_without_id_norm_column_value(monkeypatch, read_only, client):
    # No precomputed id_norm: the build falls back to normalize(id).
    _base_store(monkeypatch, index_rows=[
        _index_row("Qwen/Qwen2.5-7B-Instruct", with_norm=False)])
    r = client.post("/api/v1/resolve", json={
        "raw_value": "qwen/qwen2.5-7b-instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["resolution_source"] == "hub_stats_index"


# --- hf_repo_id populated on both paths ------------------------------------

def test_hf_repo_id_populated_both_paths(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    reg = client.post("/api/v1/resolve", json={
        "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model"}).json()
    idx = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"}).json()
    assert reg["resolution_detail"]["hf_repo_id"] == "meta-llama/Llama-3.1-8B"
    assert idx["resolution_detail"]["hf_repo_id"] == "Qwen/Qwen2.5-7B-Instruct"


# --- precedence: exact index confirmation beats a FUZZY registry match ------

def _fuzzy_collision_store(monkeypatch, *, include_variant_in_index: bool):
    """A base canonical that `acme/widget-7b-fp8` fuzzy-matches; the index
    optionally holds the exact variant id."""
    store = hf_store.RegistryStore()
    tables = {name: s.empty(name) for name in _ALL_TABLES}
    tables["canonical_models"] = pd.DataFrame([
        _row("canonical_models", id="acme/widget-7b", display_name="Widget 7B",
             org_id="acme", resolution_source="hf", parents="[]"),
    ])
    tables["aliases"] = pd.DataFrame([_alias("acme/widget-7b", "model", "acme/widget-7b")])
    tables["hub_stats_index"] = (
        pd.DataFrame([_index_row("acme/widget-7b-fp8")])
        if include_variant_in_index else s.empty("hub_stats_index")
    )
    store._tables = tables
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


def test_exact_index_overrides_fuzzy_registry_match(monkeypatch, read_only, client):
    # Else a real-but-unminted HF id is shadowed by the slug base.
    _fuzzy_collision_store(monkeypatch, include_variant_in_index=True)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "acme/widget-7b-fp8", "entity_type": "model"})
    d = r.json()
    assert d["resolution_source"] == "hub_stats_index"
    assert d["canonical_id"] == "acme/widget-7b-fp8"
    assert d["resolution_detail"]["hf_repo_id"] == "acme/widget-7b-fp8"
    assert d["strategy"] == "exact"


def test_fuzzy_registry_match_kept_when_not_in_index(monkeypatch, read_only, client):
    # Index precedence must not turn an existing fuzzy match into no_match.
    _fuzzy_collision_store(monkeypatch, include_variant_in_index=False)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "acme/widget-7b-fp8", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] == "acme/widget-7b"
    assert d["strategy"] == "fuzzy"


# --- drift guard: SQL id_norm == Python normalize() -------------------------

def test_sql_id_norm_matches_python_normalize():
    """The build script computes id_norm in SQL and the resolver looks up with
    Python normalize(); drift silently breaks variant confirmation."""
    import importlib.util
    from pathlib import Path

    import duckdb

    root = Path(__file__).resolve().parents[1]
    frozen = root / "curation" / "hub_stats_frozen.parquet"
    if not frozen.exists():
        pytest.skip("frozen hub-stats parquet not present")
    spec = importlib.util.spec_from_file_location(
        "build_hub_stats_index", root / "scripts" / "build_hub_stats_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q = mod.build_query([str(frozen)], limit=300)
    rows = duckdb.connect().execute(q).fetchall()
    assert rows, "no rows from frozen parquet"
    mismatches = [(r[0], r[1]) for r in rows if r[1] != normalize(r[0])]
    assert not mismatches, f"SQL id_norm drifted from normalize(): {mismatches[:5]}"
