"""Write-mode model-id policy: a verbatim HF id confirmation beats stored
aliases and mints the HF-true id (never a slug); a disagreeing alias is
flagged for review, never repointed, and a confirmed one is never demoted,
even on rerun; exact mode writes nothing; the resolve memo is keyed by mode.
The index tier runs against a real `hub_stats_index`; the live tier is off.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eval_entity_registry.main import app
from eval_entity_registry.services.log_writer import ResolveLogWriter
from eval_entity_registry.services.resolution_service import ResolutionService
from eval_entity_registry.store import hf_store, queries, schemas as s
from eval_entity_registry.services.hub_stats import normalize as _hsnorm

_ALL_TABLES = [
    "canonical_models", "canonical_benchmarks", "canonical_metrics",
    "eval_harnesses", "canonical_orgs", "canonical_families",
    "canonical_composites", "aliases", "resolution_log", "eval_results",
    "sync_runs",
]


def _row(table: str, **vals) -> dict:
    base = {col: None for col in s._SCHEMAS[table]}
    base.update({"review_status": "reviewed",
                 "created_at": "2026-01-01T00:00:00+00:00",
                 "updated_at": "2026-01-01T00:00:00+00:00"})
    for k, v in vals.items():
        base[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return base


def _index_row(hf_id: str) -> dict:
    base = {col: None for col in s._SCHEMAS["hub_stats_index"]}
    base.update({"id": hf_id, "id_norm": _hsnorm(hf_id), "open_weights": True})
    return base


def _store(monkeypatch, *, index_ids=(), models=(), aliases=()):
    store = hf_store.RegistryStore()
    tables = {name: s.empty(name) for name in _ALL_TABLES}
    if models:
        tables["canonical_models"] = pd.DataFrame([
            _row("canonical_models", id=mid, display_name=mid,
                 org_id=None, resolution_source=None, parents="[]")
            for mid in models
        ])
    if index_ids:
        tables["hub_stats_index"] = pd.DataFrame(
            [_index_row(i) for i in index_ids])
    store._tables = tables
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    svc = ResolutionService(store)
    for raw, cid, status in aliases:
        queries.add_alias(store, {
            "raw_value": raw, "entity_type": "model", "canonical_id": cid,
            "source_config": None, "source_field": None, "status": status,
            "strategy": "exact", "confidence": 1.0, "notes": None,
        })
    return store, svc


def _model_ids(store) -> set:
    queries.flush_pending(store)
    df = store.table("canonical_models")
    return set(df["id"]) if not df.empty else set()


def test_unregistered_index_hit_mints_hf_true_id(monkeypatch):
    store, svc = _store(monkeypatch, index_ids=["NousResearch/Hermes-4-70B"])
    d = svc.resolve("nousresearch/hermes-4-70b", "model", None, None)
    assert d["canonical_id"] == "NousResearch/Hermes-4-70B"
    assert d["strategy"] == "exact"
    assert d["confidence"] == 1.0
    assert d["created_new"] is True
    ids = _model_ids(store)
    assert "NousResearch/Hermes-4-70B" in ids
    assert "nousresearch/hermes-4-70b" not in ids
    alias = queries.get_alias(store, "nousresearch/hermes-4-70b", "model", None)
    assert alias and alias["canonical_id"] == "NousResearch/Hermes-4-70B"


def test_disagreeing_auto_alias_flagged_and_hf_id_wins(monkeypatch):
    store, svc = _store(
        monkeypatch,
        index_ids=["acme/Widget-7B-FP8"],
        models=["acme/widget-7b"],
        aliases=[("acme/Widget-7B-FP8", "acme/widget-7b", "auto")],
    )
    d = svc.resolve("acme/Widget-7B-FP8", "model", None, None)
    assert d["canonical_id"] == "acme/Widget-7B-FP8"
    assert d["created_new"] is True
    alias = queries.get_alias(store, "acme/Widget-7B-FP8", "model", None)
    # Flagged for review, not silently repointed.
    assert alias["canonical_id"] == "acme/widget-7b"
    assert alias["status"] == "uncertain"
    assert "hf-id-check disagreement" in (alias["notes"] or "")


def test_disagreeing_confirmed_alias_keeps_status(monkeypatch):
    store, svc = _store(
        monkeypatch,
        index_ids=["acme/Widget-7B-FP8"],
        models=["acme/widget-7b"],
        aliases=[("acme/Widget-7B-FP8", "acme/widget-7b", "confirmed")],
    )
    d = svc.resolve("acme/Widget-7B-FP8", "model", None, None)
    assert d["canonical_id"] == "acme/Widget-7B-FP8"
    alias = queries.get_alias(store, "acme/Widget-7B-FP8", "model", None)
    assert alias["status"] == "confirmed"
    assert "hf-id-check disagreement" in (alias["notes"] or "")


def test_agreeing_alias_skips_nothing_and_keeps_result(monkeypatch):
    store, svc = _store(
        monkeypatch,
        index_ids=["acme/Widget-7B"],
        models=["acme/Widget-7B"],
        aliases=[("acme/Widget-7B", "acme/Widget-7B", "confirmed")],
    )
    d = svc.resolve("acme/Widget-7B", "model", None, None)
    assert d["canonical_id"] == "acme/Widget-7B"
    assert d["created_new"] is False


def test_exact_mode_creates_nothing(monkeypatch):
    store, svc = _store(monkeypatch)
    d = svc.resolve("someorg/unknown-model", "model", None, None, mode="exact")
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"
    assert d["created_new"] is False
    queries.flush_pending(store)
    assert store.table("canonical_models").empty
    assert store.table("aliases").empty


def test_exact_mode_no_fuzzy_but_resolver_mode_fuzzy(monkeypatch):
    store, svc = _store(
        monkeypatch,
        models=["acme/widget-7b"],
        aliases=[("acme/widget-7b", "acme/widget-7b", "confirmed")],
    )
    exact = svc.resolve("acme/widget-7b-fp8", "model", None, None, mode="exact")
    assert exact["canonical_id"] is None
    # The memo is keyed by mode.
    fuzzy = svc.resolve("acme/widget-7b-fp8", "model", None, None)
    assert fuzzy["canonical_id"] == "acme/widget-7b"
    assert fuzzy["strategy"] == "fuzzy"


def test_exact_mode_match_writes_nothing(monkeypatch):
    # Exact mode is side-effect-free on a match too.
    store, svc = _store(
        monkeypatch,
        models=["acme/widget-7b"],
        aliases=[("Widget 7B", "acme/widget-7b", "confirmed")],
    )
    d = svc.resolve("widget-7b", "model", None, None, mode="exact")
    assert d["canonical_id"] == "acme/widget-7b"
    assert d["strategy"] == "normalized"
    queries.flush_pending(store)
    assert len(store.table("aliases")) == 1


def test_exact_mode_checker_hit_serves_attestation_without_minting(monkeypatch):
    store, svc = _store(monkeypatch, index_ids=["NousResearch/Hermes-4-70B"])
    d = svc.resolve(
        "NousResearch/Hermes-4-70B", "model", None, None, mode="exact")
    assert d["canonical_id"] == "NousResearch/Hermes-4-70B"
    assert d["created_new"] is False
    queries.flush_pending(store)
    assert store.table("canonical_models").empty
    assert store.table("aliases").empty
    assert store.table("canonical_orgs").empty


def test_attested_unregistered_reuses_case_variant_canonical(monkeypatch):
    # A lowercased draft already exists; the HF-true casing must reuse it.
    store, svc = _store(
        monkeypatch,
        index_ids=["acme/Widget-7B"],
        models=["acme/widget-7b"],
        aliases=[("acme/widget-7b", "acme/widget-7b", "auto")],
    )
    d = svc.resolve("acme/widget-7b", "model", None, None)
    assert d["canonical_id"] == "acme/widget-7b"
    assert d["created_new"] is False
    ids = _model_ids(store)
    assert ids == {"acme/widget-7b"}


def test_rerun_never_repoints_confirmed_alias(monkeypatch):
    store, svc = _store(
        monkeypatch,
        index_ids=["acme/Widget-7B-FP8"],
        models=["acme/widget-7b"],
        aliases=[("acme/Widget-7B-FP8", "acme/widget-7b", "confirmed")],
    )
    d = svc.resolve(
        "acme/Widget-7B-FP8", "model", None, None,
        sync_run_id="run-1", rerun=True,
    )
    assert d["canonical_id"] == "acme/Widget-7B-FP8"
    alias = queries.get_alias(store, "acme/Widget-7B-FP8", "model", None)
    assert alias["canonical_id"] == "acme/widget-7b"
    assert alias["status"] == "confirmed"
    assert "disagreement" in (alias["notes"] or "")


def test_api_mode_param_threads(monkeypatch):
    store, _svc = _store(monkeypatch)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    client = TestClient(app, raise_server_exceptions=True)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "someorg/unknown-model", "entity_type": "model",
        "mode": "exact"})
    assert r.status_code == 200
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"
    # The core fields are always serialized; ancestry/resolution_detail never null.
    for f in ("raw_value", "entity_type", "canonical_id", "strategy",
              "confidence", "created_new", "resolution_source",
              "review_status", "ancestry", "resolution_detail"):
        assert f in d
    assert d["ancestry"] == []
    assert d["resolution_detail"] == {}
    queries.flush_pending(store)
    assert store.table("canonical_models").empty


def test_api_rejects_bad_mode(monkeypatch):
    store, _svc = _store(monkeypatch)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    client = TestClient(app, raise_server_exceptions=True)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "x/y", "entity_type": "model", "mode": "wild"})
    assert r.status_code == 422
