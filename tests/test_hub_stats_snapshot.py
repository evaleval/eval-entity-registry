"""Pinned hub-stats snapshot helper (services/hub_stats_snapshot.py) — offline.

Pins: (1) one metadata HEAD yields commit sha + LFS sha256 + size, and missing
pieces fail loudly; (2) the download is pinned to that sha and size-checked;
(3) the DuckDB-native pinned source never names the converter endpoint or a
moving ref; (4) no bulk script references the converter URL any more (the
whole point of specs/hub-stats-snapshot-fetch.md).
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_entity_registry.services import hub_stats_snapshot as snapmod
from eval_entity_registry.services.hub_stats_snapshot import (
    HUB_STATS_FILE,
    HUB_STATS_REPO,
    HubStatsSnapshot,
    download_snapshot,
    hf_source,
    is_remote_source,
    resolve_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SHA = "c2dc24e395df9df1052f067fcda6314e86ecd84c"
ETAG = "7fd4aaf31072a5f502622a629749177c03e7d620f8488977d8003a403e9a4974"


def _patch_metadata(monkeypatch, **fields):
    meta = SimpleNamespace(commit_hash=SHA, etag=f'"{ETAG}"', size=1622261921, xet_file_data=None)
    for k, v in fields.items():
        setattr(meta, k, v)
    calls = []

    def fake_get(url, token=None):
        calls.append((url, token))
        return meta

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", fake_get)
    return calls


def test_resolve_snapshot_maps_metadata(monkeypatch):
    calls = _patch_metadata(monkeypatch)
    snap = resolve_snapshot(token="tok")
    assert snap == HubStatsSnapshot(
        repo_id=HUB_STATS_REPO, revision=SHA, filename=HUB_STATS_FILE,
        sha256=ETAG, size=1622261921, local_path=None,
    )
    # One request, against the raw resolve URL (resolvers bucket), with the token.
    assert len(calls) == 1
    url, tok = calls[0]
    assert url == f"https://huggingface.co/datasets/{HUB_STATS_REPO}/resolve/main/{HUB_STATS_FILE}"
    assert tok == "tok"
    assert "api/datasets" not in url and "parquet/" not in url


@pytest.mark.parametrize("missing", ["commit_hash", "etag", "size"])
def test_resolve_snapshot_fails_loudly_on_incomplete_metadata(monkeypatch, missing):
    _patch_metadata(monkeypatch, **{missing: None})
    with pytest.raises(RuntimeError, match="incomplete metadata"):
        resolve_snapshot()


def test_download_snapshot_pins_revision_and_sets_path(monkeypatch, tmp_path):
    blob = tmp_path / "models.parquet"
    blob.write_bytes(b"x" * 10)
    seen = {}

    def fake_download(repo_id, filename, **kw):
        seen.update(repo_id=repo_id, filename=filename, **kw)
        return str(blob)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    snap = HubStatsSnapshot(HUB_STATS_REPO, SHA, HUB_STATS_FILE, ETAG, 10)
    out = download_snapshot(snap, token="tok", cache_dir=tmp_path)
    assert out.local_path == blob
    assert out.revision == SHA and out.sha256 == ETAG  # unchanged identity
    assert seen["repo_id"] == HUB_STATS_REPO
    assert seen["filename"] == HUB_STATS_FILE
    assert seen["repo_type"] == "dataset"
    assert seen["revision"] == SHA  # pinned, never "main"
    assert seen["token"] == "tok"


def test_download_snapshot_rejects_size_mismatch(monkeypatch, tmp_path):
    blob = tmp_path / "models.parquet"
    blob.write_bytes(b"x" * 9)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(blob))
    snap = HubStatsSnapshot(HUB_STATS_REPO, SHA, HUB_STATS_FILE, ETAG, 10)
    with pytest.raises(RuntimeError, match="downloaded size 9 != expected 10"):
        download_snapshot(snap)


def test_hf_source_is_pinned_and_never_the_converter():
    snap = HubStatsSnapshot(HUB_STATS_REPO, SHA, HUB_STATS_FILE, ETAG, 1)
    src = hf_source(snap)
    assert src == f"hf://datasets/{HUB_STATS_REPO}@{SHA}/{HUB_STATS_FILE}"
    for forbidden in ("/api/", "refs%2Fconvert", "@main", "parquet/models/train"):
        assert forbidden not in src
    assert is_remote_source(src)
    assert is_remote_source("https://huggingface.co/x")
    assert not is_remote_source("/tmp/models.parquet")


def test_no_bulk_script_references_the_converter_endpoint():
    """Nothing in the package, the scripts or the workflows may name the
    converter endpoint; every reader goes through the pinned snapshot."""
    offenders: list[str] = []
    py_files = list((REPO_ROOT / "scripts").glob("*.py")) + list(
        (REPO_ROOT / "src" / "eval_entity_registry").rglob("*.py")
    )
    for path in sorted(py_files):
        text = path.read_text()
        if re.search(r"\bPARQUET_URL\b", text) or "hub-stats/parquet/" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if "hub-stats/parquet/" in path.read_text():
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"converter endpoint referenced by: {offenders}"



# ---------- live HubStatsClient (write-mode draft enrichment) ----------

class _Recorder:
    def __init__(self):
        self.sql: list[str] = []

    def execute(self, sql, *a, **k):
        self.sql.append(sql)
        return self


def test_live_client_pins_hf_source_and_uses_huggingface_secret(monkeypatch):
    """Construction is network-free; first use resolves ONE pinned snapshot,
    reads `hf://…@<sha>/models.parquet` with a HUGGINGFACE secret, and never
    names the converter endpoint."""
    from eval_entity_registry.services import hub_stats as hs

    snap = HubStatsSnapshot(HUB_STATS_REPO, SHA, HUB_STATS_FILE, ETAG, 1)
    calls = []
    monkeypatch.setattr(snapmod, "resolve_snapshot", lambda token=None, **_: (calls.append(token), snap)[1])
    rec = _Recorder()
    import duckdb
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: rec)
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.delenv(hs.HUB_STATS_LOCAL_PARQUET_ENV, raising=False)

    client = hs.HubStatsClient()
    assert client.parquet_url is None and calls == []  # nothing resolved yet
    con = client._ensure_con()
    assert con is rec
    assert calls == ["tok"]
    assert client.parquet_url == f"hf://datasets/{HUB_STATS_REPO}@{SHA}/{HUB_STATS_FILE}"
    sql = "\n".join(rec.sql)
    assert "LOAD httpfs" in sql
    assert "TYPE HUGGINGFACE" in sql and "TOKEN 'tok'" in sql
    assert "TYPE HTTP" not in sql and "huggingface.co/api" not in sql
    # Second call reuses the connection; no re-resolve.
    assert client._ensure_con() is rec and calls == ["tok"]


def test_live_client_offline_path_skips_network(monkeypatch, tmp_path):
    from eval_entity_registry.services import hub_stats as hs

    local = tmp_path / "models.parquet"
    local.write_bytes(b"")
    monkeypatch.setenv(hs.HUB_STATS_LOCAL_PARQUET_ENV, str(local))
    monkeypatch.setattr(snapmod, "resolve_snapshot", lambda **_: (_ for _ in ()).throw(AssertionError("no network")))
    rec = _Recorder()
    import duckdb
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: rec)
    client = hs.HubStatsClient()
    assert client.parquet_url == str(local)
    client._ensure_con()
    assert rec.sql == []  # no httpfs, no secret


def test_live_client_lookup_degrades_to_none_when_snapshot_unresolvable(monkeypatch):
    from eval_entity_registry.services import hub_stats as hs

    monkeypatch.delenv(hs.HUB_STATS_LOCAL_PARQUET_ENV, raising=False)
    monkeypatch.setattr(snapmod, "resolve_snapshot", lambda **_: (_ for _ in ()).throw(RuntimeError("hub down")))
    client = hs.HubStatsClient()
    assert client.lookup("org/model") is None  # logged, never raised
