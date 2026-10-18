"""The publish script's logical content hash, seed tree hash and skip rule."""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from conftest import load_script_module


@pytest.fixture(scope="module")
def mod():
    return load_script_module("publish_registry_data")


_LOCAL = {
    "seed_git_sha": "a" * 40,
    "schema_version": "registry.3.3",
    "content_hash": "c1",
    "resolver_tree_sha": "r1",
    "seed_tree_sha": "s1",
}


def test_skip_when_content_and_tree_shas_match(mod):
    remote = {**_LOCAL, "seed_git_sha": "b" * 40}
    assert mod._skip_reason(_LOCAL, remote) == "content and tree shas match"


@pytest.mark.parametrize("key", ["resolver_tree_sha", "seed_tree_sha", "content_hash", "schema_version"])
def test_upload_when_any_key_differs(mod, key):
    remote = {**_LOCAL, "seed_git_sha": "b" * 40, key: "changed"}
    assert mod._skip_reason(_LOCAL, remote) is None


def test_upload_when_remote_lacks_new_keys(mod):
    remote = {"seed_git_sha": "b" * 40, "content_hash": "c1"}
    assert mod._skip_reason(_LOCAL, remote) is None


def test_skip_when_seed_git_sha_matches_even_if_content_differs(mod):
    remote = {"seed_git_sha": "a" * 40, "content_hash": "other"}
    assert mod._skip_reason(_LOCAL, remote) == "seed_git_sha matches"


def test_upload_when_no_remote_or_unknown_git_sha(mod):
    assert mod._skip_reason(_LOCAL, None) is None
    local = {**_LOCAL, "seed_git_sha": None}
    assert mod._skip_reason(local, {"seed_git_sha": None}) is None


def _write(tmp_path, name, df):
    df.to_parquet(tmp_path / name, index=False)


def _frames(created="2026-01-01", updated="2026-01-02", name="x"):
    a = pd.DataFrame({
        "id": ["m1", "m2"],
        "name": [name, "y"],
        "created_at": [created, created],
        "updated_at": [updated, updated],
    })
    b = pd.DataFrame({"id": ["b1"], "score": [1.5]})
    return a, b


def test_content_hash_ignores_timestamps_only(mod, tmp_path):
    a, b = _frames()
    _write(tmp_path, "canonical_models.parquet", a)
    _write(tmp_path, "canonical_benchmarks.parquet", b)
    base = mod._content_hash(tmp_path)
    assert mod._content_hash(tmp_path) == base

    a2, _ = _frames(created="2027-05-05", updated="2027-06-06")
    _write(tmp_path, "canonical_models.parquet", a2)
    assert mod._content_hash(tmp_path) == base

    a3, _ = _frames(name="changed")
    _write(tmp_path, "canonical_models.parquet", a3)
    assert mod._content_hash(tmp_path) != base

    _write(tmp_path, "canonical_models.parquet", a)
    assert mod._content_hash(tmp_path) == base
    (tmp_path / "slice_overrides.yaml").write_text("a: 1\n")
    assert mod._content_hash(tmp_path) != base


_LS_TREE = (
    "100644 blob 1111111111111111111111111111111111111111\tseed/orgs.yaml\n"
    "100644 blob 2222222222222222222222222222222222222222\tseed/models/core.yaml\n"
    "100644 blob 3333333333333333333333333333333333333333\tseed/models/sources/hub_stats.state.json\n"
)


def test_seed_tree_sha_ignores_state_json(mod, monkeypatch):
    listing = {"out": _LS_TREE}
    monkeypatch.setattr(mod, "_git_out", lambda *a: listing["out"])
    base = mod._seed_tree_sha()

    listing["out"] = _LS_TREE.replace("3333", "9999")
    assert mod._seed_tree_sha() == base
    listing["out"] = _LS_TREE + (
        "100644 blob 4444444444444444444444444444444444444444\tseed/other.state.json\n"
    )
    assert mod._seed_tree_sha() == base

    listing["out"] = _LS_TREE.replace("2222", "8888")
    assert mod._seed_tree_sha() != base
    listing["out"] = _LS_TREE + (
        "100644 blob 5555555555555555555555555555555555555555\tseed/new.yaml\n"
    )
    assert mod._seed_tree_sha() != base


@pytest.mark.parametrize("matching", [True, False])
def test_dry_run_prints_verdict_and_never_pushes(mod, monkeypatch, tmp_path, capsys, matching):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    a, _ = _frames()
    _write(fixtures, "canonical_models.parquet", a)
    monkeypatch.setattr(mod, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(mod, "MANIFEST_PATH", fixtures / "manifest.json")
    monkeypatch.setattr(mod, "SEED_DIR", tmp_path / "seed")
    monkeypatch.setattr(mod, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(mod, "_resolver_tree_sha", lambda: "r1")
    monkeypatch.setattr(mod, "_seed_tree_sha", lambda: "s1")

    def no_push(*a, **k):
        raise AssertionError("dry-run must never push")

    monkeypatch.setattr(mod, "_push", no_push)
    remote = {
        "seed_git_sha": "b" * 40,
        "schema_version": mod.SCHEMA_VERSION,
        "content_hash": mod._content_hash(fixtures) if matching else "old",
        "resolver_tree_sha": "r1",
        "seed_tree_sha": "s1",
    }
    monkeypatch.setattr(mod, "_read_remote_manifest", lambda: remote)
    monkeypatch.setattr(sys, "argv", ["publish_registry_data.py", "--skip-seed", "--dry-run"])
    assert mod.main() == 0
    err = capsys.readouterr().err
    if matching:
        assert "dry-run verdict: would skip (content and tree shas match)" in err
    else:
        assert "dry-run verdict: would upload" in err


def test_dry_run_survives_unreadable_remote(mod, monkeypatch, tmp_path, capsys):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    monkeypatch.setattr(mod, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(mod, "MANIFEST_PATH", fixtures / "manifest.json")
    monkeypatch.setattr(mod, "SEED_DIR", tmp_path / "seed")
    monkeypatch.setattr(mod, "_push", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(mod, "_read_remote_manifest", boom)
    monkeypatch.setattr(sys, "argv", ["publish_registry_data.py", "--skip-seed", "--dry-run"])
    assert mod.main() == 0
    err = capsys.readouterr().err
    assert "remote manifest: unavailable" in err
    assert "dry-run verdict: would upload" in err
