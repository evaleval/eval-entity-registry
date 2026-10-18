"""Non-local store load and push: every table comes from one pinned revision,
only the Hub's own not-found starts a table empty, any other failure aborts,
and the push is one commit."""
from __future__ import annotations

import httpx
import pandas as pd
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

from eval_entity_registry.config import settings
from eval_entity_registry.store import hf_store, schemas

SHA = "f" * 40


def _remote_not_found() -> RemoteEntryNotFoundError:
    resp = httpx.Response(404, request=httpx.Request("HEAD", "https://hub.test/x"))
    return RemoteEntryNotFoundError("missing", response=resp)


class FakeApi:
    created: list[dict] = []

    def __init__(self, token=None):
        pass

    def dataset_info(self, repo_id):
        class Info:
            sha = SHA
        return Info()

    def create_commit(self, **kwargs):
        FakeApi.created.append(kwargs)

    def upload_file(self, **kwargs):
        raise AssertionError("per-table upload_file must not be used")


@pytest.fixture
def remote_mode(monkeypatch):
    import huggingface_hub
    monkeypatch.delenv("LOCAL_MODE", raising=False)
    monkeypatch.setattr(settings, "local_mode", False)
    monkeypatch.setattr(settings, "hf_dataset_repo", "org/registry-data")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.created = []


def _fake_download(tmp_path, failing: dict[str, Exception], seen: list[str]):
    def download(repo_id, filename, repo_type, revision, token):
        seen.append(revision)
        table = filename.split("/")[0]
        if table in failing:
            raise failing[table]
        df = schemas.empty(table)
        if "id" in df.columns:
            df = pd.concat([df, pd.DataFrame([{"id": f"{table}-row"}])], ignore_index=True)
        p = tmp_path / f"{table}.parquet"
        df.to_parquet(p, index=False)
        return str(p)
    return download


def test_missing_hub_file_starts_table_empty_at_pinned_sha(remote_mode, monkeypatch, tmp_path, caplog):
    import huggingface_hub
    seen: list[str] = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        _fake_download(tmp_path, {"hub_stats_index": _remote_not_found()}, seen))
    store = hf_store.RegistryStore()
    with caplog.at_level("WARNING"):
        store.load(tables=["canonical_models", "hub_stats_index"])
    assert len(store.table("canonical_models")) == 1
    assert len(store.table("hub_stats_index")) == 0
    assert "hub_stats_index/part-0.parquet" in caplog.text
    assert seen == [SHA, SHA]


@pytest.mark.parametrize("exc", [
    LocalEntryNotFoundError("hub unreachable"),
    ConnectionError("boom"),
    OSError("corrupt"),
])
def test_any_other_download_failure_aborts_load(remote_mode, monkeypatch, tmp_path, exc):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        _fake_download(tmp_path, {"eval_results": exc}, []))
    store = hf_store.RegistryStore()
    with pytest.raises(RuntimeError, match="eval_results/part-0.parquet"):
        store.load(tables=["canonical_models", "eval_results"])
    assert not store.loaded


def test_head_resolution_failure_aborts_load(remote_mode, monkeypatch):
    def broken_info(self, repo_id):
        raise ConnectionError("hub down")
    monkeypatch.setattr(FakeApi, "dataset_info", broken_info)
    store = hf_store.RegistryStore()
    with pytest.raises(RuntimeError, match="HEAD"):
        store.load(tables=["canonical_models"])


def test_push_to_hub_is_one_commit_covering_every_table(remote_mode):
    store = hf_store.RegistryStore()
    for table in hf_store.TABLE_NAMES:
        store.set_table(table, schemas.empty(table))
    store.push_to_hub()

    assert len(FakeApi.created) == 1
    call = FakeApi.created[0]
    paths = sorted(op.path_in_repo for op in call["operations"])
    expected = sorted(f"{t}/part-0.parquet" for t in hf_store.TABLE_NAMES if t not in hf_store._CRON_OWNED_TABLES)
    assert paths == expected
    assert call["repo_id"] == "org/registry-data"
    assert call["repo_type"] == "dataset"
