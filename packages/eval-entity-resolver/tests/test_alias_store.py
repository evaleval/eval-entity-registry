"""AliasStore constructor failures fall back to an empty store and are
logged, never silently swallowed.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from eval_entity_resolver import AliasStore


def test_from_hf_logs_warning_on_missing_repo(caplog):
    """A missing HF repo yields an empty store plus a WARNING naming the repo."""
    with caplog.at_level(logging.WARNING, logger="eval_entity_resolver.alias_store"):
        store = AliasStore.from_hf(repo_id="nonexistent-org/nonexistent-repo-xyz-123")

    assert store is not None
    assert store.lookup("anything", "model", None) is None
    assert store.to_dataframe().empty

    assert any(
        "from_hf" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected a WARNING with 'from_hf' in the message; got: {[(r.levelname, r.message) for r in caplog.records]}"
    assert any(
        "nonexistent-org/nonexistent-repo-xyz-123" in r.message
        for r in caplog.records
    )


def test_from_parquet_handles_missing_dir(tmp_path, caplog):
    """A missing dir is the legitimate fresh-store case: empty store, INFO log."""
    missing = tmp_path / "does-not-exist"

    with caplog.at_level(logging.INFO, logger="eval_entity_resolver.alias_store"):
        store = AliasStore.from_parquet(missing)

    assert store is not None
    assert store.to_dataframe().empty
    assert any(
        "from_parquet" in r.message and "not found" in r.message
        for r in caplog.records
    ), f"expected an INFO 'not found' log; got: {[(r.levelname, r.message) for r in caplog.records]}"


def test_from_parquet_logs_warning_on_corrupt_parquet(tmp_path, caplog):
    """A corrupt aliases.parquet logs a WARNING and yields an empty store
    rather than failing resolver init."""
    (tmp_path / "aliases.parquet").write_bytes(b"this is not a parquet file")

    with caplog.at_level(logging.WARNING, logger="eval_entity_resolver.alias_store"):
        store = AliasStore.from_parquet(tmp_path)

    assert store is not None
    assert store.to_dataframe().empty
    assert any(
        "from_parquet" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected a WARNING from from_parquet; got: {[(r.levelname, r.message) for r in caplog.records]}"


def test_from_parquet_loads_real_file(tmp_path):
    """A valid aliases.parquet round-trips without the fallback path."""
    df = pd.DataFrame(
        [
            {
                "id": "row-1",
                "raw_value": "IFEval",
                "entity_type": "benchmark",
                "canonical_id": "ifeval",
                "source_config": None,
                "source_field": None,
                "status": "confirmed",
                "strategy": "seed",
                "confidence": 1.0,
                "notes": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    df.to_parquet(tmp_path / "aliases.parquet")

    store = AliasStore.from_parquet(tmp_path)
    assert store.lookup("IFEval", "benchmark", None) == "ifeval"
