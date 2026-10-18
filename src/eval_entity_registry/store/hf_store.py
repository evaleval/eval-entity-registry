"""In-memory store backed by HF Dataset parquet (or local fixtures in LOCAL_MODE).

Writes update memory immediately; the Hub is pushed only at the end of a run
via push_to_hub().
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from eval_entity_registry.config import settings
from eval_entity_registry.store import schemas

log = logging.getLogger(__name__)


def _local_mode() -> bool:
    """Check LOCAL_MODE at call time, not import time, so CLI --local flag works."""
    return os.environ.get("LOCAL_MODE", "").lower() in ("true", "1") or settings.local_mode


def _fixtures_path() -> Path:
    return Path(os.environ.get("FIXTURES_PATH", settings.fixtures_path))


TABLE_NAMES = [
    "canonical_orgs",
    "canonical_models",
    "canonical_benchmarks",
    "canonical_families",
    "canonical_composites",
    "canonical_metrics",
    "benchmark_metric_folds",
    "eval_harnesses",
    "canonical_inference_platforms",
    "aliases",
    "resolution_log",
    "eval_results",
    "sync_runs",
    # Cron-owned; listed so a non-local seed load() round-trips it intact.
    "hub_stats_index",
]


def _reconcile_schema(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Align a freshly-loaded parquet with the current schema: add missing
    columns, drop retired ones, apply one-way data migrations.

    Migration shims become no-ops once the parquet is re-saved; keep each for
    at least one release cycle so read-only Spaces survive a schema rollout.
    """
    import json as _json

    expected = schemas._SCHEMAS[table]
    df = df.copy()

    if table == "canonical_models" and "parent_model_id" in df.columns and "parents" not in df.columns:
        def _to_parents(legacy):
            try:
                if pd.isna(legacy):
                    return None
            except (TypeError, ValueError):
                pass
            if legacy is None or legacy == "":
                return None
            return _json.dumps([{"id": str(legacy), "relationship": "variant", "axis": "size"}])
        df["parents"] = df["parent_model_id"].map(_to_parents).astype(pd.StringDtype())

    # Fill with `None`, not `pd.NA`: pandas 3.x rejects pd.NA when building a
    # float64 Series, while None coerces to both NA and NaN.
    for col, dtype in expected.items():
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype=dtype)

    # Schema order gives deterministic parquet output.
    keep = [c for c in expected if c in df.columns]
    return df[keep]

# Tables needed for query-only (read-only) mode
QUERY_TABLE_NAMES = [
    "canonical_orgs",
    "canonical_models",
    "canonical_benchmarks",
    "canonical_families",
    "canonical_composites",
    "canonical_metrics",
    # No CRUD route, but the query-only Space serves it.
    "benchmark_metric_folds",
    "eval_harnesses",
    "canonical_inference_platforms",
    "aliases",
    # Consulted by the resolve path.
    "hub_stats_index",
]

# Written only by a dedicated cron; push_to_hub() and _flush_to_fixtures()
# must not overwrite them.
_CRON_OWNED_TABLES = frozenset({"hub_stats_index"})


class RegistryStore:
    """Holds all tables in memory. Single instance per process."""

    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, tables: list[str] | None = None) -> None:
        """Load tables (all of TABLE_NAMES by default) from HF Hub or local fixtures."""
        names = tables or TABLE_NAMES
        if _local_mode():
            self._load_from_fixtures(_fixtures_path(), names)
        else:
            self._load_from_hf(settings.hf_dataset_repo, names)
        self._loaded = True

    def _load_from_fixtures(self, path: Path, names: list[str]) -> None:
        for table in names:
            p = path / f"{table}.parquet"
            if p.exists():
                self._tables[table] = _reconcile_schema(table, pd.read_parquet(p))
            else:
                self._tables[table] = schemas.empty(table)

    def _load_from_hf(self, repo_id: str, names: list[str]) -> None:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        token = settings.hf_token or None
        # One sha for every table: the tables then come from a single revision,
        # and a cache hit at an exact sha is current by construction (at a moving
        # ref, the client silently serves the cached copy when the Hub is down).
        try:
            sha = HfApi(token=token).dataset_info(repo_id).sha
        except Exception as exc:
            raise RuntimeError(f"could not resolve {repo_id} HEAD: {exc!r}; aborting") from exc

        for table in names:
            filename = f"{table}/part-0.parquet"
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=sha,
                    token=token,
                )
                df = pd.read_parquet(local)
            except RemoteEntryNotFoundError:
                # Only the Hub's own "file does not exist" starts a table empty
                # (a table nobody has written yet). Any other failure, including
                # the client's LocalEntryNotFoundError for an unreachable Hub,
                # must abort, or push_to_hub would overwrite the remote with it.
                log.warning("%s@%s has no %s; starting with an empty table", repo_id, sha[:12], filename)
                self._tables[table] = schemas.empty(table)
                continue
            except Exception as exc:
                raise RuntimeError(
                    f"could not load {filename} from {repo_id}@{sha[:12]}: {exc!r}; "
                    "aborting so no empty table can be pushed back"
                ) from exc
            self._tables[table] = _reconcile_schema(table, df)

    def push_to_hub(self) -> None:
        """Push all in-memory tables to HF Hub as one commit, so readers never
        see a revision with tables from two different runs."""
        if _local_mode():
            self._flush_to_fixtures(_fixtures_path())
            return

        import tempfile
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=settings.hf_token or None)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            operations = []
            for table, df in self._tables.items():
                if table in _CRON_OWNED_TABLES:
                    continue
                p = tmp / f"{table}.parquet"
                df.to_parquet(p, index=False)
                operations.append(CommitOperationAdd(
                    path_in_repo=f"{table}/part-0.parquet", path_or_fileobj=str(p),
                ))
            api.create_commit(
                repo_id=settings.hf_dataset_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Push {len(operations)} registry tables",
            )

    def _flush_to_fixtures(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for table, df in self._tables.items():
            if table in _CRON_OWNED_TABLES:
                continue
            df.to_parquet(path / f"{table}.parquet", index=False)

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def table(self, name: str) -> pd.DataFrame:
        return self._tables[name]

    def has_table(self, name: str) -> bool:
        return name in self._tables

    def set_table(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df

    @property
    def loaded(self) -> bool:
        return self._loaded


_store: Optional[RegistryStore] = None


def get_store() -> RegistryStore:
    global _store
    if _store is None:
        _store = RegistryStore()
    return _store
