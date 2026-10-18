from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


_SCHEMA = {
    "id": pd.StringDtype(),
    "raw_value": pd.StringDtype(),
    "entity_type": pd.StringDtype(),
    "canonical_id": pd.StringDtype(),
    "source_config": pd.StringDtype(),
    "source_field": pd.StringDtype(),
    "status": pd.StringDtype(),
    "strategy": pd.StringDtype(),
    "confidence": "float64",
    "notes": pd.StringDtype(),
    "created_at": pd.StringDtype(),
    "updated_at": pd.StringDtype(),
}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in _SCHEMA.items()})


class AliasStore:
    """Wraps the aliases table. Loaded into memory; writes are in-memory only."""

    def __init__(self, df: pd.DataFrame, read_only: bool = False) -> None:
        self._df = df.copy()
        self.read_only = read_only
        # Lazy per-(entity_type, source_config) caches.
        self._normalized_cache: dict[tuple[str, Optional[str]], dict[str, str]] = {}
        self._candidates_cache: dict[tuple[str, Optional[str]], list[tuple[str, str]]] = {}
        self._lookup_index: dict[tuple[str, str, Optional[str]], str] | None = None

    def _ensure_lookup_index(self) -> None:
        if self._lookup_index is not None:
            return
        self._lookup_index = {}
        df = self._df[self._df["status"] != "rejected"]
        for _, row in df.iterrows():
            # NaN != None as a dict key: global aliases must be keyed by None
            # or `lookup` misses every one of them.
            sc = row.get("source_config")
            if pd.isna(sc):
                sc = None
            key = (row["raw_value"], row["entity_type"], sc)
            self._lookup_index[key] = row["canonical_id"]

    def _invalidate_caches(self) -> None:
        self._normalized_cache.clear()
        self._candidates_cache.clear()
        self._lookup_index = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path, read_only: bool = False) -> "AliasStore":
        p = Path(path) / "aliases.parquet"
        if not p.exists():
            # A missing file is the legitimate fresh-store case: INFO, not WARNING.
            logger.info(
                "AliasStore.from_parquet: %s not found; falling back to empty store",
                p,
            )
            return cls(_empty_df(), read_only=read_only)
        try:
            df = pd.read_parquet(p)
        except (OSError, ValueError) as exc:
            # pyarrow's ArrowInvalid (corrupt parquet) subclasses ValueError.
            logger.warning(
                "AliasStore.from_parquet: failed to read %s (%s: %s); "
                "falling back to empty store",
                p,
                type(exc).__name__,
                exc,
            )
            df = _empty_df()
        return cls(df, read_only=read_only)

    @classmethod
    def from_hf(cls, repo_id: str, read_only: bool = False) -> "AliasStore":
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename="aliases/part-0.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(local)
        except (
            RepositoryNotFoundError,
            EntryNotFoundError,
            HfHubHTTPError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as exc:
            # RepositoryNotFoundError also covers an invalid token (HF returns
            # 401 disguised as 404); ValueError covers pyarrow's ArrowInvalid.
            # Construction must succeed, so fall back to an empty store but warn.
            logger.warning(
                "AliasStore.from_hf: failed to load aliases from %r (%s: %s); "
                "falling back to empty store",
                repo_id,
                type(exc).__name__,
                exc,
            )
            df = _empty_df()
        return cls(df, read_only=read_only)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def iter_alias_pairs(self, entity_type: str):
        """Yield (raw_value, canonical_id) for every non-rejected alias of
        `entity_type`, global and scoped alike."""
        df = self._df
        sub = df[(df["entity_type"] == entity_type) & (df["status"] != "rejected")]
        for raw, cid in zip(sub["raw_value"], sub["canonical_id"]):
            if isinstance(raw, str) and isinstance(cid, str):
                yield raw, cid

    def lookup(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
    ) -> Optional[str]:
        """canonical_id of the first non-rejected match, config-scoped before global."""
        self._ensure_lookup_index()
        if source_config:
            result = self._lookup_index.get((raw_value, entity_type, source_config))
            if result is not None:
                return result
        return self._lookup_index.get((raw_value, entity_type, None))

    # ------------------------------------------------------------------
    # Writes (in-memory only; caller is responsible for persistence)
    # ------------------------------------------------------------------

    def add_alias(
        self,
        raw_value: str,
        entity_type: str,
        canonical_id: str,
        source_config: Optional[str],
        source_field: Optional[str],
        status: str,
        strategy: str,
        confidence: float,
    ) -> None:
        if self.read_only:
            raise RuntimeError("AliasStore is read-only")
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "id": str(uuid.uuid4()),
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": source_field,
            "status": status,
            "strategy": strategy,
            "confidence": confidence,
            "notes": None,
            "created_at": now,
            "updated_at": now,
        }
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)
        self._invalidate_caches()

    def update_alias(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        canonical_id: str,
        status: str,
        strategy: str,
        confidence: float,
    ) -> None:
        """Upsert: update existing alias row or add new one."""
        if self.read_only:
            raise RuntimeError("AliasStore is read-only")
        df = self._df
        mask = (df["raw_value"] == raw_value) & (df["entity_type"] == entity_type)
        if source_config:
            mask = mask & (df["source_config"] == source_config)
        else:
            mask = mask & df["source_config"].isna()
        if mask.any():
            now = datetime.now(timezone.utc).isoformat()
            self._df.loc[mask, "canonical_id"] = canonical_id
            self._df.loc[mask, "status"] = status
            self._df.loc[mask, "strategy"] = strategy
            self._df.loc[mask, "confidence"] = confidence
            self._df.loc[mask, "updated_at"] = now
            self._invalidate_caches()
        else:
            self.add_alias(raw_value, entity_type, canonical_id, source_config, None, status, strategy, confidence)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def get_normalized_lookup(
        self, entity_type: str, source_config: Optional[str] = None
    ) -> dict[str, str]:
        """{normalized_raw_value: canonical_id}. With ``source_config``, that
        config's scoped aliases overlay the global ones; without it only global
        aliases are included, so scoped aliases never leak into other configs.
        """
        key = (entity_type, source_config)
        if key in self._normalized_cache:
            return self._normalized_cache[key]

        from eval_entity_resolver.normalization import normalize

        base = self._df[(self._df["entity_type"] == entity_type) & (self._df["status"] != "rejected")]
        global_df = base[base["source_config"].isna()]
        result: dict[str, str] = {}
        for _, row in global_df.iterrows():
            result[normalize(row["raw_value"])] = row["canonical_id"]
        if source_config:
            scoped_df = base[base["source_config"] == source_config]
            for _, row in scoped_df.iterrows():
                result[normalize(row["raw_value"])] = row["canonical_id"]
        self._normalized_cache[key] = result
        return result

    def get_all_for_type(
        self, entity_type: str, source_config: Optional[str] = None
    ) -> list[tuple[str, str]]:
        """[(raw_value, canonical_id)] for non-rejected aliases of ``entity_type``,
        filtered like ``get_normalized_lookup``. Cached."""
        key = (entity_type, source_config)
        if key in self._candidates_cache:
            return self._candidates_cache[key]

        base = self._df[(self._df["entity_type"] == entity_type) & (self._df["status"] != "rejected")]
        if source_config:
            mask = base["source_config"].isna() | (base["source_config"] == source_config)
            df = base[mask]
        else:
            df = base[base["source_config"].isna()]
        result = list(zip(df["raw_value"].tolist(), df["canonical_id"].tolist()))
        self._candidates_cache[key] = result
        return result
