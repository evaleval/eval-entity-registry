"""Read-only access to the canonical entity tables, used by the resolver to
enrich a match with metadata and hierarchy. Mirrors `AliasStore`: parquet/HF
loading with an empty-table fallback. Writes are a service-side concern."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Parquet filenames per entity type (registry fixtures / HF dataset naming).
_TABLES = {
    "model": "canonical_models",
    "benchmark": "canonical_benchmarks",
    "metric": "canonical_metrics",
    "harness": "eval_harnesses",
    "org": "canonical_orgs",
    "family": "canonical_families",
    "composite": "canonical_composites",
}

# Scalar parent column for non-model types; models carry a typed `parents`
# JSON list instead (`decode_parents`).
_PARENT_FIELD = {
    "benchmark": "parent_benchmark_id",
    "org": "parent_org_id",
}


# ---------------------------------------------------------------------------
# Helpers (pure; the service wrapper imports these)
# ---------------------------------------------------------------------------

def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _na_to_none(value: Any) -> Any:
    return None if _is_na(value) else value


def _safe_json_load(s: str, default: Any = None) -> Any:
    """`json.loads` that returns `default` on any decode/type error."""
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return default


def _hf_repo_id_of(ent: Optional[dict], canonical_id: Optional[str]) -> Optional[str]:
    """The model's HF repo id when the registry can ATTEST it, else None.

    Attested = HF oracle provenance (`resolution_source == "hf"`) or hub-stats
    confirmation of that exact id (`metadata.hf_id == canonical_id`); a shadow
    slug whose `metadata.hf_id` names a DIFFERENT repo is excluded. None means
    "not attested", not "not on HF": a non-null value is reliable, a null is not
    a negative claim."""
    if not ent or not canonical_id:
        return None
    if _na_to_none(ent.get("resolution_source")) == "hf":
        return canonical_id
    md = _na_to_none(ent.get("metadata"))
    if isinstance(md, str) and md.strip():
        # metadata is free-form JSON and may parse to a non-dict.
        md_obj = _safe_json_load(md, {})
        if isinstance(md_obj, dict) and md_obj.get("hf_id") == canonical_id:
            return canonical_id
    return None


def decode_parents(value) -> list[dict]:
    """Decode `canonical_models.parents` (JSON list of edges); accepts NA, None,
    empty strings and pre-decoded lists, returning [] for anything unparseable."""
    if _is_na(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null"):
            return []
        decoded = _safe_json_load(s, default=[])
        return list(decoded) if isinstance(decoded, list) else []
    return []


def variant_parent_id(parents: list[dict]) -> Optional[str]:
    """Id of the first `variant` edge in a parents list, or None."""
    for edge in parents:
        if isinstance(edge, dict) and edge.get("relationship") == "variant":
            pid = edge.get("id")
            if pid:
                return pid
    return None


def _kwarg_for(entity_type: str) -> str:
    """Constructor kwarg for an entity_type (English plurals, so no `<type>s_df`
    rule: `harnesses_df`, `families_df`)."""
    return {
        "model": "models_df",
        "benchmark": "benchmarks_df",
        "metric": "metrics_df",
        "harness": "harnesses_df",
        "org": "orgs_df",
        "family": "families_df",
        "composite": "composites_df",
    }[entity_type]


# ---------------------------------------------------------------------------
# CanonicalStore
# ---------------------------------------------------------------------------

class CanonicalStore:
    """One DataFrame per entity type with O(1) `lookup(entity_type, id)`.
    Empty tables are valid; `lookup` returns None."""

    def __init__(
        self,
        models_df: Optional[pd.DataFrame] = None,
        benchmarks_df: Optional[pd.DataFrame] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        harnesses_df: Optional[pd.DataFrame] = None,
        orgs_df: Optional[pd.DataFrame] = None,
        families_df: Optional[pd.DataFrame] = None,
        composites_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self._tables: dict[str, pd.DataFrame] = {
            "model": models_df if models_df is not None else pd.DataFrame(),
            "benchmark": benchmarks_df if benchmarks_df is not None else pd.DataFrame(),
            "metric": metrics_df if metrics_df is not None else pd.DataFrame(),
            "harness": harnesses_df if harnesses_df is not None else pd.DataFrame(),
            "org": orgs_df if orgs_df is not None else pd.DataFrame(),
            "family": families_df if families_df is not None else pd.DataFrame(),
            "composite": composites_df if composites_df is not None else pd.DataFrame(),
        }
        self._index: dict[str, dict[str, dict]] = {}
        # benchmark_id → family_id, built lazily from canonical_families.
        self._benchmark_to_family: Optional[dict[str, str]] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path) -> "CanonicalStore":
        """Load every `_TABLES` entry from `<path>/<table>.parquet`; missing or
        unreadable files become empty tables."""
        p = Path(path)
        kwargs: dict[str, pd.DataFrame] = {}
        for entity_type, fname in _TABLES.items():
            file = p / f"{fname}.parquet"
            if not file.exists():
                logger.info(
                    "CanonicalStore.from_parquet: %s not found; using empty table",
                    file,
                )
                continue
            try:
                df = pd.read_parquet(file)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "CanonicalStore.from_parquet: failed to read %s (%s: %s); "
                    "falling back to empty table",
                    file, type(exc).__name__, exc,
                )
                continue
            kwargs[_kwarg_for(entity_type)] = df
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo_id: str) -> "CanonicalStore":
        """Load every `_TABLES` entry from `<table>/part-0.parquet` in a HF
        Dataset repo; missing tables become empty tables."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        kwargs: dict[str, pd.DataFrame] = {}
        for entity_type, fname in _TABLES.items():
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{fname}/part-0.parquet",
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
                logger.warning(
                    "CanonicalStore.from_hf: failed to load %s from %r (%s: %s); "
                    "using empty table",
                    fname, repo_id, type(exc).__name__, exc,
                )
                continue
            kwargs[_kwarg_for(entity_type)] = df
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _ensure_index(self, entity_type: str) -> dict[str, dict]:
        if entity_type in self._index:
            return self._index[entity_type]
        df = self._tables.get(entity_type)
        idx: dict[str, dict] = {}
        if df is not None and not df.empty and "id" in df.columns:
            for _, row in df.iterrows():
                cid = row["id"]
                if isinstance(cid, str):
                    idx[cid] = {k: _na_to_none(v) for k, v in row.items()}
        self._index[entity_type] = idx
        return idx

    def lookup(self, entity_type: str, canonical_id: str) -> Optional[dict]:
        """The canonical row as a dict (NaN coerced to None), or None."""
        if not canonical_id:
            return None
        return self._ensure_index(entity_type).get(canonical_id)

    @property
    def org_dev_map(self) -> dict[str, str]:
        """HF-namespace -> developer-org map: `_ORG_ALIASES` unioned with the
        loaded `canonical_orgs` (`id` / `hf_org` / `aliases`). Cached."""
        cached = getattr(self, "_org_dev_map", None)
        if cached is not None:
            return cached
        from eval_entity_resolver.fold import build_curated_org_map

        df = self._tables.get("org")
        records: list[dict] = []
        if df is not None and not df.empty:
            for rec in df.to_dict("records"):
                rec = {k: _na_to_none(v) for k, v in rec.items()}
                # `aliases` is a JSON-encoded VARCHAR in the parquet tables.
                al = rec.get("aliases")
                if isinstance(al, str):
                    rec["aliases"] = _safe_json_load(al, default=[])
                records.append(rec)
        self._org_dev_map = build_curated_org_map(records)
        return self._org_dev_map

    # ------------------------------------------------------------------
    # Enrichment (pure functions of the loaded tables)
    # ------------------------------------------------------------------

    def benchmark_family_enrichment(
        self, benchmark_id: Optional[str]
    ) -> dict:
        """Family/category fields for a matched benchmark.

          - `family_key`: id of the canonical_families row listing
            `benchmark_id` (a slice inherits its parent's family); falls back
            to the parent-walk terminus, i.e. `benchmark_id` itself for a
            singleton.
          - `category`: the family's curated category, or None.
          - `composite_keys`: always [] here; which composites a benchmark
            appears in is a fact only the producer's view layer has.
        """
        if not benchmark_id:
            return {"family_key": None, "category": None, "composite_keys": []}

        if self._benchmark_to_family is None:
            self._benchmark_to_family = self._build_benchmark_to_family_index()

        family_key = self._benchmark_to_family.get(benchmark_id)

        if family_key is None:
            visited: set[str] = {benchmark_id}
            cur = benchmark_id
            while True:
                bench_row = self.lookup("benchmark", cur)
                if bench_row is None:
                    break
                parent = _na_to_none(bench_row.get("parent_benchmark_id"))
                if not parent or parent == cur or parent in visited:
                    break
                visited.add(parent)
                cur = parent
                fam = self._benchmark_to_family.get(parent)
                if fam:
                    family_key = fam
                    break
            if family_key is None:
                family_key = cur

        family_row = self.lookup("family", family_key)
        category = (
            _na_to_none(family_row.get("category"))
            if family_row is not None
            else None
        )
        return {
            "family_key": family_key,
            "category": category,
            "composite_keys": [],
        }

    # ------------------------------------------------------------------
    # Hierarchy: ancestry + typed resolution_detail
    # ------------------------------------------------------------------

    def _family_to_composite(self, family_id: Optional[str]) -> Optional[str]:
        """The family's first `composite_keys` entry, else the first composite
        whose `family_id` points back at it; None for a root family."""
        if not family_id:
            return None
        family_row = self.lookup("family", family_id)
        if family_row is not None:
            keys = family_row.get("composite_keys")
            if isinstance(keys, str):
                keys = _safe_json_load(keys, default=[])
            if isinstance(keys, list):
                for k in keys:
                    if isinstance(k, str) and k:
                        return k
        comp_df = self._tables.get("composite")
        if comp_df is not None and not comp_df.empty and "family_id" in comp_df.columns:
            hit = comp_df[comp_df["family_id"] == family_id]
            if not hit.empty:
                cid = hit.iloc[0].get("id")
                if isinstance(cid, str):
                    return cid
        return None

    def compute_ancestry(
        self, entity_type: str, canonical_id: Optional[str],
        matched_entity: Optional[dict] = None,
    ) -> list[dict]:
        """Ordered `[{canonical_id, level}]` from the matched entity's immediate
        parent up to the root; `[]` for a root.

        - model: group (when != leaf) then family (when distinct from both).
        - benchmark: family (when != self) then that family's composite.
        - family: its composite.
        - composite/metric/harness/org: [].
        """
        if not canonical_id:
            return []
        out: list[dict] = []
        if entity_type == "model":
            ent = matched_entity if matched_entity is not None else self.lookup("model", canonical_id)
            if not ent:
                return []
            group = _na_to_none(ent.get("model_group_id"))
            family = _na_to_none(ent.get("model_family_id"))
            if group and group != canonical_id:
                out.append({"canonical_id": group, "level": "group"})
            if family and family != canonical_id and family != group:
                out.append({"canonical_id": family, "level": "family"})
            return out
        if entity_type == "benchmark":
            fam = self.benchmark_family_enrichment(canonical_id)
            family_key = fam.get("family_key")
            if family_key and family_key != canonical_id:
                out.append({"canonical_id": family_key, "level": "family"})
            composite = self._family_to_composite(family_key)
            if composite and composite != canonical_id:
                out.append({"canonical_id": composite, "level": "composite"})
            return out
        if entity_type == "family":
            composite = self._family_to_composite(canonical_id)
            if composite and composite != canonical_id:
                out.append({"canonical_id": composite, "level": "composite"})
            return out
        return out

    def resolution_detail(
        self, entity_type: str, canonical_id: Optional[str],
        raw_value: Optional[str] = None,
        matched_entity: Optional[dict] = None,
    ) -> dict:
        """Typed resolution-detail sub-object keyed by entity_type.

        - model:     {"granularity": variant|group|family,
                      "hf_repo_id": str|None (see `_hf_repo_id_of`)}
        - benchmark: {"level": composite|family|benchmark|slice,
                      "matched_subset": str|None}
        - harness:   {} here; the resolver fills {harness_version_stripped,
                     bare_name, bare_tier} on a version-stripped match.
        - others:    {}
        """
        if entity_type == "model":
            ent = matched_entity if matched_entity is not None else self.lookup("model", canonical_id)
            gran = _na_to_none((ent or {}).get("resolution_granularity")) if ent else None
            return {"granularity": gran, "hf_repo_id": _hf_repo_id_of(ent, canonical_id)}
        if entity_type == "benchmark":
            ent = matched_entity if matched_entity is not None else self.lookup("benchmark", canonical_id)
            level = "benchmark"
            matched_subset: Optional[str] = None
            if ent:
                parent = _na_to_none(ent.get("parent_benchmark_id"))
                if parent and parent != canonical_id:
                    level = "slice"
            # An alias-fold match (`Anatomy` -> mmlu) carries the raw value
            # through so downstream can map it to the folded subset.
            if raw_value and canonical_id and raw_value.strip().lower() != canonical_id.lower():
                matched_subset = raw_value
            return {"level": level, "matched_subset": matched_subset}
        return {}

    def _build_benchmark_to_family_index(self) -> dict[str, str]:
        """benchmark_id → family_id from canonical_families; empty when the
        table is absent so benchmarks still resolve without family ancestry."""
        out: dict[str, str] = {}
        df = self._tables.get("family")
        if df is None or df.empty or "id" not in df.columns:
            return out
        for _, row in df.iterrows():
            family_id = row.get("id")
            if not isinstance(family_id, str):
                continue
            raw = row.get("benchmark_ids")
            if _is_na(raw) or raw is None:
                continue
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, str):
                s = raw.strip()
                if not s or s in ("[]", "null"):
                    continue
                items = _safe_json_load(s, default=[])
            else:
                continue
            for bid in items:
                if isinstance(bid, str):
                    # Seed validation rejects multi-family benchmarks, so
                    # first-write-wins is safe.
                    out.setdefault(bid, family_id)
        return out

    def parent_canonical_id(
        self, entity_type: str, entity: Optional[dict]
    ) -> Optional[str]:
        """Immediate parent id: the first `variant` edge for models, the
        `parent_*_id` column for benchmarks/orgs."""
        if not entity:
            return None
        if entity_type == "model":
            return variant_parent_id(decode_parents(entity.get("parents")))
        field = _PARENT_FIELD.get(entity_type)
        if not field:
            return None
        return _na_to_none(entity.get(field))

    def model_metadata_fields(
        self, matched_canonical_id: str, matched_entity: Optional[dict]
    ) -> dict:
        """Model-specific response fields.

        `canonical_id` is the matched LEAF (snapshot, precision and mode all
        distinct); `resolved_leaf_id` equals it (compat). `model_group_id` is
        group membership, a total partition, so it is always set: the group
        root, or the leaf itself for a singleton. The deprecated `root_model_id`
        keeps null-at-root semantics: the group root only when the leaf
        collapses into a larger group, else None. Lineage and metadata fields
        are read off the leaf row, where seed time materialised them."""
        if not matched_entity:
            return {
                "canonical_id": matched_canonical_id,
                "resolved_leaf_id": matched_canonical_id,
                "root_model_id": None,
                "lineage_origin_org_id": None,
                "model_group_id": None,
                "model_family_id": None,
                "lineage_origin_model_id": None,
                "lineage_origin_model_org_id": None,
                "inference_platform": None,
                "resolution_source": None,
                "resolution_granularity": None,
                "parents": None,
                "open_weights": None,
                "release_date": None,
                "params_billions": None,
            }

        group_id = _na_to_none(matched_entity.get("model_group_id"))
        leaf_id = matched_canonical_id
        parents_decoded = decode_parents(matched_entity.get("parents")) or None
        leaf_family = _na_to_none(matched_entity.get("model_family_id"))
        leaf_lineage_model = _na_to_none(matched_entity.get("lineage_origin_model_id"))
        leaf_lineage_org = _na_to_none(matched_entity.get("lineage_origin_model_org_id"))

        collapses = group_id is not None and group_id != leaf_id
        return {
            "canonical_id": leaf_id,
            "resolved_leaf_id": leaf_id,
            "root_model_id": group_id if collapses else None,
            "lineage_origin_org_id": leaf_lineage_org,
            "model_group_id": group_id,
            "model_family_id": leaf_family,
            "lineage_origin_model_id": leaf_lineage_model,
            "lineage_origin_model_org_id": leaf_lineage_org,
            "inference_platform": None,
            "resolution_source": _na_to_none(matched_entity.get("resolution_source")),
            "resolution_granularity": _na_to_none(matched_entity.get("resolution_granularity")),
            "parents": parents_decoded,
            "open_weights": _na_to_none(matched_entity.get("open_weights")),
            "release_date": _na_to_none(matched_entity.get("release_date")),
            "params_billions": _na_to_none(matched_entity.get("params_billions")),
        }
