"""Read/write helpers over RegistryStore's in-memory DataFrames.

Tight-loop writers (add_alias, upsert_eval_result, append_resolution_log)
accumulate rows in a pending buffer; flush_pending(store) applies them in
one pd.concat per table.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from eval_entity_registry.store.hf_store import RegistryStore
from eval_entity_registry.store import schemas


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Snapshot-date suffix shapes: `-YYYY-MM-DD` (gpt-4o-2024-08-06),
# `-YYYYMMDD` (claude-sonnet-4-20250514), `-YYYY-MM` (gpt-5-2025-08).
_DATE_ISO_FULL_RE = re.compile(r"-(\d{4})-(\d{2})-(\d{2})$")
_DATE_PACKED_RE = re.compile(r"-(\d{4})(\d{2})(\d{2})$")
_DATE_ISO_MONTH_RE = re.compile(r"-(\d{4})-(\d{2})$")

# Keeps 4-digit tails (param counts, batch numbers) from reading as a year.
_VALID_YEAR_RANGE = (2015, 2035)


def _derive_release_date_from_id(canonical_id: str) -> Optional[str]:
    """ISO YYYY-MM-DD parsed off a date-suffixed canonical id; None when no
    suffix is present or a component fails its plausibility guard."""
    if not canonical_id:
        return None

    def _ok_year(s: str) -> bool:
        try:
            return _VALID_YEAR_RANGE[0] <= int(s) <= _VALID_YEAR_RANGE[1]
        except ValueError:
            return False

    m = _DATE_ISO_FULL_RE.search(canonical_id)
    if m:
        y, mo, d = m.groups()
        if _ok_year(y) and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
        return None

    m = _DATE_PACKED_RE.search(canonical_id)
    if m:
        y, mo, d = m.groups()
        if _ok_year(y) and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
        return None

    m = _DATE_ISO_MONTH_RE.search(canonical_id)
    if m:
        y, mo = m.groups()
        if _ok_year(y) and 1 <= int(mo) <= 12:
            # Month-grained snapshot: day defaults to 01.
            return f"{y}-{mo}-01"
        return None

    return None


def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _source_config_key(value) -> Optional[str]:
    """Normalize nullable source_config values for alias-index keys."""
    return None if _is_na(value) else value


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a Series to dict, coercing pandas NA/NaN/NaT to None for JSON.
    Uses Series.to_dict() so numpy scalars are unboxed to Python types."""
    return {k: (None if _is_na(v) else v) for k, v in row.to_dict().items()}


def _records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to list-of-dicts, coercing NA/NaN to None for JSON."""
    if df.empty:
        return []
    return df.astype(object).mask(df.isna(), None).to_dict(orient="records")


# ------------------------------------------------------------------
# Pending-row buffer  (avoids O(n²) pd.concat-per-row)
# ------------------------------------------------------------------

def _get_pending(store: RegistryStore, table: str) -> list[dict]:
    """Return the pending-row list for *table*, creating it if needed."""
    if not hasattr(store, "_pending"):
        store._pending = {}
    return store._pending.setdefault(table, [])


def decode_parents(value) -> list[dict]:
    """Decode `canonical_models.parents` (JSON-encoded list-of-edges) to a
    Python list. Tolerant of NA/NaN, None, empty strings, and pre-decoded lists."""
    if _is_na(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null"):
            return []
        try:
            decoded = json.loads(s)
            return list(decoded) if isinstance(decoded, list) else []
        except (ValueError, TypeError):
            return []
    return []


def derive_model_lineage_fields(store: RegistryStore) -> dict[str, int]:
    """Populate the denormalized lineage columns from `canonical_models.parents`
    (caches recomputed on every seed/refresh; returns counts for logging).

    - `model_group_id`: root via same-org `quantized` and `variant axis in
      {version, mode}` edges (chat/instruct carry `training_stage` and stay
      out). Always set, self at the root.
    - `model_family_id`: root via same-org `quantized` + `variant axis in
      {version, mode, training_stage, size, tier}`; stops at finetune/merge/
      adapter. Version lines are never crossed because `variant·version`
      edges only link a snapshot to its pointer. Always set, self at the root.
    - `lineage_origin_model_id`: deepest ancestor via non-`variant` edges;
      NULL at the origin. `lineage_origin_model_org_id`: its `org_id`, with
      self-fallback.
    - `open_weights`: explicit value wins; else inherited through `variant`
      + `quantized` edges, never across finetune/merge/adapter.
    """
    df = store.table("canonical_models")
    if df.empty:
        return {
            "group_set": 0,
            "family_set": 0,
            "lineage_model_set": 0,
            "lineage_org_set": 0,
            "open_weights_inherited": 0,
            "release_date_derived_from_id": 0,
        }

    parents_by_id: dict[str, list[dict]] = {}
    org_by_id: dict[str, Optional[str]] = {}
    open_by_id: dict[str, Optional[bool]] = {}
    release_by_id: dict[str, Optional[str]] = {}
    for _, row in df.iterrows():
        cid = row["id"]
        # A self-referencing edge (malformed seed) would be a degenerate
        # cycle in every walk.
        parents_by_id[cid] = [
            p
            for p in decode_parents(row.get("parents"))
            if not (isinstance(p, dict) and p.get("id") == cid)
        ]
        org = row.get("org_id")
        org_by_id[cid] = None if _is_na(org) else org
        ow = row.get("open_weights")
        open_by_id[cid] = None if _is_na(ow) else bool(ow)
        rd = row.get("release_date")
        release_by_id[cid] = None if _is_na(rd) else str(rd)

    # An HF-shaped id with no org_id gets its developer from the same
    # store-backed org map the resolver uses (org alias rows included, since
    # canonical_orgs has no aliases column). Never overrides an existing
    # org_id and never invents an org for a bare id.
    from eval_entity_resolver.fold import build_org_dev_map_from_store

    _adf = store.table("aliases")
    _org_alias_pairs = (
        zip(_adf[_adf["entity_type"] == "org"]["raw_value"],
            _adf[_adf["entity_type"] == "org"]["canonical_id"])
        if _adf is not None and not _adf.empty else []
    )
    hf_to_dev = build_org_dev_map_from_store(
        store.table("canonical_orgs").to_dict("records"), _org_alias_pairs
    )
    org_prefix_updates: dict[str, str] = {}
    for cid in parents_by_id:
        if org_by_id.get(cid) is None and isinstance(cid, str) and "/" in cid:
            prefix = cid.split("/", 1)[0]
            dev = hf_to_dev.get(prefix.lower(), prefix)
            if dev:
                org_by_id[cid] = dev  # feed the lineage_origin_org walk below
                org_prefix_updates[cid] = dev

    def _walk(
        start: str,
        edge_ok: Callable[[dict], bool],
        org_conditional: bool = False,
    ) -> str:
        """Deepest id reachable through edges where `edge_ok(edge)`; stops on
        no-match or cycle. Multiple matching edges (a merge) pick min-id so the
        result does not depend on YAML edge order."""
        visited = {start}
        current = start
        while True:
            cur_org = org_by_id.get(current)
            edges = parents_by_id.get(current, []) or []
            candidate_ids = [
                p["id"] for p in edges
                if isinstance(p, dict) and edge_ok(p) and p.get("id")
                # Org-conditional: a third-party quant keeps its own family;
                # the link survives via lineage_origin.
                and (not org_conditional or (cur_org is not None and org_by_id.get(p["id"]) == cur_org))
            ]
            next_id = min(candidate_ids) if candidate_ids else None
            if not next_id or next_id in visited or next_id not in parents_by_id:
                return current
            visited.add(next_id)
            current = next_id

    def _is_identity_edge(p: dict) -> bool:
        rel = p.get("relationship")
        if rel == "quantized":
            return True
        # `version` = dated snapshot of the pointer; `mode` = runtime reasoning
        # toggle of the same weights. chat/instruct carry `training_stage`.
        if rel == "variant" and p.get("axis") in ("version", "mode"):
            return True
        return False

    def _walk_group(start: str) -> str:
        """`_walk(_is_identity_edge)` restricted to same-org edges: a community
        quant (`unsloth/...-bnb-4bit` of `microsoft/phi-4`) keeps its own group
        so its scores never merge into the base lab's model; the edge still
        feeds `lineage_origin_model_id`."""
        visited = {start}
        current = start
        while True:
            cur_org = org_by_id.get(current)
            edges = parents_by_id.get(current, []) or []
            candidate_ids = [
                p["id"] for p in edges
                if isinstance(p, dict) and _is_identity_edge(p) and p.get("id")
                and cur_org is not None
                and org_by_id.get(p["id"]) == cur_org
            ]
            next_id = min(candidate_ids) if candidate_ids else None
            if not next_id or next_id in visited or next_id not in parents_by_id:
                return current
            visited.add(next_id)
            current = next_id

    def _is_lineage_edge(p: dict) -> bool:
        return p.get("relationship") in {"quantized", "finetune", "merge", "adapter"}

    def _is_family_fold_edge(p: dict) -> bool:
        """`quantized` plus `variant` axes {version, mode, training_stage, size,
        tier}; a modality/domain sibling is a distinct artifact and stays at the leaf."""
        rel = p.get("relationship")
        if rel == "quantized":
            return True
        if rel == "variant":
            return p.get("axis") in {"version", "mode", "training_stage", "size", "tier"}
        return False

    def _inherit_open_from_ancestors(start: str) -> Optional[bool]:
        """First explicit `open_weights` on an ancestor (self excluded) reached
        through `variant` + `quantized` edges; None when none has it set."""
        visited = {start}
        current = start
        while True:
            edges = parents_by_id.get(current, []) or []
            cands = [
                p["id"] for p in edges
                if isinstance(p, dict) and p.get("relationship") in {"variant", "quantized"} and p.get("id")
            ]
            next_id = min(cands) if cands else None
            if not next_id or next_id in visited or next_id not in parents_by_id:
                return None
            visited.add(next_id)
            current = next_id
            v = open_by_id.get(current)
            if v is not None:
                return v

    group_updates: dict[str, Optional[str]] = {}
    family_updates: dict[str, Optional[str]] = {}
    lineage_model_updates: dict[str, Optional[str]] = {}
    lineage_updates: dict[str, Optional[str]] = {}
    open_updates: dict[str, Optional[bool]] = {}
    release_updates: dict[str, Optional[str]] = {}
    inherited_count = 0
    release_derived_count = 0
    for cid in parents_by_id:
        group = _walk_group(cid)
        group_updates[cid] = group
        family = _walk(cid, _is_family_fold_edge, org_conditional=True)
        family_updates[cid] = family
        # `ancestor` serves the model_id column (no self-fallback) and the
        # org column (self-fallback).
        ancestor = _walk(cid, _is_lineage_edge)
        lineage_model_updates[cid] = ancestor if ancestor != cid else None
        # Explicit None check, not `or`: an upstream lab with an empty org_id
        # must not flip to self.org_id and mis-attribute a finetune as same-org.
        ancestor_org = org_by_id.get(ancestor)
        lineage_updates[cid] = ancestor_org if ancestor_org is not None else org_by_id.get(cid)
        explicit = open_by_id.get(cid)
        if explicit is not None:
            open_updates[cid] = explicit
        else:
            inherited = _inherit_open_from_ancestors(cid)
            open_updates[cid] = inherited
            if inherited is not None:
                inherited_count += 1

        # An explicit release_date wins; a dated id is the fallback.
        explicit_release = release_by_id.get(cid)
        if explicit_release is not None and explicit_release.strip():
            release_updates[cid] = explicit_release
        else:
            derived = _derive_release_date_from_id(cid)
            release_updates[cid] = derived
            if derived is not None:
                release_derived_count += 1

    df = df.copy()
    df["model_group_id"] = df["id"].map(group_updates).astype(pd.StringDtype())
    df["model_family_id"] = df["id"].map(family_updates).astype(pd.StringDtype())
    df["lineage_origin_model_id"] = df["id"].map(lineage_model_updates).astype(pd.StringDtype())
    df["lineage_origin_model_org_id"] = df["id"].map(lineage_updates).astype(pd.StringDtype())
    df["open_weights"] = df["id"].map(open_updates).astype(pd.BooleanDtype())
    df["release_date"] = df["id"].map(release_updates).astype(pd.StringDtype())
    if org_prefix_updates:
        df["org_id"] = df.apply(
            lambda r: org_prefix_updates.get(r["id"], r.get("org_id")), axis=1
        ).astype(pd.StringDtype())

    # The `org-unknown` tag tracks org-less rows (null or the `unknown`
    # sentinel) in both directions, so a stale tag cannot outlive a resolved org.
    def _sync_org_unknown_tag(row) -> Any:
        raw = row.get("tags")
        oid = row.get("org_id")
        try:
            is_na = bool(pd.isna(oid))
        except (ValueError, TypeError):
            is_na = False
        orgless = is_na or oid == "unknown"
        try:
            cur = json.loads(raw) if isinstance(raw, str) else []
        except (ValueError, TypeError):
            cur = []
        if not isinstance(cur, list):
            cur = []
        has = "org-unknown" in cur
        if orgless and not has:
            cur = [*cur, "org-unknown"]
        elif not orgless and has:
            cur = [c for c in cur if c != "org-unknown"]
        else:
            return raw
        return json.dumps(cur)

    df["tags"] = df.apply(_sync_org_unknown_tag, axis=1).astype(pd.StringDtype())
    store.set_table("canonical_models", df)

    # Every org the prefix rule derived must exist in canonical_orgs; an
    # uncatalogued uploader gets a community row (as in `_ensure_hf_org`).
    if org_prefix_updates:
        odf = store.table("canonical_orgs")
        have = set(odf["id"])
        missing = sorted({o for o in org_prefix_updates.values() if o not in have})
        if missing:
            cols = list(odf.columns)
            new_rows = []
            for o in missing:
                r = {c: None for c in cols}
                r.update({"id": o, "display_name": o, "hf_org": o, "kind": "community"})
                for c, v in (("tags", "[]"), ("metadata", "{}"), ("review_status", "reviewed")):
                    if c in r:
                        r[c] = v
                new_rows.append(r)
            store.set_table(
                "canonical_orgs", pd.concat([odf, pd.DataFrame(new_rows)], ignore_index=True)
            )

    return {
        "group_set": int(df["model_group_id"].notna().sum()),
        "family_set": int(df["model_family_id"].notna().sum()),
        "lineage_model_set": int(df["lineage_origin_model_id"].notna().sum()),
        "lineage_org_set": int(df["lineage_origin_model_org_id"].notna().sum()),
        "open_weights_inherited": inherited_count,
        "release_date_derived_from_id": release_derived_count,
    }


def flush_pending(store: RegistryStore) -> None:
    """Concat all buffered rows into their respective tables in one shot."""
    pending = getattr(store, "_pending", {})
    for table, rows in pending.items():
        if not rows:
            continue
        df = store.table(table)
        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)
        store.set_table(table, df)
    store._pending = {}


# ------------------------------------------------------------------
# Generic entity helpers
# ------------------------------------------------------------------

def get_entity(store: RegistryStore, table: str, entity_id: str) -> Optional[dict]:
    df = store.table(table)
    row = df[df["id"] == entity_id]
    if row.empty:
        for pending_row in _get_pending(store, table):
            if pending_row.get("id") == entity_id:
                return pending_row
        return None
    return _row_to_dict(row.iloc[0])


def list_entities(
    store: RegistryStore,
    table: str,
    search: Optional[str] = None,
    review_status: Optional[str] = None,
    **filters: Any,
) -> list[dict]:
    df = store.table(table)
    if search:
        mask = df["id"].str.contains(search, case=False, na=False)
        if "display_name" in df.columns:
            mask = mask | df["display_name"].str.contains(search, case=False, na=False)
        df = df[mask]
    if review_status:
        df = df[df["review_status"] == review_status]
    for col, val in filters.items():
        if col in df.columns and val is not None:
            df = df[df[col] == val]
    return _records(df)


def upsert_entity(store: RegistryStore, table: str, data: dict, buffered: bool = False) -> dict:
    """Insert or update an entity row. `data` must contain `id`.
    If buffered=True, new rows go to the pending buffer (flushed by flush_pending).
    """
    df = store.table(table)
    entity_id = data["id"]
    now = _now()
    existing = df[df["id"] == entity_id]

    if existing.empty:
        pending = _get_pending(store, table)
        for p in pending:
            if p.get("id") == entity_id:
                p.update({k: v for k, v in data.items() if k != "id"})
                p["updated_at"] = now
                return p
        row = {**data, "created_at": now, "updated_at": now}
        if buffered:
            pending.append(row)
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            store.set_table(table, df)
        return row
    else:
        for col, val in data.items():
            if col != "id" and col in df.columns:
                df.loc[df["id"] == entity_id, col] = val
        df.loc[df["id"] == entity_id, "updated_at"] = now
        store.set_table(table, df)
        return _row_to_dict(df[df["id"] == entity_id].iloc[0])


# ------------------------------------------------------------------
# Alias helpers
# ------------------------------------------------------------------

# Sync-time alias index: (entity_type, raw_value, source_config_or_None) -> row dict.
_alias_index: dict[tuple, dict] = {}


def _rebuild_alias_index(store: RegistryStore) -> None:
    """Rebuild the in-memory alias index from the aliases table."""
    global _alias_index
    _alias_index = {}
    df = store.table("aliases")
    for _, row in df.iterrows():
        if row.get("status") != "rejected":
            row_dict = _row_to_dict(row)
            key = (
                row_dict["entity_type"],
                row_dict["raw_value"],
                _source_config_key(row_dict.get("source_config")),
            )
            _alias_index[key] = row_dict
    for pending_row in _get_pending(store, "aliases"):
        if pending_row.get("status") != "rejected":
            key = (
                pending_row["entity_type"],
                pending_row["raw_value"],
                _source_config_key(pending_row.get("source_config")),
            )
            _alias_index[key] = pending_row


def get_alias(
    store: RegistryStore,
    raw_value: str,
    entity_type: str,
    source_config: Optional[str],
) -> Optional[dict]:
    source_config = _source_config_key(source_config)
    if _alias_index:
        if source_config:
            scoped = _alias_index.get((entity_type, raw_value, source_config))
            if scoped:
                return scoped
        global_ = _alias_index.get((entity_type, raw_value, None))
        if global_:
            return global_
        return None

    df = store.table("aliases")
    mask = (
        (df["raw_value"] == raw_value)
        & (df["entity_type"] == entity_type)
        & (df["status"] != "rejected")
    )
    if source_config:
        scoped = df[mask & (df["source_config"] == source_config)]
        if not scoped.empty:
            return _row_to_dict(scoped.iloc[0])
    global_ = df[mask & df["source_config"].isna()]
    if not global_.empty:
        return _row_to_dict(global_.iloc[0])
    return None


def seed_alias_id(entity_type: str, raw_value: str, source_config) -> str:
    """Content-derived alias id over the uniqueness key, so a from-scratch seed
    is reproducible; the caller must fall back to a random id when it is taken."""
    key = f"{entity_type}\0{raw_value}\0{_source_config_key(source_config) or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def add_alias(store: RegistryStore, data: dict, buffered: bool = False) -> dict:
    """Insert an alias row. Raises ValueError when a non-rejected alias already
    exists for (entity_type, raw_value, source_config). buffered=True defers
    the write to flush_pending. A caller-supplied `id` is used as-is."""
    raw_value = data["raw_value"]
    entity_type = data["entity_type"]
    source_config = _source_config_key(data.get("source_config"))
    key = (entity_type, raw_value, source_config)

    if _alias_index and key in _alias_index:
        raise ValueError(
            f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
            "Use update_alias() to modify an existing alias."
        )

    df = store.table("aliases")
    mask = (
        (df["raw_value"] == raw_value)
        & (df["entity_type"] == entity_type)
        & (df["status"] != "rejected")
    )
    if source_config is not None:
        mask = mask & (df["source_config"] == source_config)
    else:
        mask = mask & df["source_config"].isna()
    if mask.any():
        raise ValueError(
            f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
            "Use update_alias() to modify an existing alias."
        )

    for p in _get_pending(store, "aliases"):
        if (p["entity_type"] == entity_type and p["raw_value"] == raw_value
                and _source_config_key(p.get("source_config")) == source_config
                and p.get("status") != "rejected"):
            raise ValueError(
                f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
                "Use update_alias() to modify an existing alias."
            )

    now = _now()
    row = {
        **data,
        "source_config": source_config,
        "id": data.get("id") or str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
    }

    if buffered:
        _get_pending(store, "aliases").append(row)
    else:
        df = store.table("aliases")  # re-read in case it changed
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        store.set_table("aliases", df)

    # Only an already-built index is updated; a partial one would mislead get_alias.
    if _alias_index and row.get("status") != "rejected":
        _alias_index[key] = row
    return row


def update_alias(store: RegistryStore, alias_id: str, updates: dict) -> Optional[dict]:
    df = store.table("aliases")
    if not (df["id"] == alias_id).any():
        return None
    for col, val in updates.items():
        if col in df.columns:
            df.loc[df["id"] == alias_id, col] = val
    df.loc[df["id"] == alias_id, "updated_at"] = _now()
    store.set_table("aliases", df)
    updated = _row_to_dict(df[df["id"] == alias_id].iloc[0])
    # Keep a built index in sync so a follow-up lookup does not see stale data.
    if _alias_index:
        key = (
            updated["entity_type"],
            updated["raw_value"],
            _source_config_key(updated.get("source_config")),
        )
        if updated.get("status") != "rejected":
            _alias_index[key] = updated
        else:
            _alias_index.pop(key, None)
    return updated


# ------------------------------------------------------------------
# Eval results (mapping table: one row per EEE evaluation result)
# ------------------------------------------------------------------

def _eval_result_id(evaluation_id: str, result_index: int) -> str:
    """Deterministic ID from evaluation_id + result_index."""
    key = f"{evaluation_id}:{result_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# Ids already in the pending buffer, so an upsert finds them without a scan.
_pending_result_ids: set[str] = set()


def upsert_eval_result(store: RegistryStore, data: dict) -> dict:
    """Insert or update an eval_results row. Uses deterministic ID from evaluation_id + result_index."""
    row_id = _eval_result_id(data["evaluation_id"], data["result_index"])
    now = _now()

    if row_id in _pending_result_ids:
        pending = _get_pending(store, "eval_results")
        for p in pending:
            if p["id"] == row_id:
                for col, val in data.items():
                    if col != "id":
                        p[col] = val
                p["updated_at"] = now
                return p

    df = store.table("eval_results")
    existing = df[df["id"] == row_id]
    if not existing.empty:
        for col, val in data.items():
            if col != "id" and col in df.columns:
                df.loc[df["id"] == row_id, col] = val
        df.loc[df["id"] == row_id, "updated_at"] = now
        store.set_table("eval_results", df)
        return _row_to_dict(df[df["id"] == row_id].iloc[0])

    row = {**data, "id": row_id, "created_at": now, "updated_at": now}
    _get_pending(store, "eval_results").append(row)
    _pending_result_ids.add(row_id)
    return row


def get_eval_results(
    store: RegistryStore,
    model_id: Optional[str] = None,
    benchmark_id: Optional[str] = None,
    source_config: Optional[str] = None,
) -> list[dict]:
    """Query eval_results with optional filters."""
    df = store.table("eval_results")
    if model_id:
        df = df[df["model_id"] == model_id]
    if benchmark_id:
        df = df[df["benchmark_id"] == benchmark_id]
    if source_config:
        df = df[df["source_config"] == source_config]
    return _records(df)


# ------------------------------------------------------------------
# Resolution log
# ------------------------------------------------------------------

def append_resolution_log(store: RegistryStore, entry: dict) -> None:
    row = {**entry, "id": str(uuid.uuid4()), "timestamp": _now()}
    _get_pending(store, "resolution_log").append(row)


# ------------------------------------------------------------------
# Sync runs
# ------------------------------------------------------------------

def start_sync_run(
    store: RegistryStore, source_config: str, rerun: bool
) -> str:
    run_id = str(uuid.uuid4())
    df = store.table("sync_runs")
    row = {
        "id": run_id,
        "source_config": source_config,
        "started_at": _now(),
        "completed_at": None,
        "status": "running",
        "rerun": rerun,
        "entities_created": 0,
        "entities_updated": 0,
        "aliases_created": 0,
        "aliases_updated": 0,
        "errors": json.dumps([]),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    store.set_table("sync_runs", df)
    return run_id


def finish_sync_run(store: RegistryStore, run_id: str, counts: dict, errors: list) -> None:
    df = store.table("sync_runs")
    df.loc[df["id"] == run_id, "completed_at"] = _now()
    df.loc[df["id"] == run_id, "status"] = "failed" if errors else "completed"
    for col, val in counts.items():
        if col in df.columns:
            df.loc[df["id"] == run_id, col] = val
    df.loc[df["id"] == run_id, "errors"] = json.dumps(errors)
    store.set_table("sync_runs", df)
