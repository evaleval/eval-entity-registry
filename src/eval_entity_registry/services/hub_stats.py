"""Shared hub-stats integration: HF metadata normalizers plus a DuckDB-backed
`HubStatsClient` for single-id lookups against the `cfahlgren1/hub-stats`
parquet. The bulk refresh script and live draft enrichment share one
row-shape parser so they cannot drift."""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, date
from typing import Optional

# Offline read path: a local hub-stats parquet replaces the pinned remote
# snapshot. Read at call time so tests can set/unset it per test.
HUB_STATS_LOCAL_PARQUET_ENV = "HUB_STATS_LOCAL_PARQUET"


def local_parquet_path() -> Optional[str]:
    """`$HUB_STATS_LOCAL_PARQUET` when set and the file exists; else None."""
    local = os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
    if local and os.path.exists(local):
        return local
    return None


def is_local_parquet() -> bool:
    """True when the offline local-parquet read path is active."""
    return local_parquet_path() is not None

# Shared so the bulk and lookup paths parse the same row shape.
QUERY_COLUMNS = (
    "id, author, createdAt, lastModified, downloads, downloadsAllTime, "
    "likes, trendingScore, tags, cardData, safetensors, gguf, baseModels, "
    "pipeline_tag, library_name"
)


# ---------------------------------------------------------------------------
# Slug + normalization helpers
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase + collapse separators (-/_./:) to single dashes."""
    return re.sub(r"[/_.:\-]+", "-", s.lower()).strip("-")


def slugify(value: str) -> str:
    """Lowercase, drop display-only punctuation, collapse whitespace and
    underscores to dashes. Preserves dots so canonical ids like
    `claude-opus-4.5` stay legible."""
    s = value.strip().lower()
    s = re.sub(r"[()\[\]{}]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def hf_id_to_canonical(
    hf_id: str,
    org_alias_map: dict[str, str],
) -> tuple[str, str]:
    """`'meta-llama/Llama-3.1-70B'` -> `('meta/llama-3.1-70b', 'meta')`: the
    HF org maps via `org_alias_map`, an unknown org becomes its own slug.
    Ids without `/` get the resolver's `unknown` org placeholder."""
    if "/" not in hf_id:
        return f"unknown/{slugify(hf_id)}", "unknown"
    org_part, name_part = hf_id.split("/", 1)
    org_norm = normalize(org_part)
    org_id = org_alias_map.get(org_norm, org_norm)
    name_slug = slugify(name_part)
    return f"{org_id}/{name_slug}", org_id


def hf_id_to_canonical_cased(
    hf_id: str,
    hf_to_dev: dict[str, str],
) -> tuple[str, str]:
    """Canonical id = the real HF repo id verbatim (`Qwen/Qwen2-7B-Instruct`);
    org_id = the curated parent when the HF org maps to one (qwen->alibaba),
    else the HF org itself. The org spelling is never folded into the id, so
    `huggingface.co/{id}` stays valid. `hf_to_dev` keys are lowercase HF org
    spellings. Ids without `/` fall back to `unknown/{HF-NAME}`."""
    if "/" not in hf_id:
        return f"unknown/{hf_id}", "unknown"
    org_part, name_part = hf_id.split("/", 1)
    org_id = hf_to_dev.get(org_part.lower(), org_part)
    return f"{org_part}/{name_part}", org_id


# ---------------------------------------------------------------------------
# Field extraction from a single hub-stats row
# ---------------------------------------------------------------------------

def coerce_date(value) -> Optional[str]:
    """Normalize createdAt to YYYY-MM-DD (drops time portion)."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else None
    if isinstance(value, str):
        return value[:10] if len(value) >= 10 else value
    return None


def has_downloadable_weights(row: dict) -> bool:
    """Open weights iff the row carries safetensors and/or gguf data; a row
    with neither is an inference-only endpoint or a deleted/empty repo."""
    if row.get("safetensors"):
        return True
    gguf = row.get("gguf")
    return bool(gguf)


def approx_params_billions(safetensors) -> Optional[float]:
    """Parameter count in billions. hub-stats' `safetensors.total` is the
    parameter count, not bytes (`Llama-3.1-8B` -> 8_030_261_248), so
    billions = total / 1e9."""
    if safetensors is None:
        return None
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
    else:
        total = getattr(safetensors, "total", None)
    if not total:
        return None
    return round(total / 1e9, 2) if total > 0 else None


def extract_license(card_data) -> Optional[str]:
    if not card_data:
        return None
    if isinstance(card_data, str):
        try:
            card_data = json.loads(card_data)
        except (ValueError, TypeError):
            return None
    if isinstance(card_data, dict):
        lic = card_data.get("license")
        return str(lic) if lic else None
    return None


def filter_useful_tags(raw_tags) -> list[str]:
    """Keep language codes, eval-results, license and format markers; drop
    noisy library/region/pipeline tags."""
    if not raw_tags:
        return []
    tags = list(raw_tags) if not isinstance(raw_tags, str) else [raw_tags]
    keep: list[str] = []
    for t in tags:
        t = str(t)
        if t == "eval-results":
            keep.append(t)
        elif t.startswith("license:"):
            keep.append(t)
        elif len(t) <= 3 and t.isalpha():  # language codes (en, zh, fr)
            keep.append(t)
        elif t in ("safetensors", "gguf"):
            keep.append(t)
    return sorted(set(keep))


# ---------------------------------------------------------------------------
# Family-version parent inference
# ---------------------------------------------------------------------------
#
# Hub-stats `baseModels` records upstream lineage only, never the edge from
# a dated snapshot to its moving-pointer canonical (`Olmo-3-1125-32B` ->
# `allenai/olmo-3-32b`): the pointer exists only in this registry. Without
# inference, such snapshots auto-create as orphan canonicals.

_INTERNAL_DATE_RE = re.compile(r"^(.+?)-(\d{4})-([^-].*)$")
_TRAILING_4DIGIT_RE = re.compile(r"^(.+)-(\d{4})$")
_TRAILING_6DIGIT_RE = re.compile(r"^(.+)-(\d{6})$")
_TRAILING_8DIGIT_RE = re.compile(r"^(.+)-(\d{8})$")
# Strict component widths keep non-date numeric tails from being peeled.
_ISO_FULL_DATE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH_DATE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})$")
_ISO_YEAR_DATE_RE = re.compile(r"^(.+)-(\d{4})$")

# Keeps 4-digit tails (param counts, batch numbers) from reading as a year.
_VALID_YEAR_RANGE = (2015, 2035)


def _looks_like_mmdd(token: str) -> bool:
    """4-digit MMDD with MM in [01,12] and DD in [01,31]; rejects size/version
    tokens like `8000`."""
    if len(token) != 4 or not token.isdigit():
        return False
    mm, dd = int(token[:2]), int(token[2:])
    return 1 <= mm <= 12 and 1 <= dd <= 31


def _looks_like_yyyymm(token: str) -> bool:
    """6-digit YYYYMM, the Stepfun-style release tag (`step-2-16k-202411`)."""
    if len(token) != 6 or not token.isdigit():
        return False
    yyyy, mm = int(token[:4]), int(token[4:])
    return _VALID_YEAR_RANGE[0] <= yyyy <= _VALID_YEAR_RANGE[1] and 1 <= mm <= 12


def _looks_like_yyyymmdd(token: str) -> bool:
    if len(token) != 8 or not token.isdigit():
        return False
    yyyy, mm, dd = int(token[:4]), int(token[4:6]), int(token[6:])
    return (
        _VALID_YEAR_RANGE[0] <= yyyy <= _VALID_YEAR_RANGE[1]
        and 1 <= mm <= 12
        and 1 <= dd <= 31
    )


def _looks_like_release_year(token: str) -> bool:
    if len(token) != 4 or not token.isdigit():
        return False
    return _VALID_YEAR_RANGE[0] <= int(token) <= _VALID_YEAR_RANGE[1]


def infer_family_parent_edge(
    hf_id: str,
    aliases_to_canonical: dict[str, str],
    target_canonical: Optional[str] = None,
) -> Optional[dict]:
    """`{id, relationship: variant, axis: version}` edge for a snapshot-shaped
    id whose stripped form resolves through the alias index; None otherwise.
    Single-pass strips (no composing with mode/quant suffixes): internal MMDD
    (`Olmo-3-1125-32B`), trailing MMDD (`kimi-k2-0905`), YYYYMM
    (`step-2-16k-202411`), YYYYMMDD (`claude-haiku-4-5-20251001`) and the ISO
    ladder (`gpt-5-2025-08-07` -> `-08` -> `-2025` -> `gpt-5`).
    `target_canonical` suppresses a self-edge: the bulk refresh may alias an
    HF id to its own family pointer, and a self-edge breaks the lineage walker."""
    candidates: list[str] = []

    # Internal-token strips first: a more specific lookup target than trailing strips.
    m = _INTERNAL_DATE_RE.match(hf_id)
    if m and _looks_like_mmdd(m.group(2)):
        prefix, _, suffix = m.groups()
        candidates.append(f"{prefix}-{suffix}")

    # The three ISO tail shapes are mutually exclusive; at most one branch fires.
    m = _ISO_FULL_DATE_RE.match(hf_id)
    if m:
        prefix, y, mo, d = m.groups()
        if (_looks_like_release_year(y) and 1 <= int(mo) <= 12
                and 1 <= int(d) <= 31):
            candidates.append(f"{prefix}-{y}-{mo}")
            candidates.append(f"{prefix}-{y}")
            candidates.append(prefix)
    else:
        m = _ISO_MONTH_DATE_RE.match(hf_id)
        if m:
            prefix, y, mo = m.groups()
            if _looks_like_release_year(y) and 1 <= int(mo) <= 12:
                candidates.append(f"{prefix}-{y}")
                candidates.append(prefix)
        else:
            m = _ISO_YEAR_DATE_RE.match(hf_id)
            if m:
                prefix, y = m.groups()
                if _looks_like_release_year(y):
                    candidates.append(prefix)

    # Trailing YYYYMMDD (Anthropic/xAI/Tencent style).
    m = _TRAILING_8DIGIT_RE.match(hf_id)
    if m and _looks_like_yyyymmdd(m.group(2)):
        candidates.append(m.group(1))

    # Trailing YYYYMM (Stepfun and several Chinese-lab release tags).
    m = _TRAILING_6DIGIT_RE.match(hf_id)
    if m and _looks_like_yyyymm(m.group(2)):
        candidates.append(m.group(1))

    # Trailing 4-digit MMDD (Moonshot/Kimi, Google -exp tags).
    m = _TRAILING_4DIGIT_RE.match(hf_id)
    if m and _looks_like_mmdd(m.group(2)):
        candidates.append(m.group(1))

    for cand in candidates:
        canonical = aliases_to_canonical.get(normalize(cand))
        if not canonical:
            continue
        if target_canonical is not None and canonical == target_canonical:
            continue
        return {"id": canonical, "relationship": "variant", "axis": "version"}
    return None


def extract_base_models(base_models) -> list[dict]:
    """`[{id, relationship}, ...]` from the `baseModels` struct; ids are raw
    HF ids for the caller to resolve."""
    if base_models is None:
        return []
    if isinstance(base_models, dict):
        relation = base_models.get("relation")
        models_list = base_models.get("models") or []
    else:
        relation = getattr(base_models, "relation", None)
        models_list = getattr(base_models, "models", None) or []
    if not relation:
        return []
    out: list[dict] = []
    for m in models_list:
        base_hf = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        if base_hf:
            out.append({"id": base_hf, "relationship": relation})
    return out


# ---------------------------------------------------------------------------
# Live lookup client
# ---------------------------------------------------------------------------

class HubStatsClient:
    """Single-id lookup against the hub-stats parquet via DuckDB (write-mode
    draft enrichment; the read-only Space uses `hub_stats_index`). Source: a
    pinned commit resolved on first use, or `$HUB_STATS_LOCAL_PARQUET`,
    materialized into a local table (one bulk fetch instead of thousands of
    per-id range reads) with per-id remote queries as fallback. `lookup()`
    never raises."""

    def __init__(self, parquet_url: Optional[str] = None) -> None:
        # Explicit source > local file > pinned snapshot (resolved lazily;
        # construction never touches the network).
        self.parquet_url: Optional[str] = parquet_url or local_parquet_path()
        self._con = None
        self._local_table_ready: bool = False
        self._local_table_failed: bool = False
        self._cache: dict[str, Optional[dict]] = {}
        self._lock = threading.Lock()

    def _ensure_con(self):
        if self._con is not None:
            return self._con
        # Lazy: processes that never call lookup() skip the duckdb import cost.
        import os
        import duckdb
        from eval_entity_registry.services.hub_stats_snapshot import (
            hf_source,
            is_remote_source,
            resolve_snapshot,
        )

        hf_token = os.environ.get("HF_TOKEN")
        if self.parquet_url is None:
            self.parquet_url = hf_source(resolve_snapshot(token=hf_token))
        con = duckdb.connect()
        if not is_remote_source(self.parquet_url):
            self._con = con  # local file: no httpfs, no auth
            return con
        con.execute("INSTALL httpfs; LOAD httpfs;")
        if hf_token:
            escaped = hf_token.replace("'", "''")
            con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{escaped}');")
        self._con = con
        return con

    def _ensure_local_table(self, con) -> bool:
        """Materialize the remote parquet into a local table; False when the
        bulk fetch failed (caller falls back to per-id remote queries).
        Idempotent. Takes the connection so it never calls `_ensure_con`
        itself, which tests patch as a call counter."""
        if self._local_table_ready:
            return True
        if self._local_table_failed:
            return False
        try:
            import logging
            log = logging.getLogger(__name__)
            log.info(
                "hub-stats: bulk-loading remote parquet into local table "
                "(one-time cost; avoids per-id rate limits)..."
            )
            # Only QUERY_COLUMNS, to keep the in-memory table small.
            con.execute(
                f"CREATE TABLE hub_stats AS "
                f"SELECT {QUERY_COLUMNS} FROM read_parquet('{self.parquet_url}')"
            )
            con.execute("CREATE INDEX hub_stats_id_idx ON hub_stats(id)")
            row_count = con.execute("SELECT COUNT(*) FROM hub_stats").fetchone()[0]
            log.info("hub-stats: local table loaded (%d rows)", row_count)
            self._local_table_ready = True
            return True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "hub-stats: bulk load failed (%s: %s); falling back to "
                "per-id remote queries (rate-limit-prone)",
                type(exc).__name__, exc,
            )
            self._local_table_failed = True
            return False

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None
            self._local_table_ready = False
            self._local_table_failed = False

    def lookup(self, hf_id: str) -> Optional[dict]:
        """Row dict for one HF id, or None when not found or on any error.
        Thread-safe and cached."""
        with self._lock:
            if hf_id in self._cache:
                return self._cache[hf_id]
        try:
            con = self._ensure_con()
            use_local = self._ensure_local_table(con)
            # HF ids keep the author's casing while EEE values arrive in mixed
            # casing; an exact `=` would silently miss and leave the draft unenriched.
            escaped = hf_id.lower().replace("'", "''")
            if use_local:
                sql = f"SELECT * FROM hub_stats WHERE LOWER(id) = '{escaped}' LIMIT 1"
            else:
                sql = (
                    f"SELECT {QUERY_COLUMNS} "
                    f"FROM read_parquet('{self.parquet_url}') "
                    f"WHERE LOWER(id) = '{escaped}' LIMIT 1"
                )
            cursor = con.execute(sql)
            cols = [d[0] for d in cursor.description]
            row = cursor.fetchone()
            result = dict(zip(cols, row)) if row else None
        except Exception as e:
            # Best-effort enrichment never blocks draft creation.
            import logging
            logging.getLogger(__name__).warning(
                "hub-stats lookup failed for %r: %s", hf_id, e
            )
            result = None
        with self._lock:
            self._cache[hf_id] = result
        return result


# ---------------------------------------------------------------------------
# Draft enrichment
# ---------------------------------------------------------------------------

def enrich_draft_from_row(
    row: dict,
    aliases_to_canonical: dict[str, str],
    org_alias_map: dict[str, str],
    target_canonical: Optional[str] = None,
) -> dict:
    """Partial canonical_models dict from one hub-stats row (release_date,
    params_billions, tags, parents, lineage_origin_model_org_id, metadata,
    hf_id). Nothing is forced: the caller decides what to write. Empty dict
    when the row carries no usable info."""
    out: dict = {}

    release = coerce_date(row.get("createdAt"))
    if release:
        out["release_date"] = release

    params = approx_params_billions(row.get("safetensors"))
    if params is not None:
        out["params_billions"] = params

    # Downloadable artifacts on HF mean open weights.
    if has_downloadable_weights(row):
        out["open_weights"] = True

    useful_tags = filter_useful_tags(row.get("tags"))
    if useful_tags:
        out["tags"] = json.dumps(useful_tags)

    # Unresolvable base ids are dropped so no parent edge dangles.
    parents: list[dict] = []
    lineage_origin_org_id: Optional[str] = None
    for edge in extract_base_models(row.get("baseModels")):
        base_hf = edge["id"]
        n = normalize(base_hf)
        parent_canonical = aliases_to_canonical.get(n)
        if parent_canonical is None:
            # Second chance for an HF id not yet aliased but whose org is known.
            slugified, _ = hf_id_to_canonical(base_hf, org_alias_map)
            parent_canonical = aliases_to_canonical.get(normalize(slugified))
        if parent_canonical:
            parents.append({"id": parent_canonical, "relationship": edge["relationship"]})
            # First resolved non-variant edge sets the origin, as in
            # `derive_model_lineage_fields`.
            if lineage_origin_org_id is None and edge["relationship"] != "variant":
                if "/" in parent_canonical:
                    # The prefix is the HF org; fold to the curated developer
                    # (meta-llama -> meta).
                    _pp = parent_canonical.split("/", 1)[0]
                    lineage_origin_org_id = org_alias_map.get(_pp.lower(), _pp)

    hf_id = row.get("id")
    if isinstance(hf_id, str) and not any(
        p.get("relationship") == "variant" and p.get("axis") == "version"
        for p in parents
    ):
        version_edge = infer_family_parent_edge(
            hf_id, aliases_to_canonical, target_canonical=target_canonical,
        )
        if version_edge is not None:
            parents.append(version_edge)

    if parents:
        out["parents"] = json.dumps(parents)
    if lineage_origin_org_id:
        out["lineage_origin_model_org_id"] = lineage_origin_org_id

    metadata: dict = {"source": "hub_stats", "hf_id": row.get("id")}
    if row.get("downloadsAllTime") is not None:
        metadata["downloads_all_time"] = int(row["downloadsAllTime"])
    if row.get("likes") is not None:
        metadata["likes"] = int(row["likes"])
    if row.get("library_name"):
        metadata["library_name"] = row["library_name"]
    if row.get("pipeline_tag"):
        metadata["pipeline_tag"] = row["pipeline_tag"]
    license_str = extract_license(row.get("cardData"))
    if license_str:
        metadata["license"] = license_str
    if len(metadata) > 2:  # has more than just source + hf_id
        out["metadata"] = json.dumps(metadata, sort_keys=True)

    # Always set, independent of the metadata guard: `_auto_create_entity`
    # reads it to mint the canonical id and display_name in HF-true casing.
    if isinstance(hf_id, str) and hf_id.strip():
        out["hf_id"] = hf_id

    return out
