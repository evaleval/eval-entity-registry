"""
Shared hub-stats integration: helpers for normalizing HF metadata, plus
a `HubStatsClient` that wraps a DuckDB connection for single-id lookups
against the `cfahlgren1/hub-stats` parquet.

Used by:
  - `scripts/refresh_from_hub_stats.py` — bulk backfill of existing
    canonicals' metadata
  - `services/resolution_service.py` — on-demand enrichment of model
    drafts at auto-create time

Both paths share the same row-shape parsing so behavior stays consistent
between the bulk pre-load and the live lookup.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, date
from typing import Optional

# Live-lookup source for the deployed Space's `HubStatsClient` ONLY. This is
# the HF parquet-converter endpoint, which answers 400 for several minutes after
# ANY commit to the upstream dataset (daily, ~13:18–13:36 UTC) while the
# dataset-viewer rebuilds `refs/convert/parquet`. The bulk scripts (refresh /
# index / freeze) therefore do NOT read it: they download a pinned snapshot of
# `main/models.parquet` via `services/hub_stats_snapshot.py` (same bytes; see
# specs/hub-stats-snapshot-fetch.md). The client tolerates the window because a
# lookup failure degrades to "no enrichment".
PARQUET_URL = (
    "https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet"
)

# Offline read path: when HUB_STATS_LOCAL_PARQUET points at a local hub-stats
# parquet, every
# hub-stats query (bulk refresh + live HubStatsClient lookup) reads that file
# instead of streaming the live URL. CI / the deployed Space leave the env unset
# and keep the live path. The env var is read at call time (not import time) so
# tests can set/unset it per-test without reimporting the module.
HUB_STATS_LOCAL_PARQUET_ENV = "HUB_STATS_LOCAL_PARQUET"


def resolve_parquet_source(parquet_url: str = PARQUET_URL) -> str:
    """Return the parquet source DuckDB should `read_parquet()` from.

    Prefers the local file at `$HUB_STATS_LOCAL_PARQUET` when that env var is
    set and the file exists (offline mode); otherwise the passed-in live URL.
    A set-but-missing path falls through to the live URL so a stale env var
    never silently breaks the live path."""
    local = os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
    if local and os.path.exists(local):
        return local
    return parquet_url


def is_local_parquet() -> bool:
    """True when the offline local-parquet read path is active."""
    local = os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
    return bool(local and os.path.exists(local))

# Columns the queries fetch. Centralized so the bulk and lookup paths
# parse the same row shape.
QUERY_COLUMNS = (
    "id, author, createdAt, lastModified, downloads, downloadsAllTime, "
    "likes, trendingScore, tags, cardData, safetensors, gguf, baseModels, "
    "pipeline_tag, library_name"
)


# ---------------------------------------------------------------------------
# Slug + normalization helpers (mirror the resolver/promotion script style)
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
    """Convert an HF id like `'meta-llama/Llama-3.1-70B'` into our
    canonical form `('meta/llama-3.1-70b', 'meta')`. The HF org segment
    maps via `org_alias_map`; unknown orgs slug-normalize their author
    name and become their own org id (auto-org).

    Ids without `/` use `unknown` as the org placeholder — matches the
    resolver's convention for raw values without a dev prefix.
    """
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
    """Canonical id = the real HF repo id verbatim; org_id = the canonical
    parent. The id and the org are DECOUPLED — the id is the model's true HF
    identity (a consumer can build `huggingface.co/{id}`), while `org_id`
    carries developer grouping. The HF org spelling is never folded into the
    id:

      - id  = `{HF-ORG}/{HF-NAME}` verbatim (e.g. `Qwen/Qwen2-7B-Instruct`);
      - org_id = the curated parent if the HF org maps to one
        (meta-llama->meta, qwen->alibaba, facebook->meta, ...), else the
        HF org itself.

    `hf_to_dev` keys are LOWERCASE HF org spellings (built single-sourced from
    `canonical_orgs.hf_org` + `_ORG_ALIASES`). Ids without `/` fall back to
    `unknown/{HF-NAME}`.
    """
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
    """An HF model row in hub-stats has open weights iff it carries
    safetensors and/or gguf data — the formats users actually download.
    Models without either are typically inference-only API endpoints
    (rare in the HF index) or deleted/empty repos."""
    if row.get("safetensors"):
        return True
    gguf = row.get("gguf")
    return bool(gguf)


def approx_params_billions(safetensors) -> Optional[float]:
    """Parameter count in billions from safetensors metadata.

    hub-stats' `safetensors.total` is the total PARAMETER COUNT (not bytes) —
    e.g. `meta-llama/Llama-3.1-8B` -> 8_030_261_248, `…-70B` -> 70_553_706_496
    (verified against the hub-stats parquet). So billions = total / 1e9.
    (A `/2` bytes-at-BF16 assumption would be wrong for this schema and
    would produce half the true count.)"""
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
    """Pick only tags worth surfacing on the canonical: language codes,
    eval-results, license markers, format markers. Drop noisy
    library/region/pipeline tags."""
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
# Hub-stats `baseModels` records *upstream* lineage (finetune / quantized /
# merge / adapter), never the family-version relationship between a dated
# snapshot and its moving pointer canonical (`Olmo-3-1125-32B` ↔ our
# `allenai/olmo-3-32b`). The pointer isn't an HF id — it only exists in our
# registry — so HF can't surface that edge. Without inference here, dated
# snapshots auto-create as orphaned canonicals: `release_date` lands fine
# but `parents`/`model_group_id` stay empty, root-collapse never fires, and
# the snapshot shows up as a separate model in consumers.

_INTERNAL_DATE_RE = re.compile(r"^(.+?)-(\d{4})-([^-].*)$")
_TRAILING_4DIGIT_RE = re.compile(r"^(.+)-(\d{4})$")
_TRAILING_6DIGIT_RE = re.compile(r"^(.+)-(\d{6})$")
_TRAILING_8DIGIT_RE = re.compile(r"^(.+)-(\d{8})$")
# ISO date patterns (anchored, full-string). Strict component widths
# stop us from peeling tokens that aren't dates (a 5-digit numeric tail
# won't match `\d{4}-\d{2}`).
_ISO_FULL_DATE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH_DATE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})$")
_ISO_YEAR_DATE_RE = re.compile(r"^(.+)-(\d{4})$")

# Plausible release-year window; guards against 4-digit tails (param counts,
# batch numbers) being mis-read as a release year.
_VALID_YEAR_RANGE = (2015, 2035)


def _looks_like_mmdd(token: str) -> bool:
    """4-digit MMDD where MM ∈ [01,12] and DD ∈ [01,31]. Used to gate
    snapshot-token stripping on shapes that actually look like dates,
    avoiding false-positives on numeric size/version tokens like `8000`."""
    if len(token) != 4 or not token.isdigit():
        return False
    mm, dd = int(token[:2]), int(token[2:])
    return 1 <= mm <= 12 and 1 <= dd <= 31


def _looks_like_yyyymm(token: str) -> bool:
    """6-digit YYYYMM (year+month). Stepfun and several Chinese-lab
    release tags use this convention, e.g. `step-2-16k-202411`."""
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
    """Detect snapshot-shape ids whose stripped form matches an existing
    canonical, and return a `{id, relationship: variant, axis: version}`
    edge pointing at it. Returns None when the id has no snapshot shape
    or the stripped form doesn't match any known canonical/alias.

    Patterns recognized (single-pass strip — does NOT compose with
    mode/quant suffix stripping):
      - internal MMDD token: `Olmo-3-1125-32B` → `Olmo-3-32B`
        also `Olmo-3-1125-7B-Instruct` → `Olmo-3-7B-Instruct`
      - trailing MMDD token: `kimi-k2-0905` → `kimi-k2`
      - trailing YYYYMM token: `step-2-16k-202411` → `step-2-16k`
      - trailing YYYYMMDD: `claude-haiku-4-5-20251001` → `claude-haiku-4-5`
      - trailing ISO date ladder: `gpt-5-2025-08-07` →
        `gpt-5-2025-08` → `gpt-5-2025` → `gpt-5`

    Only fires when the candidate stripped form resolves through the
    alias index — no false matches manufactured by stripping alone.
    For compound mode+date inputs (`claude-4-5-thinking-20251001`), the
    strip resolves to the mode-promoted canonical iff one exists; if
    not, returns None (the snapshot still gets `release_date` from
    hub-stats but lands without a parent edge).

    `target_canonical` is the canonical id the inferred edge will be
    attached to. When provided, suppresses self-edges (matters in the
    bulk-refresh path where an HF id may be aliased directly to its
    family pointer rather than a separate snapshot canonical — without
    this guard the family pointer gains a parent edge to itself,
    breaking the lineage walker). Live auto-create can also pass the
    proposed draft id; it just makes the guard tighter.
    """
    candidates: list[str] = []

    # Internal MMDD: `Olmo-3-1125-32B` shape. Tries first because
    # internal-token strips give a more specific lookup target than
    # trailing-token strips.
    m = _INTERNAL_DATE_RE.match(hf_id)
    if m and _looks_like_mmdd(m.group(2)):
        prefix, _, suffix = m.groups()
        candidates.append(f"{prefix}-{suffix}")

    # ISO ladder (full → month → year). The three regexes match
    # mutually exclusive tail shapes (`-YYYY-MM-DD` vs `-YYYY-MM` vs
    # `-YYYY`), so each input fires at most one branch.
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
    """Decode the `baseModels` struct into a list of typed parent edges.
    Returns `[{id, relationship}, ...]` — caller resolves each id to our
    canonical via the alias index. Empty list when no baseModels."""
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
    """Single-id lookup against the hub-stats parquet via DuckDB.

    Materializes the remote parquet into a local DuckDB table on first
    use, then serves all subsequent lookups against the local copy.
    Trade-off: one bulk fetch (large but a single HTTP transaction)
    instead of N per-id range-fetch queries (small but rate-limit-prone
    when N is in the thousands — `huggingface.co` 429s aggressively on
    the parquet API even with auth, and DuckDB's range-fetch pattern
    counts as multiple HTTP HEAD/GET requests per logical query).

    On bulk-fetch failure, falls back to per-id remote queries (older
    behaviour) so the client degrades gracefully rather than going
    silent. Per-id fallback retains the same in-process cache.

    Failure mode: any DuckDB error (network down, parquet schema drift,
    etc.) returns None from `lookup()` and is logged. Callers must
    handle None as "no enrichment data available" — never raise."""

    def __init__(self, parquet_url: str = PARQUET_URL) -> None:
        # Resolve the offline local-parquet path here (at construction) AND
        # again at query time — the env-aware `resolve_parquet_source` makes the
        # offline switch transparent to all the `read_parquet('{self.parquet_url}')`
        # call sites below without touching their SQL.
        self.parquet_url = resolve_parquet_source(parquet_url)
        self._con = None
        self._local_table_ready: bool = False
        self._local_table_failed: bool = False
        self._cache: dict[str, Optional[dict]] = {}
        self._lock = threading.Lock()

    def _ensure_con(self):
        if self._con is not None:
            return self._con
        # Import lazily so processes that never call lookup() don't pay
        # the duckdb import cost.
        import os
        import duckdb
        con = duckdb.connect()
        # Reading a LOCAL parquet needs no httpfs / HF auth — skip the network
        # extension install so the offline path works with no connectivity.
        if is_local_parquet():
            self._con = con
            return con
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # Authenticate parquet fetches when HF_TOKEN is in the environment
        # (typical on the deployed Space). Unauth limit is 500 req/5min;
        # one DuckDB read_parquet against the remote file streams via
        # several range requests and a sync that auto-creates many drafts
        # can brush that ceiling. With auth the ceiling is ~30k/5min.
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            escaped = hf_token.replace("'", "''")
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, BEARER_TOKEN '{escaped}', "
                f"SCOPE 'https://huggingface.co');"
            )
        self._con = con
        return con

    def _ensure_local_table(self, con) -> bool:
        """Materialize the remote parquet into a local DuckDB table. Returns
        True when the table is queryable, False when the bulk fetch failed
        (caller should fall through to per-id remote query). Idempotent.

        Caller must pass an already-prepared connection so this method
        doesn't double-count `_ensure_con` invocations under tests that
        patch the latter as a call counter."""
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
            # `id` is included implicitly via QUERY_COLUMNS. Materializing
            # only the columns we need keeps the local table small enough
            # to live in memory comfortably.
            con.execute(
                f"CREATE TABLE hub_stats AS "
                f"SELECT {QUERY_COLUMNS} FROM read_parquet('{self.parquet_url}')"
            )
            # Indexed lookup — DuckDB handles a single-column equality
            # filter on a string column efficiently without an explicit
            # index, but adding one explicitly costs ~ms and makes the
            # plan unambiguous to anyone reading EXPLAIN later.
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
        """Query hub-stats for one HF id. Returns the row as a dict, or
        None if not found / on any error. Threadsafe + cached.

        Prefers the locally-materialized table (one bulk fetch upfront);
        falls back to a per-id remote query if the bulk fetch failed."""
        with self._lock:
            if hf_id in self._cache:
                return self._cache[hf_id]
        try:
            con = self._ensure_con()
            use_local = self._ensure_local_table(con)
            # Case-insensitive match — HF stores ids with the upstream
            # author's original casing (`allenai/Olmo-3-1125-32B`); EEE
            # surfaces values in mixed conventions (some leaderboards
            # lowercase, some preserve). An exact-case `=` filter
            # silently misses any casing mismatch and the draft lands
            # without enrichment metadata. LOWER() forces a match
            # regardless of the surface form.
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
            # Log-and-continue: live lookup is best-effort enrichment,
            # never blocks draft creation.
            import logging
            logging.getLogger(__name__).warning(
                "hub-stats lookup failed for %r: %s", hf_id, e
            )
            result = None
        with self._lock:
            self._cache[hf_id] = result
        return result


# ---------------------------------------------------------------------------
# High-level enrichment: take a hub-stats row + the registry's alias index
# and return a dict of fields ready to merge into a draft canonical_models row.
# ---------------------------------------------------------------------------

def enrich_draft_from_row(
    row: dict,
    aliases_to_canonical: dict[str, str],
    org_alias_map: dict[str, str],
    target_canonical: Optional[str] = None,
) -> dict:
    """Convert one hub-stats row into a partial canonical_models dict
    suitable for merging into an auto-created draft. Computes:
      - release_date (from createdAt)
      - params_billions (approx from safetensors)
      - tags (filtered list)
      - parents (resolved against our canonical alias index when possible)
      - lineage_origin_model_org_id (= upstream lab when a parent edge resolves)
      - metadata (license, downloads, etc.)

    Caller decides what to actually write — none of these fields are
    forced. Returns an empty dict if the row carries no usable info.
    """
    out: dict = {}

    release = coerce_date(row.get("createdAt"))
    if release:
        out["release_date"] = release

    params = approx_params_billions(row.get("safetensors"))
    if params is not None:
        out["params_billions"] = params

    # Anything we can pull from hub-stats with downloadable artifacts is
    # by definition open weights. Closed-API models aren't on HF.
    if has_downloadable_weights(row):
        out["open_weights"] = True

    useful_tags = filter_useful_tags(row.get("tags"))
    if useful_tags:
        out["tags"] = json.dumps(useful_tags)

    # Resolve baseModels parents to OUR canonical ids when possible.
    # Drop edges we can't resolve — a parents edge pointing at an HF id
    # we don't track would dangle.
    parents: list[dict] = []
    lineage_origin_org_id: Optional[str] = None
    for edge in extract_base_models(row.get("baseModels")):
        base_hf = edge["id"]
        n = normalize(base_hf)
        parent_canonical = aliases_to_canonical.get(n)
        if parent_canonical is None:
            # Try slugifying via org map (covers HF ids that aren't yet
            # in our aliases but whose org we know).
            slugified, _ = hf_id_to_canonical(base_hf, org_alias_map)
            parent_canonical = aliases_to_canonical.get(normalize(slugified))
        if parent_canonical:
            parents.append({"id": parent_canonical, "relationship": edge["relationship"]})
            # First resolved non-variant edge sets the lineage origin
            # (matches `derive_model_lineage_fields` semantics).
            if lineage_origin_org_id is None and edge["relationship"] != "variant":
                if "/" in parent_canonical:
                    # The parent canonical id is the real HF repo id, so its
                    # prefix is the HF org — fold it to the canonical developer
                    # (meta-llama -> meta) for the lineage-origin ORG field.
                    _pp = parent_canonical.split("/", 1)[0]
                    lineage_origin_org_id = org_alias_map.get(_pp.lower(), _pp)

    # Family-version inference: hub-stats `baseModels` only records
    # upstream-lineage edges (finetune/quantized/merge/adapter), never
    # the dated-snapshot ↔ moving-pointer relationship that lives only
    # in our registry. Without this, snapshots like `Olmo-3-1125-32B`
    # auto-create as orphan canonicals — release_date lands but parents
    # stays empty and root-collapse never fires.
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
        # Output key matches the renamed canonical_models column so it lands
        # in the right column when merged into an auto-created draft.
        out["lineage_origin_model_org_id"] = lineage_origin_org_id

    # Stash extra hub-stats context in metadata so consumers can find it
    # without re-querying.
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

    # Promote the HF-true repo id as a first-class key UNCONDITIONALLY
    # (bypasses the `len(metadata) > 2` guard that otherwise drops it for
    # sparse rows). `_auto_create_entity` reads this to mint the canonical id
    # + display_name in HF-true casing instead of a lowercased slug. Without
    # it, HF-confirmed models land under a stale lowercase id.
    if isinstance(hf_id, str) and hf_id.strip():
        out["hf_id"] = hf_id

    return out
