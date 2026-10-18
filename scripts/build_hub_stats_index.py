#!/usr/bin/env python3
"""
Build the local hub-stats id INDEX consulted by the read-only resolve path.

The registry resolves raw model strings to canonical entities. For an HF model
id that was never minted into the registry, the read-only API still wants to
CONFIRM that the exact id is a real HF repo (returning a `hub_stats_index`
confirmation in the standard ResolveResponse shape) — without minting or
persisting anything. This script materializes that index: one DuckDB `COPY`
over cfahlgren1/hub-stats, filtered to repos with downloadable weights.

Source: a pinned, downloaded snapshot of `main/models.parquet`
(see `services/hub_stats_snapshot.py`).

Output:
    fixtures/hub_stats_index.parquet  (single file; published to the dataset's
    `hub_stats_index/part-0.parquet` subdir by the refresh-hub-stats-index cron)

Columns (mirror store/schemas.py `hub_stats_index`):
    id (HF-true), id_norm, release_date, pipeline_tag, params_billions,
    open_weights (always true), downloads

`id_norm` is computed in SQL to mirror `services.hub_stats.normalize` EXACTLY:
    lower(id) -> collapse runs of [/_.:-] to a single '-' -> trim leading/
    trailing '-'.

Usage:
    HF_TOKEN=... uv run python scripts/build_hub_stats_index.py        # pinned snapshot download
    HUB_STATS_LOCAL_PARQUET=path uv run python scripts/build_hub_stats_index.py --limit 50  # offline test
    uv run python scripts/build_hub_stats_index.py --source path/to.parquet --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

from eval_entity_registry.services.hub_stats import HUB_STATS_LOCAL_PARQUET_ENV
from eval_entity_registry.services.hub_stats_snapshot import (
    download_snapshot,
    is_remote_source,
    resolve_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "fixtures" / "hub_stats_index.parquet"

# A full pull must not shrink the live index — abort below this floor without
# writing. The full safetensors/gguf-filtered set is well over 500k rows.
ROW_FLOOR = 500_000

# id_norm mirrors services.hub_stats.normalize:
#   re.sub(r"[/_.:\-]+", "-", s.lower()).strip("-")
# DuckDB equivalent: lower -> regexp_replace collapses any run of separator
# chars [/_.:-] (mixed runs included) to a single dash -> trim leading/trailing
# dashes. Equivalence is asserted by tests/test_hub_stats_index_confirm.py.
_ID_NORM_SQL = "trim(regexp_replace(lower(id), '[/_.:-]+', '-', 'g'), '-')"


def build_query(source: str | list[str], limit: int | None) -> str:
    """`source` is one parquet path/URL, or a list of them (read as one table)."""
    if isinstance(source, str):
        src = "'" + source.replace("'", "''") + "'"
    else:
        src = "[" + ", ".join("'" + s.replace("'", "''") + "'" for s in source) + "]"
    body = f"""
    SELECT
        id,
        {_ID_NORM_SQL} AS id_norm,
        substr(CAST(createdAt AS VARCHAR), 1, 10) AS release_date,
        pipeline_tag,
        CAST(safetensors.total AS DOUBLE) / 1e9 AS params_billions,
        CAST(downloads AS BIGINT) AS downloads,
        true AS open_weights
    FROM read_parquet({src})
    WHERE safetensors IS NOT NULL OR gguf IS NOT NULL
"""
    if limit is not None:
        body = f"{body}\n    LIMIT {int(limit)}"
    return body


def select_source(explicit: str | None, token: str | None) -> str:
    """`--source` > existing `$HUB_STATS_LOCAL_PARQUET` > pinned snapshot download."""
    if explicit:
        return explicit
    local = os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
    if local and os.path.exists(local):
        return local
    snap = resolve_snapshot(token=token)
    print(f"[index] snapshot {snap.short}", file=sys.stderr)
    snap = download_snapshot(snap, token=token)
    return str(snap.local_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override parquet source (offline path or URL). Defaults to "
        "$HUB_STATS_LOCAL_PARQUET if set, else a pinned snapshot download.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows (for offline testing only; bypasses the row floor).",
    )
    args = p.parse_args()

    tok = os.environ.get("HF_TOKEN")
    src = select_source(args.source, tok)

    con = duckdb.connect()
    if is_remote_source(src):  # only when --source / the env var is a URL
        con.execute("INSTALL httpfs; LOAD httpfs;")
        if tok:
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, "
                f"BEARER_TOKEN '{tok.replace(chr(39), chr(39) * 2)}', "
                f"SCOPE 'https://huggingface.co');"
            )
    print(f"[index] reading {src}", file=sys.stderr)

    query = build_query(src, args.limit)

    # Count first so the floor check can abort WITHOUT writing a partial index.
    n = con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
    print(f"[index] rows matched: {n}", file=sys.stderr)
    if args.limit is None and n < ROW_FLOOR:
        print(
            f"[index] ERROR: {n} rows < floor {ROW_FLOOR} — a partial pull must "
            f"not shrink the live index. Aborting (no file written).",
            file=sys.stderr,
        )
        return 1
    if n == 0:
        print("[index] ERROR: 0 rows matched — aborting (no file written).", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{args.out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    written = con.execute(f"SELECT count(*) FROM read_parquet('{args.out}')").fetchone()[0]
    print(f"[index] wrote {written} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
