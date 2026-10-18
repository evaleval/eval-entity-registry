#!/usr/bin/env python3
"""
Freeze the candidate subset of cfahlgren1/hub-stats into a committed, offline
parquet cache so the lineage/enrichment layer regenerates REPRODUCIBLY without
hitting live HuggingFace.

Why this exists: ~2000+ HF-bulk rows' `parents` / `model_group_id` /
`model_family_id` / `lineage_origin` are derived (in `enrich_draft_from_row`)
from the parquet's `baseModels` column. Live HF queries are flaky and can
mis-classify real repos as absent, so a clean regen against live HF would
silently lose lineage. Freezing the actual upstream rows (incl. baseModels) to a
committed parquet lets `refresh_from_hub_stats.py` reproduce the SAME enrichment
offline, byte-for-byte, via its existing local-parquet path
(`HUB_STATS_LOCAL_PARQUET`).

This is a CACHE-REFRESH tool — run it occasionally (or in a dedicated cron) to
re-pull the upstream rows for our covered canonicals. It is NETWORK; the normal
`refresh_from_hub_stats.py` regen reads the frozen cache offline.

Source: a pinned, downloaded snapshot of `main/models.parquet`
(see `services/hub_stats_snapshot.py`). Its commit sha and content sha256 are
written into the cache's parquet key-value metadata.

Output:
    curation/hub_stats_frozen.parquet  (committed; the durable offline cache)

Usage:
    HF_TOKEN=... uv run python scripts/freeze_hub_stats_cache.py
    uv run python scripts/freeze_hub_stats_cache.py --out path/to.parquet
    HUB_STATS_LOCAL_PARQUET=path uv run python scripts/freeze_hub_stats_cache.py   # from a local file
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

from eval_entity_registry.lib.seed_io import build_hf_to_dev_from_orgs_yaml, load_entries_from_yaml
from eval_entity_registry.services.hub_stats import HUB_STATS_LOCAL_PARQUET_ENV, QUERY_COLUMNS
from eval_entity_registry.services.hub_stats_snapshot import (
    HubStatsSnapshot,
    download_snapshot,
    is_remote_source,
    resolve_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ORGS_PATH = REPO_ROOT / "seed" / "orgs.yaml"
FROZEN_CACHE_PATH = REPO_ROOT / "curation" / "hub_stats_frozen.parquet"

# Same model sources refresh_from_hub_stats.py reads candidates from.
_MODEL_SOURCES = (
    REPO_ROOT / "seed" / "models" / "core.yaml",
    REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml",
    REPO_ROOT / "seed" / "models" / "sources" / "hf_oracle.generated.yaml",
    REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml",
    REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml",
    REPO_ROOT / "seed" / "models" / "sources" / "tier3_inferred.generated.yaml",
)


def build_candidate_ids() -> set[str]:
    """Every HF-shaped (`org/name`) id + alias on a known canonical, plus the
    big-dev re-mapped repo id (`meta/Llama-3.1-8B` -> `meta-llama/Llama-3.1-8B`)
    so the parquet lookup finds the real-namespace row. Mirrors the candidate
    construction in refresh_from_hub_stats.main()."""
    hf_to_dev = build_hf_to_dev_from_orgs_yaml(ORGS_PATH)
    dev_to_hf_orgs: dict[str, set[str]] = {}
    for hf_org, dev in hf_to_dev.items():
        dev_to_hf_orgs.setdefault(dev, set()).add(hf_org)

    initial: set[str] = set()
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            if not isinstance(e, dict):
                continue
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    initial.add(a)
            cid = e.get("id")
            if isinstance(cid, str) and "/" in cid:
                initial.add(cid)
                org_part, name_part = cid.split("/", 1)
                for hf_org in dev_to_hf_orgs.get(org_part, ()):
                    initial.add(f"{hf_org}/{name_part}")
    return initial


def select_source(token: str | None) -> tuple[str, HubStatsSnapshot | None]:
    """Existing `$HUB_STATS_LOCAL_PARQUET` > pinned snapshot download."""
    local = os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
    if local and os.path.exists(local):
        return local, None
    snap = resolve_snapshot(token=token)
    print(f"[freeze] snapshot {snap.short}", file=sys.stderr)
    snap = download_snapshot(snap, token=token)
    return str(snap.local_path), snap


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=FROZEN_CACHE_PATH)
    args = p.parse_args()

    candidates = build_candidate_ids()
    print(f"[freeze] candidate HF ids: {len(candidates)}", file=sys.stderr)
    if not candidates:
        print("[freeze] ERROR: no candidates found — seed sources empty?", file=sys.stderr)
        return 1

    tok = os.environ.get("HF_TOKEN")
    src, snap = select_source(tok)

    con = duckdb.connect()
    if is_remote_source(src):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        if tok:
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, BEARER_TOKEN '{tok.replace(chr(39), chr(39)*2)}', "
                f"SCOPE 'https://huggingface.co');"
            )
    quoted = ", ".join(f"'{i.replace(chr(39), chr(39) * 2)}'" for i in candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if snap is not None:
        kv = {"source_repo": snap.repo_id, "source_revision": snap.revision, "source_sha256": snap.sha256}
    else:
        kv = {"source_path": src}
    kv_sql = ", ".join(f"'{k}': '{v.replace(chr(39), chr(39) * 2)}'" for k, v in kv.items())
    # COPY the candidate subset (case-sensitive id match) to a local parquet.
    con.execute(
        f"COPY (SELECT {QUERY_COLUMNS} FROM read_parquet('{src.replace(chr(39), chr(39) * 2)}') "
        f"WHERE id IN ({quoted})) TO '{args.out}' (FORMAT PARQUET, KV_METADATA {{{kv_sql}}})"
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{args.out}')").fetchone()[0]
    print(f"[freeze] wrote {n} rows -> {args.out}", file=sys.stderr)
    if n == 0:
        print("[freeze] ERROR: 0 rows matched — aborting (no cache written).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
