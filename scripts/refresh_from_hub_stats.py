#!/usr/bin/env python3
"""
Generate seed/models/sources/hub_stats.generated.yaml from
HuggingFace's `cfahlgren1/hub-stats` dataset.

Backfill only: for HF ids (`org/name`) already aliased to one of our
canonicals, emit an enrichment entry (`release_date` from `createdAt`,
`params_billions` from the safetensors total, license, useful tags, `parents`
from `baseModels`) that the seed loader's field-level merge fills in without
overriding curated values in `seed/models/core.yaml`. Lineage descendants of
covered models are not pre-loaded; the live hub-stats lookup enriches those
drafts on demand.

Source: by default the committed offline cache
`curation/hub_stats_frozen.parquet` (refreshed by
`scripts/freeze_hub_stats_cache.py`), so a clean regen is reproducible;
`--live` reads a pinned snapshot of `main/models.parquet` instead (see
`services/hub_stats_snapshot.py`). On the live path a watermark in
`seed/models/sources/hub_stats.state.json` (upstream content sha256 + the
candidate ids checked against it) skips the download when nothing changed.

Output:
    seed/models/sources/hub_stats.generated.yaml — enrichment entries
    that merge into existing canonicals at seed time.
    seed/models/sources/hub_stats.state.json     — live-path watermark.

Usage:
    python scripts/refresh_from_hub_stats.py             # offline cache (reproducible)
    python scripts/refresh_from_hub_stats.py --live      # pinned upstream snapshot
    python scripts/refresh_from_hub_stats.py --dry-run   # preview only
    python scripts/refresh_from_hub_stats.py --force     # ignore watermark (live only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import duckdb
import yaml

from eval_entity_registry.lib.seed_io import build_hf_to_dev_from_orgs_yaml, load_entries_from_yaml

from eval_entity_registry.services.hub_stats import (
    HUB_STATS_LOCAL_PARQUET_ENV,
    QUERY_COLUMNS,
    approx_params_billions as _approx_params_billions,
    coerce_date as _coerce_date,
    enrich_draft_from_row,
    extract_license as _extract_license,
    filter_useful_tags as _filter_useful_tags,
    hf_id_to_canonical_cased,
    normalize as _normalize,
    slugify as _slugify,
)
from eval_entity_registry.services.hub_stats_snapshot import (
    HubStatsSnapshot,
    download_snapshot,
    is_remote_source,
    resolve_snapshot,
)

from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REPO_ROOT = Path(__file__).resolve().parent.parent
ORGS_PATH = REPO_ROOT / "seed" / "orgs.yaml"
MODELS_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
STATE_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.state.json"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
MODELS_DEV_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"
HF_ORACLE_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "hf_oracle.generated.yaml"
MODELS_DEV_CATALOG_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml"

# Offline cache frozen by scripts/freeze_hub_stats_cache.py; the default source.
FROZEN_CACHE_PATH = REPO_ROOT / "curation" / "hub_stats_frozen.parquet"

_MODEL_SOURCES = (
    CORE_PATH,
    MODELS_DEV_GENERATED,
    HF_ORACLE_GENERATED,
    MODELS_DEV_CATALOG_GENERATED,
)


def build_hf_to_dev() -> dict[str, str]:
    """HF-org-lowercase -> developer slug; see seed_io.build_hf_to_dev_from_orgs_yaml."""
    return build_hf_to_dev_from_orgs_yaml(ORGS_PATH)


# ---------------------------------------------------------------------------
# Snapshot watermark — short-circuits the daily cron on no-op cycles.
# ---------------------------------------------------------------------------

STATE_SHA_KEY = "parquet_sha256"  # upstream file's LFS content sha256
STATE_ROWS_KEY = "rows_checked_at_sha256"


def load_state() -> dict:
    """The watermark state, or {} (re-check everything) when the file is
    missing, unreadable, or schema-invalid."""
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text())
        if not isinstance(data, dict):
            return {}
        if not isinstance(data.get(STATE_SHA_KEY), str):
            return {}
        if not isinstance(data.get(STATE_ROWS_KEY), list):
            return {}
        return data
    except (OSError, ValueError):
        return {}


def write_state(sha256: str, rows_checked: list[str]) -> None:
    """Write the watermark (sorted for diff-clean commits). Must run after the
    YAML write succeeds, so a crash never marks rows checked whose data did
    not land."""
    payload = {
        STATE_SHA_KEY: sha256,
        STATE_ROWS_KEY: sorted(set(rows_checked)),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Registry index that gates which hub-stats rows are looked up.
# ---------------------------------------------------------------------------

def load_org_alias_to_canonical() -> dict[str, str]:
    """{normalized HF org alias -> canonical org id}, e.g. `'meta-llama' -> 'meta'`."""
    with open(ORGS_PATH) as f:
        orgs = yaml.safe_load(f) or []
    out: dict[str, str] = {}
    for o in orgs:
        cid = o.get("id")
        if not cid:
            continue
        out[_normalize(cid)] = cid
        if o.get("hf_org"):
            out[_normalize(o["hf_org"])] = cid
        for a in (o.get("aliases") or []):
            if isinstance(a, str):
                out[_normalize(a)] = cid
    return out


def load_existing_canonical_aliases() -> dict[str, str]:
    """{normalized HF-shaped surface form -> canonical id} over every model
    source's ids and `/`-containing aliases."""
    out: dict[str, str] = {}

    # A canonical's own id outranks another canonical's alias for the same
    # normalized form, so ids are registered first.
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if cid:
                out.setdefault(_normalize(cid), cid)
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if not cid:
                continue
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    out.setdefault(_normalize(a), cid)
    return out


def load_ambiguous_canonicals() -> set[str]:
    """Canonical ids whose normalized id/name/alias form is also claimed by a
    different canonical. Enriching one of such a near-duplicate pair can flip
    the resolver's normalized-match tie-break, so they are skipped until
    curation de-dups the pair."""
    norm_owners: dict[str, set[str]] = {}

    def _claim(form: str, cid: str) -> None:
        if isinstance(form, str) and form:
            norm_owners.setdefault(_normalize(form), set()).add(cid)

    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            _claim(cid, cid)
            if "/" in cid:
                _claim(cid.split("/", 1)[1], cid)
            for a in (e.get("aliases") or []):
                _claim(a, cid)
    ambiguous: set[str] = set()
    for owners in norm_owners.values():
        if len(owners) > 1:
            ambiguous.update(owners)
    return ambiguous


def load_canonical_id_norms() -> dict[str, str]:
    """{normalized full id or bare name -> canonical id} for every canonical.
    An enrichment alias colliding with a different canonical's id or name is
    a double claim the seed validator rejects, and one that lets normalized
    match steal that canonical's raw id."""
    out: dict[str, str] = {}
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            out.setdefault(_normalize(cid), cid)
            if "/" in cid:
                out.setdefault(_normalize(cid.split("/", 1)[1]), cid)
    return out


def query_hub_stats(
    con: duckdb.DuckDBPyConnection,
    candidate_hf_ids: set[str],
    source: str,
) -> list[dict]:
    """Hub-stats rows whose `id` is in `candidate_hf_ids`, from the parquet at
    `source` (a local path, or a URL on a connection with httpfs loaded)."""
    if not candidate_hf_ids:
        return []
    quoted = ", ".join(f"'{i.replace(chr(39), chr(39)*2)}'" for i in candidate_hf_ids)
    source = source.replace("'", "''")
    sql = f"""
        SELECT {QUERY_COLUMNS}
        FROM read_parquet('{source}')
        WHERE id IN ({quoted})
    """
    cursor = con.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def build_entry(
    row: dict,
    org_alias_map: dict[str, str],
    aliases_to_canonical: dict[str, str],
    hf_to_dev: Optional[dict[str, str]] = None,
    canonical_id_norms: Optional[dict[str, str]] = None,
    ambiguous_canonicals: Optional[set[str]] = None,
) -> Optional[dict]:
    """Seed entry for a hub-stats row, or None when the row's HF id is not
    aliased to a covered canonical.

    All hub-stats-derived fields come from `enrich_draft_from_row` so this
    script and the live-lookup path extract identically; the loader unions
    `parents` by id, so generated parents compose with curated ones.
    """
    hf_id = row["id"]
    # HF casing is preserved so the id matches what the oracle generator and
    # live auto-create mint; the gate is on the normalized form.
    if hf_to_dev is None:
        hf_to_dev = build_hf_to_dev()
    canonical_id, _ = hf_id_to_canonical_cased(hf_id, hf_to_dev)
    norm_canon = _normalize(canonical_id)
    if norm_canon not in aliases_to_canonical:
        return None
    # The registry's own spelling of the canonical id.
    canonical_id = aliases_to_canonical[norm_canon]

    if ambiguous_canonicals and canonical_id in ambiguous_canonicals:
        return None

    cid_norms = canonical_id_norms or {}

    # Only the raw HF repo id is a candidate alias: normalized match already
    # collapses case/separators, and a fabricated lowercase slug tends to
    # collide with another canonical's alias. Dropped when another canonical
    # owns the exact string or its normalized form.
    def _alias_ok(a: str) -> bool:
        if a == canonical_id:
            return False
        n = _normalize(a)
        if aliases_to_canonical.get(n, canonical_id) != canonical_id:
            return False
        if cid_norms.get(n, canonical_id) != canonical_id:
            return False
        if "/" in a:
            name_n = _normalize(a.split("/", 1)[1])
            if cid_norms.get(name_n, canonical_id) != canonical_id:
                return False
        return True

    aliases = sorted(a for a in {hf_id} if _alias_ok(a))

    # target_canonical lets enrich_draft_from_row suppress a self-edge when the
    # HF id is aliased directly to its family pointer.
    # org_alias_map lacks some curated HF prefixes (e.g. ``CohereLabs``); union
    # the full map so this cron and the models.dev reconcile emit the same
    # lineage-origin org ids instead of rewriting each other's rows.
    lineage_org_map = {**org_alias_map, **hf_to_dev}
    enrichment = enrich_draft_from_row(
        row, aliases_to_canonical, lineage_org_map,
        target_canonical=canonical_id,
    )

    # `metadata.hf_id` asserts the canonical IS that HF repo. When the
    # enrichment lands on a differently spelled canonical and the real id is
    # not itself a canonical, the claim would trip the canonical-id-equals-hf-id
    # gate; the alias still resolves, so drop only the claim.
    if "metadata" in enrichment and hf_id != canonical_id:
        if aliases_to_canonical.get(_normalize(hf_id)) != hf_id:
            try:
                meta = json.loads(enrichment["metadata"])
            except (ValueError, TypeError):
                meta = None
            if isinstance(meta, dict) and "hf_id" in meta:
                meta.pop("hf_id")
                enrichment["metadata"] = json.dumps(meta, sort_keys=True)

    # List-form tags keep the generated YAML diff reviewable; the loader
    # accepts either form.
    if "tags" in enrichment:
        try:
            enrichment["tags"] = json.loads(enrichment["tags"])
        except (ValueError, TypeError):
            pass

    # No display_name: the owning source already carries the authoritative
    # one, and a humanized bare name would collide across orgs as a global
    # alias. No org_id: the seed-time org-from-prefix rule derives it, and a
    # stamped raw HF prefix overrides deliberately org-less canonicals and
    # leaves dangling org FKs.
    entry: dict = {
        "id": canonical_id,
    }
    if "release_date" in enrichment:
        entry["release_date"] = enrichment["release_date"]
    if "tags" in enrichment:
        entry["tags"] = enrichment["tags"]
    entry["aliases"] = [a for a in aliases if a != canonical_id]
    if "metadata" in enrichment:
        entry["metadata"] = enrichment["metadata"]
    entry["review_status"] = "reviewed"
    if "params_billions" in enrichment:
        entry["params_billions"] = enrichment["params_billions"]
    if "open_weights" in enrichment:
        entry["open_weights"] = enrichment["open_weights"]
    if "parents" in enrichment:
        entry["parents"] = json.loads(enrichment["parents"])
    if "lineage_origin_model_org_id" in enrichment:
        entry["lineage_origin_model_org_id"] = enrichment["lineage_origin_model_org_id"]
    return entry


def _carry_forward_committed(fresh: list[dict], committed: list[dict]) -> list[dict]:
    """Merge the committed file into the fresh rewrite so a model that fell out
    of the fetch (parquet shrank, repo gated/deleted) keeps its enrichment.

      * id in the fresh output: the fresh entry wins (clearing any prior
        `upstream_status: removed` tag);
      * id absent: the committed entry is retained, tagged
        `metadata.upstream_status: removed`, minus `org_id` (the seed-time
        org-from-prefix rule recovers it).

    Malformed committed metadata degrades to {} with a warning. Output is
    id-sorted."""
    fresh_ids = {e["id"] for e in fresh if e.get("id")}
    out = list(fresh)
    retained = 0
    for c in committed:
        cid = c.get("id")
        if not cid or cid in fresh_ids:
            continue
        rc = {k: v for k, v in c.items() if k != "org_id"}
        try:
            meta = json.loads(rc.get("metadata") or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            print(f"[refresh] WARN: malformed committed metadata on {cid}; treating as empty", file=sys.stderr)
            meta = {}
        meta["upstream_status"] = "removed"
        rc["metadata"] = json.dumps(meta, sort_keys=True)
        out.append(rc)
        retained += 1
    if retained:
        print(
            f"[refresh] carry-forward: retained {retained} committed entr(ies) "
            f"absent from today's fetch",
            file=sys.stderr,
        )
    return sorted(out, key=lambda e: e["id"])


# ---------------------------------------------------------------------------
# Output writing.
# ---------------------------------------------------------------------------

_HEADER = """# Generated from cfahlgren1/hub-stats — DO NOT EDIT BY HAND.
# To update: run `python scripts/refresh_from_hub_stats.py`.
#
# Source: https://huggingface.co/datasets/cfahlgren1/hub-stats (Apache-2.0)
# Last refresh date is in git history.
#
# Backfill-only: every entry here MERGES into an existing canonical_models
# row at seed time. Fills in release_date, params_billions, license, tags,
# etc. via the seed loader's field-level merge (curated values in
# core.yaml take precedence on conflict).
#
# Entries tagged `metadata.upstream_status: removed` are carry-forwards:
# their id fell out of the upstream fetch, so the committed enrichment is
# retained rather than deleted. A reappearing id replaces the carried entry.
#
# Lineage descendant pre-load (community quants/finetunes of our covered
# models) is not done here. EEE drafts get on-demand enrichment via the
# live hub-stats lookup at draft creation.
"""


def write_models_yaml(entries: list[dict]) -> None:
    MODELS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=200)
    MODELS_OUT_PATH.write_text(_HEADER + "\n" + body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print summary; don't write files")
    p.add_argument("--force", action="store_true", help="ignore the snapshot watermark; re-fetch unconditionally")
    p.add_argument(
        "--live",
        action="store_true",
        help="download a pinned snapshot of cfahlgren1/hub-stats main/models.parquet "
        "instead of reading the committed offline cache "
        "(curation/hub_stats_frozen.parquet). Use to refresh; the default is the "
        "reproducible offline cache. Refresh the cache itself with "
        "scripts/freeze_hub_stats_cache.py.",
    )
    args = p.parse_args()

    # The --live source is resolved after the candidate set is known so the
    # watermark can short-circuit before any download.
    snap: Optional[HubStatsSnapshot] = None
    source: Optional[str] = None
    if not args.live:
        local = os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
        if local and os.path.exists(local):
            source = local
            print(f"[refresh] reading local parquet {source}", file=sys.stderr)
        elif FROZEN_CACHE_PATH.exists():
            source = str(FROZEN_CACHE_PATH)
            print(
                f"[refresh] reading frozen offline cache {FROZEN_CACHE_PATH} "
                f"(pass --live for a fresh upstream snapshot)",
                file=sys.stderr,
            )
        else:
            print(
                f"[refresh] ERROR: no offline source — {FROZEN_CACHE_PATH} missing and "
                f"${HUB_STATS_LOCAL_PARQUET_ENV} unset; pass --live or freeze the cache.",
                file=sys.stderr,
            )
            return 1

    org_alias_map = load_org_alias_to_canonical()
    aliases_to_canonical = load_existing_canonical_aliases()
    canonical_id_norms = load_canonical_id_norms()
    ambiguous_canonicals = load_ambiguous_canonicals()
    hf_to_dev = build_hf_to_dev()
    if not org_alias_map:
        print("[refresh] ERROR: seed/orgs.yaml is empty or missing.", file=sys.stderr)
        return 1
    if not aliases_to_canonical:
        print("[refresh] ERROR: no canonical models found in seed/. Seed first.", file=sys.stderr)
        return 1

    # dev slug -> HF org spellings, to reconstruct the real HF repo id for a
    # big-dev canonical whose id prefix is the dev slug.
    dev_to_hf_orgs: dict[str, set[str]] = {}
    for hf_org, dev in hf_to_dev.items():
        dev_to_hf_orgs.setdefault(dev, set()).add(hf_org)

    # Candidates keep their original casing: the parquet `id` column is
    # case-sensitive.
    initial: set[str] = set()
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    initial.add(a)
            cid = e.get("id")
            if isinstance(cid, str) and "/" in cid:
                initial.add(cid)
                org_part, name_part = cid.split("/", 1)
                for hf_org in dev_to_hf_orgs.get(org_part, ()):
                    initial.add(f"{hf_org}/{name_part}")
    print(f"[refresh] HF-id candidates to look up: {len(initial)}", file=sys.stderr)

    # Short-circuit only when the content hash is unchanged, every candidate
    # was already checked, and the YAML still exists (a deleted YAML must
    # regenerate).
    hf_token = os.environ.get("HF_TOKEN")
    if args.live:
        snap = resolve_snapshot(token=hf_token)
        print(f"[refresh] snapshot {snap.short}", file=sys.stderr)
        state = load_state()
        can_short_circuit = (
            not args.force
            and not args.dry_run
            and state.get(STATE_SHA_KEY) == snap.sha256
            and MODELS_OUT_PATH.exists()
            and initial.issubset(set(state.get(STATE_ROWS_KEY, [])))
        )
        if can_short_circuit:
            print(
                f"[refresh] no-op: upstream sha256 unchanged ({snap.sha256[:12]}...) "
                f"and all {len(initial)} candidates already checked",
                file=sys.stderr,
            )
            return 0
        # A download failure fails the run; there is no remote fallback.
        snap = download_snapshot(snap, token=hf_token)
        source = str(snap.local_path)
        print(f"[refresh] downloaded snapshot -> {source}", file=sys.stderr)
    assert source is not None

    con = duckdb.connect()
    if is_remote_source(source):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        if hf_token:
            escaped = hf_token.replace("'", "''")
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, BEARER_TOKEN '{escaped}', "
                f"SCOPE 'https://huggingface.co');"
            )
    rows = query_hub_stats(con, initial, source)
    print(f"[refresh] hub-stats rows fetched: {len(rows)}", file=sys.stderr)

    entries: list[dict] = []
    for row in rows:
        e = build_entry(
            row, org_alias_map, aliases_to_canonical, hf_to_dev,
            canonical_id_norms, ambiguous_canonicals,
        )
        if e is not None:
            entries.append(e)
    entries.sort(key=lambda e: e["id"])

    # Several HF ids can map to one canonical (the fuzzy strip collapses
    # version suffixes): latest release_date wins and aliases are unioned so
    # every HF id stays addressable.
    by_id: dict[str, dict] = {}
    collisions = 0
    for e in entries:
        eid = e["id"]
        prev = by_id.get(eid)
        if prev is None:
            by_id[eid] = e
            continue
        collisions += 1
        prev_date = prev.get("release_date") or ""
        new_date = e.get("release_date") or ""
        winner = e if new_date > prev_date else prev
        loser = prev if winner is e else e
        merged_aliases = sorted(set(winner.get("aliases", [])) | set(loser.get("aliases", [])))
        winner["aliases"] = merged_aliases
        by_id[eid] = winner
    if collisions:
        print(
            f"[refresh] deduped {collisions} collision(s) by canonical id "
            f"(latest release_date wins; aliases unioned)",
            file=sys.stderr,
        )
    entries = sorted(by_id.values(), key=lambda e: e["id"])

    committed = (
        yaml.safe_load(MODELS_OUT_PATH.read_text()) or []
        if MODELS_OUT_PATH.exists() else []
    )
    entries = _carry_forward_committed(entries, committed)
    print(f"[refresh] enrichment entries: {len(entries)}", file=sys.stderr)

    if args.dry_run:
        print("[refresh] [dry-run] no files written")
        if entries:
            print("\nFirst 5 entries:")
            for e in entries[:5]:
                print(f"  {e['id']}  release_date={e.get('release_date')}  params={e.get('params_billions')}")
        return 0

    write_models_yaml(entries)
    print(f"[refresh] wrote {MODELS_OUT_PATH}", file=sys.stderr)

    if snap is not None:
        write_state(snap.sha256, sorted(initial))
        print(f"[refresh] wrote {STATE_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
