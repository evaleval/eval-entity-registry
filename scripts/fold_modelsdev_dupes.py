#!/usr/bin/env python3
"""Fold models.dev-minted duplicate canonicals into their real HF id.

For every `resolution_source: models_dev` canonical in seed/models/core.yaml,
decide whether a real HF id already exists for the same model. HF targets are
the frozen HF oracle's fixed_exact/fixed_near_miss ids plus the HF-true
canonicals already in core.yaml; the match policy (exact / alias / normalized /
fuzzy, never across different developers) is `eval_entity_resolver.fold.decide_fold`.

The dry run only reports. --apply, given confirmed mint ids, deletes each mint
entry, merges providers/release_date/open_weights onto the HF target, appends
the mint id + display_name as aliases on the target, and rewrites any
parents[].id edge pointing at a folded id to the HF target.

Usage:
    LOCAL_MODE=true uv run python scripts/fold_modelsdev_dupes.py            # dry run
    LOCAL_MODE=true uv run python scripts/fold_modelsdev_dupes.py --json     # dry run, JSON out
    LOCAL_MODE=true uv run python scripts/fold_modelsdev_dupes.py --apply --confirm <id> [<id> ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from eval_entity_registry.lib.seed_io import (
    build_hf_to_dev_from_orgs_yaml,
    resolve_oracle_path,
)

from eval_entity_resolver.normalization import normalize as nz
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES
# Re-exported for tests/test_gate_invariants.py.
from eval_entity_resolver.fold import (  # noqa: F401
    brand_tokens_for,
    build_hf_index,
    decide_fold,
    dev_org_of_prefix,
    name_norm,
    strip_brand_prefix,
)

REGISTRY_ROOT = Path(__file__).resolve().parents[1]
ORACLE = resolve_oracle_path()
CORE_YAML = REGISTRY_ROOT / "seed" / "models" / "core.yaml"
ORGS_YAML = REGISTRY_ROOT / "seed" / "orgs.yaml"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_core() -> tuple[dict, list[dict]]:
    doc = yaml.safe_load(CORE_YAML.read_text())
    if isinstance(doc, dict):
        return doc, doc.get("entries", []) or []
    return {"entries": doc}, doc


def build_hf_to_dev() -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug; see seed_io.build_hf_to_dev_from_orgs_yaml."""
    return build_hf_to_dev_from_orgs_yaml(ORGS_YAML)


# ---------------------------------------------------------------------------
# HF target authority
# ---------------------------------------------------------------------------
def build_hf_targets(entries: list[dict], hf_to_dev: dict[str, str]):
    """eval_entity_resolver.fold.build_hf_index with the frozen oracle's
    fixed_exact/near_miss HF ids added as real-HF targets."""
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    fixed_ids = frozenset(
        v["fixed_hf_model_id"]
        for v in oracle.values()
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
        and isinstance(v.get("fixed_hf_model_id"), str) and "/" in v["fixed_hf_model_id"]
    )
    return build_hf_index(entries, hf_to_dev, fixed_ids)


# ---------------------------------------------------------------------------
# parents edge detection
# ---------------------------------------------------------------------------
def parent_refs(entries: list[dict]) -> dict[str, list[str]]:
    """id -> list of entry ids that reference it as a parents[].id edge."""
    refs: dict[str, list[str]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        for p in e.get("parents") or []:
            if isinstance(p, dict) and isinstance(p.get("id"), str):
                refs.setdefault(p["id"], []).append(e.get("id"))
    return refs


# ---------------------------------------------------------------------------
# --apply
# ---------------------------------------------------------------------------
def apply_folds(doc: dict, entries: list[dict], confirm_ids: set[str], folds: list[dict]):
    """Apply the confirmed folds to `entries` in place and rewrite core.yaml."""
    fold_by_mint = {f["mint_id"]: f for f in folds if f["mint_id"] in confirm_ids}
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}

    rename: dict[str, str] = {}
    for mint_id, f in fold_by_mint.items():
        tgt = f["hf_target"]
        mint = by_id.get(mint_id)
        target = by_id.get(tgt)
        if mint is None or target is None:
            print(f"  SKIP {mint_id}: mint or target missing in entries", file=sys.stderr)
            continue
        rename[mint_id] = tgt

        tmd = _load_md(target)
        mmd = _load_md(mint)
        tp = list(tmd.get("providers") or [])
        for p in mmd.get("providers") or []:
            if p not in tp:
                tp.append(p)
        if tp:
            tmd["providers"] = tp
        target["metadata"] = json.dumps(tmd)
        if not target.get("release_date") and mint.get("release_date"):
            target["release_date"] = mint["release_date"]
        if target.get("open_weights") is None and mint.get("open_weights") is not None:
            target["open_weights"] = mint["open_weights"]

        aliases = list(target.get("aliases") or [])
        seen = {a.lower() for a in aliases if isinstance(a, str)}
        for extra in [mint_id, mint.get("display_name"), *(mint.get("aliases") or [])]:
            if isinstance(extra, str) and extra and extra != tgt and extra.lower() not in seen:
                aliases.append(extra)
                seen.add(extra.lower())
        target["aliases"] = aliases

    kept = [e for e in entries if not (isinstance(e, dict) and e.get("id") in rename)]

    for e in kept:
        if not isinstance(e, dict):
            continue
        for p in e.get("parents") or []:
            if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"] in rename:
                p["id"] = rename[p["id"]]

    if isinstance(doc, dict):
        doc["entries"] = kept
    CORE_YAML.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000)
    )
    print(f"applied {len(rename)} folds; core.yaml now {len(kept)} entries")


def _load_md(entry: dict) -> dict:
    md = entry.get("metadata")
    if isinstance(md, str):
        try:
            return json.loads(md) or {}
        except (ValueError, TypeError):
            return {}
    if isinstance(md, dict):
        return md
    return {}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confirm", nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    doc, entries = load_core()
    hf_to_dev = build_hf_to_dev()
    mints = [e for e in entries if isinstance(e, dict) and e.get("resolution_source") == "models_dev"]
    hf_ids, alias_to_hf, by_org_name, _hf_by_id = build_hf_targets(entries, hf_to_dev)

    folds = []
    for m in mints:
        f = decide_fold(m, hf_ids, alias_to_hf, by_org_name, hf_to_dev)
        if f is not None:
            folds.append(f)

    refs = parent_refs(entries)
    fold_ids = {f["mint_id"] for f in folds}
    for f in folds:
        f["referenced_as_parent"] = f["mint_id"] in refs
    parent_referenced = sorted(fid for fid in fold_ids if fid in refs)

    if args.apply:
        confirm = set(args.confirm or [])
        if not confirm:
            print("--apply requires --confirm <id> [...]", file=sys.stderr)
            sys.exit(2)
        apply_folds(doc, entries, confirm, folds)
        return

    report = {
        "candidates_examined": len(mints),
        "fold_count": len(folds),
        "no_fold_count": len(mints) - len(folds),
        "folds": folds,
        "parent_referenced_fold_ids": parent_referenced,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"candidates_examined: {report['candidates_examined']}")
        print(f"fold_count:          {report['fold_count']}")
        print(f"no_fold_count:       {report['no_fold_count']}")
        print(f"parent_referenced:   {parent_referenced}")
        from collections import Counter
        print("by match_type:", dict(Counter(f["match_type"] for f in folds)))
        for f in folds:
            print(f"  {f['mint_id']}  ->  {f['hf_target']}  [{f['match_type']}] "
                  f"org {f['mint_org']}=={f['hf_org']}? {f['org_agreement']} "
                  f"parent_ref={f['referenced_as_parent']}  :: {f['evidence']}")


if __name__ == "__main__":
    main()
