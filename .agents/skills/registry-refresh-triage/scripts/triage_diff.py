#!/usr/bin/env python3
"""Summarize models.dev refresh side-effects in a worktree: added/removed IDs,
review_status flips (which affect cli.py lab_leaf_owner), and first-party
models.dev author IDs resolving to no_match."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml_entries(raw: str) -> dict[str, dict]:
    doc = yaml.safe_load(raw) or []
    entries = doc.get("entries", doc) if isinstance(doc, dict) else doc
    return {e["id"]: e for e in (entries or []) if isinstance(e, dict) and e.get("id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path, help="Path to the registry worktree")
    parser.add_argument(
        "--api-json",
        type=Path,
        default=Path("/tmp/modelsdev_api.json"),
        help="Cached models.dev api.json for first-party no_match audit",
    )
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="Git ref to diff models_dev.generated.yaml against (default: HEAD)",
    )
    args = parser.parse_args()
    wt = args.worktree.resolve()

    rel_path = "seed/models/sources/models_dev.generated.yaml"
    old_raw = subprocess.check_output(
        ["git", "-C", str(wt), "show", f"{args.base_ref}:{rel_path}"],
        text=True,
    )
    old_by_id = _load_yaml_entries(old_raw)
    new_by_id = _load_yaml_entries((wt / rel_path).read_text())

    added = sorted(set(new_by_id) - set(old_by_id))
    removed = sorted(set(old_by_id) - set(new_by_id))
    flips: list[tuple[str, str | None, str | None]] = []
    for cid in sorted(set(old_by_id) & set(new_by_id)):
        o_rs = old_by_id[cid].get("review_status")
        n_rs = new_by_id[cid].get("review_status")
        if o_rs != n_rs:
            flips.append((cid, o_rs, n_rs))

    print(f"=== models_dev.generated.yaml vs {args.base_ref} ===")
    print(f"added ({len(added)}): {added}")
    print(f"removed ({len(removed)}): {removed}")
    print(f"review_status flips ({len(flips)}):")
    for cid, o_rs, n_rs in flips:
        print(f"  {cid}: {o_rs} -> {n_rs}")

    fixtures = wt / "fixtures"
    if not (fixtures / "aliases.parquet").exists():
        print("\n[skip] fixtures/aliases.parquet missing; seed --local first for no_match audit")
        return 0
    if not args.api_json.exists():
        print(f"\n[skip] {args.api_json} missing; pass --api-json for first-party no_match audit")
        return 0

    sys.path.insert(0, str(wt / "scripts"))
    sys.path.insert(0, str(wt / "src"))
    sys.path.insert(0, str(wt / "packages" / "eval-entity-resolver" / "src"))
    import refresh_from_modelsdev as rfm
    from eval_entity_resolver import Resolver

    resolver = Resolver.from_parquet(str(fixtures))
    api = json.loads(args.api_json.read_text())
    no_match: list[tuple[str, str, str | None]] = []
    for prov in sorted(rfm.PROVIDER_TO_ORG):
        for mid, mr in sorted(((api.get(prov) or {}).get("models") or {}).items()):
            if "/" in mid:
                continue
            if resolver.resolve(mid, "model").canonical_id is None:
                no_match.append((prov, mid, mr.get("name")))

    print(f"\n=== first-party author IDs resolving to no_match ({len(no_match)}) ===")
    for prov, mid, name in no_match:
        print(f"  {prov:12s} {mid:40s} name={name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
