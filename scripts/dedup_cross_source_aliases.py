#!/usr/bin/env python3
"""
Cross-source alias reconciliation: the finalize step after the model seed
sources are (re)generated, before seeding.

The seed loader fails fast when two canonicals declare the same alias. Each
generator cleans its own output, but a surface form can still be claimed
across different sources; this pass resolves every such collision
deterministically so a clean regen always produces a seedable source set.

Ownership priority for a contested form (highest first):
  1. the canonical whose id == the form;
  2. curated core over a generated source;
  3. the canonical whose normalized name == the form;
  4. lexicographically-first id.
The owner keeps the form; it is stripped from every other canonical's aliases /
alias_platforms, and a losing display_name is re-derived from the id tail. Core
is never stripped: a core-vs-core collision is reported, not auto-resolved.

Idempotent: re-running on a deduped tree is a no-op.

Usage:
    LOCAL_MODE=true uv run python scripts/dedup_cross_source_aliases.py
    ... --dry-run    # report strips, write nothing
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from eval_entity_resolver.normalization import normalize as _nz

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "seed" / "models" / "core.yaml"
SOURCES = [
    REPO / "seed" / "models" / "sources" / name
    for name in (
        "hf_oracle.generated.yaml",
        "models_dev.generated.yaml",
        "hub_stats.generated.yaml",
        "models_dev_catalog.generated.yaml",
        "tier3_inferred.generated.yaml",
    )
]


def _load(path: Path):
    if not path.exists():
        return None, []
    doc = yaml.safe_load(path.read_text())
    entries = (doc.get("entries") if isinstance(doc, dict) else doc) or []
    return doc, [e for e in entries if isinstance(e, dict)]


def _name_part(cid: str) -> str:
    return cid.split("/", 1)[-1] if "/" in cid else cid


def build_owner_map(core_entries, source_entries):
    """Return ({surface_form -> owner_id}, all ids) over every id/display_name/alias
    across core + sources, using the module docstring's ownership priority."""
    all_entries = [(e, True) for e in core_entries] + [(e, False) for e in source_entries]
    ids = {e["id"] for e, _ in all_entries if isinstance(e.get("id"), str)}

    cands: dict[str, list[tuple]] = defaultdict(list)

    def consider(form: str, e: dict, is_core: bool):
        if not isinstance(form, str) or not form:
            return
        cid = e["id"]
        is_id_owner = form == cid
        is_natural = _nz(form) == _nz(_name_part(cid))
        # Lower sorts first; tuple order is the docstring's priority order.
        key = (
            0 if is_id_owner else 1,
            0 if is_core else 1,
            0 if is_natural else 1,
            cid,
        )
        cands[form].append((key, cid))

    for e, is_core in all_entries:
        cid = e.get("id")
        if not isinstance(cid, str):
            continue
        consider(cid, e, is_core)
        dn = e.get("display_name")
        if isinstance(dn, str):
            consider(dn, e, is_core)
        for a in e.get("aliases") or []:
            if isinstance(a, str):
                consider(a, e, is_core)

    owner: dict[str, str] = {}
    for form, lst in cands.items():
        owner[form] = min(lst)[1]
    return owner, ids


def reconcile(dry_run: bool = False) -> int:
    _core_doc, core_entries = _load(CORE)
    loaded = [(p, *_load(p)) for p in SOURCES]
    source_entries = [e for _p, _doc, ents in loaded for e in ents]

    owner, _ids = build_owner_map(core_entries, source_entries)

    core_ids = {e["id"] for e in core_entries}
    strips: list[str] = []

    def clean_entry(e: dict) -> bool:
        """Strip forms this entry does not own. Returns True if mutated."""
        cid = e["id"]
        mutated = False
        kept = []
        for a in e.get("aliases") or []:
            if isinstance(a, str) and a != cid and owner.get(a, cid) != cid:
                strips.append(f"{cid}: drop alias {a!r} (owned by {owner.get(a)!r})")
                mutated = True
                continue
            kept.append(a)
        if mutated:
            e["aliases"] = kept
        ap = e.get("alias_platforms")
        if isinstance(ap, dict):
            new_ap = {k: v for k, v in ap.items() if k == cid or owner.get(k, cid) == cid}
            if len(new_ap) != len(ap):
                mutated = True
                e["alias_platforms"] = new_ap
        dn = e.get("display_name")
        if isinstance(dn, str) and dn and owner.get(dn, cid) != cid:
            e["display_name"] = _name_part(cid)
            strips.append(f"{cid}: re-derive display_name (was {dn!r}, owned by {owner.get(dn)!r})")
            mutated = True
        return mutated

    changed_paths = []
    for path, doc, ents in loaded:
        any_changed = False
        for e in ents:
            if clean_entry(e):
                any_changed = True
        if any_changed and not dry_run:
            out = ents if not isinstance(doc, dict) else {**doc, "entries": ents}
            header = ""
            text = path.read_text()
            if text.startswith("#"):
                header = "\n".join(
                    ln for ln in text.splitlines() if ln.startswith("#")
                ) + "\n"
            path.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))
            changed_paths.append(path.name)

    for s in strips:
        print(f"[dedup] {s}", file=sys.stderr)
    print(
        f"[dedup] {'(dry-run) ' if dry_run else ''}stripped {len(strips)} contested "
        f"form(s) across {len(changed_paths)} source file(s): {changed_paths}",
        file=sys.stderr,
    )
    core_conflicts = []
    for e in core_entries:
        cid = e["id"]
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a != cid and owner.get(a) != cid and owner.get(a) in core_ids:
                core_conflicts.append((cid, a, owner.get(a)))
    if core_conflicts:
        print(f"[dedup] WARNING: {len(core_conflicts)} core-vs-core alias conflict(s) "
              f"(NOT auto-stripped — fix curation): {core_conflicts[:5]}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="report strips; write nothing")
    args = p.parse_args()
    return reconcile(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
