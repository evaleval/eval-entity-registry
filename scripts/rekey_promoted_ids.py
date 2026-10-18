#!/usr/bin/env python3
"""Re-key seed references to ids the models.dev refresh promoted to HF-true ids.

When the refresh folds an invented `{org}/{slug}` canonical onto its real HF
repo, the old id survives only as an alias, so any seed record still keyed by
it or naming it in a `parents` edge points at an alias and the gates fail
closed. Edits are text-level so curated comments survive.

Per file:
  core.yaml, enrichments/parents.yaml, hub_stats.generated.yaml,
  tier3_inferred.generated.yaml   -> parent-edge ids rewritten old -> new
  enrichments/aliases.yaml, enrichments/parents.yaml, hub_stats.generated.yaml
                                  -> record keys rewritten old -> new (the
                                     loader unions same-id records)
  tier3_inferred.generated.yaml   -> a record keyed by an old id is dropped and
                                     its id bridged onto the HF entry in
                                     enrichments/aliases.yaml (the raw EEE
                                     spelling must keep resolving)

The old -> new map is read from the regenerated models.dev sources: every
`org/...`-shaped alias on an HF-true entry (`resolution_source: hf` or
`metadata.hf_deferred`), minus curated umbrella ids (`non_lineage_bases`).

Usage:
    uv run python scripts/rekey_promoted_ids.py            # plan only
    uv run python scripts/rekey_promoted_ids.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "seed" / "models"
GENERATED = [
    MODELS / "sources" / "models_dev.generated.yaml",
    MODELS / "sources" / "models_dev_catalog.generated.yaml",
]
CORE = MODELS / "core.yaml"
ALIASES = MODELS / "enrichments" / "aliases.yaml"
PARENTS = MODELS / "enrichments" / "parents.yaml"
HUB_STATS = MODELS / "sources" / "hub_stats.generated.yaml"
TIER3 = MODELS / "sources" / "tier3_inferred.generated.yaml"
HF_ORACLE = MODELS / "sources" / "hf_oracle.generated.yaml"
OVERRIDES = MODELS / "collision_overrides.yaml"

EDGE_FILES = (CORE, PARENTS, HUB_STATS, TIER3)
KEY_FILES = (ALIASES, PARENTS, HUB_STATS)


def _entries(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or []
    if isinstance(data, dict):
        data = data.get("entries") or []
    return [e for e in data if isinstance(e, dict)]


def _meta(e: dict) -> dict:
    m = e.get("metadata")
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except ValueError:
            return {}
    return m if isinstance(m, dict) else {}


def _hf_true_ids() -> set[str]:
    ids = {e["id"] for e in _entries(HF_ORACLE) if e.get("resolution_source") == "hf"}
    for path in GENERATED:
        for e in _entries(path):
            if e.get("resolution_source") == "hf" or _meta(e).get("hf_deferred") is True:
                ids.add(e["id"])
    return ids


def _refresh_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "refresh_from_modelsdev", REPO_ROOT / "scripts" / "refresh_from_modelsdev.py"
    )
    rfm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rfm)
    return rfm


def promotion_map() -> dict[str, str]:
    rfm = _refresh_module()
    # Curated umbrellas are never re-keyed away; a multi-child family root only
    # onto an HF repo that carries its own bare name.
    umbrellas = set((yaml.safe_load(OVERRIDES.read_text()) or {}).get("non_lineage_bases") or [])
    roots = set(rfm._multi_child_roots())
    hf_true = _hf_true_ids()
    side_keys = {
        m.group(1)
        for path in (*KEY_FILES, TIER3)
        for m in re.finditer(r"(?m)^- id: (\S+)$", path.read_text())
    }
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from eval_entity_registry.lib.collision_fold import _bsizes
    from eval_entity_registry.lib.seed_io import build_hf_to_dev_from_orgs_yaml

    hf_to_dev = build_hf_to_dev_from_orgs_yaml(REPO_ROOT / "seed" / "orgs.yaml")

    def _key(cid: str):
        org, name = cid.split("/", 1) if "/" in cid else ("", cid)
        return (hf_to_dev.get(org.lower(), org.lower()), rfm._identity_sig(name), _bsizes(name))

    out: dict[str, str] = {}
    # Aliases on an HF-true entry that are themselves ids: `org/...`-shaped,
    # or org-less but still keyed by a side file.
    for path in GENERATED:
        for e in _entries(path):
            if e["id"] not in hf_true:
                continue
            for a in e.get("aliases") or []:
                if a == e["id"] or a in umbrellas or a in hf_true:
                    continue
                if a in roots and _key(a) != _key(e["id"]):
                    continue
                if "/" in a or a in side_keys:
                    out.setdefault(a, e["id"])
    # A tier3 draft matching an HF-true entry under the folded-identity gate's
    # own key folds too.
    by_key: dict = {}
    for hf_id in hf_true:
        if "/" in hf_id:
            by_key.setdefault(_key(hf_id), hf_id)
    for e in _entries(TIER3):
        cid = e["id"]
        if "/" not in cid or cid in out or cid in hf_true or cid in umbrellas:
            continue
        hit = by_key.get(_key(cid))
        if hit is not None:
            out[cid] = hit
    return out


_RECORD_RE = re.compile(r"^- id: (\S+)\s*(#.*)?$")
_EDGE_RE = re.compile(r"^( +- id: )(\S+)(\s*#.*)?$")
_KEY_RE = re.compile(r"^( *)([A-Za-z_][A-Za-z0-9_]*):\s*(#.*)?$")


def rewrite_edges(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Rewrite parent-edge ids: the indented `- id:` items under a record's
    `parents:` key. Record keys sit at column 0; any other nested `- id:`
    list is left alone."""
    n = 0
    out: list[str] = []
    in_parents = False
    for line in text.splitlines(keepends=True):
        s = line.rstrip("\n")
        if _RECORD_RE.match(s):
            in_parents = False
        else:
            k = _KEY_RE.match(s)
            if k:
                in_parents = k.group(2) == "parents"
        e = _EDGE_RE.match(s) if in_parents else None
        if e and e.group(2) in mapping:
            n += 1
            line = f"{e.group(1)}{mapping[e.group(2)]}{e.group(3) or ''}\n"
        out.append(line)
    return "".join(out), n


def strip_self_edges(text: str) -> tuple[str, int]:
    """After a rename a record may name itself as a parent (the old mint stood
    in as the HF repo's family root): drop such edges with their
    relationship/axis lines; a `parents:` left empty becomes `parents: []`."""
    n = 0
    out: list[str] = []
    record_id = None
    in_parents = False
    parents_line = None
    parents_has_edge = False
    skipping = False

    def _close_parents() -> None:
        nonlocal parents_line
        if parents_line is not None and not parents_has_edge:
            out[parents_line] = out[parents_line].rstrip("\n") + " []\n"
        parents_line = None

    for line in text.splitlines(keepends=True):
        s = line.rstrip("\n")
        if _RECORD_RE.match(s):
            _close_parents()
            record_id = _RECORD_RE.match(s).group(1)
            in_parents = skipping = False
            out.append(line)
            continue
        k = _KEY_RE.match(s)
        if k:
            _close_parents()
            in_parents = k.group(2) == "parents"
            skipping = False
            out.append(line)
            if in_parents:
                parents_line, parents_has_edge = len(out) - 1, False
            continue
        e = _EDGE_RE.match(s) if in_parents else None
        if e:
            skipping = e.group(2) == record_id
            if skipping:
                n += 1
                continue
            parents_has_edge = True
            out.append(line)
            continue
        if skipping and (re.match(r"^ +(relationship|axis): ", s) or s.lstrip().startswith("#") or not s.strip()):
            continue
        skipping = False
        out.append(line)
    _close_parents()
    return "".join(out), n


def rewrite_keys(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        old = m.group(1)
        if old in mapping:
            n += 1
            return f"- id: {mapping[old]}{m.group(2) or ''}"
        return m.group(0)

    return re.sub(r"^- id: (\S+)(\s*#.*)?$", sub, text, flags=re.M), n


def drop_tier3_records(text: str, mapping: dict[str, str]) -> tuple[str, list[tuple[str, list[str]]]]:
    """Remove whole records keyed by a promoted id; return (id, aliases) of
    each, so every raw spelling the draft carried can be bridged."""
    blocks = re.split(r"(?m)^(?=- id: )", text)
    kept, dropped = [], []
    for b in blocks:
        m = re.match(r"- id: (\S+)", b)
        if m and m.group(1) in mapping:
            rec = yaml.safe_load(b)
            aliases = [a for a in ((rec[0] if isinstance(rec, list) else rec).get("aliases") or []) if isinstance(a, str)]
            dropped.append((m.group(1), aliases))
        else:
            kept.append(b)
    return "".join(kept), dropped


def _merge_record(tgt: dict, rec: dict) -> None:
    """Union a same-id record into `tgt`: lists union (parents by edge id),
    dict-valued or JSON-string metadata merges per key, a scalar fills only
    where `tgt` has none. Mirrors the loader's same-id merge."""
    for k, v in rec.items():
        if k == "id":
            continue
        cur = tgt.get(k)
        if k == "parents":
            have = {p.get("id") for p in cur or [] if isinstance(p, dict)}
            for p in v or []:
                if isinstance(p, dict) and p.get("id") not in have:
                    tgt.setdefault("parents", []).append(p)
                    have.add(p.get("id"))
        elif isinstance(v, list):
            lst = list(cur) if isinstance(cur, list) else []
            lst += [x for x in v if x not in lst]
            tgt[k] = lst
        elif isinstance(v, (dict, str)) and k == "metadata":
            def _as_dict(x):
                if isinstance(x, dict):
                    return dict(x)
                try:
                    d = json.loads(x) if isinstance(x, str) else None
                except ValueError:
                    d = None
                return d if isinstance(d, dict) else None
            a, b = _as_dict(cur), _as_dict(v)
            if a is not None and b is not None:
                for mk, mv in b.items():
                    a.setdefault(mk, mv)
                tgt[k] = json.dumps(a, sort_keys=True) if isinstance(cur, str) else a
            elif cur in (None, ""):
                tgt[k] = v
        elif cur in (None, ""):
            tgt[k] = v


def merge_duplicate_records(text: str) -> tuple[str, int]:
    """Fold later same-id records into the first one (the seed files keep one
    record per id). Only the first duplicate's block is re-emitted, so comments
    elsewhere survive."""
    blocks = re.split(r"(?m)^(?=- id: )", text)
    first: dict[str, int] = {}
    merged: dict[int, dict] = {}
    drop: set[int] = set()
    for i, b in enumerate(blocks):
        m = re.match(r"- id: (\S+)", b)
        if not m:
            continue
        cid = m.group(1)
        if cid not in first:
            first[cid] = i
            continue
        j = first[cid]
        if j not in merged:
            merged[j] = yaml.safe_load(blocks[j])[0]
        rec = yaml.safe_load(b)[0]
        _merge_record(merged[j], rec)
        drop.add(i)
    if not drop:
        return text, 0
    out = []
    for i, b in enumerate(blocks):
        if i in drop:
            continue
        if i in merged:
            # Comment lines at the end of a block introduce the NEXT record;
            # keep them after the re-emitted one.
            lines = b.splitlines(keepends=True)
            k = len(lines)
            while k > 0 and (lines[k - 1].startswith("#") or not lines[k - 1].strip()):
                k -= 1
            dumped = yaml.safe_dump([merged[i]], sort_keys=False, allow_unicode=True, width=1000)
            out.append(dumped + "".join(lines[k:]))
        else:
            out.append(b)
    return "".join(out), len(drop)


def bridge_aliases(text: str, bridges: dict[str, list[str]]) -> str:
    """Append `{id: hf, aliases: [...]}` records for dropped tier3 ids."""
    if not bridges:
        return text
    out = text if text.endswith("\n") else text + "\n"
    out += "# Raw EEE spellings whose tier3 draft folded onto the HF-true repo\n"
    out += "# (scripts/rekey_promoted_ids.py); the raw must keep resolving.\n"
    records = [
        {"id": hf_id, "aliases": sorted(set(bridges[hf_id]))} for hf_id in sorted(bridges)
    ]
    return out + yaml.safe_dump(records, sort_keys=False, allow_unicode=True, width=1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    mapping = promotion_map()
    print(f"[rekey] {len(mapping)} promoted id(s) in the generated sources", file=sys.stderr)

    pending: dict[Path, str] = {}

    for path in EDGE_FILES:
        text = pending.get(path, path.read_text())
        text, n = rewrite_edges(text, mapping)
        print(f"[rekey] {path.relative_to(REPO_ROOT)}: {n} parent edge(s)", file=sys.stderr)
        pending[path] = text

    for path in KEY_FILES:
        text = pending.get(path, path.read_text())
        text, n = rewrite_keys(text, mapping)
        print(f"[rekey] {path.relative_to(REPO_ROOT)}: {n} record key(s)", file=sys.stderr)
        pending[path] = text

    for path in EDGE_FILES:
        pending[path], n = strip_self_edges(pending[path])
        if n:
            print(f"[rekey] {path.relative_to(REPO_ROOT)}: {n} self-edge(s) removed", file=sys.stderr)

    tier3_text, dropped = drop_tier3_records(pending[TIER3], mapping)
    pending[TIER3] = tier3_text
    bridges: dict[str, list[str]] = {}
    for old, aliases in dropped:
        bridges.setdefault(mapping[old], []).extend([old, *aliases])
    print(f"[rekey] {TIER3.relative_to(REPO_ROOT)}: {len(dropped)} record(s) dropped, "
          f"bridged onto {len(bridges)} HF id(s): {[d for d, _ in dropped]}", file=sys.stderr)
    pending[ALIASES] = bridge_aliases(pending[ALIASES], bridges)
    for path in KEY_FILES:
        pending[path], n = merge_duplicate_records(pending[path])
        print(f"[rekey] {path.relative_to(REPO_ROOT)}: {n} duplicate record(s) merged", file=sys.stderr)

    if not args.apply:
        print("[rekey] plan only; pass --apply to write", file=sys.stderr)
        return 0
    for path, text in pending.items():
        if text != path.read_text():
            path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
