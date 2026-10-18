"""Seed-time normalize-collision fold: sources mint the same model under
different separator spellings (`gemini-1.5-pro` vs `gemini-1-5-pro`), so each
collision group folds into one winner (ids, display_name and aliases move
onto it; parent edges are repointed). A separator difference can change a
parameter size (`opt-1.3b` vs `opt-13b`), so size-conflicting groups never
fold; `never_fold` covers other look-alikes, `prefer` pins a winner."""
import json as _json
import re
from collections import defaultdict

_SEP_RE = re.compile(r"[-_./]+")
_BSIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b\b")
# Curated / HF entries beat generated ones.
_SRC_RANK = {None: 3, "NA": 3, "hf": 3, "models_dev": 2, "inferred": 1}
# Unresolved-org draft prefixes: stripped for collision keying so
# `unknown/perplexity-sonar-reasoning` buckets with `perplexity/sonar-reasoning`.
_PLACEHOLDER_ORGS = frozenset({"unknown", "none", "na", ""})


def _strip_placeholder(s: str) -> str:
    if "/" in s:
        org, rest = s.split("/", 1)
        if org.lower() in _PLACEHOLDER_ORGS:
            return rest
    return s


def _has_real_org(s: str) -> bool:
    return "/" in s and s.split("/", 1)[0].lower() not in _PLACEHOLDER_ORGS


def collision_key(s: str) -> str:
    """Separator-, case- and placeholder-org-agnostic key (strips a leading
    unknown//none/ prefix, then - _ . /), so every spelling of the same name —
    including org-less and unresolved-org drafts — collapses into one bucket."""
    return _SEP_RE.sub("", _strip_placeholder(str(s)).lower())


def _bsizes(s: str):
    return tuple(sorted(float(m) for m in _BSIZE_RE.findall(str(s).lower())))


def _winner_rank(e: dict):
    id_ = e["id"]
    name = id_.split("/")[-1]
    return (
        _has_real_org(id_),                          # a real org prefix beats unknown//bare
        _SRC_RANK.get(e.get("resolution_source"), 0),
        "." in name,                                 # prefer a dotted version spelling
        -len(id_),                                   # then the shorter id
        id_,                                         # deterministic final tiebreak
    )


def _edges(v):
    if isinstance(v, str):
        try:
            return _json.loads(v) or []
        except Exception:
            return []
    return list(v) if isinstance(v, list) else []


def fold_collisions(entries: list[dict], never_fold=(), prefer=None, force_merge=None,
                    non_lineage_bases=None):
    """Fold normalize-colliding model entries into one winner each.

    `force_merge` is an explicit {loser_id -> winner_id} map for duplicates
    that do not share a collision key; they fold through the same path.
    Returns (surviving_entries, remap: loser id -> winner id). Mutates
    winners in place."""
    prefer = prefer or {}
    non_lineage_bases = set(non_lineage_bases or ())
    never = [frozenset(p) for p in never_fold]
    by_id = {e["id"]: e for e in entries}

    groups: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        groups[collision_key(e["id"])].append(e["id"])

    remap: dict[str, str] = {
        l: w for l, w in (force_merge or {}).items() if l in by_id and w in by_id and l != w
    }
    for key, gids in groups.items():
        if len(gids) < 2:
            continue
        if sum(1 for i in gids if by_id[i].get("resolution_source") == "hf") > 1:
            # Two real HF repos that normalize-collide are distinct uploads.
            continue
        if len({_bsizes(i) for i in gids}) > 1:          # size-conflict guard
            continue
        if any(nf <= set(gids) for nf in never):          # curated never-fold
            continue
        if key in prefer and prefer[key] in gids:
            winner = prefer[key]
        else:
            winner = max(gids, key=lambda i: _winner_rank(by_id[i]))
        for lid in gids:
            if lid != winner and lid not in remap:
                remap[lid] = winner
    if not remap:
        return entries, {}

    # A group-fold winner may itself be a force_merge loser; resolve chains to
    # the final survivor so the transfer pass never looks up a folded-away id.
    for lid in list(remap):
        w, seen = remap[lid], {lid}
        while w in remap and w not in seen:
            seen.add(w)
            w = remap[w]
        remap[lid] = w

    surviving = [e for e in entries if e["id"] not in remap]
    surv_by_id = {e["id"]: e for e in surviving}

    # The loser's display_name moves too: the seed promotes display_name to a
    # global alias, so dropping it would delete a resolvable surface form.
    for lid, wid in remap.items():
        le, we = by_id[lid], surv_by_id[wid]
        aliases = list(we.get("aliases") or [])
        ldn = le.get("display_name")
        forms = [lid, *(le.get("aliases") or [])]
        if isinstance(ldn, str) and ldn and ldn != we.get("display_name"):
            forms.append(ldn)
        for a in forms:
            if a and a != wid and a not in aliases:
                aliases.append(a)
        we["aliases"] = aliases
        # A reviewed winner that declares its own parents is authoritative and
        # does not absorb an inferred loser's edges. Scoped to NON-EMPTY curated
        # parents: a reviewed root still inherits a loser's inferred base, which
        # is often the only lineage it has.
        winner_curated_lineage = (
            we.get("review_status") == "reviewed" and _edges(we.get("parents"))
        )
        loser_inferred = (
            le.get("review_status") == "draft" or le.get("resolution_source") == "inferred"
        )
        # Never inherit an edge to an umbrella base (`deepseek/deepseek`); only
        # fold-inherited edges are filtered, pre-existing ones stay.
        le_edges = [e for e in _edges(le.get("parents"))
                    if not (isinstance(e, dict) and e.get("id") in non_lineage_bases)]
        if le_edges and not (winner_curated_lineage and loser_inferred):
            raw = we.get("parents")
            combined = _edges(raw) + le_edges
            we["parents"] = _json.dumps(combined) if isinstance(raw, str) else combined
        for f in ("release_date", "params_billions", "open_weights", "family", "architecture"):
            if we.get(f) in (None, "") and le.get(f) not in (None, ""):
                we[f] = le[f]

    for e in surviving:
        raw = e.get("parents")
        edges = _edges(raw)
        if not edges:
            continue
        out, seen, changed = [], set(), False
        for ed in edges:
            if not isinstance(ed, dict):
                out.append(ed)
                continue
            pid = ed.get("id")
            if pid in remap:
                pid = remap[pid]
                ed = {**ed, "id": pid}
                changed = True
            if pid == e["id"] or pid in seen:
                changed = True
                continue
            seen.add(pid)
            out.append(ed)
        if changed:
            e["parents"] = _json.dumps(out) if isinstance(raw, str) else out

    return surviving, remap
