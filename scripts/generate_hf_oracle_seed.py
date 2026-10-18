#!/usr/bin/env python3
"""
Generate the HF source-of-truth seed from the frozen HuggingFace oracle
(`hf_model_id_resolution.json`).

Offline and deterministic. The oracle carries, for each raw EEE `model_id`,
the HF-true repo id (`fixed_hf_model_id`) and a `resolution_status` in
{fixed_exact, fixed_near_miss, unresolved_not_found_or_inaccessible}; the
HF-present ids (fixed_exact + fixed_near_miss) are materialised.

Canonical id rule: `{HF-ORG}/{HF-NAME}` verbatim (HF casing preserved, org
never folded into the id); `org_id` = the curated developer slug when the HF
org maps to one (meta-llama->meta, qwen->alibaba, ...), else the HF org.

Each oracle entry's target is canon_id(fixed_hf_model_id) and the raw EEE id
becomes a confirmed alias of it (for fixed_near_miss the target is the HF
redirect repo, possibly a different org). Targets are partitioned against the
current seed YAML: present -> nothing minted; present under a casing variant
-> re-keyed in place; otherwise -> minted.

Writes:
   (a) seed/models/sources/hf_oracle.generated.yaml — the minted canonicals
       (HF-cased id, org_id, display_name, resolution_source: hf,
       review_status: reviewed, aliases incl. raw EEE id(s) + lowercase form).
       Sparse metadata; no invented parents.
   (b) re-keys colliding lowercase canonicals in place wherever they live
       (core.yaml / sources/models_dev.generated.yaml /
       sources/hub_stats.generated.yaml), appending the old id and raw EEE
       id(s) to that entry's aliases, and rewrites every parents[].id /
       model_group_id / model_family_id / lineage_origin_model_id reference
       across all three files so no edge dangles.
   (c) seed/orgs.generated.yaml — new HF-cased orgs (kind community,
       review_status reviewed, hf_org set). Curated orgs.yaml wins on collision.

Usage:
    LOCAL_MODE=true uv run python scripts/generate_hf_oracle_seed.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from eval_entity_registry.lib.seed_io import (
    build_hf_to_dev_from_orgs_yaml,
    resolve_oracle_path,
)

from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REGISTRY_ROOT = Path(__file__).resolve().parents[1]
ORACLE = resolve_oracle_path()

SEED = REGISTRY_ROOT / "seed"
FIXTURES = REGISTRY_ROOT / "fixtures"
MODELS_DIR = SEED / "models"
CORE_YAML = MODELS_DIR / "core.yaml"
SOURCES_DIR = MODELS_DIR / "sources"
HF_ORACLE_YAML = SOURCES_DIR / "hf_oracle.generated.yaml"
MODELS_DEV_YAML = SOURCES_DIR / "models_dev.generated.yaml"
HUB_STATS_YAML = SOURCES_DIR / "hub_stats.generated.yaml"
ORGS_YAML = SEED / "orgs.yaml"
ORGS_GENERATED_YAML = SEED / "orgs.generated.yaml"

# Files whose model entries may carry edge references rewritten on re-key.
EDIT_FILES = [CORE_YAML, MODELS_DEV_YAML, HUB_STATS_YAML]
EDGE_ID_KEYS = ("model_group_id", "model_family_id", "lineage_origin_model_id")


def _load_yaml_entries(path: Path) -> tuple[object, list[dict]]:
    """Return (raw_doc, entries-list-view). core.yaml uses a
    {skip_ids, skip_source_ids, entries} dict; sources are flat lists."""
    if not path.exists():
        return None, []
    with open(path) as f:
        doc = yaml.safe_load(f) or []
    if isinstance(doc, dict):
        return doc, doc.get("entries", []) or []
    return doc, doc


def _entries_view(doc: object) -> list[dict]:
    if isinstance(doc, dict):
        return doc.get("entries", []) or []
    return doc if isinstance(doc, list) else []


def _load_curated_orgs() -> list[dict]:
    """Curated orgs from seed/orgs.yaml; fixtures are a build artifact and are
    never read here."""
    if not ORGS_YAML.exists():
        return []
    with open(ORGS_YAML) as f:
        return [e for e in (yaml.safe_load(f) or []) if isinstance(e, dict)]


def build_hf_to_dev(curated_orgs: list[dict]) -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug (see
    seed_io.build_hf_to_dev_from_orgs_yaml). `curated_orgs` is unused; kept
    for call-site compatibility."""
    return build_hf_to_dev_from_orgs_yaml(ORGS_YAML)


def canon_id(hf_model_id: str, hf_to_dev: dict[str, str]) -> tuple[str, str]:
    """Returns (canonical_id, org_id). canonical_id = the real HF repo id
    verbatim (org is never folded into the id); org_id = the curated parent
    if the HF org maps to one, else the HF org itself."""
    org_part, name_part = hf_model_id.split("/", 1)
    org_id = hf_to_dev.get(org_part.lower(), org_part)
    return f"{org_part}/{name_part}", org_id


# --- near-miss identity guard ----------------------------------------------
# HF `fixed_near_miss` redirects are did-you-mean matches that can map a raw
# EEE string to a different model (size/generation/uploader swap); such a
# redirect must not become a confirmed alias. Blocked by the curated denylist
# (curation/audit_bad_nearmiss.json), a genuine size-token change, or a
# cross-developer org. Version-vs-size (`Yi-1.5-34B`) and MoE active-param
# (`A22B`) tokens are not size changes.
_NM_SIZE_RE = re.compile(r'(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])', re.I)
_NM_ACTIVE_RE = re.compile(r'(?<![a-z0-9])a\d+(?:\.\d+)?b', re.I)


def _nm_sizes(name: str) -> set[str]:
    return set(_NM_SIZE_RE.findall(_NM_ACTIVE_RE.sub(" ", name.lower())))


def _load_audit_bad_nearmiss() -> frozenset[str]:
    p = REGISTRY_ROOT  / "curation" / "audit_bad_nearmiss.json"
    try:
        return frozenset(json.loads(p.read_text()).get("raws", []))
    except FileNotFoundError:  # pragma: no cover
        return frozenset()


_AUDIT_BAD_NEARMISS = _load_audit_bad_nearmiss()


def _fold_org(org: str, hf_to_dev: dict[str, str]) -> str:
    """Developer identity of an HF org spelling: curated remap, then separators
    and case collapsed, so a same-uploader rename folds equal (`DeepAutoAI` ==
    `DeepAuto-AI`) while a cross-uploader move stays distinct."""
    return re.sub(r"[^a-z0-9]", "", hf_to_dev.get(org.lower(), org.lower()))


def nearmiss_changes_identity(raw: str, fixed: str, hf_to_dev: dict[str, str]) -> str | None:
    """Reason a fixed_near_miss raw->fixed redirect must not be aliased (it
    changes model identity), else None. Signals: audit denylist; genuine size
    change; cross-developer org (per `_fold_org`, so a same-uploader rename is
    not flagged)."""
    if raw in _AUDIT_BAD_NEARMISS:
        return "audit-confirmed wrong redirect"
    rn, fn = raw.split("/", 1)[-1], fixed.split("/", 1)[-1]
    rs, fs = _nm_sizes(rn), _nm_sizes(fn)
    if rs and fs and rs.isdisjoint(fs):
        return f"size change {sorted(rs)}->{sorted(fs)}"
    if "/" in raw and "/" in fixed:
        ro = raw.split("/", 1)[0]
        if ro.lower() != "unknown":
            rd = _fold_org(ro, hf_to_dev)
            fd = _fold_org(fixed.split("/", 1)[0], hf_to_dev)
            if rd != fd:
                return f"cross-dev-org {rd}!={fd}"
    return None


def authoritative_metadata(hf_repo_id: str) -> str:
    """metadata JSON marking a mint as the authoritative real HF repo
    (`hf_id == id`): such an id must never be rewritten or suppressed in favor
    of a colliding dev-org slug."""
    return json.dumps({"hf_id": hf_repo_id}, sort_keys=True)


def main() -> None:
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    curated_orgs = _load_curated_orgs()

    hf_to_dev = build_hf_to_dev(curated_orgs)

    # Only the seed YAML counts as existing: fixtures/*.parquet already hold
    # prior mints after a seed, which would make a re-run see everything as
    # present.
    existing_ids: set[str] = set()
    for path in EDIT_FILES:
        _doc, _entries = _load_yaml_entries(path)
        for e in _entries:
            # Only defining records (with display_name) count: an enrich
            # record defines nothing, and counting it would make a re-run
            # classify every target as present.
            if (
                isinstance(e, dict)
                and isinstance(e.get("id"), str)
                and "display_name" in e
            ):
                existing_ids.add(e["id"])
    existing_lower: dict[str, str] = {}
    for cid in sorted(existing_ids):
        existing_lower.setdefault(cid.lower(), cid)

    existing_org_ids: set[str] = {
        e["id"] for e in curated_orgs if isinstance(e.get("id"), str)
    }

    # Incumbent index: the canonical each HF/raw id resolves to today, from
    # the seed YAML (fixtures lag it).
    from eval_entity_registry.services.hub_stats import normalize as _nz

    incumbent_exact: dict[str, str] = {}
    incumbent_norm: dict[str, str] = {}

    def _claim_incumbent(raw_value: str, cid: str) -> None:
        if not isinstance(raw_value, str) or not raw_value:
            return
        incumbent_exact.setdefault(raw_value, cid)
        incumbent_norm.setdefault(_nz(raw_value), cid)

    for path in EDIT_FILES:
        _doc, _entries = _load_yaml_entries(path)
        for e in _entries:
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            _claim_incumbent(cid, cid)
            dn = e.get("display_name")
            if isinstance(dn, str):
                _claim_incumbent(dn, cid)
            for a in e.get("aliases") or []:
                if isinstance(a, str):
                    _claim_incumbent(a, cid)

    def incumbent_for(*ids: str) -> Optional[str]:
        for i in ids:
            if i in incumbent_exact:
                return incumbent_exact[i]
        for i in ids:
            n = _nz(i)
            if n in incumbent_norm:
                return incumbent_norm[n]
        return None

    # --- Compute targets ---------------------------------------------------
    targets: dict[str, dict] = {}
    skipped_nearmiss: list[tuple[str, str, str]] = []
    for raw, meta in oracle.items():
        status = meta.get("resolution_status")
        if status not in ("fixed_exact", "fixed_near_miss"):
            continue
        fixed = meta.get("fixed_hf_model_id")
        if not isinstance(fixed, str) or "/" not in fixed:
            continue
        tgt, org_id = canon_id(fixed, hf_to_dev)
        name = fixed.split("/", 1)[1]
        bucket = targets.setdefault(
            tgt, {"org_id": org_id, "name": name, "raws": set(), "fixed": set()}
        )
        bucket["fixed"].add(fixed)
        # A near_miss raw is a real alias unless the redirect changes model
        # identity; those raws resolve via Tier-3 instead.
        if raw != tgt:
            reason = (nearmiss_changes_identity(raw, fixed, hf_to_dev)
                      if status == "fixed_near_miss" else None)
            if reason:
                skipped_nearmiss.append((raw, fixed, reason))
            else:
                bucket["raws"].add(raw)
    if skipped_nearmiss:
        print(f"[hf-oracle] guarded {len(skipped_nearmiss)} identity-changing near_miss "
              f"redirect(s) (NOT aliased): e.g. {skipped_nearmiss[:5]}")

    # --- Partition: ok / re-key / mint ------------------------------------
    # By the incumbent (the canonical the target's raw/fixed ids resolve to):
    #   incumbent == target -> ok; a casing variant (name-token-equal under
    #   normalize) -> re-key it to the HF casing; token-different or none ->
    #   mint, stripping the HF/raw aliases off the incumbent so resolve(raw)
    #   reaches the new canonical.
    ok_targets: dict[str, dict] = {}
    rekey_targets: dict[str, dict] = {}
    mint_targets: dict[str, dict] = {}
    incumbent_alias_strip: dict[str, set[str]] = {}

    for tgt, info in targets.items():
        hf_name = tgt.split("/", 1)[1]
        lookup_ids = sorted(info["fixed"]) + sorted(info["raws"]) + [tgt]
        inc = incumbent_for(*lookup_ids)

        # The re-key adopts the HF name casing but keeps the incumbent's id
        # prefix; the curated developer goes on `org_id` only. Folding the
        # prefix into the dev slug (`zai-org/x` -> `zai/x`) would re-mint a
        # shadow id that collides with the committed enrichment alias and abort
        # the seed.
        if inc is not None and _nz(inc.split("/", 1)[-1]) == _nz(hf_name):
            inc_org = inc.split("/", 1)[0]
            dev_org = hf_to_dev.get(inc_org.lower(), inc_org)
            rekey_tgt = f"{inc_org}/{hf_name}"
            if rekey_tgt == inc:
                # Already correctly cased. "ok" only when a defining record
                # carries the id; an incumbency claimed solely by enrich
                # records means this file is the definer and must re-emit.
                if rekey_tgt in existing_ids:
                    ok_targets[rekey_tgt] = info
                else:
                    mint_targets[rekey_tgt] = info
                continue
            # One re-key per incumbent: further targets fold in as aliases.
            existing_rk = next(
                (t for t, i in rekey_targets.items() if i["old_id"] == inc),
                None,
            )
            if existing_rk is not None:
                rekey_targets[existing_rk]["raws"] |= info["raws"]
                rekey_targets[existing_rk]["fixed"] |= info["fixed"]
                continue
            rk = dict(
                info,
                old_id=inc,
                org_id=dev_org,
                raws=set(info["raws"]),
                fixed=set(info["fixed"]),
            )
            rekey_targets[rekey_tgt] = rk
            continue

        if tgt in existing_ids:
            ok_targets[tgt] = info
            continue
        if tgt.lower() in existing_lower and existing_lower[tgt.lower()] != tgt:
            # Pure case-variant id exists (no alias incumbent) -> re-key it.
            rekey_targets[tgt] = dict(info, old_id=existing_lower[tgt.lower()])
            continue

        # near_miss where the incumbent's id IS the raw repo (HF renamed it):
        # resolve(raw) hits the incumbent by exact id, so stripping aliases is
        # not enough; merge it into the target by re-keying.
        all_src_ids = {x.lower() for x in (info["fixed"] | info["raws"])}
        if (
            inc is not None
            and inc.lower() in all_src_ids
            and not any(i["old_id"] == inc for i in rekey_targets.values())
            and tgt not in rekey_targets
        ):
            rekey_targets[tgt] = dict(
                info, old_id=inc, raws=set(info["raws"]), fixed=set(info["fixed"])
            )
            continue

        # Mint; strip the HF/raw ids off a token-different incumbent.
        if inc is not None and inc != tgt:
            strip = incumbent_alias_strip.setdefault(inc, set())
            for x in info["fixed"] | info["raws"]:
                strip.add(x)
        mint_targets[tgt] = info

    rename_map: dict[str, str] = {
        info["old_id"]: tgt for tgt, info in rekey_targets.items()
    }

    # A re-key/mint can case-collide with a different existing id; fold the
    # straggler into the HF-cased winner so no case-duplicate pair survives.
    final_targets = (
        set(rekey_targets) | set(mint_targets) | set(ok_targets)
    )
    final_by_lower = {t.lower(): t for t in final_targets}
    for cid in existing_ids:
        if cid in final_targets:
            continue
        winner = final_by_lower.get(cid.lower())
        if winner is not None and winner != cid and cid not in rename_map:
            rename_map[cid] = winner
            for tgt, info in rekey_targets.items():
                if tgt == winner:
                    info.setdefault("raws", set()).add(cid)
                    break

    # --- Re-key in place across all edit files ----------------------------
    rekey_by_old = {info["old_id"]: (tgt, info) for tgt, info in rekey_targets.items()}
    rekeyed_count = 0

    import copy as _copy

    docs: dict[Path, object] = {}
    docs_before: dict[Path, object] = {}
    for path in EDIT_FILES:
        doc, _ = _load_yaml_entries(path)
        docs[path] = doc
        docs_before[path] = _copy.deepcopy(doc)

    def _rewrite_edges_in_entry(entry: dict) -> None:
        parents = entry.get("parents")
        if isinstance(parents, list):
            for p in parents:
                if isinstance(p, dict) and isinstance(p.get("id"), str):
                    if p["id"] in rename_map:
                        p["id"] = rename_map[p["id"]]
        for key in EDGE_ID_KEYS:
            v = entry.get(key)
            if isinstance(v, str) and v in rename_map:
                entry[key] = rename_map[v]

    stripped_alias_count = 0
    for path in EDIT_FILES:
        entries = _entries_view(docs[path])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("id")
            if isinstance(cid, str) and cid in rekey_by_old:
                tgt, info = rekey_by_old[cid]
                entry["id"] = tgt
                aliases = list(entry.get("aliases") or [])
                seen = {a.lower() for a in aliases if isinstance(a, str)}
                for extra in [cid, *sorted(info["raws"]), *sorted(info["fixed"])]:
                    if extra and extra != tgt and extra.lower() not in seen:
                        aliases.append(extra)
                        seen.add(extra.lower())
                entry["aliases"] = aliases
                if isinstance(entry.get("org_id"), str):
                    entry["org_id"] = info["org_id"]
                rekeyed_count += 1
            # Case-collision straggler: fold into the winner.
            elif isinstance(cid, str) and cid in rename_map:
                winner = rename_map[cid]
                entry["id"] = winner
                aliases = list(entry.get("aliases") or [])
                seen = {a.lower() for a in aliases if isinstance(a, str)}
                if cid.lower() not in seen and cid != winner:
                    aliases.append(cid)
                entry["aliases"] = aliases
                if isinstance(entry.get("org_id"), str):
                    win_org = winner.split("/", 1)[0]
                    entry["org_id"] = hf_to_dev.get(win_org.lower(), win_org)
                rekeyed_count += 1
            # Aliases a mint takes over from a token-different incumbent.
            elif isinstance(cid, str) and cid in incumbent_alias_strip:
                strip = incumbent_alias_strip[cid]
                aliases = list(entry.get("aliases") or [])
                kept = [
                    a for a in aliases
                    if not (isinstance(a, str) and a in strip)
                ]
                stripped_alias_count += len(aliases) - len(kept)
                entry["aliases"] = kept
            # Any entry may point at a re-keyed parent.
            _rewrite_edges_in_entry(entry)

    # --- Global claimed-alias set ----------------------------------------
    # Aliases the seed CLI will emit from surviving entries (id + display_name
    # + aliases). The seed aborts on a duplicate alias, so mints must not
    # re-emit one; the check is exact and case-sensitive.
    claimed: dict[str, str] = {}

    def _claim(alias: str, owner: str) -> None:
        if isinstance(alias, str) and alias:
            claimed.setdefault(alias, owner)

    for path in EDIT_FILES:
        for e in _entries_view(docs[path]):
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            _claim(cid, cid)
            dn = e.get("display_name")
            if isinstance(dn, str):
                _claim(dn, cid)
            for a in e.get("aliases") or []:
                _claim(a, cid)

    # --- (a) Mint brand-new canonicals into hf_oracle.generated.yaml ------
    mint_entries: list[dict] = []
    for tgt in sorted(mint_targets):
        info = mint_targets[tgt]
        name = info["name"]
        org_id = info["org_id"]

        # A bare HF name already claimed by another canonical (cross-org
        # clash) is qualified with the org so the auto-emitted display alias
        # stays unique.
        if name in claimed and claimed[name] != tgt:
            display = tgt
        else:
            display = name
            _claim(name, tgt)

        aliases: list[str] = []
        seen: set[str] = {tgt.lower(), display.lower()}

        def _add_alias(a: str) -> None:
            nonlocal aliases
            if not a or a.lower() in seen:
                return
            if a in claimed and claimed[a] != tgt:
                return
            aliases.append(a)
            seen.add(a.lower())
            _claim(a, tgt)

        for a in sorted(info["raws"]) + sorted(info["fixed"]):
            _add_alias(a)
        low = tgt.lower()
        if low != tgt:
            _add_alias(low)

        entry = {
            "id": tgt,
            "display_name": display,
            "org_id": org_id,
            "resolution_source": "hf",
            "review_status": "reviewed",
            "metadata": authoritative_metadata(tgt),
        }
        if aliases:
            entry["aliases"] = aliases
        mint_entries.append(entry)
        _claim(tgt, tgt)

    # --- Global de-collision pass ----------------------------------------
    # The seed CLI aborts if two canonicals declare the same exact alias. Keep
    # the first declarer (core > models_dev > hub_stats > mints); an entry's
    # own id is never dropped and beats any alias.
    decollide_owner: dict[str, str] = {}
    dropped_collisions = 0
    all_entry_groups = [_entries_view(docs[p]) for p in EDIT_FILES] + [mint_entries]

    # Ids claim first: an id is immutable and beats any display_name/alias.
    for grp in all_entry_groups:
        for e in grp:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                decollide_owner.setdefault(e["id"], e["id"])

    def _decollide(entries: list[dict]) -> None:
        nonlocal dropped_collisions
        for e in entries:
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            dn = e.get("display_name")
            if isinstance(dn, str) and dn != cid:
                owner = decollide_owner.get(dn)
                if owner is not None and owner != cid:
                    e["display_name"] = cid
                else:
                    decollide_owner.setdefault(dn, cid)
            new_aliases = []
            for a in e.get("aliases") or []:
                if not isinstance(a, str) or not a:
                    continue
                owner = decollide_owner.get(a)
                if owner is not None and owner != cid:
                    dropped_collisions += 1
                    continue
                decollide_owner.setdefault(a, cid)
                new_aliases.append(a)
            if "aliases" in e:
                e["aliases"] = new_aliases

    for grp in all_entry_groups:
        _decollide(grp)

    # Rewrite only files whose entries changed: a yaml re-dump destroys
    # comments (core.yaml carries curated ones).
    for path in EDIT_FILES:
        if docs[path] != docs_before[path]:
            with open(path, "w") as f:
                yaml.safe_dump(
                    docs[path], f, sort_keys=False, allow_unicode=True, width=10_000
                )

    header = (
        "# AUTO-GENERATED by scripts/generate_hf_oracle_seed.py — DO NOT HAND-EDIT.\n"
        "# HF source-of-truth canonicals (Tier-1), minted from the frozen oracle\n"
        "# hf_model_id_resolution.json. Ids carry HF-true casing per the two-tier\n"
        "# org rule. Metadata is sparse (params/release/open_weights\n"
        "# enriched later online); NO invented parents.\n"
    )
    with open(HF_ORACLE_YAML, "w") as f:
        f.write(header)
        yaml.safe_dump(
            mint_entries, f, sort_keys=False, allow_unicode=True, width=10_000
        )

    # --- New HF-cased orgs -----------------------------------------------
    needed_orgs: set[str] = set()
    for info in list(mint_targets.values()) + list(rekey_targets.values()):
        oid = info["org_id"]
        if oid and oid not in existing_org_ids:
            needed_orgs.add(oid)

    # The seed CLI aborts if a new org id duplicates a curated org's
    # id/display_name/hf_org/alias, so those are excluded.
    curated_org_claims: set[str] = set()
    if ORGS_YAML.exists():
        with open(ORGS_YAML) as f:
            for e in yaml.safe_load(f) or []:
                if not isinstance(e, dict):
                    continue
                for key in ("id", "display_name", "hf_org"):
                    v = e.get(key)
                    if isinstance(v, str) and v:
                        curated_org_claims.add(v)
                for a in e.get("aliases") or []:
                    if isinstance(a, str) and a:
                        curated_org_claims.add(a)
    needed_orgs -= curated_org_claims

    org_entries = []
    for oid in sorted(needed_orgs):
        org_entries.append(
            {
                "id": oid,
                "display_name": oid,
                "hf_org": oid,
                "kind": "community",
                "tags": "[]",
                "metadata": "{}",
                "review_status": "reviewed",
            }
        )
    org_header = (
        "# AUTO-GENERATED by scripts/generate_hf_oracle_seed.py — DO NOT HAND-EDIT.\n"
        "# HF-derived orgs (Tier-1 community namespaces). Curated seed/orgs.yaml\n"
        "# wins on id collision (loader merge in cli.py:_load_orgs_merged).\n"
    )
    with open(ORGS_GENERATED_YAML, "w") as f:
        f.write(org_header)
        yaml.safe_dump(
            org_entries, f, sort_keys=False, allow_unicode=True, width=10_000
        )

    # --- Report -----------------------------------------------------------
    print("=== generate_hf_oracle_seed ===")
    print(f"oracle fixed_exact+near_miss entries: "
          f"{sum(1 for v in oracle.values() if v.get('resolution_status') in ('fixed_exact','fixed_near_miss'))}")
    print(f"distinct target canonicals: {len(targets)}")
    print(f"  already present (ok, no work): {len(ok_targets)}")
    print(f"  re-keyed in place (collision): {len(rekey_targets)} "
          f"(entries rewritten: {rekeyed_count})")
    print(f"  brand-new canonicals (mint):   {len(mint_targets)}")
    print(f"wrote {HF_ORACLE_YAML.relative_to(REGISTRY_ROOT)}: {len(mint_entries)} entries")
    print(f"global rename map (old_id -> hf_cased): {len(rename_map)} entries")
    print(f"incumbent aliases stripped (taken over by mints): {stripped_alias_count}")
    print(f"residual collision aliases dropped (de-collision pass): {dropped_collisions}")
    print(f"wrote {ORGS_GENERATED_YAML.relative_to(REGISTRY_ROOT)}: "
          f"{len(org_entries)} new HF orgs")


if __name__ == "__main__":
    main()
