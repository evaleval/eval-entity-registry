"""Org-aware fold decision: does a minted off-HF canonical refer to the SAME
model as a real HF repo already in the registry? Single owner of this rule for
both the seed generators (a mint defers to the HF id) and the gate tests.

A fold requires ORG AGREEMENT after the curated developer remap (meta-llama ->
meta, Qwen -> alibaba): a mint with no resolvable org, or whose name matches
only under a DIFFERENT developer, never folds. Tiers, strongest first: exact
id; mint string already an alias of an HF entry; `name_norm` equality + org
agreement; brand-prefix-stripped name + org agreement (`qwen-qwq-32b` ->
`QwQ-32B`).
"""
from __future__ import annotations

import re
from typing import Optional

from eval_entity_resolver.normalization import normalize as _nz


# Curated HF-namespace -> developer-org remap (lowercase keys). Affects org_id
# only; a canonical_id keeps its real HF repo prefix.
_ORG_ALIASES: dict[str, str] = {
    "deepseek-ai": "deepseek",
    "cohereforai": "cohere",
    "cohere-labs": "cohere",
    "coherelabs": "cohere",
    "huggingfacetb": "huggingface",
    "baichuan-inc": "baichuan",
    "minimaxai": "minimax",
    "sarvamai": "sarvam",
    "tii-uae": "tiiuae",
    "meta-llama": "meta",
    "mistral-ai": "mistralai",
    "nvidia-nemo": "nvidia",
    # `THUDM` is the legacy HF org for GLM/ChatGLM; HF now publishes under `zai-org`.
    "zhipu": "zai",
    "zhipu-ai": "zai",
    "z-ai": "zai",
    "zai-org": "zai",
    "thudm": "zai",
    "moonshot": "moonshotai",
    "moonshot-ai": "moonshotai",
    # One-way: `alibaba/` also hosts non-Qwen uploads, so alibaba -> qwen is unsafe.
    "qwen": "alibaba",
    "facebook": "meta",
    "mistral": "mistralai",
    "mosaicml": "databricks",  # MosaicML acquired by Databricks
    "databricks-mosaic-research": "databricks",
    "alibaba-aidc": "alibaba",
    "alibaba-nlp": "alibaba",
    "aws-prototyping": "amazon",
    "ibm-research": "ibm",
    "ibm-granite": "ibm",
    "bytedance-seed": "bytedance",
}


def build_curated_org_map(orgs_yaml_entries: list[dict]) -> dict[str, str]:
    """`_ORG_ALIASES` unioned with every curated org's id, `hf_org` and
    `aliases`, keyed lowercase -> curated id; the curated seed wins on conflict.
    Every generator and the resolver build their fold map here."""
    m: dict[str, str] = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    for e in orgs_yaml_entries or []:
        if not isinstance(e, dict):
            continue
        oid = e.get("id")
        if not isinstance(oid, str) or not oid:
            continue
        m[oid.lower()] = oid
        hf_org = e.get("hf_org")
        if isinstance(hf_org, str) and hf_org.strip():
            m[hf_org.lower()] = oid
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a.strip():
                m[a.lower()] = oid
    return m


def build_org_dev_map_from_store(org_records, org_alias_pairs) -> dict[str, str]:
    """`build_curated_org_map` for store-backed callers without orgs.yaml.
    `canonical_orgs` has no `aliases` column; the alias tier arrives as
    `org_alias_pairs` ((raw, canonical_id) rows of entity_type=org)."""
    m = build_curated_org_map([
        {"id": r.get("id"), "hf_org": r.get("hf_org")} for r in (org_records or [])
    ])
    for raw, cid in org_alias_pairs or []:
        if isinstance(raw, str) and raw and isinstance(cid, str) and cid:
            m[raw.lower()] = cid
    return m


def _norm_org_key(org: str) -> str:
    """Separator/case-insensitive one-token org key."""
    return re.sub(r"[^a-z0-9]", "", org.lower())


def build_community_casing(org_prefixes: list[str]) -> dict[str, str]:
    """`_norm_org_key` -> one HF-true casing from real HF org prefixes, so
    `Sao10K`/`sao10k` collapse to one canonical_orgs row. Sorted tie-break so a
    refresh never flips a previously chosen casing."""
    by_key: dict[str, set[str]] = {}
    for p in org_prefixes:
        if isinstance(p, str) and p.strip():
            by_key.setdefault(_norm_org_key(p), set()).add(p)
    return {k: sorted(v)[0] for k, v in by_key.items() if v}


def canonicalize_org(
    prefix: str,
    curated_map: dict[str, str],
    community_casing: Optional[dict[str, str]] = None,
    distinct_allowlist: Optional[set[str]] = None,
) -> str:
    """Canonical org id for an HF org spelling: the curated developer id, else
    the HF-true community casing (unless the spelling is on the distinct-org
    allowlist, which keeps verified separate uploaders apart), else the prefix
    verbatim."""
    if not prefix:
        return prefix
    curated = curated_map.get(prefix.lower())
    if curated is not None:
        return curated
    if distinct_allowlist and prefix in distinct_allowlist:
        return prefix
    if community_casing:
        cased = community_casing.get(_norm_org_key(prefix))
        if cased is not None:
            return cased
    return prefix


def dev_org_of_prefix(prefix: str, hf_to_dev: dict[str, str]) -> str:
    """Remap an id-prefix org to its curated developer slug (else itself)."""
    return hf_to_dev.get(prefix.lower(), prefix)


def name_norm(name: str) -> str:
    """Normalized name with all separators removed, so `Qwen2.5-14B-Instruct`
    and `qwen-2-5-14b-instruct` map to one token."""
    return _nz(name).replace(" ", "")


def brand_tokens_for(dev_org: str, hf_to_dev: dict[str, str]) -> set[str]:
    """Brand tokens a models.dev key may glue onto a model name for this
    developer: the dev slug, every HF alias mapping to it (meta -> {meta,
    metallama, facebook}) and a few brands absent from the org map."""
    toks: set[str] = set()
    d = dev_org.lower()
    toks.add(name_norm(d))
    for hf_alias, dev in hf_to_dev.items():
        if dev.lower() == d:
            toks.add(name_norm(hf_alias))
    extra = {
        "alibaba": {"qwen"},
        "meta": {"llama"},
        "minimax": {"minimax"},
        "google": {"gemini", "gemma"},
        "deepseek": {"deepseek"},
    }
    toks |= extra.get(d, set())
    return {t for t in toks if t}


def strip_brand_prefix(norm_name_tok: str, brands: set[str]) -> set[str]:
    """Name tokens with leading brand tokens removed, repeatedly
    (`qwen-qwen-...`); always includes the original."""
    out = {norm_name_tok}
    cur = norm_name_tok
    changed = True
    while changed:
        changed = False
        for b in sorted(brands, key=len, reverse=True):
            if b and cur.startswith(b) and len(cur) > len(b):
                cur = cur[len(b):]
                out.add(cur)
                changed = True
                break
    return out


def build_hf_index(
    entries: list[dict],
    hf_to_dev: dict[str, str],
    fixed_ids: frozenset[str] = frozenset(),
):
    """HF target index from registry `entries` plus extra real-HF `fixed_ids`:
      - hf_ids: every real-HF canonical id (resolution_source == 'hf' or in fixed_ids).
      - alias_to_hf: every id/display/alias string on an HF entry -> that HF id.
      - by_org_name: (dev_org, name_norm) -> hf_id.
      - hf_entry_by_id: id -> entry.
    """
    hf_entry_by_id: dict[str, dict] = {}
    hf_ids: set[str] = set(fixed_ids)
    for e in entries:
        if not isinstance(e, dict):
            continue
        cid = e.get("id")
        if not isinstance(cid, str):
            continue
        if e.get("resolution_source") == "hf" or cid in fixed_ids:
            hf_ids.add(cid)
            hf_entry_by_id[cid] = e

    alias_to_hf: dict[str, str] = {}
    by_org_name: dict[tuple[str, str], str] = {}

    def index_target(cid: str, entry: Optional[dict]) -> None:
        if "/" not in cid:
            return
        org, name = cid.split("/", 1)
        dev = dev_org_of_prefix(org, hf_to_dev)
        by_org_name.setdefault((dev, name_norm(name)), cid)
        alias_to_hf.setdefault(cid, cid)
        if entry is not None:
            dn = entry.get("display_name")
            if isinstance(dn, str):
                alias_to_hf.setdefault(dn, cid)
            for a in entry.get("aliases") or []:
                if isinstance(a, str):
                    alias_to_hf.setdefault(a, cid)

    for cid, e in hf_entry_by_id.items():
        index_target(cid, e)
    for cid in fixed_ids:
        if cid not in hf_entry_by_id:
            index_target(cid, None)

    return hf_ids, alias_to_hf, by_org_name, hf_entry_by_id


def decide_fold(mint: dict, hf_ids, alias_to_hf, by_org_name, hf_to_dev) -> Optional[dict]:
    """Fold dict (mint_id, hf_target, match_type, org agreement, evidence) when
    `mint` confidently refers to the same model as a real HF id; else None,
    never a cross-developer merge."""
    cid = mint.get("id")
    if not isinstance(cid, str):
        return None
    mint_org = mint.get("org_id")
    mint_org = mint_org if isinstance(mint_org, str) and mint_org else None
    prefix_org = dev_org_of_prefix(cid.split("/", 1)[0], hf_to_dev) if "/" in cid else None
    eff_org = mint_org or prefix_org

    mint_strings = [cid]
    dn = mint.get("display_name")
    if isinstance(dn, str):
        mint_strings.append(dn)
    for a in mint.get("aliases") or []:
        if isinstance(a, str):
            mint_strings.append(a)

    for s in mint_strings:
        if s in hf_ids and s != cid:
            return _mk(mint, s, "exact", eff_org, hf_to_dev, f"mint string {s!r} is a real HF id")
    # A bare alias (`gemma-3-4b-it`) is legitimately carried by distinct
    # developers (google and unsloth's re-upload), so the alias tier requires
    # org agreement whenever both orgs are known. The exact tier above matches
    # a full org/model id and is unambiguous, so it stays unguarded.
    for s in mint_strings:
        tgt = alias_to_hf.get(s)
        if tgt and tgt != cid and tgt in hf_ids:
            tgt_org = dev_org_of_prefix(tgt.split("/", 1)[0], hf_to_dev) if "/" in tgt else None
            if eff_org and tgt_org and eff_org.lower() != tgt_org.lower():
                continue
            return _mk(mint, tgt, "alias", eff_org, hf_to_dev, f"mint string {s!r} declared on {tgt!r}")

    if eff_org is None:
        return None  # no org agreement possible

    cand_names = {(s.split("/", 1)[1] if "/" in s else s) for s in mint_strings}
    cand_names = {name_norm(n) for n in cand_names}

    for nm in cand_names:
        tgt = by_org_name.get((eff_org, nm))
        if tgt and tgt != cid:
            return _mk(mint, tgt, "normalized", eff_org, hf_to_dev,
                       f"org={eff_org} + name {nm!r} == {tgt!r}")
    brands = brand_tokens_for(eff_org, hf_to_dev)
    stripped: set[str] = set()
    for nm in cand_names:
        stripped |= strip_brand_prefix(nm, brands)
    for nm in stripped - cand_names:
        tgt = by_org_name.get((eff_org, nm))
        if tgt and tgt != cid:
            return _mk(mint, tgt, "fuzzy", eff_org, hf_to_dev,
                       f"org={eff_org} + brand-stripped {nm!r} == {tgt!r}")
    return None


def _mk(mint, hf_target, match_type, mint_dev_org, hf_to_dev, evidence) -> dict:
    hf_org = (hf_to_dev.get(hf_target.split("/", 1)[0].lower(), hf_target.split("/", 1)[0])
              if "/" in hf_target else hf_target)
    return {
        "mint_id": mint["id"],
        "hf_target": hf_target,
        "match_type": match_type,
        "mint_org": mint_dev_org or "",
        "hf_org": hf_org,
        "org_agreement": (mint_dev_org or "").lower() == (hf_org or "").lower(),
        "evidence": evidence,
    }
