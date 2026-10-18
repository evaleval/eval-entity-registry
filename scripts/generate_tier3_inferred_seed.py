#!/usr/bin/env python3
"""
Generate the Tier-3 name-based-inference seed.

Offline and deterministic. The resolution gate is resolve-based (no
auto-create), so every EEE id that still returns no_match after the Tier-1/2
sources + curation must be seeded.

1. Resolves every EEE id (the keys of `hf_model_id_resolution.json`) against
   the current fixtures + seed YAML and collects the no_match residual.
2. Mints a stable canonical for each residual id, with the raw EEE id as an
   exact alias:
   - org present (`org/name`): canonical id `{org}/{name}` verbatim, org_id
     via the curated org map, so a name-only id never collides across orgs;
   - org-less: a stable lowercase slug, `org_id = None`,
     `tags: [org-unknown]`, surfaced to `org_unknown_review.json`; an org is
     never guessed.
3. Base inference: detect a base family in the name tokens, build candidate
   base ids, and emit a typed parent edge (`finetune`, or `variant` with an
   axis) only when a candidate alias-confirms to an existing canonical.
4. Writes `seed/models/sources/tier3_inferred.generated.yaml`
   (`resolution_source: inferred`, `review_status: draft`) and
   `curation/org_unknown_review.json` (org-less mints plus a proposed org from
   an inferred base, not auto-applied).

Usage:
    LOCAL_MODE=true uv run python scripts/generate_tier3_inferred_seed.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from eval_entity_registry.lib.seed_io import (
    build_hf_to_dev_from_orgs_yaml,
    resolve_oracle_path,
)

from eval_entity_resolver.resolver import Resolver
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REGISTRY_ROOT = Path(__file__).resolve().parents[1]
ORACLE = resolve_oracle_path()

SEED = REGISTRY_ROOT / "seed"
FIXTURES = REGISTRY_ROOT / "fixtures"
MODELS_DIR = SEED / "models"
SOURCES_DIR = MODELS_DIR / "sources"
TIER3_YAML = SOURCES_DIR / "tier3_inferred.generated.yaml"
ORGS_YAML = SEED / "orgs.yaml"
SPEC_DIR = REGISTRY_ROOT  / "curation"
REVIEW_JSON = SPEC_DIR / "org_unknown_review.json"


# --- Tier-3 lexicon ---------------------------------------------------------
# Base family tokens; a detected base must still alias-confirm against the
# registry before any edge is emitted.
BASE_FAMILY_TOKENS = [
    "tinyllama", "openhermes", "openchat", "mixtral",
    "llama", "mistral", "qwen", "gemma", "yi", "phi", "gpt", "claude",
    "gemini", "falcon", "bloom", "deepseek", "baichuan", "cohere", "command",
    "neural", "solar", "nous", "zephyr", "orca", "dolphin",
]

# Not in `BASE_FAMILY_TOKENS`: the o-series pass runs only where the
# family-token pass found no edge, so existing edges never move.
O_SERIES_TOKENS = {"o1", "o3", "o4"}

# Serving-mode tokens: an `axis: mode` edge, not a `finetune`. Honoured only
# after a validated date (`split_date_mode_suffix`); on their own, `medium`
# and friends are product tiers in other families.
EFFORT_MODE_TOKENS = {
    "high", "low", "medium", "minimal", "xhigh", "none",
    "fc", "prompt", "thinking", "reasoning",
}

# Placeholder prefixes an upstream harness wrote when it did not know the
# developer; treated as org-less.
SENTINEL_ORGS = {"unknown"}

# Presence of any marker signals a derivation (finetune, no axis).
DERIVATION_MARKERS = [
    "dpo", "sft", "ft", "lora", "qlora", "merge", "slerp", "orpo", "kto",
    "uncensored", "abliterated", "ablated", "dare", "ties", "finetune",
    "instruct", "chat", "it", "base", "reasoning", "extended", "special",
]


def _non_tier3_claim_owners() -> dict[str, str]:
    """{surface form -> owning record id} over every id / display_name / alias
    claimed by a non-tier3 seed layer (other generated sources, core.yaml,
    enrichments/*.yaml); first writer wins."""
    owners: dict[str, str] = {}

    def _add(rec: dict) -> None:
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            return
        for f in [rid, rec.get("display_name"), *(rec.get("aliases") or [])]:
            if isinstance(f, str) and f:
                owners.setdefault(f, rid)

    for n in ("hf_oracle", "models_dev", "hub_stats", "models_dev_catalog"):
        p = SOURCES_DIR / f"{n}.generated.yaml"
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text())
        for e in (d.get("entries") if isinstance(d, dict) else d) or []:
            if isinstance(e, dict):
                _add(e)
    core_doc = yaml.safe_load((MODELS_DIR / "core.yaml").read_text()) or {}
    for e in _core_entries(core_doc):
        _add(e)
    enrich_dir = MODELS_DIR / "enrichments"
    if enrich_dir.exists():
        for p in sorted(enrich_dir.glob("*.yaml")):
            doc = yaml.safe_load(p.read_text()) or []
            doc = doc.get("entries", doc) if isinstance(doc, dict) else doc
            for e in doc or []:
                if isinstance(e, dict):
                    _add(e)
    return owners


def _non_tier3_defining_ids() -> set[str]:
    """Canonical ids DEFINED (display_name-bearing entry, or any core entry)
    by a non-tier3 seed layer. The curated-keyed retention pass skips ids
    another source already defines."""
    ids: set[str] = set()
    for n in ("hf_oracle", "models_dev", "hub_stats", "models_dev_catalog"):
        p = SOURCES_DIR / f"{n}.generated.yaml"
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text())
        for e in (d.get("entries") if isinstance(d, dict) else d) or []:
            if isinstance(e, dict) and e.get("id") and "display_name" in e:
                ids.add(e["id"])
    core_doc = yaml.safe_load((MODELS_DIR / "core.yaml").read_text()) or {}
    for e in _core_entries(core_doc):
        if e.get("id"):
            ids.add(e["id"])
    return ids


def _clean_resolver() -> Resolver:
    """Resolver from fixtures with this generator's own prior output
    (`resolution_source == 'inferred'` rows and their alias rows) filtered
    out, so a re-run resolves against the same Tier-1/2 + curation universe."""
    import pandas as pd
    from eval_entity_resolver import AliasStore, CanonicalStore

    cs = CanonicalStore.from_parquet(str(FIXTURES))
    models = cs._tables.get("model")
    inferred_ids: set[str] = set()
    if models is not None and not models.empty and "resolution_source" in models:
        mask = models["resolution_source"] == "inferred"
        inferred_ids = set(models.loc[mask, "id"].astype(str))
        cs._tables["model"] = models.loc[~mask].reset_index(drop=True)
        cs._index.clear()

    aliases_df = pd.read_parquet(FIXTURES / "aliases.parquet")
    if inferred_ids:
        aliases_df = aliases_df[
            ~aliases_df["canonical_id"].astype(str).isin(inferred_ids)
        ].reset_index(drop=True)
    # Also strip alias rows whose raw form originated from this generator's
    # committed output (enrich-record aliases on non-inferred canonicals, and
    # forms the seed's collision_fold moved onto a twin); the inferred-row
    # filter misses both, and leaving them in makes the residual oscillate
    # run-to-run.
    if TIER3_YAML.exists():
        prior = yaml.safe_load(TIER3_YAML.read_text()) or []
        prior = prior.get("entries", prior) if isinstance(prior, dict) else prior
        own_forms: set[str] = set()
        for e in prior:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            if "display_name" in e:
                own_forms.add(str(e["id"]))
            for a in e.get("aliases") or []:
                if a:
                    own_forms.add(str(a))
        # A form another layer also claims is not tier3-originated; stripping
        # it would re-mint a shadow of an existing entity.
        own_forms -= set(_non_tier3_claim_owners())
        if own_forms:
            keep = [str(rv) not in own_forms for rv in aliases_df["raw_value"]]
            aliases_df = aliases_df.loc[keep].reset_index(drop=True)
    alias_store = AliasStore(aliases_df, read_only=True)
    return Resolver(alias_store, canonical_store=cs)


def _load_curated_orgs() -> list[dict]:
    if not ORGS_YAML.exists():
        return []
    with open(ORGS_YAML) as f:
        return [e for e in (yaml.safe_load(f) or []) if isinstance(e, dict)]


def build_hf_to_dev(curated_orgs: list[dict]) -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug (see
    seed_io.build_hf_to_dev_from_orgs_yaml). `curated_orgs` is unused; kept
    for call-site compatibility."""
    return build_hf_to_dev_from_orgs_yaml(ORGS_YAML)


def _core_entries(core_doc) -> list[dict]:
    """Normalize the two `core.yaml` shapes (flat list OR
    `{skip_ids, skip_source_ids, entries}` dict) to a list of entry dicts."""
    if isinstance(core_doc, dict):
        items = core_doc.get("entries") or []
    elif isinstance(core_doc, list):
        items = core_doc
    else:
        items = []
    return [e for e in items if isinstance(e, dict)]


def build_core_norm_index(core_doc) -> dict[str, str]:
    """{normalized surface form -> curated core canonical id} over core.yaml
    ids / display_names / aliases; first writer wins. Reads the YAML directly
    so a curated entry not yet re-seeded into fixtures still wins."""
    from eval_entity_resolver.normalization import normalize as _rnz

    idx: dict[str, str] = {}

    def _add(form, cid: str) -> None:
        if isinstance(form, str) and form.strip():
            idx.setdefault(_rnz(form), cid)

    for e in _core_entries(core_doc):
        cid = e.get("id")
        if not isinstance(cid, str) or not cid:
            continue
        _add(cid, cid)
        _add(e.get("display_name"), cid)
        for a in (e.get("aliases") or []):
            _add(a, cid)
    return idx


def core_steals(cid: str, core_norm_index: dict[str, str]) -> bool:
    """True iff `cid` normalized-collides with a curated core canonical under a
    different id, i.e. it is the same model and must not be minted as a twin."""
    from eval_entity_resolver.normalization import normalize as _rnz

    owner = core_norm_index.get(_rnz(cid))
    return owner is not None and owner != cid


def mint_collision_decision(
    cid: str,
    core_skip_ids: set[str],
    core_norm_index: dict[str, str],
    resolver_hit: bool,
) -> str:
    """Collision policy for a residual tier-3 mint id `cid`:

    - 'skip'     — a curated core canonical owns `cid`'s normalized identity
                   (same model under another id, or core defines this id and a
                   tier3 twin would contaminate it with `resolution_source:
                   inferred`); the raw resolves to the curated canonical.
    - 'inferred' — `cid` clashes with a core `skip_ids` entry (the loader would
                   drop it) or with a different existing canonical (unrelated
                   models sharing a slug); suffix-disambiguate.
    - 'mint'     — no collision.

    `resolver_hit` is `resolver.resolve(cid, "model").canonical_id is not None`,
    passed in so the policy is testable without a resolver."""
    from eval_entity_resolver.normalization import normalize as _rnz

    if core_norm_index.get(_rnz(cid)) is not None:
        return "skip"
    if cid in core_skip_ids or resolver_hit:
        return "inferred"
    return "mint"


def canon_id_for_org_present(raw: str, hf_to_dev: dict[str, str]) -> tuple[str, str]:
    """`org/name` -> (canonical_id, org_id). canonical_id keeps the real
    `org/name` verbatim (org never folded into the id); org_id = the curated
    parent if the org maps to one, else the org verbatim."""
    org_part, name_part = raw.split("/", 1)
    org_id = hf_to_dev.get(org_part.lower(), org_part)
    return f"{org_part}/{name_part}", org_id


def _slug(value: str) -> str:
    """Stable lowercase slug for an org-less mint id."""
    s = re.sub(r"[^\w\s\-./]", "", value.lower().strip())
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-/")
    return s or "model"


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s\-_/.:]+", name.lower()) if t]


def detect_base_token(name: str) -> Optional[str]:
    toks = _tokens(name)
    tokset = set(toks)
    for fam in BASE_FAMILY_TOKENS:
        if fam in tokset:
            return fam
        # token may carry an attached version digit, e.g. `llama3`, `qwen2`.
        for t in toks:
            if t.startswith(fam) and t[len(fam):][:1].isdigit():
                return fam
    return None


def has_derivation_marker(name: str) -> bool:
    tokset = set(_tokens(name))
    return any(m in tokset for m in DERIVATION_MARKERS)


def _is_pure_version_suffix(tokens: list[str]) -> bool:
    """True iff the suffix tokens (the delta between a child id and its confirmed
    base) are purely a release-snapshot date or version string (`2026-03-05`,
    `20250929`, `2025-08`, `v0-3`): such an id is the same model identity along
    the version line and earns `variant/version` rather than `finetune`."""
    if not tokens:
        return False
    joined = "-".join(tokens)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}|\d{4}-\d{2}|\d{6}", joined):
        return True
    # 4-digit vendor tags: YYMM (`2411`, Mistral) or MMDD (`0514`, Gemini);
    # range-checked so a param count or opaque version is not read as a date.
    if re.fullmatch(r"\d{4}", joined):
        a, b = int(joined[:2]), int(joined[2:])
        if (20 <= a <= 29 and 1 <= b <= 12) or (1 <= a <= 12 and 1 <= b <= 31):
            return True
    # slugified v0.3 -> v0-3
    if re.fullmatch(r"v\d+(?:-\d+)*", joined):
        return True
    return False


def _valid_ymd(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
    except ValueError:
        return False
    return True


def is_validated_date_suffix(tokens: list[str]) -> bool:
    """Like `_is_pure_version_suffix`, but the calendar form must be a real
    date (`2025-99-99`, `20250230`, `999999` are rejected): a tail that only
    looks like a date must not assert two models are the same release. Kept
    separate so the family-token path's output does not move."""
    if not tokens:
        return False
    # slugified v0.3 -> v0-3
    if re.fullmatch(r"v\d+(?:-\d+)*", "-".join(tokens)):
        return True
    if not all(t.isdigit() for t in tokens):
        return False
    if len(tokens) == 3:
        y, m, d = tokens
        return len(y) == 4 and len(m) == 2 and len(d) == 2 and _valid_ymd(int(y), int(m), int(d))
    if len(tokens) == 2:
        y, m = tokens
        return len(y) == 4 and len(m) == 2 and _valid_ymd(int(y), int(m), 1)
    if len(tokens) != 1:
        return False
    t = tokens[0]
    if len(t) == 8:
        return _valid_ymd(int(t[:4]), int(t[4:6]), int(t[6:]))
    if len(t) == 6:
        # YYYYMM (`202608`) or YYMMDD (`250416`); only a leading `19`/`20`
        # reads as a four-digit year.
        if t[:2] in ("19", "20") and _valid_ymd(int(t[:4]), int(t[4:]), 1):
            return True
        return _valid_ymd(2000 + int(t[:2]), int(t[2:4]), int(t[4:]))
    if len(t) == 4:
        # YYMM (`2411`, Mistral) or MMDD (`0613`); range-checked so a param
        # count or opaque version (`4096`) is not read as a date.
        a, b = int(t[:2]), int(t[2:])
        if 20 <= a <= 29 and 1 <= b <= 12:
            return True
        return _valid_ymd(2000, a, b)  # 2000 is a leap year
    return False


def split_date_mode_suffix(tokens: list[str]) -> Optional[tuple[list[str], list[str]]]:
    """(date tokens, mode tokens) when the suffix is a validated release date
    followed by effort/mode tokens (`2025-04-16-high` ->
    `(['2025','04','16'], ['high'])`): the dated snapshot served at one effort,
    which belongs on the dated sibling under `axis: mode`. None otherwise; a
    bare effort token without a date is ambiguous and stays parentless."""
    for cut in range(len(tokens) - 1, 0, -1):
        rest = tokens[cut:]
        if all(t in EFFORT_MODE_TOKENS for t in rest) and is_validated_date_suffix(tokens[:cut]):
            return tokens[:cut], rest
    return None


def strip_mode_suffix(cid: str, mode_tokens: list[str]) -> Optional[str]:
    """`cid` with `mode_tokens` removed from its right-hand side, keeping the
    id's own separators and casing (`openai/o3-2025-04-16-high` ->
    `openai/o3-2025-04-16`). None when the tokens are not the literal tail."""
    out = cid
    for tok in reversed(mode_tokens):
        m = re.search(r"[-_. ]" + re.escape(tok) + r"$", out, flags=re.IGNORECASE)
        if m is None:
            return None
        out = out[: m.start()]
    return out or None


def same_dev_org(a: str, b: str, hf_to_dev: dict[str, str]) -> bool:
    """True iff two org-qualified ids belong to the same developer after the
    curated org fold. A sentinel prefix never agrees with anything: openness
    inheritance walks `variant` edges without an org check, so an `unknown/…`
    row must not acquire a real lab's verdict."""
    if "/" not in a or "/" not in b:
        return False
    oa, ob = a.split("/", 1)[0].lower(), b.split("/", 1)[0].lower()
    if oa in SENTINEL_ORGS or ob in SENTINEL_ORGS:
        return False
    return hf_to_dev.get(oa, oa).lower() == hf_to_dev.get(ob, ob).lower()


def o_series_edge(
    raw: str,
    hf_to_dev: dict[str, str],
    confirm,
) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """Parent inference for a residual whose name leads with an OpenAI o-series
    token. Returns `(parent_edge, dated_sibling, confirmed_base)`.

    Emits only: a validated date/version tail -> `variant/version` on the
    base; a validated date followed by effort/mode tokens -> a deferred
    `variant/mode` on the dated sibling (`resolve_mode_edges`). Everything
    else stays parentless: there is no `finetune` fallback, since an o-series
    model has no published weights to derive from. Candidates are
    org-qualified and the hit is org-checked, so a name-only match never links
    two labs."""
    from eval_entity_resolver.normalization import normalize as _rnz

    if "/" not in raw:
        return None, None, None
    org_slug, name = raw.split("/", 1)
    if not org_slug.strip() or not name.strip():
        return None, None, None
    if org_slug.strip().lower() in SENTINEL_ORGS:
        return None, None, None
    toks = _tokens(name)
    # Leading position only: a mid-name `o1` (`cursor-o1-7b`) is the
    # uploader's own label.
    if not toks or toks[0] not in O_SERIES_TOKENS:
        return None, None, None

    dev = hf_to_dev.get(org_slug.lower(), org_slug)
    raw_nz, name_nz = _rnz(raw), _rnz(name)
    for end in range(len(toks) - 1, 0, -1):
        stem = "-".join(toks[:end])
        for cand in dict.fromkeys((f"{dev}/{stem}", f"{org_slug}/{stem}")):
            if _rnz(cand) == raw_nz:
                continue
            hit = confirm(cand)
            if not hit or not same_dev_org(raw, hit, hf_to_dev):
                continue
            hit_name = hit.split("/", 1)[1] if "/" in hit else hit
            if _rnz(hit) == raw_nz or _rnz(hit_name) == name_nz:
                continue  # self-edge
            suffix = toks[end:]
            if is_validated_date_suffix(suffix):
                return (
                    {"id": hit, "relationship": "variant", "axis": "version"},
                    None,
                    hit,
                )
            split = split_date_mode_suffix(suffix)
            if split is None:
                return None, None, hit
            return None, strip_mode_suffix(raw, split[1]), hit
    return None, None, None


def resolve_mode_edges(pending, minted_by_id: dict[str, dict], confirm) -> None:
    """Attach the deferred `variant/mode` edges: the dated sibling is usually
    minted in the same unordered run, so its final id is known only after the
    mint loop. A missing sibling leaves the child parentless. A normalized
    match is accepted (siblings often disagree on separators) only when it is
    unique among same-run mints."""
    from eval_entity_resolver.normalization import normalize as _rnz

    by_norm: dict[str, set[str]] = {}
    for entry in minted_by_id.values():
        by_norm.setdefault(_rnz(entry["id"]), set()).add(entry["id"])
    for entry, dated in pending:
        exact = minted_by_id.get(dated.lower())
        if exact is not None:
            target = exact["id"]
        else:
            same_norm = by_norm.get(_rnz(dated)) or set()
            if len(same_norm) > 1:
                continue  # ambiguous
            target = next(iter(same_norm)) if same_norm else confirm(dated)
        if not target or target == entry["id"] or entry.get("parents"):
            continue
        entry["parents"] = [{
            "id": target,
            "relationship": "variant",
            "axis": "mode",
        }]


def _prior_committed_records() -> list[dict]:
    """The previously committed tier3 output (full entries and enrich records),
    read before this run overwrites the file; input to the carry-forward floor."""
    if not TIER3_YAML.exists():
        return []
    prior = yaml.safe_load(TIER3_YAML.read_text()) or []
    prior = prior.get("entries", prior) if isinstance(prior, dict) else prior
    return [e for e in prior if isinstance(e, dict) and e.get("id")]


def _curated_claim_forms() -> set[str]:
    """Surface forms the curated layers (core.yaml + enrichments/*.yaml) claim.
    A second claim on one of these (possibly placed on a tier3 canonical the
    clean resolver cannot see) would be an alias collision at seed time."""
    forms: set[str] = set()

    def _add(rec: dict) -> None:
        for f in [rec.get("display_name"), *(rec.get("aliases") or [])]:
            if isinstance(f, str) and f:
                forms.add(f)

    core_doc = yaml.safe_load((MODELS_DIR / "core.yaml").read_text()) or {}
    for e in _core_entries(core_doc):
        _add(e)
    enrich_dir = MODELS_DIR / "enrichments"
    if enrich_dir.exists():
        for p in sorted(enrich_dir.glob("*.yaml")):
            doc = yaml.safe_load(p.read_text()) or []
            doc = doc.get("entries", doc) if isinstance(doc, dict) else doc
            for e in doc or []:
                if isinstance(e, dict):
                    _add(e)
    return forms


def main() -> None:
    from eval_entity_resolver.normalization import normalize as _rnz_main

    oracle = json.loads(ORACLE.read_text())["resolutions"]
    curated_orgs = _load_curated_orgs()
    hf_to_dev = build_hf_to_dev(curated_orgs)
    prior_records = _prior_committed_records()

    # A mint colliding with a core `skip_ids` entry would be dropped by the
    # loader and its raw would stay no_match.
    core_skip_ids: set[str] = set()
    core_doc = yaml.safe_load((MODELS_DIR / "core.yaml").read_text()) or {}
    if isinstance(core_doc, dict):
        core_skip_ids = set(core_doc.get("skip_ids") or [])

    core_norm_index = build_core_norm_index(core_doc)

    r = _clean_resolver()

    # A residual that is the same model as a real HF repo under a
    # dev-org-decoupled id (`cohere/c4ai...` vs `CohereLabs/c4ai...`) defers to
    # it (raw becomes an alias) rather than minting a shadow slug; same
    # decide_fold as the other generators and the gate.
    from eval_entity_resolver.fold import build_hf_index as _build_hf_index, decide_fold as _decide_fold

    def _hf_source_entries() -> list[dict]:
        out: list[dict] = []
        for n in ("hf_oracle", "models_dev", "hub_stats", "models_dev_catalog"):
            p = SOURCES_DIR / f"{n}.generated.yaml"
            if not p.exists():
                continue
            d = yaml.safe_load(p.read_text())
            out.extend((d.get("entries") if isinstance(d, dict) else d) or [])
        return [e for e in out if isinstance(e, dict)]

    _oracle_fixed = frozenset(
        v["fixed_hf_model_id"] for v in oracle.values()
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
        and isinstance(v.get("fixed_hf_model_id"), str) and "/" in v["fixed_hf_model_id"]
    )
    _hf_ids, _alias_to_hf, _by_org_name, _ = _build_hf_index(_hf_source_entries(), hf_to_dev, _oracle_fixed)

    def fold_to_real_hf(cid: str, org_id: Optional[str], raw: str) -> Optional[str]:
        """Real HF id this residual folds onto (same model, org agreement), or None."""
        mint = {"id": cid, "org_id": org_id, "display_name": raw, "aliases": [raw]}
        f = _decide_fold(mint, _hf_ids, _alias_to_hf, _by_org_name, hf_to_dev)
        return f["hf_target"] if f and f["hf_target"] != cid else None

    # {real_hf_id -> raw aliases}, emitted as enrich records.
    fold_enrich: dict[str, set[str]] = {}

    # Parity with the seed's fold_collisions: a draft it would merge anyway
    # (`cohere/CommandA` -> `cohere/command-a`) becomes a stable enrich record
    # now, or the row oscillates (minted, folded away, re-minted).
    from eval_entity_registry.lib.collision_fold import _bsizes as _ck_bsizes
    from eval_entity_registry.lib.collision_fold import collision_key as _ck

    ck_index: dict[str, str] = {}
    _clean_models = r.canonical_store._tables.get("model")
    if _clean_models is not None and not _clean_models.empty:
        for _cid in _clean_models["id"].astype(str):
            ck_index.setdefault(_ck(_cid), _cid)

    def fold_to_collision_twin(cid: str) -> Optional[str]:
        hit = ck_index.get(_ck(cid))
        if hit is not None and hit != cid and _ck_bsizes(hit) == _ck_bsizes(cid):
            return hit
        return None

    # Same-run dev-org twins (`kimi/kimi-k2` vs `moonshotai/kimi-k2`, separator
    # respellings): two full entries with one folded identity fail the adoption
    # gate, and the seed fold cannot merge cross-org spellings. Key mirrors the
    # seed fold (dev org, stripped name, b-size signature).
    minted_by_devkey: dict[tuple, dict] = {}

    def _dev_key(c: str) -> tuple:
        if "/" in c:
            o, n = c.split("/", 1)
            return (hf_to_dev.get(o.lower(), o.lower()), _ck(n), _ck_bsizes(n))
        return ("", _ck(c), _ck_bsizes(c))

    # Ids curated records key by: in a twin pair that spelling must win the id
    # slot, or the curated aliases/parents strand on a stand-in.
    curated_record_ids: set[str] = set()
    for _e in _core_entries(core_doc):
        if isinstance(_e.get("id"), str):
            curated_record_ids.add(_e["id"])
    _enrich_dir = MODELS_DIR / "enrichments"
    if _enrich_dir.exists():
        for _p in sorted(_enrich_dir.glob("*.yaml")):
            _doc = yaml.safe_load(_p.read_text()) or []
            _doc = _doc.get("entries", _doc) if isinstance(_doc, dict) else _doc
            for _e in _doc or []:
                if isinstance(_e, dict) and isinstance(_e.get("id"), str):
                    curated_record_ids.add(_e["id"])

    def _twin_id_wins(challenger: str, incumbent: str, org_id_: Optional[str]) -> bool:
        """True iff `challenger` is the better spelling for a same-run twin pair:
        curated-record-keyed id, then real developer-org prefix, then the seed
        fold's `_winner_rank` preferences (dotted version, shorter id)."""

        def _score(c: str) -> tuple:
            org = c.split("/", 1)[0].lower() if "/" in c else ""
            name = c.rsplit("/", 1)[-1]
            return (
                c in curated_record_ids,
                org == (org_id_ or "").lower(),
                "." in name,
                -len(c),
                c,
            )

        return _score(challenger) > _score(incumbent)

    from eval_entity_registry.services.hub_stats import normalize as _nz

    def _confirm_base(candidate: str) -> Optional[str]:
        """Canonical id of a candidate base, accepting only an exact or
        normalized match so no cross-version / cross-family edge is invented."""
        res = r.resolve(candidate, "model")
        if res.canonical_id and res.strategy in ("exact", "normalized"):
            return res.canonical_id
        return None

    # --- residual ------------------------------------------------------------
    # Sentinel-org ids: the placeholder prefix is dropped and the bare name
    # re-resolved; an exact/normalized hit aliases the raw onto the true
    # canonical instead of a shadow `unknown/` mint. Opaque ones fall through
    # to the residual and mint org-less.
    minted: list[dict] = []
    minted_by_id: dict[str, dict] = {}
    # (entry, dated sibling id) for `{date}-{mode}` children; see resolve_mode_edges.
    pending_mode_edges: list[tuple[dict, str]] = []
    review: list[dict] = []
    buckets = Counter()

    # A residual raw that curation claims must never mint: the curated target
    # may be a tier3 canonical this clean resolver cannot see, and a mint would
    # double-claim the raw at seed time.
    curated_claims = _curated_claim_forms()

    # --- curated-keyed entity retention --------------------------------------
    # A prior full entry whose id a curated record keys is an entity by
    # curation's declaration (enrich records never materialize a canonical),
    # so it is retained unless another source now defines the id. Registered
    # up front so the residual pass folds respelled raws onto it.
    other_defined = _non_tier3_defining_ids()
    claim_owner = _non_tier3_claim_owners()
    core_dropped_ids: set[str] = set(core_skip_ids)
    if isinstance(core_doc, dict):
        core_dropped_ids |= set(core_doc.get("skip_source_ids") or [])
    for rec in prior_records:
        if "display_name" not in rec:
            continue
        rid = rec["id"]
        if rid not in curated_record_ids or rid in other_defined:
            continue
        if rid in core_dropped_ids or rid.lower() in minted_by_id:
            continue
        if claim_owner.get(rid, rid) != rid:
            # Claimed by a different canonical's record: deliberately superseded.
            continue
        if r.resolve(rid, "model").canonical_id is None:
            # Still no_match: the residual pass re-mints it with fresh
            # inference; retention is only for ids that pass would drop.
            continue
        entry = dict(rec)
        entry["aliases"] = [
            a for a in (rec.get("aliases") or [])
            if a == rid or claim_owner.get(a, rid) == rid
        ]
        minted.append(entry)
        minted_by_id[rid.lower()] = entry
        minted_by_devkey.setdefault(_dev_key(rid), entry)
        buckets["retained-curated-keyed"] += 1

    residual: list[str] = []
    for raw in oracle:
        if "/" in raw and raw.split("/", 1)[0].strip().lower() in SENTINEL_ORGS:
            sres = r.resolve(raw.split("/", 1)[1], "model")
            if sres.canonical_id and sres.strategy in ("exact", "normalized"):
                fold_enrich.setdefault(sres.canonical_id, set()).add(raw)
                buckets["folded-sentinel-org"] += 1
                continue
        if r.resolve(raw, "model").canonical_id is None:
            residual.append(raw)

    for raw in residual:
        if raw in curated_claims:
            buckets["curated-claimed"] += 1
            continue
        # A sentinel id reaching here is genuinely opaque; it takes the org-less
        # path rather than minting with org `unknown`. The mint id keeps the
        # `unknown/` prefix: an org-qualified id avoids bare-name alias
        # collisions.
        sentinel_org = "/" in raw and raw.split("/", 1)[0].strip().lower() in SENTINEL_ORGS
        has_org = (
            "/" in raw
            and all(p.strip() for p in raw.split("/", 1))
            and not sentinel_org
        )

        # Base detection keys off the model name, never a placeholder org.
        name_for_infer = raw.split("/", 1)[1] if (has_org or sentinel_org) else raw

        base_tok = detect_base_token(name_for_infer)
        deriv = has_derivation_marker(name_for_infer)

        # Candidate bases: `{org}/{stem}` before bare `{stem}`, so a name-only
        # match never merges across orgs.
        parent_edge: Optional[dict] = None
        confirmed_base: Optional[str] = None
        dated_sibling: Optional[str] = None
        if base_tok:
            # Progressively drop trailing tokens (`llama-3.1-8b-instruct-dpo`
            # -> `…-instruct`, `…-8b`, `llama-3.1`, …).
            name_toks = _tokens(name_for_infer)
            try:
                start = next(
                    i for i, t in enumerate(name_toks)
                    if t == base_tok or (t.startswith(base_tok) and t[len(base_tok):][:1].isdigit())
                )
            except StopIteration:
                start = 0
            tail = name_toks[start:]
            candidates: list[tuple[str, int]] = []
            for end in range(len(tail), 0, -1):
                stem = "-".join(tail[:end])
                if has_org:
                    org_slug = raw.split("/", 1)[0]
                    dev = hf_to_dev.get(org_slug.lower(), org_slug)
                    candidates.append((f"{dev}/{stem}", end))
                candidates.append((stem, end))
            raw_name_nz = _nz(name_for_infer)
            seen = set()
            confirmed_end: Optional[int] = None
            for cand, end in candidates:
                if cand in seen or _nz(cand) == _nz(raw):
                    continue
                seen.add(cand)
                hit = _confirm_base(cand)
                if not hit:
                    continue
                # Same full id or same model-name part as raw is a self-edge,
                # not a base.
                hit_name = hit.split("/", 1)[1] if "/" in hit else hit
                if _nz(hit) == _nz(raw) or _nz(hit_name) == raw_name_nz:
                    continue
                confirmed_base = hit
                confirmed_end = end
                break
            if confirmed_base:
                # A delta that is purely a date/version is the same identity
                # along the version line (`variant/version`, folds into the
                # base's group); anything else is a conservative `finetune`.
                suffix_toks = tail[confirmed_end:] if confirmed_end is not None else []
                if _is_pure_version_suffix(suffix_toks):
                    parent_edge = {
                        "id": confirmed_base,
                        "relationship": "variant",
                        "axis": "version",
                    }
                else:
                    parent_edge = {"id": confirmed_base, "relationship": "finetune"}

        # Gated on the family-token pass finding nothing, so no existing
        # classification moves.
        if parent_edge is None and confirmed_base is None:
            parent_edge, dated_sibling, confirmed_base = o_series_edge(
                raw, hf_to_dev, _confirm_base
            )

        # --- mint id + org -------------------------------------------------
        if has_org:
            cid, org_id = canon_id_for_org_present(raw, hf_to_dev)
            bucket = "inferable-base" if confirmed_base else "opaque"
        else:
            # Never guess the org, even when a base confirms. Slug the full raw
            # (incl. any `unknown/` prefix) so the mint stays org-qualified: a
            # bare slug can normalized-collide with a real org-prefixed
            # canonical at seed time.
            cid = _slug(raw)
            org_id = None
            bucket = "org-less"

        # Same model as a real HF canonical: alias the raw onto it instead of
        # minting a shadow slug.
        fold_hf = fold_to_real_hf(cid, org_id, raw)
        if fold_hf is not None:
            fold_enrich.setdefault(fold_hf, set()).add(raw)
            buckets["folded-to-hf"] += 1
            continue

        # Case/separator twin the seed's fold_collisions would merge anyway.
        fold_ck = fold_to_collision_twin(cid)
        if fold_ck is not None:
            fold_enrich.setdefault(fold_ck, set()).add(raw)
            buckets["folded-collision"] += 1
            continue

        # A case-insensitive id collision is the same model under different
        # casing (the org part is identical), so the raw folds as an alias on
        # the first-seen casing rather than minting a case-variant duplicate
        # or dropping it.
        def _fold_dup(target_cid: str) -> bool:
            existing = minted_by_id.get(target_cid.lower())
            if existing is None:
                return False
            if raw not in existing["aliases"]:
                existing["aliases"].append(raw)
            return True

        if _fold_dup(cid):
            continue
        # Same-run dev-org twin: fold onto it; the better spelling takes the id
        # slot and the loser id + raw stay aliases.
        twin = minted_by_devkey.get(_dev_key(cid))
        if twin is not None:
            for f in (raw, cid):
                if f != twin["id"] and f not in twin["aliases"]:
                    twin["aliases"].append(f)
            if _twin_id_wins(cid, twin["id"], org_id):
                old_id = twin["id"]
                twin["id"] = cid
                twin["display_name"] = raw
                if old_id not in twin["aliases"]:
                    twin["aliases"].append(old_id)
                twin["aliases"] = [a for a in twin["aliases"] if a != cid]
                minted_by_id[cid.lower()] = twin
            if parent_edge is not None and not twin.get("parents"):
                twin["parents"] = [parent_edge]
            buckets["folded-same-run-twin"] += 1
            continue
        decision = mint_collision_decision(
            cid,
            core_skip_ids,
            core_norm_index,
            resolver_hit=r.resolve(cid, "model").canonical_id is not None,
        )
        if decision == "skip":
            # A raw whose normalized form does not collapse onto the owner
            # (`+`/parens) would go unresolvable on a silent skip; union it as
            # an explicit alias.
            skip_owner = core_norm_index[_rnz_main(cid)]
            if raw != skip_owner and _rnz_main(raw) != _rnz_main(skip_owner):
                fold_enrich.setdefault(skip_owner, set()).add(raw)
                buckets["folded-core-owner"] += 1
            continue
        if decision == "inferred":
            cid = f"{cid}-inferred"
            if _fold_dup(cid):
                continue

        # display_name is the raw id verbatim: the seed promotes display_name
        # to a global alias, and a bare name frequently already belongs to
        # another canonical.
        entry: dict = {
            "id": cid,
            "display_name": raw,
            "resolution_source": "inferred",
            "review_status": "draft",
            "resolution_granularity": "variant",
            "metadata": "{}",
            "aliases": [raw],
        }
        if org_id is not None:
            entry["org_id"] = org_id
        else:
            entry["tags"] = ["org-unknown"]
        if parent_edge is not None:
            entry["parents"] = [parent_edge]

        minted.append(entry)
        minted_by_id[cid.lower()] = entry
        minted_by_devkey.setdefault(_dev_key(cid), entry)
        buckets[bucket] += 1
        if dated_sibling is not None:
            pending_mode_edges.append((entry, dated_sibling))

        if org_id is None:
            proposed = (
                confirmed_base.split("/", 1)[0]
                if confirmed_base and "/" in confirmed_base else None
            )
            review.append({
                "raw_value": raw,
                "minted_canonical_id": cid,
                "inferred_base": confirmed_base,
                "proposed_org": proposed,
                "rationale": (
                    "inferred base org (review before applying)" if proposed
                    else "no recognizable org/base token"
                ),
                "status": "unreviewed",
            })

    resolve_mode_edges(pending_mode_edges, minted_by_id, _confirm_base)

    # --- carry-forward floor ("nothing is removed") --------------------------
    # A draft whose raw now resolves only fuzzily would be dropped, degrading
    # the raw from an exact hit and deleting its other surface forms. Every
    # form the committed file carried must resolve exact/normalized, be
    # re-emitted, be claimed by a curated layer, or be carried as an enrich
    # alias onto the entity its raw now resolves to.
    emitted: set[str] = set()
    for e in minted:
        emitted.add(e["id"])
        emitted.update(e.get("aliases") or [])
    for hf, raws in fold_enrich.items():
        emitted.add(hf)
        emitted.update(raws)
    carried_home: dict[str, str] = {}
    lost_forms: list[str] = []
    for rec in prior_records:
        home = rec["id"]
        if home in core_dropped_ids:
            continue  # removed by curation on purpose
        forms = ([home] if "display_name" in rec else []) + list(rec.get("aliases") or [])
        for form in forms:
            if not isinstance(form, str) or not form or form in emitted:
                continue
            if form in curated_claims:
                continue  # a floor claim would collide with curation's
            if form in curated_record_ids:
                # A curated record keyed by this form keeps it alive as a
                # canonical id; a floor alias claim elsewhere would collide.
                continue
            if core_norm_index.get(_rnz_main(form)) is not None:
                # A core-defined canonical owns the normalized identity (core
                # scaffold entities are invisible to the clean resolver).
                continue
            res = r.resolve(form, "model")
            if res.canonical_id and res.strategy in ("exact", "normalized"):
                continue  # handed off to another source's claim
            if res.canonical_id:
                target = res.canonical_id  # fuzzy-only: pin the form as an alias
            elif home in carried_home:
                target = carried_home[home]  # keep the record's forms together
            else:
                hres = r.resolve(home, "model")
                if hres.canonical_id and hres.strategy in ("exact", "normalized"):
                    target = hres.canonical_id
                elif home.lower() in minted_by_id:
                    ent = minted_by_id[home.lower()]
                    if form != ent["id"] and form not in ent["aliases"]:
                        ent["aliases"].append(form)
                    emitted.add(form)
                    buckets["carried-committed-floor"] += 1
                    continue
                else:
                    lost_forms.append(form)
                    continue
            if form != target:
                fold_enrich.setdefault(target, set()).add(form)
            carried_home.setdefault(home, target)
            emitted.add(form)
            buckets["carried-committed-floor"] += 1
    if lost_forms:
        print(
            f"WARNING: {len(lost_forms)} previously-committed form(s) have no "
            f"surviving owner and were NOT carried (floor violation — fix the "
            f"drop): {lost_forms[:10]}",
            file=sys.stderr,
        )

    # --- write outputs -------------------------------------------------------
    header = (
        "# AUTO-GENERATED by scripts/generate_tier3_inferred_seed.py — DO NOT HAND-EDIT.\n"
        "# Tier-3 name-based inference. Mints a stable canonical for every\n"
        "# EEE id that still no_match-es after Tier 1/2 + curation, so the\n"
        "# resolve-based gate returns non-null. resolution_source=inferred,\n"
        "# review_status=draft. Base edges are alias-confirmed only (NO invented\n"
        "# edges); org-less ids carry org_id=None + tags:[org-unknown] and are\n"
        "# surfaced to org_unknown_review.json (never auto-guessed).\n"
        "# Enrich records ({id, aliases}) union surface forms onto existing\n"
        "# canonicals: org-aware HF folds, collision twins, sentinel-org folds,\n"
        "# and the carry-forward floor (a dropped draft's forms pinned onto the\n"
        "# entity its raw resolves to — nothing is removed).\n"
    )
    # Enrich records: the loader unions these raws onto the existing canonical.
    enrich_records = [
        {"id": hf, "aliases": sorted(raws)}
        for hf, raws in sorted(fold_enrich.items())
        if raws
    ]
    minted.sort(key=lambda e: e["id"])
    out_records = minted + enrich_records
    with open(TIER3_YAML, "w") as f:
        f.write(header)
        yaml.safe_dump(out_records, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    REVIEW_JSON.write_text(json.dumps(
        {
            "_note": (
                "Org-less Tier-3 mints (org_id=None + tags:[org-unknown]). "
                "proposed_org is a SUGGESTION from an inferred base only; NOT "
                "auto-applied. A reviewer / EEE upstream fix must confirm."
            ),
            "count": len(review),
            "entries": sorted(review, key=lambda e: e["raw_value"]),
        },
        indent=1,
    ))

    print("Tier-3 residual (current no_match over 6,720):", len(residual))
    print("minted:", len(minted))
    print(f"folded-to-HF (enrich records): {len(enrich_records)}")
    print("buckets:", dict(buckets))
    print("org-less review entries:", len(review))
    print("wrote", TIER3_YAML.relative_to(REGISTRY_ROOT))
    print("wrote", REVIEW_JSON.relative_to(REGISTRY_ROOT))


if __name__ == "__main__":
    main()
