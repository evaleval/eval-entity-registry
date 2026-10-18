"""Acceptance-gate invariants for model resolution and canonical-graph integrity.

Runs offline against the committed `fixtures/` parquet and the frozen HF oracle
`hf_model_id_resolution.json`. Pins: every EEE id resolves; every oracle HF id
resolves org-aware to its real repo id; no duplicate, dangling or cross-org
graph state; snapshot resolutions, edges and lineage never regress. The full
sweeps are `@pytest.mark.slow` but stay runnable in CI.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eval_entity_registry.lib.seed_io import resolve_oracle_path

from eval_entity_resolver.resolver import Resolver
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES


# --------------------------------------------------------------------------
# Locations (offline; everything is committed)
# --------------------------------------------------------------------------
REGISTRY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REGISTRY_ROOT / "fixtures"
ORGS_YAML = REGISTRY_ROOT / "seed" / "orgs.yaml"
# Shared helper so the gate and the generator scripts resolve the same oracle file.
ORACLE_PATH = resolve_oracle_path()

# Gate floor numbers (the registry's measured baseline).
EXPECTED_TOTAL = 6720   # total EEE ids in the frozen hf_model_id_resolution.json oracle
EXPECTED_ORACLE = 4074  # fixed_exact + fixed_near_miss subset of that oracle


# --------------------------------------------------------------------------
# Module-scoped fixtures (load the warehouse + oracle once)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def resolver() -> Resolver:
    """Resolver over the committed parquet fixtures."""
    assert FIXTURES_DIR.exists(), f"missing fixtures dir: {FIXTURES_DIR}"
    return Resolver.from_parquet(str(FIXTURES_DIR))


@pytest.fixture(scope="module")
def models_df() -> pd.DataFrame:
    """canonical_models as seeded (derived walk columns populated)."""
    df = pd.read_parquet(FIXTURES_DIR / "canonical_models.parquet")
    assert not df.empty
    return df


@pytest.fixture(scope="module")
def oracle() -> dict[str, dict]:
    assert ORACLE_PATH.exists(), f"missing oracle: {ORACLE_PATH}"
    return json.loads(ORACLE_PATH.read_text())["resolutions"]


@pytest.fixture(scope="module")
def hf_to_dev() -> dict[str, str]:
    """HF-namespace -> developer map, built by the same `build_curated_org_map`
    the resolver uses so the gate folds orgs exactly as real resolution does."""
    from eval_entity_resolver.fold import build_curated_org_map
    return build_curated_org_map(yaml.safe_load(ORGS_YAML.read_text()) or [])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _org_of(cid: str) -> str:
    return cid.split("/", 1)[0] if "/" in cid else ""


def _name_of(cid: str) -> str:
    return cid.split("/", 1)[1] if "/" in cid else cid


def _parse_parents(value) -> list[dict]:
    """`parents` is stored as a JSON string; may also arrive as a list or
    pandas NA. Normalise to a list of edge dicts."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except (ValueError, TypeError):
            return []
    # array-like (pyarrow list) or scalar NA
    try:
        if pd.isna(value):  # scalar NA
            return []
    except (ValueError, TypeError):
        pass  # array-like → not a scalar NA
    return [e for e in list(value) if isinstance(e, dict)]


def _parse_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except (ValueError, TypeError):
            return []
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return list(value)


# --------------------------------------------------------------------------
# Sanity: the oracle has the expected shape (guards against a swapped/empty file)
# --------------------------------------------------------------------------
def test_oracle_has_expected_population(oracle):
    assert len(oracle) == EXPECTED_TOTAL, (
        f"oracle key count drifted: {len(oracle)} != {EXPECTED_TOTAL}"
    )
    n_oracle = sum(
        1 for v in oracle.values()
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
    )
    assert n_oracle == EXPECTED_ORACLE, (
        f"fixed_exact+fixed_near_miss count drifted: {n_oracle} != {EXPECTED_ORACLE}"
    )


# --------------------------------------------------------------------------
# 1. COVERAGE — every one of the 6,720 EEE ids resolves to a non-null canonical
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_coverage_all_eee_ids_resolve_non_null(resolver, oracle):
    """Every EEE id in the oracle resolves to a non-null canonical; a single
    no_match is a gate break."""
    no_match = [raw for raw in oracle if resolver.resolve(raw, "model").canonical_id is None]
    assert no_match == [], (
        f"{len(no_match)}/{len(oracle)} EEE ids resolved to no_match (gate floor "
        f"is 0). First few: {no_match[:10]}"
    )


# --------------------------------------------------------------------------
# 2. ORACLE — the 4,074 HF ids resolve ORG-AWARE to their fixed_hf_model_id
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_oracle_org_aware_match(resolver, oracle, hf_to_dev):
    """Each fixed_exact / fixed_near_miss oracle id resolves to a canonical whose
    model name equals the oracle's case-sensitively (HF repo names are
    case-sensitive) and whose org folds to the same developer as the HF org."""
    fixed = {
        raw: meta for raw, meta in oracle.items()
        if meta.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
    }

    # Near-miss redirects that change identity (a wrong HF "did-you-mean") must
    # NOT be followed. Derived from the generator's `nearmiss_changes_identity`
    # so the gate and the generator cannot disagree on which redirects are bad.
    _nearmiss_changes_identity, _hf_oracle_hf_to_dev = _import_nearmiss_rule()
    nm_hf_to_dev = _hf_oracle_hf_to_dev()
    AUDIT_CORRECTED = frozenset(
        raw for raw, meta in fixed.items()
        if meta.get("resolution_status") == "fixed_near_miss"
        and isinstance(meta.get("fixed_hf_model_id"), str)
        and _nearmiss_changes_identity(raw, meta["fixed_hf_model_id"], nm_hf_to_dev)
    )

    def org_aware_equal(resolved_org: str, hf_org: str) -> bool:
        # Two HF orgs of one developer (THUDM / zai-org) must fold to the same developer.
        if resolved_org.lower() == hf_org.lower():
            return True
        fold = lambda x: hf_to_dev.get(x.lower(), x.lower())
        return fold(resolved_org) == fold(hf_org)

    # Compared by resolved canonical, not name: cross-uploader corrections share
    # the redirect's model name but differ in org.
    not_corrected = []
    for raw in (AUDIT_CORRECTED & set(fixed)):
        bad_id = fixed[raw]["fixed_hf_model_id"]
        cid = resolver.resolve(raw, "model").canonical_id
        bad_cid = resolver.resolve(bad_id, "model").canonical_id
        if cid is None or cid == bad_cid:
            not_corrected.append((raw, bad_id, cid))
    assert not_corrected == [], (
        f"{len(not_corrected)} audit-corrected raw(s) still resolve to the oracle's "
        f"bad redirect (the bad-redirect correction regressed): {not_corrected[:10]}"
    )

    # Every other oracle id must org-aware-match its fixed_hf_model_id.
    checkable = {raw: meta for raw, meta in fixed.items() if raw not in AUDIT_CORRECTED}
    passed = 0
    buckets: Counter = Counter()
    samples: list[tuple] = []
    for raw, meta in checkable.items():
        hf_id = meta["fixed_hf_model_id"]
        hf_org, hf_name = _org_of(hf_id), _name_of(hf_id)
        cid = resolver.resolve(raw, "model").canonical_id
        if cid is None:
            buckets["no_match"] += 1
            if len(samples) < 12:
                samples.append((raw, hf_id, None))
            continue
        name_ok = _name_of(cid) == hf_name           # case-sensitive
        org_ok = org_aware_equal(_org_of(cid), hf_org)
        if name_ok and org_ok:
            passed += 1
        else:
            key = "name+org" if (not name_ok and not org_ok) else ("name" if not name_ok else "org")
            buckets[f"{key}_mismatch"] += 1
            if len(samples) < 12:
                samples.append((raw, hf_id, cid))

    assert passed == len(checkable) == EXPECTED_ORACLE - len(AUDIT_CORRECTED & set(fixed)), (
        f"ORACLE org-aware match: {passed}/{len(checkable)} (floor "
        f"{EXPECTED_ORACLE - len(AUDIT_CORRECTED & set(fixed))} = {EXPECTED_ORACLE} - "
        f"{len(AUDIT_CORRECTED & set(fixed))} audit-corrected). Failure buckets: "
        f"{dict(buckets)}. Samples (raw, hf, resolved): {samples}"
    )


# --------------------------------------------------------------------------
# 3. DEDUP — 0 case-insensitive duplicate canonical ids
# --------------------------------------------------------------------------
def test_no_case_insensitive_duplicate_canonical_ids(models_df):
    """Two canonical ids differing only by case is a dedup failure: the HF-cased
    id wins and the lowercase spelling must be an alias on it."""
    lc = Counter(i.lower() for i in models_df["id"])
    dups = {k: v for k, v in lc.items() if v > 1}
    detail = {}
    if dups:
        by_lower = defaultdict(list)
        for i in models_df["id"]:
            if i.lower() in dups:
                by_lower[i.lower()].append(i)
        detail = dict(by_lower)
    assert dups == {}, f"case-insensitive duplicate canonical ids: {detail}"


# --------------------------------------------------------------------------
# 3b. NO ENRICHMENT-MINT — enrichments/*.yaml bridge, never create a canonical
# --------------------------------------------------------------------------
def test_alias_enrichments_never_mint_canonicals():
    """Every `enrichments/*.yaml` record id must be a def id in core.yaml or
    sources/*.generated.yaml: for an unknown id the loader (`cli.py::_absorb`)
    mints a bare, metadata-less canonical that can shadow a real one.
    `skip_source_ids` are subtracted because the loader drops them, never mints."""
    seed_models = REGISTRY_ROOT / "seed" / "models"

    def _entry_ids(obj) -> set[str]:
        entries = obj.get("entries", []) if isinstance(obj, dict) else obj
        return {e["id"] for e in (entries or []) if isinstance(e, dict) and e.get("id")}

    core = yaml.safe_load((seed_models / "core.yaml").read_text()) or {}
    def_ids = _entry_ids(core)
    for src in sorted((seed_models / "sources").glob("*.generated.yaml")):
        def_ids |= _entry_ids(yaml.safe_load(src.read_text()) or [])
    skip = set(core.get("skip_source_ids") or []) if isinstance(core, dict) else set()

    minted = {}
    for enr in sorted((seed_models / "enrichments").glob("*.yaml")):
        bad = sorted(_entry_ids(yaml.safe_load(enr.read_text()) or []) - def_ids - skip)
        if bad:
            minted[enr.name] = bad
    assert minted == {}, (
        f"enrichment record(s) MINT a new canonical (id absent from core/sources "
        f"and not in skip_source_ids): {minted}"
    )


# --------------------------------------------------------------------------
# 4. NO DANGLING — every parents[].id references an existing canonical id
# --------------------------------------------------------------------------
def test_no_dangling_parent_edges(models_df):
    """Every `parents[].id` edge across canonical_models must point at a row
    that exists as a canonical id. A dangling parent edge breaks the
    group/family/lineage walks and the producer's lineage derivation."""
    idset = set(models_df["id"])
    dangling: list[tuple[str, str]] = []
    for cid, parents in zip(models_df["id"], models_df["parents"]):
        for edge in _parse_parents(parents):
            target = edge.get("id")
            if target is not None and target not in idset:
                dangling.append((cid, target))
    assert dangling == [], (
        f"{len(dangling)} dangling parent edges (parent id not a canonical). "
        f"First few: {dangling[:10]}"
    )


# --------------------------------------------------------------------------
# 5. MEMBERSHIP — group/family non-null (self for singletons);
#    lineage_origin_model_id null-at-origin (NOT self)
# --------------------------------------------------------------------------
def test_group_and_family_membership_is_total(models_df):
    """`model_group_id` and `model_family_id` are total: non-null for every
    model, self for singletons and roots."""
    null_group = models_df[models_df["model_group_id"].isna()]
    null_family = models_df[models_df["model_family_id"].isna()]
    assert null_group.empty, (
        f"{len(null_group)} models have null model_group_id "
        f"(must be self at root). e.g. {null_group['id'].head(10).tolist()}"
    )
    assert null_family.empty, (
        f"{len(null_family)} models have null model_family_id "
        f"(must be self at root). e.g. {null_family['id'].head(10).tolist()}"
    )
    # Guard against non-null being satisfied trivially while self-at-root regressed.
    self_group = (models_df["model_group_id"] == models_df["id"]).sum()
    assert self_group > 0, "no model is its own group root — self-fallback regressed"


def test_lineage_origin_model_id_is_null_at_origin_not_self(models_df):
    """`lineage_origin_model_id` is NULL at origin with no self-fallback (unlike
    `model_group_id`); a self-pointing row regresses that semantics."""
    self_pointing = models_df[models_df["lineage_origin_model_id"] == models_df["id"]]
    assert self_pointing.empty, (
        f"{len(self_pointing)} models have lineage_origin_model_id == self "
        f"(must be null at origin). e.g. {self_pointing['id'].head(10).tolist()}"
    )
    # Some rows must carry an origin, or the check above is vacuous.
    has_origin = models_df["lineage_origin_model_id"].notna().sum()
    assert has_origin > 0, (
        "no model carries a lineage_origin_model_id — the lineage walk produced "
        "nothing, so the null-at-origin check is trivially green"
    )


# --------------------------------------------------------------------------
# 6. NO NAME-ONLY CROSS-ORG MERGE — a same model-name under two orgs stays distinct
# --------------------------------------------------------------------------
def test_no_name_only_cross_org_merge(resolver, models_df):
    """A model name shared by two orgs must not collapse to one canonical: both
    known `Llama-3-Instruct-8B-SimPO` uploads exist as distinct canonicals and
    resolve to themselves."""
    a = "haoranxu/Llama-3-Instruct-8B-SimPO"
    b = "princeton-nlp/Llama-3-Instruct-8B-SimPO"
    idset = set(models_df["id"])
    assert a in idset and b in idset, (
        "spot-check canonicals missing from fixtures — pick a fresh known "
        f"multi-org name. have a={a in idset} b={b in idset}"
    )
    assert _name_of(a) == _name_of(b)
    assert _org_of(a) != _org_of(b)

    res_a = resolver.resolve(a, "model").canonical_id
    res_b = resolver.resolve(b, "model").canonical_id
    assert res_a == a, f"{a} resolved to {res_a}, not itself (cross-org merge?)"
    assert res_b == b, f"{b} resolved to {res_b}, not itself (cross-org merge?)"
    assert res_a != res_b, "two distinct-org same-name models merged to one canonical"


# --------------------------------------------------------------------------
# 7. TIER-3 HONESTY — inferred rows are draft; org-less inferred are org-unknown
# --------------------------------------------------------------------------
def test_tier3_inferred_rows_are_draft(models_df):
    """Every Tier-3 mint (`resolution_source=inferred`) must be `review_status
    = draft` — the registry never silently promotes a name-inferred guess to
    reviewed."""
    inferred = models_df[models_df["resolution_source"] == "inferred"]
    assert not inferred.empty, "no inferred rows in fixtures — Tier-3 mints vanished"
    non_draft = inferred[inferred["review_status"] != "draft"]
    assert non_draft.empty, (
        f"{len(non_draft)} inferred rows are not review_status=draft "
        f"(Tier-3 honesty break). e.g. {non_draft['id'].head(10).tolist()}"
    )


def test_orgless_inferred_rows_are_tagged_org_unknown(models_df):
    """Inferred rows without an extractable org land in the `unknown` sentinel
    org and must carry the `org-unknown` tag, so the bucket is visible whether
    a consumer filters on the FK or the tag."""
    inferred = models_df[models_df["resolution_source"] == "inferred"]
    orgless = inferred[inferred["org_id"].isna() | (inferred["org_id"] == "unknown")]
    assert not orgless.empty, (
        "no org-less inferred rows in fixtures — the org-less bucket vanished; "
        "if intentional, update this test"
    )
    missing_tag = [
        cid for cid, tags in zip(orgless["id"], orgless["tags"])
        if "org-unknown" not in _parse_tags(tags)
    ]
    assert missing_tag == [], (
        f"{len(missing_tag)} org-less inferred rows lack the 'org-unknown' tag "
        f"(org-less honesty break). e.g. {missing_tag[:10]}"
    )


# --------------------------------------------------------------------------
# ORG-INTEGRITY — org FKs resolve, one org row per developer, oracle ids are
# real HF repos.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def orgs_df() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURES_DIR / "canonical_orgs.parquet")
    assert not df.empty
    return df


def test_no_dangling_org_fk(models_df, orgs_df):
    """Every `org_id` AND `lineage_origin_model_org_id` on a model must name a
    real `canonical_orgs` row. A dangling org FK = a model whose developer
    can't be resolved downstream (null `developer` in the producer)."""
    org_ids = set(orgs_df["id"].astype(str))
    org_fk = {x for x in models_df["org_id"].astype("string").dropna().unique() if x not in org_ids}
    lin_fk = {
        x for x in models_df["lineage_origin_model_org_id"].astype("string").dropna().unique()
        if x not in org_ids
    }
    assert org_fk == set(), f"{len(org_fk)} dangling org_id FK(s): {sorted(org_fk)[:10]}"
    assert lin_fk == set(), f"{len(lin_fk)} dangling lineage-org FK(s): {sorted(lin_fk)[:10]}"


def test_no_case_split_orgs(orgs_df):
    """One canonical_orgs row per developer: no two org ids may differ only by
    case (`Tencent` vs `tencent`), which would fragment the developer in every
    downstream org list/facet."""
    ci: Counter = Counter(str(i).lower() for i in orgs_df["id"])
    split = {lo: [i for i in orgs_df["id"] if str(i).lower() == lo] for lo, n in ci.items() if n > 1}
    assert split == {}, f"{len(split)} case-split org(s): {list(split.values())[:8]}"


@pytest.mark.slow
def test_oracle_canonical_id_is_real_hf_repo_id(resolver, oracle):
    """Each oracle `fixed_hf_model_id` (a real HF repo id) must resolve: directly
    as the canonical id, or via a confirmed alias when the canonical sits under
    the original publisher's namespace."""
    fixed = [
        m["fixed_hf_model_id"] for m in oracle.values()
        if isinstance(m.get("fixed_hf_model_id"), str)
    ]
    unreachable = [
        fx for fx in fixed if resolver.resolve(fx, "model").canonical_id is None
    ]
    assert unreachable == [], (
        f"{len(unreachable)}/{len(fixed)} oracle real-HF ids unreachable: {unreachable[:10]}"
    )


def test_canonical_id_equals_metadata_hf_id(models_df):
    """A model recording a real HF repo id in `metadata.hf_id` must use it as its
    canonical_id (unless that id is already another canonical); otherwise the
    canonical is a synthetic slug that 404s on HF."""
    ids = set(models_df["id"].astype(str))
    viol = []
    for r in models_df.itertuples():
        md = getattr(r, "metadata", None)
        if not isinstance(md, str):
            continue
        try:
            h = json.loads(md).get("hf_id")
        except Exception:
            h = None
        if isinstance(h, str) and "/" in h and h != str(r.id) and h not in ids:
            viol.append((str(r.id), h))
    assert viol == [], (
        f"{len(viol)} models whose canonical_id != metadata.hf_id (real absent): {viol[:10]}"
    )


@pytest.mark.slow
def test_old_folded_form_resolves_to_real_hf_id(resolver, models_df):
    """The org-folded spelling (`alibaba/Qwen2.5-7B`) resolves to the real-HF
    canonical via an alias; that alias is what stops `regenerate_catalog`
    (refresh_from_modelsdev) re-minting folded canonicals."""
    ids = set(models_df["id"].astype(str))
    pairs = [
        ("alibaba/Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
        ("meta/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("deepseek/DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
    ]
    checked = 0
    for folded, real in pairs:
        if real not in ids:
            continue  # not in this fixture snapshot — skip, don't false-fail
        checked += 1
        got = resolver.resolve(folded, "model").canonical_id
        assert got == real, f"{folded!r} resolved to {got!r}, expected {real!r}"
    assert checked > 0, "no folded->real pairs present in fixtures to check"


def test_no_separator_split_orgs(orgs_df):
    """No two org ids differ only by separator/case (`prime-intellect` vs
    `PrimeIntellect`); pairs verified as genuinely distinct HF uploaders are
    allowlisted in seed/orgs_distinct_allowlist.yaml."""
    import yaml as _yaml
    allow_path = REGISTRY_ROOT / "seed" / "orgs_distinct_allowlist.yaml"
    allow = set(_yaml.safe_load(allow_path.read_text()) or []) if allow_path.exists() else set()
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    by: dict = defaultdict(list)
    for oid in orgs_df["id"]:
        by[norm(oid)].append(str(oid))
    split = {k: v for k, v in by.items() if len(v) > 1 and not set(v) <= allow}
    assert split == {}, f"{len(split)} separator/case-split org group(s): {list(split.values())[:8]}"


def test_no_real_hf_id_duplicated_by_slug(models_df):
    """A real HF id must never be duplicated by a non-real folded/slug variant
    (`alibaba/qwen-2-5-7b-instruct`). Two distinct real repos that normalize
    alike are allowed."""
    fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            fixed.add(fx)
    from eval_entity_resolver.fold import build_curated_org_map
    hf_to_dev = build_curated_org_map(yaml.safe_load(ORGS_YAML.read_text()) or [])

    def fold(org):
        return hf_to_dev.get(org.lower(), org)

    def ndup(s):
        s = s.lower()
        s = re.sub(r"([a-z])[-_ /]+(\d)", r"\1\2", s)
        s = re.sub(r"(\d)\.(\d)(?![bmkt])", r"\1-\2", s)
        return re.sub(r"[-_ /]+", "-", s)

    def has_real_hf_id(row):
        """Oracle fixed id, hf-sourced mint, or == its own metadata.hf_id."""
        cid = str(row.id)
        if cid in fixed:
            return True
        if isinstance(row.resolution_source, str) and row.resolution_source == "hf":
            return True
        md = getattr(row, "metadata", None)
        if isinstance(md, str):
            try:
                return json.loads(md).get("hf_id") == cid
            except Exception:
                return False
        return False

    clusters: dict = defaultdict(lambda: {"real": [], "nonreal": []})
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" not in cid:
            continue
        org, name = cid.split("/", 1)
        bucket = "real" if has_real_hf_id(row) else "nonreal"
        clusters[(fold(org), ndup(name))][bucket].append(cid)
    # Violation = a real id sharing a cluster with a NON-real slug variant.
    bad = [
        sorted(c["real"] + c["nonreal"])
        for c in clusters.values()
        if c["real"] and c["nonreal"]
    ]
    assert bad == [], f"{len(bad)} real-HF id(s) duplicated by a non-real slug: {bad[:8]}"


# --------------------------------------------------------------------------
# NO MINT-DUPE — no models.dev-minted canonical still confidently matches a
# real HF id. Re-runs the fold script's own `decide_fold` predicate, which
# catches brand-prefix + org-spelling dupes (`alibaba/qwen-qwq-32b` vs
# `Qwen/QwQ-32B`) that test_no_real_hf_id_duplicated_by_slug cannot cluster.
# --------------------------------------------------------------------------
def _import_nearmiss_rule():
    """The generator's near-miss rule + hf_to_dev builder, so the gate's
    exemption set derives from the same rule."""
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import generate_hf_oracle_seed as gho  # noqa: E402

    def _hf_to_dev():
        return gho.build_hf_to_dev(gho._load_curated_orgs())

    return gho.nearmiss_changes_identity, _hf_to_dev


def _import_fold_module():
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fold_modelsdev_dupes as fold  # noqa: E402

    return fold


def _model_aliases_by_canonical() -> dict[str, list[str]]:
    """Alias raw_values per model canonical: the surface the fold predicate reads."""
    apath = FIXTURES_DIR / "aliases.parquet"
    out: dict[str, list[str]] = defaultdict(list)
    if not apath.exists():
        return out
    adf = pd.read_parquet(apath)
    adf = adf[adf["entity_type"] == "model"]
    for row in adf.itertuples():
        cid = getattr(row, "canonical_id", None)
        rv = getattr(row, "raw_value", None)
        if isinstance(cid, str) and isinstance(rv, str) and rv:
            out[cid].append(rv)
    return out


# Reviewed models.dev mints the fold predicate flags but which are NOT dupes
# (base vs dated/quant/size variant, family pointer vs leaf). The gate fails on
# a dead (unused) waiver, so the list stays tight. Currently empty.
_KNOWN_NON_DUPE_MINTS: frozenset[str] = frozenset()


def test_no_minted_modelsdev_canonical_shadows_real_hf_id(models_df):
    """No `resolution_source=models_dev` canonical still has a confident real-HF
    match under the fold script's own `decide_fold`; every waiver in
    `_KNOWN_NON_DUPE_MINTS` must still be live."""
    fold = _import_fold_module()

    # reconstruct the entry-list shape decide_fold expects, from the fixtures
    alias_map = _model_aliases_by_canonical()
    entries: list[dict] = []
    for row in models_df.itertuples():
        cid = str(row.id)
        org_id = getattr(row, "org_id", None)
        dn = getattr(row, "display_name", None)
        rsrc = getattr(row, "resolution_source", None)
        md = getattr(row, "metadata", None)
        entries.append({
            "id": cid,
            "org_id": org_id if isinstance(org_id, str) else None,
            "display_name": dn if isinstance(dn, str) else None,
            "resolution_source": rsrc if isinstance(rsrc, str) else None,
            "aliases": alias_map.get(cid, []),
            "metadata": md if isinstance(md, str) else None,
        })

    hf_to_dev = fold.build_hf_to_dev()
    hf_ids, alias_to_hf, by_org_name, _ = fold.build_hf_targets(entries, hf_to_dev)

    mints = [e for e in entries if e["resolution_source"] == "models_dev"]
    assert mints, "no models_dev canonicals in fixtures — gate cannot run (reseed?)"

    leftover = []
    for m in mints:
        if m["id"] in _KNOWN_NON_DUPE_MINTS:
            continue
        f = fold.decide_fold(m, hf_ids, alias_to_hf, by_org_name, hf_to_dev)
        if f is not None:
            leftover.append((f["mint_id"], f["hf_target"], f["match_type"]))

    assert leftover == [], (
        f"{len(leftover)} models.dev-minted canonical(s) still confidently match "
        f"a real HF id (mint-dupe shadow not folded): {leftover[:12]}"
    )

    # A waiver that no longer matches the predicate is dead and must be removed.
    by_mint = {e["id"]: e for e in mints}
    dead = [
        a for a in _KNOWN_NON_DUPE_MINTS
        if a not in by_mint
        or fold.decide_fold(by_mint[a], hf_ids, alias_to_hf, by_org_name, hf_to_dev) is None
    ]
    assert dead == [], (
        f"{len(dead)} dead _KNOWN_NON_DUPE_MINTS waiver(s) (no longer a flagged "
        f"models_dev mint-dupe) — remove them to keep the allowlist tight: {sorted(dead)}"
    )


# --------------------------------------------------------------------------
# org-conditional quant grouping: model_group_id never crosses a developer
# --------------------------------------------------------------------------
def test_model_group_id_does_not_cross_developer_org(models_df):
    """A model and its `model_group_id` root share the same developer `org_id`:
    a third-party quant (`unsloth/...-bnb-4bit` of `microsoft/phi-4`) keeps its
    own group, so its scores never merge into the base lab's model."""
    org_by_id = dict(
        zip(models_df["id"].astype(str), models_df["org_id"].astype("string"))
    )
    bad = []
    for cid, grp in zip(
        models_df["id"].astype(str), models_df["model_group_id"].astype("string")
    ):
        if grp is None or pd.isna(grp) or grp == cid:
            continue
        o_self = org_by_id.get(cid)
        o_grp = org_by_id.get(grp)
        if (
            o_self is not None and o_grp is not None
            and not pd.isna(o_self) and not pd.isna(o_grp)
            and o_self != o_grp
        ):
            bad.append((cid, grp, o_self, o_grp))
    assert bad == [], (
        f"{len(bad)} model(s) whose model_group_id root is a DIFFERENT developer "
        f"org (cross-org identity grouping is forbidden; a community quant must "
        f"keep its own group). First few: {bad[:10]}"
    )


# --------------------------------------------------------------------------
# ORACLE SNAPSHOT — resolution never regresses an already-seen canonical/alias.
# The HF-id gates above cover only HF-resolvable ids; this guards the full
# snapshot at curation/oracle_snapshot/ (incl. closed-API / NA / curated
# canonicals): every snapshot id and alias still resolves to the same
# developer. Names need not match (id spellings may change). Intentional
# divergences are enumerated in the exemption sets below.
# --------------------------------------------------------------------------
ORACLE_SNAPSHOT_DIR = REGISTRY_ROOT / "curation" / "oracle_snapshot"

# Invented `{wrong-developer}/{slug}` ids an earlier generator minted for
# community models (a Llama finetune under `meta/`, a NIM-served model under
# `nvidia/`). Never a raw in EEE data or a models.dev key, and each mislabels
# the developer; dropped from the seed but still present in the snapshots.
_INVENTED_WRONG_ORG_IDS: frozenset = frozenset({
    "alibaba/cogview4",
    "alibaba/qwen3-coder-480b-a35b-instruct-int4-mixed-ar",
    "alibaba/qwenlong-l1-32b",
    "amazon/manta-flash-1-0",
    "amazon/manta-mini-1-0",
    "amazon/manta-pro-1-0",
    "deepseek/aion-1-0-mini",
    "deepseek/deepseek-v3-1-nex-n1",
    "meta/aion-1-0",
    "meta/dracarys-72b-instruct",
    "meta/l3-1-70b-euryale-v2-2",
    "meta/l3-1-euryale-70b",
    "meta/l3-3-70b-loki-v2-0",
    "meta/l3-3-euryale-70b",
    "meta/l3-3-ms-evalebis-70b",
    "meta/l3-3-ms-evayale-70b",
    "meta/l3-3-ms-nevoria-70b",
    "meta/l3-3-nevoria-r1-70b",
    "meta/l3-8b-lunaris",
    "meta/l3-8b-stheno-v3-2",
    "meta/l3-euryale-70b",
    "meta/l3-lunaris-8b",
    "meta/llama-3-lumimaid-70b-v0-1",
    "meta/lumimaid-v0-2-70b",
    "meta/magnum-v2-72b",
    "meta/magnum-v4-72b",
    "meta/meta-llama-3-70b-instruct-abliterated-v3-5",
    "meta/mythomax-l2-13b",
    "meta/neuraldaredevil-8b-abliterated",
    "meta/remm-slerp-l2-13b",
    "meta/the-omega-abomination-l-70b-v1-0",
    "mistralai/dolphin-mistral-24b-venice-edition",
    "mistralai/e5-mistral-7b-instruct",
    "mistralai/sorcererlm-8x22b",
    "moonshotai/kimi-k2-5-nvfp4",
    "nvidia/bge-m3",
    "nvidia/esm2-650m",
    "nvidia/esmfold",
    "nvidia/google-paligemma",
    "nvidia/llama-3-1-nemotron-70b-instruct-hf-abliterated",
    "nvidia/mistral-nemotron",
    "nvidia/mixtral-8x7b-instruct",
    "nvidia/solar-10-7b-instruct",
    "nvidia/whisper-large-v3",
    "openai/aion-labs-aion-2-0",
    "openai/dmind-1",
    "openai/dmind-1-mini",
    "openai/inflection-3-pi",
    "openai/inflection-3-productivity",
    "openai/mirothinker-v1-5-235b",
    "openai/osmosis-structure-0-6b",
    "openai/wizardlm-2-8x22b",
    "unknown/perplexity-sonar-reasoning",
    "voyageai/voyage-3-5",
    "voyageai/voyage-3-5-lite",
    "zai/intellect-3",
})

# Snapshot ids whose resolution lands on a different developer AND a different
# leaf (malformed snapshot id: embedded org or placeholder host). Reviewed.
_ORACLE_CANON_EXEMPT: frozenset = frozenset({
    "openai/aion-labs-aion-2-0",           # oracle embedded the org in the leaf -> aion-labs/aion-2-0
    "unknown/perplexity-sonar-reasoning",  # `unknown/` placeholder host -> perplexity/sonar-reasoning
    "unknown/cohere-embed-v-4-0",          # `unknown/` placeholder host -> cohere/embed-v4-0 (collision fold)
    "unknown/writer.palmyra-x4",           # `unknown/` placeholder host, org in the leaf -> writer/palmyra-x4
    # ai21 == ai21-labs; the real repo leaf gains an `ai21-` brand prefix.
    "ai21/jamba-1.5-large",
    "ai21/jamba-1.5-mini",
    # Qwen finetune mis-filed under alibaba; the real repo moves the `eva` brand
    # from leaf suffix to prefix -> EVA-UNIT-01/eva-qwen2-5-32b-v0-2.
    "alibaba/qwen2-5-32b-eva-v0-2",
    # Sao10K's model mis-attributed to `meta/`; the corrected id also moves the
    # size token -> sao10k/l3-lunaris-8b.
    "meta/l3-8b-lunaris",
}) | _INVENTED_WRONG_ORG_IDS

# Snapshot aliases whose attribution was itself wrong. Keyed by bare raw_value
# or (entity_type, raw_value).
_ORACLE_ALIAS_EXEMPT: frozenset = frozenset({
    # `vercel` is an inference platform (seed/inference_platforms.yaml), not a
    # developer; the host prefix is stripped, so no_match is correct.
    ("org", "vercel"),
}) | _INVENTED_WRONG_ORG_IDS


def _org_aware_equal(resolved_org: str, oracle_org: str, hf_to_dev: dict) -> bool:
    if resolved_org.lower() == oracle_org.lower():
        return True
    fold = lambda x: hf_to_dev.get(x.lower(), x.lower())
    return fold(resolved_org) == fold(oracle_org)


def _leaf_norm(cid: str) -> str:
    from eval_entity_resolver.normalization import normalize
    return normalize(cid.split("/", 1)[1] if "/" in cid else cid).replace(" ", "")


@pytest.mark.slow
def test_phase0_oracle_canonicals_preserved(resolver, hf_to_dev):
    """Every snapshot canonical id still resolves non-null to an org-aware-equal
    developer. A different developer is allowed only when the model leaf is
    identical (a corrected attribution, `meta/aion-1-0` -> `aion-labs/aion-1-0`);
    a different developer AND leaf is a re-attribution and fails unless exempted."""
    import pandas as pd
    oc = pd.read_parquet(ORACLE_SNAPSHOT_DIR / "canonical_models.parquet")
    no_match, wrong_dev = [], []
    for row in oc.itertuples():
        oid = str(row.id)
        if oid in _ORACLE_CANON_EXEMPT:
            continue
        cid = resolver.resolve(oid, "model").canonical_id
        if cid is None:
            no_match.append(oid)
        elif ("/" in oid and "/" in cid
              and not _org_aware_equal(_org_of(cid), _org_of(oid), hf_to_dev)
              and _leaf_norm(cid) != _leaf_norm(oid)):  # not a same-model org correction
            wrong_dev.append((oid, cid))
    assert not no_match, (
        f"{len(no_match)} oracle-snapshot canonical(s) now resolve to no_match "
        f"(already-seen resolution regressed). First few: {sorted(no_match)[:15]}"
    )
    assert not wrong_dev, (
        f"{len(wrong_dev)} oracle-snapshot canonical(s) now resolve to a DIFFERENT "
        f"developer AND a different model leaf (silent re-attribution). First few: {wrong_dev[:15]}"
    )


@pytest.mark.slow
def test_phase0_oracle_aliases_preserved(resolver):
    """Every snapshot alias raw still resolves non-null for its own entity_type
    and source_config: the backstop for non-model / non-EEE aliases."""
    import pandas as pd
    apath = ORACLE_SNAPSHOT_DIR / "aliases.parquet"
    adf = pd.read_parquet(apath)
    adf = adf[adf["status"] != "rejected"]
    no_match = set()
    for row in adf.itertuples():
        rv = getattr(row, "raw_value", None)
        et = getattr(row, "entity_type", None)
        if not isinstance(rv, str) or not rv or not isinstance(et, str):
            continue
        if rv in _ORACLE_ALIAS_EXEMPT or (et, rv) in _ORACLE_ALIAS_EXEMPT:
            continue
        sc = getattr(row, "source_config", None)
        sc = sc if isinstance(sc, str) and sc else None
        if resolver.resolve(rv, et, sc).canonical_id is None:
            no_match.add((et, rv))
    assert not no_match, (
        f"{len(no_match)} oracle-snapshot alias(es) now resolve to no_match for their "
        f"entity_type (already-seen resolution regressed). By type: "
        f"{ {t: sum(1 for tt, _ in no_match if tt == t) for t in sorted({tt for tt, _ in no_match})} }. "
        f"First few: {sorted(no_match)[:15]}"
    )


def test_model_display_names_are_humanized(models_df):
    """Every model has a non-empty display_name that is not just its id; the
    seed loader humanizes the tail, so a failure means a raw slug would render
    in the frontend."""
    dn = models_df["display_name"].fillna("").astype(str).str.strip()
    ids = models_df["id"].astype(str)
    empty = sorted(models_df.loc[dn == "", "id"].tolist())
    id_placeholder = sorted(models_df.loc[(dn != "") & (dn == ids), "id"].tolist())
    assert not empty, f"{len(empty)} model(s) have an empty display_name: {empty[:15]}"
    assert not id_placeholder, (
        f"{len(id_placeholder)} model(s) have display_name == id (a raw slug, not a "
        f"label) — the load-time humanizer should have replaced it: {id_placeholder[:15]}"
    )


# Snapshot (id, parent_id) edges allowed not to land; any other lost edge fails.
_ORACLE_EDGE_EXEMPT: frozenset = frozenset({
    # Inverted tier3 edge (the base coder as a "finetune" of Intel's int4 quant).
    ("alibaba/qwen-3-coder-480b", "Intel/Qwen3-Coder-480B-A35B-Instruct-int4-mixed-AutoRound"),
    # Real parent is deepseek-ai/DeepSeek-Coder-V2-Base (per hub-stats baseModels).
    ("deepseek-ai/DeepSeek-Coder-V2-Instruct", "deepseek/deepseek-coder-v2"),
    # Inverted: Lite-Base is the pretrained origin; keeping the edge would make it
    # a finetune of its own instruct sibling.
    ("deepseek/deepseek-coder-v2-lite-base", "deepseek/deepseek-coder-v2-lite"),
    # Dated snapshots re-typed from `finetune` to identity-preserving
    # `variant/version` (same parent): a refinement, not a lost edge.
    ("google/gemini-1.5-pro-0514", "google/gemini-1.5-pro"),
    ("google/gemini-1.5-pro-0924", "google/gemini-1.5-pro"),
    ("google/gemini-exp-1114", "google/gemini-exp"),
    ("google/gemini-exp-1121", "google/gemini-exp"),
    ("alibaba/qwen3-vl-plus-2025-09-23", "alibaba/qwen3-vl-plus"),
    ("anthropic/claude-sonnet-4-5-thinking-20250929", "anthropic/claude-sonnet-4.5-thinking"),
    ("openai/gpt-4-5-2025-02-27", "openai/gpt-4.5"),
    ("openai/gpt-4-turbo-2024-04-09", "openai/gpt-4-turbo"),
    ("openai/gpt-4.1-2025-04-14", "openai/gpt-4.1"),
    ("openai/gpt-4.1-nano-2025-04-14", "openai/gpt-4.1-nano"),
    ("openai/gpt-4.5-preview-2025-02-27", "openai/gpt-4.5"),
    ("openai/gpt-5-mini-2025-08-07", "openai/gpt-5-mini"),
    ("openai/gpt-5-nano-2025-08-07", "openai/gpt-5-nano"),
    ("openai/gpt-5-pro-2025-10-06", "openai/gpt-5-pro"),
    ("openai/gpt-5.1-high-2025-11-12", "openai/gpt-5.1"),
    ("openai/gpt-5.1-instant-2025-11-12", "openai/gpt-5.1"),
    ("openai/gpt-5.1-medium-2025-11-12", "openai/gpt-5.1"),
    ("openai/gpt-5.1-thinking-2025-11-12", "openai/gpt-5.1-thinking"),
    ("openai/gpt-5.2-2025-12-11", "openai/gpt-5.2"),
    ("openai/gpt-5.2-pro-2025-12-11", "openai/gpt-5.2-pro"),
    ("openai/gpt-5.4-2026-03-05", "openai/gpt-5.4"),
    ("openai/gpt-5.4-pro-2026-03-05", "openai/gpt-5.4-pro"),
})


@pytest.mark.slow
def test_phase0_oracle_edges_preserved(resolver, models_df):
    """Every snapshot typed parent edge (relationship + axis) on a surviving
    canonical is preserved, repointed to the parent's surviving canonical.
    Edges whose parent is absent from the registry are allowed."""
    import json as _json
    oc = pd.read_parquet(ORACLE_SNAPSHOT_DIR / "canonical_models.parquet")

    def _edges(v):
        if isinstance(v, str):
            try:
                return _json.loads(v) or []
            except Exception:
                return []
        return v if isinstance(v, list) else []

    def _edge_key(target, rel, axis):
        # `axis` is variant-only; a stray axis on another relationship (two
        # sources typing one edge differently) must not read as a lost edge.
        return (target, rel, axis if rel == "variant" else None)

    cur: dict[str, set] = {}
    for r in models_df.itertuples():
        cur[str(r.id)] = {
            _edge_key(e.get("id"), e.get("relationship"), e.get("axis"))
            for e in _edges(getattr(r, "parents", None)) if isinstance(e, dict)
        }

    home: dict[str, str] = {}

    def _home(i: str):
        if i not in home:
            home[i] = resolver.resolve(i, "model").canonical_id
        return home[i]

    missing, parent_gone = [], 0
    for row in oc.itertuples():
        oes = _edges(getattr(row, "parents", None))
        if not oes:
            continue
        t = _home(str(row.id))
        if t is None:
            continue  # canonical gone -> covered by the snapshot-canonicals gate above
        tedges = cur.get(t, set())
        for e in oes:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            tp = _home(str(e["id"]))
            if tp is None:
                parent_gone += 1
                continue
            if tp == t:
                continue
            key = _edge_key(tp, e.get("relationship"), e.get("axis"))
            if (str(row.id), str(e["id"])) in _ORACLE_EDGE_EXEMPT:
                continue
            if key not in tedges:
                missing.append((str(row.id), t, str(e["id"]), tp, e.get("relationship"), e.get("axis")))
    assert not missing, (
        f"{len(missing)} oracle-snapshot typed parent edge(s) lost on the surviving "
        f"canonical (relationship/axis link not preserved). First few: {missing[:15]}"
    )


@pytest.mark.slow
def test_resolve_surfaces_typed_edges_and_ancestry(resolver, models_df):
    """The typed edge graph is queryable at resolve time: a model with stored
    parents surfaces `parents`, and a group/family member surfaces `ancestry`."""
    missing_parents, missing_ancestry = [], []
    for r in models_df.itertuples():
        cid = str(r.id)
        if not _parse_parents(getattr(r, "parents", None)):
            continue
        res = resolver.resolve(cid, "model")
        if not res.parents:
            missing_parents.append(cid)
        roots = {
            x for x in (getattr(r, "model_family_id", None), getattr(r, "model_group_id", None))
            if isinstance(x, str) and x != cid
        }
        if roots and not res.ancestry:
            missing_ancestry.append(cid)
    assert not missing_parents, (
        f"{len(missing_parents)} model(s) with stored parents whose resolve() surfaces "
        f"NO typed edges. First few: {missing_parents[:15]}"
    )
    assert not missing_ancestry, (
        f"{len(missing_ancestry)} grouped/family model(s) whose resolve() surfaces NO "
        f"ancestry chain. First few: {missing_ancestry[:15]}"
    )


# Snapshot (field, id) lineage memberships that cannot be preserved: the root is
# no longer a distinct canonical, so the model is correctly a singleton root.
_ORACLE_LINEAGE_EXEMPT: frozenset = frozenset({
    # Dated snapshots re-typed finetune -> variant/version: same identity as the
    # family, so no separate lineage origin.
    ("lineage_origin_model_id", "google/gemini-1.5-pro-0514"),
    ("lineage_origin_model_id", "google/gemini-1.5-pro-0924"),
    ("lineage_origin_model_id", "google/gemini-exp-1114"),
    ("lineage_origin_model_id", "google/gemini-exp-1121"),
    # Family root shifted with the parent-edge correction (see _ORACLE_EDGE_EXEMPT).
    ("model_family_id", "deepseek-ai/DeepSeek-Coder-V2-Instruct"),
    # Folds into the HF-true DeepSeek-Coder-V2-Lite-Base, a clean root.
    ("lineage_origin_model_id", "deepseek/deepseek-coder-v2-lite-base"),
    # The snapshot family root was an invented mint of the same model; it folded
    # into the repo, which is now its own single-member family.
    ("model_family_id", "Qwen/Qwen3-Coder-480B-A35B-Instruct"),
    ("model_family_id", "Qwen/Qwen3.6-35B-A3B"),
    ("model_family_id", "alibaba/qwen3-coder-30b-a3b-instruct"),
    # Undated and dated spellings are the one HF repo Ministral-3-8B-Instruct-2512;
    # the version edge between them collapsed with the fold.
    ("model_group_id", "mistral/ministral-3-8b-instruct-2512"),
    ("model_group_id", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ("model_group_id", "Qwen/Qwen3-VL-32B-Thinking"),
    ("model_group_id", "Qwen/Qwen3-VL-8B-Thinking"),
    ("model_group_id", "microsoft/Phi-4-mini-reasoning"),
    ("model_family_id", "Qwen/Qwen3-30B-A3B"),
    ("model_family_id", "Qwen/Qwen3-VL-32B-Thinking"),
    ("model_family_id", "Qwen/Qwen3-VL-8B-Thinking"),
    ("model_family_id", "microsoft/Phi-4-mini-reasoning"),
    ("model_family_id", "xiaomi/mimo-v2-flash"),
    # The slug folds into nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16; the
    # snapshot's family root was a size edge to a non-existent `...-120b` base
    # (A12B is MoE active-param notation, not a size).
    ("model_family_id", "nvidia/nemotron-3-super-120b-a12b"),
    # Dated snapshots re-typed finetune -> variant/version: no finetune/quant
    # lineage origin (model_group_id is gained, not lost).
    ("lineage_origin_model_id", "alibaba/qwen3-vl-plus-2025-09-23"),
    ("lineage_origin_model_id", "anthropic/claude-sonnet-4-5-thinking-20250929"),
    ("lineage_origin_model_id", "openai/gpt-4-5-2025-02-27"),
    ("lineage_origin_model_id", "openai/gpt-4-turbo-2024-04-09"),
    ("lineage_origin_model_id", "openai/gpt-4.1-2025-04-14"),
    ("lineage_origin_model_id", "openai/gpt-4.1-nano-2025-04-14"),
    ("lineage_origin_model_id", "openai/gpt-4.5-preview-2025-02-27"),
    ("lineage_origin_model_id", "openai/gpt-5-mini-2025-08-07"),
    ("lineage_origin_model_id", "openai/gpt-5-nano-2025-08-07"),
    ("lineage_origin_model_id", "openai/gpt-5-pro-2025-10-06"),
    ("lineage_origin_model_id", "openai/gpt-5.1-high-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.1-instant-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.1-medium-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.1-thinking-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.2-2025-12-11"),
    ("lineage_origin_model_id", "openai/gpt-5.2-pro-2025-12-11"),
    ("lineage_origin_model_id", "openai/gpt-5.4-2026-03-05"),
    ("lineage_origin_model_id", "openai/gpt-5.4-pro-2026-03-05"),
    # Surviving canonical of a collision fold (lib/collision_fold.py); as version
    # snapshots they carry a variant/version edge, so no finetune origin.
    ("lineage_origin_model_id", "anthropic/claude-3.5-sonnet-2024-06-20"),
    ("lineage_origin_model_id", "anthropic/claude-3.5-sonnet-2024-10-22"),
    # Origin came from the inverted tier3 edge (see _ORACLE_EDGE_EXEMPT); the raw
    # now folds onto the real base repo, which is at origin.
    ("lineage_origin_model_id", "alibaba/qwen-3-coder-480b"),
    # Tier3 scaffolds now curated aliases of the real Llama-4 canonicals, which
    # are themselves at origin.
    ("lineage_origin_model_id", "meta/llama4-maverick"),
    ("lineage_origin_model_id", "meta/llama4-scout"),
    # The API-spelling mint merged into the real repo; the snapshot's family root
    # WAS that mint.
    ("model_family_id", "microsoft/Phi-4-multimodal-instruct"),
    # Phi-4-mini-reasoning is the mode variant of Phi-4-mini-instruct, not vice
    # versa; the snapshot's family root walked the inverted edge.
    ("model_family_id", "microsoft/Phi-4-mini-instruct"),
    # "reasoning" IS the real Thinking repo (core.yaml skip_source_ids fold), not
    # a finetune of a base; no base Qwen3-Omni-30B-A3B canonical exists.
    ("lineage_origin_model_id", "alibaba/qwen3-omni-30b-a3b-reasoning"),
})


@pytest.mark.slow
def test_phase0_oracle_lineage_no_loss(resolver, models_df):
    """No surviving snapshot canonical loses a non-trivial model_group_id /
    model_family_id / lineage_origin_model_id. Checks presence, not value: added
    edges may walk to a truer origin; exact edges are guarded by
    test_phase0_oracle_edges_preserved."""
    oc = pd.read_parquet(ORACLE_SNAPSHOT_DIR / "canonical_models.parquet")
    cur = {str(x.id): x for x in models_df.itertuples()}
    home: dict = {}

    def _home(i: str):
        if not isinstance(i, str):
            return None
        if i not in home:
            home[i] = resolver.resolve(i, "model").canonical_id
        return home[i]

    def _v(x, f):
        v = getattr(x, f, None)
        return v if isinstance(v, str) and v else None

    lost = []
    for ob in oc.itertuples():
        oid = str(ob.id)
        t = _home(oid)
        if t is None or t not in cur:
            continue  # canonical gone -> guarded elsewhere
        T = cur[t]
        for f in ("model_group_id", "model_family_id", "lineage_origin_model_id"):
            ov = _v(ob, f)
            nontrivial = ov is not None and (f == "lineage_origin_model_id" or ov != oid)
            if not nontrivial or (f, oid) in _ORACLE_LINEAGE_EXEMPT:
                continue
            tv = _v(T, f)
            t_nontrivial = tv is not None and (f == "lineage_origin_model_id" or tv != t)
            if not t_nontrivial:
                lost.append((f, oid, t))
    assert not lost, (
        f"{len(lost)} oracle canonical(s) LOST a lineage field (group/family/origin "
        f"dropped to trivial/null). By field: "
        f"{ {f: sum(1 for ff, _, _ in lost if ff == f) for f in sorted({ff for ff, _, _ in lost})} }. "
        f"First few: {lost[:12]}"
    )


# --------------------------------------------------------------------------
# BARE-NAME OWNERSHIP — when several models share a normalized leaf, the bare
# alias belongs to the owner seed precedence selects (lab first, else the
# unique confirmed entry), not to whichever entry seeds first. Guards
# cli.py `seed` (lab_leaf_owner / strong_leaf_owner) against regen flips.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aliases_df() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURES_DIR / "aliases.parquet")
    assert not df.empty
    return df


def _nz(v) -> str:
    """NA/None-safe str."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


@pytest.fixture(scope="module")
def leaf_ownership(models_df, orgs_df):
    """Leaf-ownership maps recomputed from the output tables (the post-hoc
    view of the seed's precedence)."""
    from eval_entity_resolver.normalization import normalize

    kind = dict(zip(orgs_df["id"].astype(str), orgs_df["kind"].astype(str)))
    lab_claimants: dict[str, set[str]] = {}
    strong_claimants: dict[str, set[str]] = {}
    strong_ids: set[str] = set()
    lab_ids: set[str] = set()
    for r in models_df.itertuples():
        cid = _nz(r.id)
        is_lab = kind.get(_nz(r.org_id)) == "lab"
        if is_lab:
            lab_ids.add(cid)
        if "/" not in cid:
            continue
        confirmed = _nz(r.resolution_source) == "hf" or _nz(r.review_status) == "reviewed"
        if not confirmed:
            continue
        norm = normalize(cid.split("/", 1)[1])
        strong_ids.add(cid)
        strong_claimants.setdefault(norm, set()).add(cid)
        if is_lab:
            lab_claimants.setdefault(norm, set()).add(cid)
    return {
        "lab_claimants": lab_claimants,
        "strong_claimants": strong_claimants,
        "strong_ids": strong_ids,
        "lab_ids": lab_ids,
    }


def test_lab_leaf_ownership_is_unambiguous(leaf_ownership):
    """No two confirmed LAB models may share a normalized leaf — the seed's
    lab_leaf_owner keeps the first claimant, so a second would make the bare
    name's owner load-order-dependent across regens."""
    multi = {n: sorted(c) for n, c in leaf_ownership["lab_claimants"].items() if len(c) > 1}
    assert multi == {}, f"{len(multi)} normalized leaf(s) owned by 2+ lab models: {multi}"


def test_bare_alias_points_at_lab_leaf_owner(aliases_df, leaf_ownership):
    """Every global model alias whose normalized form IS a lab-confirmed
    model's leaf must resolve to that lab model (prefer-official, tier 1)."""
    from eval_entity_resolver.normalization import normalize

    lab_claimants = leaf_ownership["lab_claimants"]
    mal = aliases_df[
        (aliases_df["entity_type"] == "model") & aliases_df["source_config"].isna()
    ]
    bad = []
    for raw, cid in zip(mal["raw_value"].astype(str), mal["canonical_id"].astype(str)):
        owners = lab_claimants.get(normalize(raw))
        if owners and len(owners) == 1 and cid not in owners:
            bad.append((raw, cid, next(iter(owners))))
    assert bad == [], (
        f"{len(bad)} bare alias(es) not pointing at their lab leaf owner "
        f"(raw, points_at, lab_owner): {bad[:10]}"
    )


def test_bare_alias_points_at_unique_confirmed_owner(aliases_df, leaf_ownership):
    """Prefer-official tier 2: an alias whose normalized form matches exactly
    ONE confirmed model's leaf (and no lab's) must resolve to it — unless the
    alias's own entry is itself confirmed or lab-owned (the seed never takes
    a name from a real repo or a lab entry, only from unconfirmed drafts)."""
    from eval_entity_resolver.normalization import normalize

    lab_claimants = leaf_ownership["lab_claimants"]
    strong_claimants = leaf_ownership["strong_claimants"]
    strong_ids = leaf_ownership["strong_ids"]
    lab_ids = leaf_ownership["lab_ids"]
    mal = aliases_df[
        (aliases_df["entity_type"] == "model") & aliases_df["source_config"].isna()
    ]
    bad = []
    for raw, cid in zip(mal["raw_value"].astype(str), mal["canonical_id"].astype(str)):
        norm = normalize(raw)
        if norm in lab_claimants:
            continue  # tier 1 governs
        owners = strong_claimants.get(norm)
        if not owners or len(owners) != 1:
            continue  # no unique confirmed owner -> no expectation
        owner = next(iter(owners))
        if cid == owner or cid in strong_ids or cid in lab_ids:
            continue  # exempt: alias belongs to a real repo / lab entry
        bad.append((raw, cid, owner))
    assert bad == [], (
        f"{len(bad)} draft-owned alias(es) shadowing a unique confirmed repo "
        f"(raw, points_at, confirmed_owner): {bad[:10]}"
    )


def test_every_model_id_is_a_resolvable_raw(models_df, aliases_df):
    """Every canonical model id appears as an alias raw_value: the anti-shadow
    display drop must never discard an entity's own id alias."""
    raws = set(
        aliases_df[aliases_df["entity_type"] == "model"]["raw_value"].astype(str)
    )
    missing = [i for i in models_df["id"].astype(str) if i not in raws]
    assert missing == [], f"{len(missing)} model id(s) with no alias row: {missing[:10]}"


# --------------------------------------------------------------------------
# OPENROUTER ID ADOPTION gates
# --------------------------------------------------------------------------
MODELSDEV_SNAPSHOT = REGISTRY_ROOT / "tests" / "fixtures" / "modelsdev_api.snapshot.json"
OPENROUTER_BEFORE_SNAPSHOT = REGISTRY_ROOT / "curation" / "openrouter_adoption_before.parquet"
SEED_MODELS_DIR = REGISTRY_ROOT / "seed" / "models"


def _import_refresh_module():
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import refresh_from_modelsdev as rfm  # noqa: E402

    return rfm


def _load_source_entries() -> list[tuple[str, dict]]:
    """(source-name, entry) for every record in core.yaml + sources/*.generated.yaml."""
    out: list[tuple[str, dict]] = []
    paths = [SEED_MODELS_DIR / "core.yaml"] + sorted(
        (SEED_MODELS_DIR / "sources").glob("*.generated.yaml")
    )
    for p in paths:
        doc = yaml.safe_load(p.read_text())
        entries = doc.get("entries", doc) if isinstance(doc, dict) else doc
        for e in entries or []:
            if isinstance(e, dict) and e.get("id"):
                out.append((p.name, e))
    return out


def _core_skip_ids() -> set[str]:
    doc = yaml.safe_load((SEED_MODELS_DIR / "core.yaml").read_text())
    if not isinstance(doc, dict):
        return set()
    return set(doc.get("skip_ids") or []) | set(doc.get("skip_source_ids") or [])


def test_no_parent_edge_names_a_renamed_away_id():
    """No `parents[].id` in core or any source names a renamed-away id: every
    edge target must be a full entry's id, never a form surviving only as an alias."""
    recs = _load_source_entries()
    skip = _core_skip_ids()
    full_ids = {e["id"] for _, e in recs if "display_name" in e and e["id"] not in skip}
    alias_owner: dict[str, str] = {}
    for _, e in recs:
        for a in e.get("aliases") or []:
            if a:
                alias_owner.setdefault(a, e["id"])
    bad = []
    for src, e in recs:
        if e["id"] in skip:
            continue
        for edge in e.get("parents") or []:
            t = edge.get("id") if isinstance(edge, dict) else None
            if t and t not in full_ids:
                bad.append((src, e["id"], t, alias_owner.get(t)))
    assert bad == [], (
        f"{len(bad)} parent edge(s) point at a non-canonical id (renamed-away "
        f"ids resolve only as aliases; 4th field = the alias's owner). "
        f"First few: {bad[:12]}"
    )


# Pinned pre-existing non-adoption duplicate clusters (spelling twins awaiting
# curation). Ratchet: may only shrink; an unlisted cluster fails as a new dup.
_PREEXISTING_DUP_TAIL: frozenset = frozenset({
    ("ai2/molmo-7b-d", "allenai/molmo-7b-d"),
    ("ai2/olmo-3-7b-instruct", "allenai/olmo-3-7b-instruct"),
    ("ai2/olmo-3-7b-think", "allenai/olmo-3-7b-think"),
    ("ai21-labs/ai21-jamba-1-5-large", "ai21-labs/jamba-1-5-large"),
    ("ai21-labs/ai21-jamba-1-5-mini", "ai21-labs/jamba-1-5-mini"),
    ("ai21/jamba-1.6-large", "ai21/jamba-large-1-6", "ai21labs/jamba-large-1.6"),
    ("ai21/jamba-1.6-mini", "ai21/jamba-mini-1-6", "ai21labs/jamba-mini-1.6"),
    ("ai21/jamba-1.7-mini", "ai21/jamba-mini-1-7"),
    ("alibaba/qwen-14b-chat", "alibaba/qwen-chat-14b"),
    ("alibaba/qwen-72b-chat", "alibaba/qwen-chat-72b"),
    ("alibaba/qwen3-4b-2507-instruct", "qwen/qwen3-4b-instruct-2507"),
    ("amazon/amazon.nova-lite-v1:0", "amazon/nova-lite", "amazon/nova-lite-v1:0", "aws/nova-lite"),
    ("amazon/amazon.nova-micro-v1:0", "amazon/nova-micro", "amazon/nova-micro-v1:0", "aws/nova-micro"),
    ("amazon/amazon.nova-pro-v1:0", "amazon/nova-pro", "amazon/nova-pro-v1:0", "aws/nova-pro"),
    ("amazon/nova-premier", "amazon/nova-premier-v1:0", "aws/nova-premier"),
    ("anthropic/Claude-3.5-Sonnet(Oct)", "anthropic/claude-3.5-sonnet", "anthropic/claude-sonnet-3.5"),
    ("anthropic/anthropic/claude-3-7-sonnet-20250219", "anthropic/claude-3-7-sonnet-20250219"),
    ("anthropic/claude-4-opus-20250514", "anthropic/claude-opus-4-20250514"),
    ("anthropic/claude-4-opus-thinking", "anthropic/claude-opus-4-thinking"),
    ("anthropic/claude-4-sonnet-20250514", "anthropic/claude-sonnet-4-20250514"),
    ("anthropic/claude-4-sonnet-thinking", "anthropic/claude-sonnet-4-thinking"),
    ("anthropic/claude-4.5-opus-thinking", "anthropic/claude-opus-4-5-thinking"),
    ("anthropic/claude-4.5-sonnet-thinking", "anthropic/claude-sonnet-4.5-thinking"),
    ("anthropic/claude-4.6-opus-thinking", "anthropic/claude-opus-4-6-thinking"),
    ("google/gemini-1.5-flash", "google/gemini-flash-1-5"),
    # Veo 3.1 vs Veo 3.1 Fast are distinct products; they cluster only because
    # the tag-strip fold eats the `fast` token.
    ("google/veo-3-1", "google/veo-3.1-fast"),
    ("google/text-bison", "google/text-bison@001"),
    ("google/text-unicorn", "google/text-unicorn@001"),
    ("lmms-lab/llava-video-7b", "lmms-lab/video-llava-7b"),
    ("meta-llama/Llama-3.1-70B-Instruct", "meta/llama-3-1-instruct-70b"),
    ("meta-llama/Llama-3.1-8B-Instruct", "meta/llama-3-1-instruct-8b"),
    ("meta-llama/Llama-3.2-1B-Instruct", "meta/llama-3-2-instruct-1b"),
    ("meta-llama/Llama-3.2-3B-Instruct", "meta/llama-3-2-instruct-3b"),
    ("meta-llama/Llama-3.2-90B-Vision-Instruct", "meta/llama-3-2-instruct-90b-vision"),
    ("meta-llama/Llama-3.3-70B-Instruct", "meta/llama-3-3-instruct-70b"),
    ("meta-llama/llama-3.1", "meta/Llama 3.1"),
    ("meta/Llama-2-13b-chat", "meta/llama-2-chat-13b"),
    ("meta/Llama-2-70b-chat", "meta/llama-2-chat-70b"),
    ("meta/Llama-2-7b-chat", "meta/llama-2-chat-7b"),
    ("meta/llama-3-1-405b-instruct", "meta/llama-3-1-instruct-405b"),
    ("mosaicml/MPT-Instruct-30B", "mosaicml/mpt-30b-instruct"),
    ("nvidia/Llama-3.1-Nemotron-70B-Instruct", "nvidia/llama-3-1-nemotron-instruct-70b"),
    ("openai/Lingma Agent + Lingma SWE-GPT 72b (v0918)", "openai/Lingma Agent + Lingma SWE-GPT 72b (v0925)"),
    ("openai/Lingma Agent + Lingma SWE-GPT 7b (v0918)", "openai/Lingma Agent + Lingma SWE-GPT 7b (v0925)"),
    ("tiiuae/Falcon-Instruct-40B", "tiiuae/falcon-40b-instruct"),
    ("tiiuae/Falcon-Instruct-7B", "tiiuae/falcon-7b-instruct"),
    ("xai/grok-3-mini", "xai/grok-3-mini-fast"),
    ("xai/grok-4", "xai/grok-4-fast"),
    ("xai/grok-4-1", "xai/grok-4-1-fast"),
})


def test_no_two_full_entries_share_folded_identity(hf_to_dev):
    """No adoption-touched cluster of full entries shares a folded identity
    (curated-org-folded, tag-stripped, order-insensitive): the dup class a
    rename would otherwise carry forward forever. A cluster is touched when a
    member is OpenRouter-adopted or its id is an OpenRouter snapshot key. The
    pre-existing non-adoption tail is pinned in `_PREEXISTING_DUP_TAIL` and may
    only shrink. All-real-HF clusters and size-signature differences are allowed."""
    rfm = _import_refresh_module()
    from eval_entity_registry.lib.collision_fold import _bsizes

    oracle_fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            oracle_fixed.add(fx)

    def _attested(src: str, e: dict) -> bool:
        if src in ("hf_oracle.generated.yaml", "hub_stats.generated.yaml"):
            return True
        if e.get("resolution_source") == "hf" or e["id"] in oracle_fixed:
            return True
        md = e.get("metadata")
        if isinstance(md, str):
            try:
                m = json.loads(md)
                if m.get("hf_deferred") or m.get("hf_id") == e["id"]:
                    return True
            except (ValueError, TypeError):
                pass
        return False

    snap = json.loads(MODELSDEV_SNAPSHOT.read_text())
    or_keys: set = set()
    for k in (snap.get("openrouter", {}).get("models") or {}):
        or_keys.add(k)
        or_keys.add(k.lstrip("~"))

    skip = _core_skip_ids()
    clusters: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    for src, e in _load_source_entries():
        if "display_name" not in e or e["id"] in skip or "/" not in e["id"]:
            continue
        org, name = e["id"].split("/", 1)
        dev = hf_to_dev.get(org.lower(), org.lower())
        clusters[(dev, rfm._identity_sig(name), _bsizes(name))].append((src, e))

    def _adopted(e: dict) -> bool:
        md = e.get("metadata")
        if isinstance(md, str):
            try:
                return json.loads(md).get("openrouter_adopted") is True
            except (ValueError, TypeError):
                return False
        return False

    dups, tail = [], []
    for key, members in clusters.items():
        ids = sorted({e["id"] for _, e in members})
        if len(ids) < 2:
            continue
        if all(_attested(src, e) for src, e in members):
            continue  # real-HF cluster — allowed
        touched = any(_adopted(e) for _, e in members) or any(
            i in or_keys for i in ids
        )
        (dups if touched else tail).append(ids)
    assert dups == [], (
        f"{len(dups)} folded-identity duplicate cluster(s) involving an "
        f"OpenRouter-adopted / OpenRouter-key id (adoption id-thrash class): "
        f"{dups[:10]}"
    )
    # Ratchet: a cluster not in the pinned set is a new duplicate; re-pin
    # deliberately when curating one away.
    new_tail = [c for c in sorted(map(tuple, tail)) if c not in _PREEXISTING_DUP_TAIL]
    assert new_tail == [], (
        f"{len(new_tail)} NEW non-adoption folded-identity dup cluster(s) not in "
        f"the pinned pre-existing tail: {new_tail[:10]}"
    )
    assert len(tail) <= len(_PREEXISTING_DUP_TAIL), (
        f"pre-existing dup tail grew: {len(tail)} > {len(_PREEXISTING_DUP_TAIL)}"
    )


@pytest.mark.slow
def test_every_openrouter_snapshot_key_resolves(resolver):
    """Every OpenRouter model key in the frozen snapshot resolves to some
    canonical. `openrouter/*` router pseudo-endpoints are routing products, excluded."""
    snap = json.loads(MODELSDEV_SNAPSHOT.read_text())
    keys = [
        k for k in (snap.get("openrouter", {}).get("models") or {})
        if not k.lstrip("~").lower().startswith("openrouter/")
    ]
    assert keys, "frozen snapshot lost its openrouter provider"
    unresolved = [
        k for k in keys if resolver.resolve(k, "model").canonical_id is None
    ]
    assert unresolved == [], (
        f"{len(unresolved)}/{len(keys)} OpenRouter snapshot key(s) resolve to "
        f"no_match: {unresolved[:15]}"
    )


_BARE_ID_CEILING = 250


def test_bare_canonical_org_ambiguity_is_explicit(models_df):
    """A bare (org-less) canonical id carries ``org-unknown`` exactly when its
    ``org_id`` is unresolved, so the ambiguous tail stays visible without
    inventing ownership. The ceiling is a growth ratchet with headroom for a
    normal upstream refresh."""
    bad = []
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" in cid:
            continue
        org_id = getattr(row, "org_id", None)
        has_org = isinstance(org_id, str) and bool(org_id)
        tagged_unknown = "org-unknown" in _parse_tags(getattr(row, "tags", None))
        if has_org == tagged_unknown:
            bad.append((cid, org_id if has_org else None, tagged_unknown))
    assert bad == [], (
        f"{len(bad)} bare canonical id(s) have inconsistent org visibility; "
        f"expected org-unknown iff org_id is null: {bad[:15]}"
    )
    bare = [str(i) for i in models_df["id"].astype(str) if "/" not in str(i)]
    assert len(bare) <= _BARE_ID_CEILING, (
        f"bare (org-less) canonical id count grew: {len(bare)} > "
        f"{_BARE_ID_CEILING}; inspect new upstream mints before re-pinning"
    )


def test_canonical_id_prefix_is_external_or_folds_to_org(models_df, hf_to_dev):
    """Every canonical id prefix is a real external namespace adopted verbatim
    (HF-attested, or an OpenRouter key whose prefix org agrees with the entry's
    developer) or folds through the curated org map to the entry's `org_id`.
    Bare ids are covered by test_bare_canonical_org_ambiguity_is_explicit."""
    from eval_entity_resolver.fold import _norm_org_key

    oracle_fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            oracle_fixed.add(fx)
    snap = json.loads(MODELSDEV_SNAPSHOT.read_text())
    or_keys = set()
    for k in (snap.get("openrouter", {}).get("models") or {}):
        or_keys.add(k)
        or_keys.add(k.lstrip("~"))

    def fold(o: str) -> str:
        return hf_to_dev.get(o.lower(), o)

    # Curated core.yaml defs are judgment calls: a raw's mis-prefixed spelling
    # may be kept as the id with the corrected developer org.
    core_doc = yaml.safe_load((SEED_MODELS_DIR / "core.yaml").read_text())
    core_defs = {
        e["id"] for e in (core_doc.get("entries") or [])
        if isinstance(e, dict) and e.get("id") and "display_name" in e
    }

    bad = []
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" not in cid:
            continue
        org_id = getattr(row, "org_id", None)
        org_id = org_id if isinstance(org_id, str) and org_id else None
        prefix = cid.split("/", 1)[0]
        # External namespaces adopted verbatim; curated core judgment calls.
        if cid in oracle_fixed or cid in core_defs:
            continue
        if cid in or_keys:
            # A key under another developer's namespace must not launder a
            # mis-attributed row (the generator's `_openrouter_org_agrees` fold).
            a0, b0 = fold(prefix), fold(org_id) if org_id else None
            if org_id and (a0 == b0 or _norm_org_key(a0) == _norm_org_key(b0)):
                continue
        rsrc = getattr(row, "resolution_source", None)
        if isinstance(rsrc, str) and rsrc == "hf":
            continue
        md = getattr(row, "metadata", None)
        if isinstance(md, str):
            try:
                m = json.loads(md)
                if m.get("hf_id") == cid or m.get("hf_deferred") or m.get("openrouter_adopted"):
                    continue
            except (ValueError, TypeError):
                pass
        # Invented / community ids: prefix must fold to the entry's org.
        if org_id is None:
            if prefix.lower() == "unknown":
                continue
            bad.append((cid, None))
            continue
        a, b = fold(prefix), fold(org_id)
        if a == b or _norm_org_key(a) == _norm_org_key(b):
            continue
        bad.append((cid, org_id))
    assert bad == [], (
        f"{len(bad)} canonical id(s) whose prefix neither is an adopted external "
        f"namespace nor folds to the entry's org_id: {bad[:15]}"
    )


# Curated corrections: raws whose pre-migration resolution was itself a
# mis-attribution. Reviewed exceptions only.
_ADOPTION_REGRESSION_EXEMPT: frozenset = frozenset({
    "z-ai-glm-5-turbo",     # turbo is a sibling product, not the GLM-5 base
    "z-ai-glm-5v-turbo",    # same correction for 5V
    # Raw now reaches the true product entity; the old home was a near-miss.
    "Veo-3.1-Fast",                        # google/veo-3.1-fast, not veo3-1
    "deepseek/deepseek-v4-pro-lightning",  # Lightning is its own product, not DeepSeek-V4-Pro
    # Same model split across two tier3 sentinel drafts (org-less and org-present).
    "unknown/amazon.nova-premier-v1:0",
    # Inverse leg of the 5V-Turbo split: base spellings resolve to the 5V base.
    "glm-5v",
    "zai/glm-5v",
    # The served "phi-4-mini" IS Phi-4-mini-instruct (API names drop -instruct).
    "Phi-4 Mini",
    "microsoft/phi-4-mini",
}) | _INVENTED_WRONG_ORG_IDS


@pytest.mark.slow
def test_openrouter_adoption_no_resolution_regression(resolver, aliases_df):
    """Every canonical id and alias that resolved before the adoption migration
    (curation/openrouter_adoption_before.parquet) still resolves to the same
    entity; an identity shift counts as the same entity only when the old id
    is an alias on, or resolves exact/normalized to, the new entry."""
    before = pd.read_parquet(OPENROUTER_BEFORE_SNAPSHOT)
    alias_pairs = set(
        zip(
            aliases_df[aliases_df["entity_type"] == "model"]["raw_value"].astype(str),
            aliases_df[aliases_df["entity_type"] == "model"]["canonical_id"].astype(str),
        )
    )
    old_home: dict[str, str | None] = {}

    def _old_home(old: str):
        # Exact/normalized only: a fuzzy hit would launder a dropped draft as a rename.
        if old not in old_home:
            res = resolver.resolve(old, "model")
            old_home[old] = (
                res.canonical_id
                if res.strategy in ("exact", "normalized")
                else None
            )
        return old_home[old]

    no_match, moved = [], []
    for raw, old in zip(before["raw_value"].astype(str), before["canonical_id"].astype(str)):
        if raw in _ADOPTION_REGRESSION_EXEMPT:
            continue
        got = resolver.resolve(raw, "model").canonical_id
        if got is None:
            no_match.append(raw)
            continue
        if got == old or (old, got) in alias_pairs:
            continue
        # Rename/fold: the old id itself follows to the new entity.
        if _old_home(old) == got:
            continue
        moved.append((raw, old, got))
    assert no_match == [], (
        f"{len(no_match)} previously-resolving raw(s) now no_match: {no_match[:15]}"
    )
    assert moved == [], (
        f"{len(moved)} raw(s) resolved to a DIFFERENT entity whose aliases do NOT "
        f"carry the old id (identity break, not a rename): {moved[:15]}"
    )


def test_prefer_official_pinned_examples(aliases_df):
    """Concrete pins for the ownership precedence; each was wrong or
    load-order-dependent without the redirect."""
    mal = aliases_df[
        (aliases_df["entity_type"] == "model") & aliases_df["source_config"].isna()
    ]
    lookup = dict(zip(mal["raw_value"].astype(str), mal["canonical_id"].astype(str)))
    pins = {
        # a fork's bare display must not shadow the official repo (tier 1)
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        # a mis-prefixed draft loses the bare name to the real repo (tier 2)
        "dolphin-2-9.2-qwen2-72b": "dphn/dolphin-2.9.2-qwen2-72b",
        # a bare-slug draft loses to the curated entry that lists it as an alias
        "llama-4-maverick-17b-128e-instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        # a name-inferred draft must not own a real community repo's bare name
        "Reflection-Llama-3.1-70B": "mattshumer/Reflection-Llama-3.1-70B",
        # an entity whose display equals its id still resolves by that id
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL",
    }
    wrong = {
        raw: (lookup[raw], want)
        for raw, want in pins.items()
        if raw in lookup and lookup[raw] != want
    }
    gone = [raw for raw in pins if raw not in lookup]
    assert wrong == {}, f"pinned bare-name owner(s) regressed (raw: (got, want)): {wrong}"
    assert gone == [], (
        f"pinned raw(s) no longer seeded at all — the contested surface form "
        f"disappeared from the seed (upstream refresh dropped it?), not an "
        f"ownership regression: {gone}"
    )


# ---------------------------------------------------------------------------
# Benchmark entity invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmarks_df() -> pd.DataFrame:
    path = REGISTRY_ROOT / "fixtures" / "canonical_benchmarks.parquet"
    if not path.exists():
        pytest.skip("fixtures/canonical_benchmarks.parquet not present (run seed --local)")
    return pd.read_parquet(path)


def test_no_separator_split_benchmarks(benchmarks_df):
    """No two benchmark ids differ only by separators/case (`wild-bench` vs
    `wildbench`), which would split one benchmark across two eval pages.
    Genuinely distinct pairs go in seed/benchmarks_distinct_allowlist.yaml."""
    allow_path = REGISTRY_ROOT / "seed" / "benchmarks_distinct_allowlist.yaml"
    allow: list[set] = []
    if allow_path.exists():
        for group in yaml.safe_load(allow_path.read_text()) or []:
            allow.append(set(group))
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    by: dict = defaultdict(list)
    for bid in benchmarks_df["id"]:
        by[norm(bid)].append(str(bid))
    split = {
        k: v for k, v in by.items()
        if len(v) > 1 and not any(set(v) <= grp for grp in allow)
    }
    assert split == {}, f"{len(split)} separator/case-split benchmark group(s): {list(split.values())[:8]}"


def test_no_slash_in_benchmark_ids(benchmarks_df):
    """'/' in a benchmark id would let a merged-view single-segment eval id
    collide with per-source composite/benchmark ids."""
    bad = [b for b in benchmarks_df["id"] if "/" in str(b)]
    assert bad == [], f"benchmark ids containing '/': {bad[:8]}"


# ---------------------------------------------------------------------------
# Dated snapshots inherit their base's openness
# ---------------------------------------------------------------------------
# A dated API snapshot (`openai/o3-2025-04-16`) is the same identity as its
# moving pointer along the version line; a reasoning-effort row is that
# snapshot served differently. Without typed edges they are parentless leaves
# with a null openness verdict. Each chain must fold into the base's group and
# family and inherit `open_weights`, while every dated id stays its own leaf.
O3_CHAINS = [
    # (id, parent id, axis, expected group/family base)
    ("openai/o3-2025-04-16", "openai/o3", "version", "openai/o3"),
    ("openai/o3-2025-04-16-fc", "openai/o3-2025-04-16", "mode", "openai/o3"),
    ("openai/o3-2025-04-16-high", "openai/o3-2025-04-16", "mode", "openai/o3"),
    ("openai/o3-2025-04-16-low", "openai/o3-2025-04-16", "mode", "openai/o3"),
    ("openai/o3-2025-04-16-medium", "openai/o3-2025-04-16", "mode", "openai/o3"),
    ("openai/o3-2025-04-16-prompt", "openai/o3-2025-04-16", "mode", "openai/o3"),
    ("openai/o3-mini-2025-01-31", "openai/o3-mini", "version", "openai/o3-mini"),
    ("openai/o3-mini-2025-01-31-high", "openai/o3-mini-2025-01-31", "mode", "openai/o3-mini"),
    ("openai/o3-mini-2025-01-31-low", "openai/o3-mini-2025-01-31", "mode", "openai/o3-mini"),
    ("openai/o3-mini-2025-01-31-medium", "openai/o3-mini-2025-01-31", "mode", "openai/o3-mini"),
]


@pytest.mark.parametrize("cid,parent,axis,base", O3_CHAINS, ids=[c[0] for c in O3_CHAINS])
def test_dated_snapshot_chain_inherits_base_openness(models_df, resolver, cid, parent, axis, base):
    row = models_df[models_df["id"] == cid]
    assert len(row) == 1, f"{cid} missing from canonical_models"
    row = row.iloc[0]

    edges = _parse_parents(row["parents"])
    assert [e.get("id") for e in edges] == [parent], (
        f"{cid} parents are {row['parents']!r}, expected exactly [{parent}]"
    )
    assert edges[0].get("relationship") == "variant", edges[0]
    assert edges[0].get("axis") == axis, edges[0]

    assert row["model_group_id"] == base, row["model_group_id"]
    assert row["model_family_id"] == base, row["model_family_id"]
    assert bool(row["open_weights"]) is False and pd.notna(row["open_weights"]), (
        f"{cid} open_weights={row['open_weights']!r}, expected inherited False"
    )

    res = resolver.resolve(cid, "model")
    assert res.canonical_id == cid, (
        f"{cid} resolves to {res.canonical_id!r} — the dated leaf must not "
        f"collapse onto an alias of {base}"
    )


def test_o3_chain_rows_have_a_single_parent(models_df):
    """Layers union edges by parent id; a second edge from another layer would
    give one row two identities (group walk vs lineage walk), so each o3 chain
    row must carry exactly one parent."""
    multi = {
        cid: models_df[models_df["id"] == cid].iloc[0]["parents"]
        for cid, _p, _a, _b in O3_CHAINS
        if len(_parse_parents(models_df[models_df["id"] == cid].iloc[0]["parents"])) > 1
    }
    assert multi == {}, f"o3 chain row(s) with more than one parent: {multi}"


# Legacy `finetune` edges carrying `axis: version` (a field-merge of two layers
# that disagreed on the relationship). Ratchet, not an endorsement.
AXIS_ON_NON_VARIANT_FLOOR = 8


def test_axis_only_ever_qualifies_a_variant_edge(models_df):
    """`axis` is meaningful only on a `variant` edge; a non-variant edge carrying
    one means two layers disagreed on the relationship. The o3 chains must be
    clean outright; the corpus must not grow new instances."""
    bad: list[tuple[str, dict]] = []
    for cid, parents in zip(models_df["id"], models_df["parents"]):
        for edge in _parse_parents(parents):
            if edge.get("axis") is not None and edge.get("relationship") != "variant":
                bad.append((str(cid), edge))
    chain_ids = {cid for cid, _p, _a, _b in O3_CHAINS}
    in_chain = [b for b in bad if b[0] in chain_ids]
    assert in_chain == [], (
        f"o3 chain edge(s) carry an axis on a non-variant relationship "
        f"(a layer disagreed on the relationship and the merge kept both "
        f"fields): {in_chain}"
    )
    assert len(bad) <= AXIS_ON_NON_VARIANT_FLOOR, (
        f"{len(bad)} edge(s) carry an axis on a non-variant relationship, up "
        f"from the known {AXIS_ON_NON_VARIANT_FLOOR}: "
        f"{[b for b in bad][:10]}"
    )
