"""Pins for the models.dev refresh generator: dedup / head pick / mint /
alias emission / carry-forward / reconciliation.

Offline: drives the pinned API snapshot (tests/fixtures/modelsdev_api.snapshot.json)
and the committed sidecars under curation/. Snapshot and underlying index are
frozen from the same pull, so group counts and roots must reproduce exactly.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from conftest import load_script_module

from eval_entity_registry.lib.seed_io import resolve_oracle_path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT  / "curation"
API_CACHE = REPO_ROOT / "tests" / "fixtures" / "modelsdev_api.snapshot.json"
UNDERLYING_INDEX = SPEC_DIR / "modelsdev_underlying_index.json"
RESCUES = SPEC_DIR / "eee_modelsdev_rescues.json"
PLATFORMS = SPEC_DIR / "inference_platforms.proposed.json"

# Frozen-pull dedup floor: distinct underlying groups in curation/modelsdev_underlying_index.json.
EXPECTED_UNDERLYING_GROUPS = 1_279
# EEE rescue floor: rows in curation/eee_modelsdev_rescues.json.
EXPECTED_EEE_RESCUES = 662
# Curated inference-platform count: entries in curation/inference_platforms.proposed.json.
EXPECTED_PLATFORM_COUNT = 138


@pytest.fixture(scope="module")
def mod():
    return load_script_module("refresh_from_modelsdev")


@pytest.fixture(scope="module")
def api():
    if not API_CACHE.exists():
        pytest.skip(f"pinned models.dev API snapshot missing at {API_CACHE}")
    return json.loads(API_CACHE.read_text())


@pytest.fixture(scope="module")
def underlying_index():
    if not UNDERLYING_INDEX.exists():
        pytest.skip(f"underlying index sidecar missing at {UNDERLYING_INDEX}")
    return json.loads(UNDERLYING_INDEX.read_text())


# ---------------------------------------------------------------------------
# 1. Dedup reproduces the committed underlying groups.
# ---------------------------------------------------------------------------
def test_dedup_reproduces_underlying_group_count(mod, api, underlying_index):
    groups = mod.build_underlying_groups(api)
    n = len(groups)
    # Same frozen pull as the index, so the count is exact (no drift band).
    assert n == EXPECTED_UNDERLYING_GROUPS, f"expected {EXPECTED_UNDERLYING_GROUPS} underlying groups, got {n}"
    assert len(underlying_index) == EXPECTED_UNDERLYING_GROUPS


def test_dedup_group_roots_match_committed_index(mod, api, underlying_index):
    """Group roots must equal the committed index exactly: any difference is a
    normalize/canon_key/safe_sig regression. To re-pin, replace the snapshot
    fixture and regenerate the index from it."""
    my_roots = set(mod.build_underlying_groups(api).keys())
    idx_roots = set(underlying_index.keys())
    assert my_roots == idx_roots, (
        f"root set diverged from the frozen index — "
        f"new={sorted(my_roots - idx_roots)[:8]} "
        f"gone={sorted(idx_roots - my_roots)[:8]}"
    )


@pytest.mark.parametrize(
    "root,expect_author,expect_org_hf",
    [
        ("claude-3-5-sonnet", True, "anthropic"),
        ("gpt-4o", True, "openai"),
        ("yi-large", False, None),          # re-host-only, org-unknown
        ("abliterated-model", False, None),  # community re-host, no author lab
    ],
)
def test_dedup_spotcheck_known_groups(mod, api, underlying_index, root, expect_author, expect_org_hf):
    groups = mod.build_underlying_groups(api)
    assert root in groups, f"group {root!r} not produced"
    head = mod.pick_underlying(root, groups[root])
    assert head["has_author_lab_entry"] is expect_author
    assert head["author_org"] == expect_org_hf
    idx_entry = underlying_index[root]
    assert idx_entry["has_author_lab_entry"] is expect_author
    assert idx_entry["author_org"] == expect_org_hf


def _reorder_api(api: dict) -> dict:
    """The same catalog with provider and per-provider model order reversed: a
    deterministic permutation of models.dev's key order."""
    out = {}
    for prov in reversed(list(api.keys())):
        pdata = dict(api[prov])
        models = pdata.get("models")
        if isinstance(models, dict):
            pdata["models"] = {k: models[k] for k in reversed(list(models.keys()))}
        out[prov] = pdata
    return out


def test_grouping_invariant_to_modelsdev_key_order(mod, api):
    """Roots and group membership must not depend on models.dev key order (the
    union-find base is the lexicographically-first key, not API order)."""
    a = mod.build_underlying_groups(api)
    b = mod.build_underlying_groups(_reorder_api(api))
    assert set(a.keys()) == set(b.keys()), (
        f"roots changed under key reorder: "
        f"new={sorted(set(b) - set(a))[:8]} gone={sorted(set(a) - set(b))[:8]}"
    )
    membership = lambda g: {root: sorted(r["raw"] for r in recs) for root, recs in g.items()}
    assert membership(a) == membership(b), "group membership changed under key reorder"


def test_head_pick_invariant_to_modelsdev_key_order(mod, api):
    """head_spelling must be stable under a key reorder, so a provider reordering
    upstream cannot churn a minted id."""
    a = mod.build_underlying_groups(api)
    b = mod.build_underlying_groups(_reorder_api(api))
    for root in a:
        ha = mod.pick_underlying(root, a[root])
        hb = mod.pick_underlying(root, b[root])
        assert ha["head_spelling"] == hb["head_spelling"], (
            f"head_spelling for {root!r} flipped under key reorder: "
            f"{ha['head_spelling']!r} != {hb['head_spelling']!r}"
        )


# ---------------------------------------------------------------------------
# 2. Head selection precedence: author-lab > re-host > null.
# ---------------------------------------------------------------------------
def test_head_pick_author_lab_beats_rehost(mod):
    """The author-lab record's org/name anchor wins over re-hosts, even when
    re-hosts disagree on date/name."""
    root = "claude-3-5-sonnet"
    recs = [
        # Re-host with an earlier (wrong) date and a vanity name.
        dict(provider="venice", raw="claude-3-5-sonnet", norm="claude-3-5-sonnet",
             key=root, family="claude", release="2023-01-01", name="Sonnet (venice)",
             open_weights=False, record={}),
        dict(provider="anthropic", raw="claude-3-5-sonnet-20240620",
             norm="claude-3-5-sonnet-20240620", key=root, family="claude",
             release="2024-06-20", name="Claude 3.5 Sonnet", open_weights=False, record={}),
        dict(provider="openrouter", raw="anthropic/claude-3.5-sonnet",
             norm="claude-3-5-sonnet", key=root, family="claude", release="2024-06-21",
             name="Claude 3.5 Sonnet", open_weights=False, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is True
    assert head["author_org"] == "anthropic"
    assert head["display_name"] == "Claude 3.5 Sonnet"
    assert head["head_spelling"] == "claude-3-5-sonnet-20240620"


def test_head_pick_rehost_only_infers_org_from_family(mod):
    """With no author-lab record, the family token infers the org;
    has_author_lab_entry stays False."""
    root = "llama-3-1-8b"
    recs = [
        dict(provider="togetherai", raw="meta-llama/Llama-3.1-8B",
             norm="llama-3-1-8b", key=root, family="llama", release="2024-07-23",
             name="Llama 3.1 8B", open_weights=True, record={}),
        dict(provider="fireworks-ai", raw="llama-v3p1-8b",
             norm="llama-3-1-8b", key=root, family="llama", release=None,
             name="Llama 3.1 8B", open_weights=True, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is False
    assert head["author_org"] == "meta-llama"   # inferred from family, HF-style
    assert head["open_weights"] is True          # any(True)


def test_head_pick_null_org_when_no_signal(mod):
    """No author lab and no family-org signal: org stays null and head spelling
    falls back to the normalised spelling."""
    root = "abliterated-model"
    recs = [
        dict(provider="abliteration-ai", raw="abliterated-model",
             norm="abliterated-model", key=root, family=None, release="2026-01-06",
             name="Abliterated Model", open_weights=True, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is False
    assert head["author_org"] is None
    assert head["head_spelling"] == "abliterated-model"


# ---------------------------------------------------------------------------
# 3. EEE rescues map to their recorded underlying groups.
# ---------------------------------------------------------------------------
def test_rescues_map_to_underlying_groups(mod, underlying_index):
    if not RESCUES.exists():
        pytest.skip(f"rescues sidecar missing at {RESCUES}")
    rescues = json.loads(RESCUES.read_text())
    assert len(rescues) == EXPECTED_EEE_RESCUES

    key_to_root = {r: r for r in underlying_index}
    mset_to_root: dict[str, list[str]] = defaultdict(list)
    for r in underlying_index:
        mset_to_root[mod.safe_sig(r)].append(r)

    def lookup(eee_id: str) -> str | None:
        n = mod.normalize_modelsdev_id(eee_id)
        k = mod.canon_key_ordered(n)
        if not k:
            return None
        if k in key_to_root:
            return k
        ms = mod.safe_sig(k)
        if ms in mset_to_root:
            return mset_to_root[ms][0]
        return None

    matched = sum(1 for eee_id, rec in rescues.items()
                  if lookup(eee_id) == rec["underlying_model_id"])
    assert matched == EXPECTED_EEE_RESCUES, (
        f"only {matched}/{EXPECTED_EEE_RESCUES} rescues reproduced"
    )


def test_rescues_sample_spotcheck(mod, underlying_index):
    """Named rescues resolve to their recorded group."""
    if not RESCUES.exists():
        pytest.skip("rescues sidecar missing")
    rescues = json.loads(RESCUES.read_text())
    sample = [eee for eee in ("01-ai/yi-large", "01-ai/Yi-1.5-34B-Chat") if eee in rescues]
    assert sample, "expected at least one known sample id in the rescues"
    for eee_id in sample:
        n = mod.normalize_modelsdev_id(eee_id)
        k = mod.canon_key_ordered(n)
        root = k if k in underlying_index else None
        if root is None:
            ms = mod.safe_sig(k)
            for r in underlying_index:
                if mod.safe_sig(r) == ms:
                    root = r
                    break
        assert root == rescues[eee_id]["underlying_model_id"]


# ---------------------------------------------------------------------------
# 4. Edge axis classification.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token,expected",
    [
        ("instruct", ("variant", "training_stage")),
        ("chat", ("variant", "training_stage")),
        ("it", ("variant", "training_stage")),
        ("base", ("variant", "training_stage")),
        ("sonnet", ("variant", "tier")),
        ("haiku", ("variant", "tier")),
        ("opus", ("variant", "tier")),
        ("mini", ("variant", "tier")),
        ("nano", ("variant", "tier")),
        ("flash", ("variant", "tier")),
        ("pro", ("variant", "tier")),
        ("7b", ("variant", "size")),
        ("70b", ("variant", "size")),
        ("405b", ("variant", "size")),
        ("8x7b", ("variant", "size")),
        ("a16b", ("variant", "size")),
        ("thinking", ("variant", "mode")),
        ("vision", ("variant", "modality")),
        ("coder", ("variant", "domain")),
    ],
)
def test_classify_token_axis(mod, token, expected):
    assert mod._classify_token(token) == expected


def test_branded_tier_never_gets_size_edge(mod):
    """A branded tier makes no scale claim: it classifies as `tier`, never
    `size`."""
    for tok in ("haiku", "sonnet", "opus", "mini", "nano", "flash", "pro",
                "small", "medium", "large"):
        rel, axis = mod._classify_token(tok)
        assert axis == "tier", f"{tok} mis-classified as {axis}"
        assert axis != "size"
        assert tok in mod._TIER_TOKENS


def test_size_token_only_for_disclosed_open_weight_tokens(mod):
    """Only a disclosed scale token (name params / MoE spec) is a size token;
    a branded tier or stage token is not."""
    assert mod._is_size_token("70b") is True
    assert mod._is_size_token("8x7b") is True
    assert mod._is_size_token("a16b") is True
    assert mod._is_size_token("1.5b") is True
    assert mod._is_size_token("mini") is False
    assert mod._is_size_token("sonnet") is False
    assert mod._is_size_token("instruct") is False


def test_suffix_chain_llama_size_then_stage(mod):
    """`70b-instruct` classifies as a size edge then a training_stage edge."""
    segments = mod._classify_suffix_segments("70b-instruct")
    assert segments == [
        ("variant", "size", "70b"),
        ("variant", "training_stage", "instruct"),
    ]


def test_instruct_suffix_is_training_stage_in_family_tree(mod):
    """A `-instruct` snapshot emits a variant/training_stage edge (not
    axis=mode) in the built family tree."""
    api = {
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-7b-instruct": {
                    "id": "mistral-7b-instruct", "name": "Mistral 7B Instruct",
                    "release_date": "2023-09-27", "open_weights": True, "family": "mistral",
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, {"mistralai"})
    by_id = {e["id"]: e for e in out}
    instruct = by_id["mistralai/mistral-7b-instruct"]
    assert instruct["parents"] == [{
        "id": "mistralai/mistral-7b", "relationship": "variant", "axis": "training_stage",
    }]


# ---------------------------------------------------------------------------
# 5. PROVIDER_TO_INFERENCE_PLATFORM ids all exist in the curated catalog.
# ---------------------------------------------------------------------------
def test_provider_to_inference_platform_ids_exist(mod):
    if not PLATFORMS.exists():
        pytest.skip(f"inference platforms sidecar missing at {PLATFORMS}")
    catalog = json.loads(PLATFORMS.read_text())
    valid_ids = {p["id"] for p in catalog["platforms"]}
    valid_providers = {p["models_dev_provider"] for p in catalog["platforms"]}

    assert mod.PROVIDER_TO_INFERENCE_PLATFORM, "map is empty"
    # Catalog entries without a models_dev_provider are not provider-map entries.
    assert len(catalog["platforms"]) == EXPECTED_PLATFORM_COUNT
    provider_backed = [p for p in catalog["platforms"] if p.get("models_dev_provider")]
    assert len(mod.PROVIDER_TO_INFERENCE_PLATFORM) == len(provider_backed)

    for provider, platform_id in mod.PROVIDER_TO_INFERENCE_PLATFORM.items():
        assert platform_id in valid_ids, f"platform id {platform_id!r} not in catalog"
        assert provider in valid_providers, f"provider {provider!r} not a known slug"


def test_strict_author_sourced_from_catalog(mod):
    """STRICT_AUTHOR is the kind=author_lab subset of the same catalog."""
    if not PLATFORMS.exists():
        pytest.skip("inference platforms sidecar missing")
    catalog = json.loads(PLATFORMS.read_text())
    author_providers = {
        p["models_dev_provider"] for p in catalog["platforms"]
        if p["kind"] == "author_lab"
    }
    assert mod.STRICT_AUTHOR == author_providers
    assert len(mod.STRICT_AUTHOR) == 29


# ---------------------------------------------------------------------------
# 6. Full-catalog generation: provider-preserving group -> mint -> alias.
# ---------------------------------------------------------------------------
def test_full_generation_seeds_closed_api_and_rehosts(mod, api):
    """The full catalog (closed-API + re-hosts) is seeded: re-host-only groups
    are draft, author trees reviewed, provider spellings aliased with their
    platform."""
    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"

    by_id = {e["id"]: e for e in out}
    assert "openai/gpt-4o" in by_id
    assert by_id["openai/gpt-4o"]["review_status"] == "reviewed"
    assert "yi-large" in by_id
    yi = by_id["yi-large"]
    assert yi["org_id"] is None
    assert yi["review_status"] == "draft"
    assert "org-unknown" in yi["tags"]

    assert all(e["id"].count("/") <= 1 for e in out)

    fin = mod._finalize_entries([dict(e) for e in out])
    fin_by_id = {e["id"]: e for e in fin}
    gpt4o = fin_by_id["openai/gpt-4o"]
    meta = json.loads(gpt4o["metadata"])
    assert meta.get("alias_platforms"), "expected provider platform provenance"
    catalog = json.loads(PLATFORMS.read_text())
    valid_ids = {p["id"] for p in catalog["platforms"]}
    for plat in meta["alias_platforms"].values():
        assert plat in valid_ids
    for e in fin:
        for a in e["aliases"]:
            assert a.count("/") <= 1
            assert not a.startswith("@")


# ---------------------------------------------------------------------------
# 7. Mint-decision rule: defer HF-resolvable groups, mint off-HF ones.
# ---------------------------------------------------------------------------
HF_ORACLE_JSON = resolve_oracle_path()


def test_mint_decision_defers_on_hf_group_and_mints_off_hf(mod):
    """A group that is a real HF repo defers to the HF id (no shadow mint); an
    off-HF group mints. Synthetic authority keeps it independent of the live
    oracle; covers `qwen-qwq-32b` vs `Qwen/QwQ-32B` (org spelling and brand
    prefix both differ)."""
    from eval_entity_resolver.normalization import normalize as _nz

    alias_index = mod._build_org_alias_index()
    # Synthetic authority: {dev_org: {normalized_name: real_hf_id}}.
    authority = {
        "alibaba": {_nz("QwQ-32B"): "Qwen/QwQ-32B"},
    }

    # (a) brand-prefixed key: org Qwen->alibaba, name qwen-qwq-32b->qwq-32b.
    deferred = mod._hf_defer_target(
        "alibaba/qwen-qwq-32b",
        "alibaba",
        ["alibaba/qwen-qwq-32b", "qwen-qwq-32b", "Qwen QwQ 32B"],
        alias_index,
        authority,
    )
    assert deferred == "Qwen/QwQ-32B", "on-HF brand-prefixed group must DEFER"

    # (b) name matches without a prefix strip.
    assert (
        mod._hf_defer_target(
            "alibaba/qwq-32b", "alibaba", ["QwQ-32B", "Qwen/QwQ-32B"], alias_index, authority
        )
        == "Qwen/QwQ-32B"
    )

    # (c) off-HF closed-API group: mint.
    assert (
        mod._hf_defer_target(
            "anthropic/claude-opus-4-5",
            "anthropic",
            ["claude-opus-4-5", "Claude Opus 4.5"],
            alias_index,
            authority,
        )
        is None
    )

    # (d) same name under a different developer never defers.
    assert (
        mod._hf_defer_target(
            "openai/qwq-32b", "openai", ["qwq-32b"], alias_index, authority
        )
        is None
    )

    # (e) org-less group never defers.
    assert (
        mod._hf_defer_target("qwq-32b", None, ["qwq-32b"], alias_index, authority)
        is None
    )


def test_mint_decision_emits_hf_deferred_record_in_generation(mod, api):
    """Against the pinned snapshot + frozen oracle: QwQ-32B defers to the real
    `Qwen/QwQ-32B` with the models.dev spellings as aliases, while closed-API
    families (Claude/Grok) still mint."""
    if not HF_ORACLE_JSON.exists():
        pytest.skip(f"HF oracle missing at {HF_ORACLE_JSON}")
    # Rebuild the cached authority from the real oracle.
    mod._HF_AUTHORITY = None

    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    by_id = {e["id"]: e for e in out}

    def _is_deferred(e):
        return json.loads(e.get("metadata") or "{}").get("hf_deferred") is True

    assert "Qwen/QwQ-32B" in by_id, "expected the real HF id to be emitted"
    qwq = by_id["Qwen/QwQ-32B"]
    assert _is_deferred(qwq)
    assert qwq["resolution_source"] == "models_dev"
    assert "alibaba/qwen-qwq-32b" not in by_id, "shadow dupe must NOT be minted"
    assert "alibaba/qwen-qwq-32b" in qwq["aliases"], "shadow spelling must be an alias"

    grok = [e for e in out if e["id"].startswith("xai/grok-3") and not _is_deferred(e)]
    assert grok, "off-HF Grok family must still mint"
    claude = [
        e for e in out if e["id"].startswith("anthropic/claude") and not _is_deferred(e)
    ]
    assert claude, "off-HF Claude family must still mint"

    deferred_ids = {e["id"] for e in out if _is_deferred(e)}
    assert "Qwen/QwQ-32B" in deferred_ids


# ---------------------------------------------------------------------------
# Reconciliation (non-catalog path): a mint that normalized-collides with an
# existing canonical defers to it, never emits a twin.
# ---------------------------------------------------------------------------
REHOST_REPOINT = SPEC_DIR / "rehost_repoint.json"


def test_reconciliation_suppresses_normalized_collision_with_core(mod, tmp_path):
    """A mint whose normalized id collides with a core canonical under a
    different id is suppressed, and its unique aliases merge onto the owner
    (merge, not drop)."""
    from eval_entity_resolver.normalization import normalize as _nz

    core = tmp_path / "core.yaml"
    core.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "Foo/Bar-7B",
                    "display_name": "Foo Bar 7B",
                    "aliases": ["foo-bar-7b"],
                }
            ]
        )
    )

    generated = [
        # Lowercase twin of the curated canonical: normalized collision.
        {
            "id": "foo/bar-7b",
            "display_name": "foo bar 7b",
            "aliases": ["foo/bar-7b", "foo-bar-7b-extra-alias"],
        },
        # No collision.
        {
            "id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "aliases": ["gpt-4o"],
        },
    ]

    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    by_id = {e["id"]: e for e in out}
    emitted_ids = set(by_id)

    assert "foo/bar-7b" not in emitted_ids, "normalized-colliding twin must be suppressed"
    assert "openai/gpt-4o" in emitted_ids, "non-colliding mint must survive"

    assert "Foo/Bar-7B" in by_id, "expected an enrich record merging onto the owner"
    assert "foo-bar-7b-extra-alias" in by_id["Foo/Bar-7B"]["aliases"], (
        "suppressed mint's unique alias must survive on the owner (merge, not drop)"
    )

    core_canon = "Foo/Bar-7B"
    core_norm = _nz(core_canon)
    for e in out:
        if _nz(e["id"]) == core_norm:
            assert e["id"] == core_canon, (
                f"emitted id {e['id']!r} normalized-collides with core "
                f"canonical {core_canon!r} under a different id"
            )


def test_tier3_is_an_existing_source_for_both_refresh_paths(mod, tmp_path):
    """Both refresh paths include Tier-3 in their owner index, so a mint that
    is a normalized twin of a Tier-3 canonical donates as enrichment instead
    of becoming a second canonical.
    """
    assert mod.TIER3_PATH in mod._NONCATALOG_EXISTING_SOURCES
    assert mod.TIER3_PATH in mod._CATALOG_EXISTING_SOURCES

    tier3 = tmp_path / "tier3.yaml"
    tier3.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "bytedance/seed-2.0-pro",
                    "display_name": "Seed 2.0 Pro",
                    "aliases": ["bytedance/seed-2.0-pro"],
                }
            ]
        )
    )
    generated = [
        {
            "id": "bytedance/seed-2-0-pro",
            "display_name": "Seed 2.0 Pro",
            "aliases": ["seed-2-0-pro", "seed-2.0-pro"],
            "release_date": "2026-02-14",
        }
    ]

    out = mod.reconcile_generated_against_existing(generated, sources=(tier3,))
    full_ids = {e["id"] for e in out if "display_name" in e}
    assert "bytedance/seed-2-0-pro" not in full_ids
    (enrichment,) = [e for e in out if e["id"] == "bytedance/seed-2.0-pro"]
    assert "seed-2-0-pro" in enrichment["aliases"]
    assert enrichment["weak"]["release_date"] == "2026-02-14"


def test_reconciliation_rewrites_surviving_parent_edges_off_suppressed_mints(mod, tmp_path):
    """A surviving entry whose parent edge pointed at a suppressed mint is
    repointed to the owner, never left dangling."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump([{
        "id": "Acme/Base-7B", "display_name": "Acme Base 7B",
        "aliases": ["acme-base-7b"],
    }]))
    generated = [
        # Normalized collision with Acme/Base-7B: suppressed.
        {"id": "acme/base-7b", "display_name": "Acme Base 7B v0", "aliases": []},
        {
            "id": "acme/base-7b-instruct",
            "display_name": "Acme Base 7B Instruct",
            "aliases": [],
            "parents": [{"id": "acme/base-7b", "relationship": "variant", "axis": "training_stage"}],
        },
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "acme/base-7b" not in by_id, "colliding base mint suppressed"
    child = by_id["acme/base-7b-instruct"]
    assert child["parents"][0]["id"] == "Acme/Base-7B", (
        "surviving child's parent edge must be repointed to the curated owner, "
        f"not left dangling: {child['parents']}"
    )


def test_reconciliation_drops_intra_batch_sibling_id_alias(mod, tmp_path):
    """A base mint must not alias a sibling that is itself a batch entry (the
    double-claim aborts the seed): the sibling's id wins and the base's other
    aliases survive. The cross-source check alone misses this since neither
    sibling is in the existing sources."""
    core = tmp_path / "core.yaml"
    core.write_text("entries: []\n")
    # Synthetic ids so the only collision is the intra-batch one.
    generated = [
        {"id": "acmecorp/Foo-70B", "display_name": "Acmecorp Foo 70B",
         "aliases": ["acmecorp/Foo-70B-Instruct", "foo-70b-bare"]},
        {"id": "acmecorp/Foo-70B-Instruct", "display_name": "Acmecorp Foo 70B Instruct",
         "aliases": []},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated], sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "acmecorp/Foo-70B" in by_id and "acmecorp/Foo-70B-Instruct" in by_id
    base_aliases = by_id["acmecorp/Foo-70B"].get("aliases") or []
    assert "acmecorp/Foo-70B-Instruct" not in base_aliases, (
        "base must not claim the sibling variant's id as an alias (would abort seed)"
    )
    assert "foo-70b-bare" in base_aliases, "a non-colliding alias must survive"


def test_reconciliation_dedups_intra_batch_shared_nonid_alias(mod, tmp_path):
    """A non-id alias shared by two survivors goes to the lexicographically-first
    claimant only, so ownership is deterministic across runs."""
    core = tmp_path / "core.yaml"
    core.write_text("entries: []\n")
    generated = [
        {"id": "zzz/model-b", "display_name": "Model B", "aliases": ["shared-free-alias"]},
        {"id": "aaa/model-a", "display_name": "Model A", "aliases": ["shared-free-alias"]},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated], sources=(core,))
    owners = sorted(e["id"] for e in out if "shared-free-alias" in (e.get("aliases") or []))
    assert owners == ["aaa/model-a"], f"shared non-id alias not deterministically owned: {owners}"


def test_reconciliation_does_not_donate_suppressed_id_owned_elsewhere(mod, tmp_path):
    """When a fold suppresses a mint whose id is itself a canonical owned
    elsewhere (base google/gemma-2-9b dragged onto google/gemma-2-9b-it via a
    mangled alias), the suppressed id is not donated as an alias: that would
    merge two distinct canonicals."""
    # Real sources: build_hf_index only folds onto real-HF ids, so a synthetic
    # fixture cannot reproduce this.
    generated = [
        {"id": "google/gemma-2-9b", "display_name": "Gemma 2 9B", "aliases": ["google/gemma2-9b-it"]},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated])
    donated_onto = [e["id"] for e in out if "google/gemma-2-9b" in (e.get("aliases") or [])]
    assert donated_onto == [], (
        f"the base canonical google/gemma-2-9b was wrongly donated as an alias onto "
        f"{donated_onto} — a distinct model merged into a variant"
    )


def test_reconciliation_no_op_when_no_collision(mod, tmp_path):
    """With no collision, reconciliation is an id-sorted pass-through."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump([{"id": "Acme/Widget-3B", "aliases": []}]))
    generated = [
        {"id": "openai/gpt-4o", "display_name": "GPT-4o", "aliases": ["gpt-4o"]},
        {"id": "anthropic/claude-opus-4-5", "display_name": "Claude Opus 4.5",
         "aliases": ["claude-opus-4-5"]},
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    assert [e["id"] for e in out] == sorted(e["id"] for e in generated)


def test_rehost_mint_never_uses_base_vendor_prefix(mod, api):
    """A minted canonical id is never a base-vendor `junk` id from
    curation/rehost_repoint.json (REKEY_REAL / CLOSED entries). Entries whose
    group is absent from the pinned snapshot are skipped.
    """
    if not REHOST_REPOINT.exists():
        pytest.skip(f"rehost repoint oracle missing at {REHOST_REPOINT}")
    repoint = json.loads(REHOST_REPOINT.read_text())

    # Rebuild the cached authority from the real oracle.
    mod._HF_AUTHORITY = None
    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    emitted_ids = {e["id"] for e in out}

    reachable_roots = set(mod.build_underlying_groups(api).keys())

    # A `junk` id that has since become a real HF repo is correctly deferred
    # to, not minted; exclude it.
    HF_ORACLE_JSON = resolve_oracle_path()
    oracle_real: set[str] = set()
    if HF_ORACLE_JSON.exists():
        for _r, _m in json.loads(HF_ORACLE_JSON.read_text()).get("resolutions", {}).items():
            if _m.get("resolution_status") in ("fixed_exact", "fixed_near_miss"):
                fx = _m.get("fixed_hf_model_id")
                if isinstance(fx, str):
                    oracle_real.add(fx)

    checked = 0
    skipped_unreachable = 0
    offenders: list[str] = []
    for e in repoint:
        if e.get("kind") not in ("REKEY_REAL", "CLOSED"):
            continue
        junk = e.get("junk")
        if not junk or junk in oracle_real:
            continue
        root = mod.canon_key_ordered(mod.normalize_modelsdev_id(junk))
        if root not in reachable_roots:
            skipped_unreachable += 1
            continue
        checked += 1
        if junk in emitted_ids:
            offenders.append(f"{e['kind']}:{junk}")

    assert checked > 0, "expected at least one snapshot-reachable REKEY/CLOSED case"
    assert not offenders, (
        f"generator minted {len(offenders)} base-vendor junk id(s) as canonicals: "
        f"{offenders[:10]}"
    )
    assert checked >= 50, (
        f"only {checked} REKEY/CLOSED cases reachable in the pinned snapshot "
        f"({skipped_unreachable} skipped) — coverage unexpectedly low"
    )


# ---------------------------------------------------------------------------
# Alias forms: a variant never emits its suffix-stripped base form (owned by
# another canonical; the double-claim aborts the seed).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "variant_raw,stolen_base",
    [
        ("gpt-4-turbo", "gpt-4"),
        ("glm-5-turbo", "glm-5"),
        ("deepseek-reasoner", "deepseek"),
        ("qwen3-next-80b-a3b-thinking", "qwen3-next-80b-a3b"),
        ("Phi-4-reasoning", "phi-4"),
        ("model-x-fp8", "model-x"),
    ],
)
def test_provider_alias_forms_never_emits_variant_stripped_base(mod, variant_raw, stolen_base):
    """A variant's alias forms keep its identity; the base slug another
    canonical owns is never emitted."""
    forms = mod._provider_alias_forms(variant_raw, "anyorg")
    assert stolen_base not in forms, f"{variant_raw} emitted base {stolen_base!r}: {forms}"
    assert f"anyorg/{stolen_base}" not in forms, f"{variant_raw} emitted org/base: {forms}"
    own = mod._slugify(mod.normalize_modelsdev_id(variant_raw, strip_variants=False)).rsplit("/", 1)[-1]
    assert own in forms, f"{variant_raw} lost its own form {own!r}: {forms}"


def test_serving_tags_still_stripped_in_alias_forms(mod):
    """Serving tags (`:free`, `-fast`) stay stripped from alias forms; only
    identity variants are preserved."""
    assert "gpt-4o" in mod._provider_alias_forms("gpt-4o:free", "openai")
    forms = mod._provider_alias_forms("some-model-fast", "x")
    assert "some-model" in forms


def test_intra_output_merge_preserves_loser_scalars_and_parents(mod):
    """The winner-merge fills empty winner scalars from the loser and unions
    parent edges (the winner is often an hf_deferred entry with empty fields)."""
    entries = [
        # Winner: empty enrichment fields.
        {
            "id": "Foo/Bar-7B",
            "org_id": "foo",
            "aliases": [],
            "parents": [],
            "input_modalities": None,
            "params_billions": None,
            "metadata": json.dumps({"hf_deferred": True}),
        },
        # Loser: same normalized id, carries metadata + lineage.
        {
            "id": "foo/bar-7b",
            "org_id": "foo",
            "aliases": ["foo-bar-7b"],
            "parents": [{"id": "foo/base-7b", "relationship": "finetune"}],
            "input_modalities": ["text", "image"],
            "params_billions": 7.0,
            "metadata": "{}",
        },
    ]
    out = mod._reconcile_intra_output(entries)
    by_id = {e["id"]: e for e in out}
    assert "Foo/Bar-7B" in by_id and "foo/bar-7b" not in by_id, "same-model dup merged into HF-true winner"
    w = by_id["Foo/Bar-7B"]
    assert w["input_modalities"] == ["text", "image"], "loser modalities must be inherited"
    assert w["params_billions"] == 7.0, "loser params must be inherited"
    assert any(edge["id"] == "foo/base-7b" for edge in w.get("parents") or []), "loser parent edge must survive"


def test_reconcile_strips_surviving_mint_foreign_alias_and_displayname(mod, tmp_path):
    """A surviving mint drops an alias or display_name owned by another
    existing canonical (the double-claim aborts the seed)."""
    core = tmp_path / "core.yaml"
    core.write_text(
        yaml.safe_dump(
            [{"id": "Existing/Canon-7B", "display_name": "Existing Canon", "aliases": ["shared-bare"]}]
        )
    )
    generated = [
        {
            "id": "other/distinct-model",   # survives
            "display_name": "Existing Canon",   # owned by core
            "aliases": ["shared-bare", "own-clean-alias"],  # shared-bare owned by core
        }
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    surv = next(e for e in out if e["id"] == "other/distinct-model")
    assert "shared-bare" not in surv["aliases"], "foreign alias must be stripped from survivor"
    assert "own-clean-alias" in surv["aliases"], "clean alias must be kept"
    assert surv["display_name"] != "Existing Canon", "foreign display_name must be re-derived"


def test_intra_output_dedupes_ambiguous_bare_alias_across_distinct_canonicals(mod):
    """A bare alias claimed by two distinct survivors is kept on exactly one
    (the double-claim aborts the seed)."""
    entries = [
        {"id": "google/gemma-3-1b-it", "org_id": "google", "aliases": ["gemma-3-1b-it"]},
        {"id": "unsloth/gemma-3-1b-it", "org_id": "unsloth", "aliases": ["gemma-3-1b-it"]},
    ]
    out = mod._reconcile_intra_output(entries)
    claimers = [e["id"] for e in out if "gemma-3-1b-it" in (e.get("aliases") or [])]
    assert len(claimers) == 1, f"ambiguous bare alias must be kept on exactly one canonical, got {claimers}"


def test_no_emitted_alias_steals_another_canonical_id(mod, api):
    """Over the pinned snapshot, no emitted entry carries another emitted
    canonical's id (exact or normalized) as an alias."""
    from collections import defaultdict

    from eval_entity_resolver.normalization import normalize as _nz

    mod._HF_AUTHORITY = None
    out, skipped = mod._generate_models(api, mod._load_known_org_ids())
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    fin = mod._finalize_entries([dict(e) for e in out])

    ids = {e["id"] for e in fin}
    id_norms = {_nz(i): i for i in ids}
    offenders = []
    for e in fin:
        for a in e.get("aliases", []):
            if a in ids and a != e["id"]:
                offenders.append((e["id"], "exact", a))
            owner = id_norms.get(_nz(a))
            if owner is not None and owner != e["id"] and a not in ids:
                offenders.append((e["id"], "norm", a, owner))
    assert not offenders, (
        f"{len(offenders)} alias(es) steal another canonical's id "
        f"(seed would abort): {offenders[:12]}"
    )


# ---------------------------------------------------------------------------
# Source-shape guard: a catalog regen is deterministic and never drops a
# committed canonical id no other source covers. The resolution gates check
# outcomes, not source shape, so they would not catch this.
# ---------------------------------------------------------------------------
def test_catalog_regen_deterministic_and_no_drop(mod, api, tmp_path, monkeypatch):
    import copy
    import yaml as _y

    known = mod._load_known_org_ids()
    full = mod._finalize_entries(mod._generate_models(api, known)[0])

    def _regen(catalog_path):
        # regenerate_catalog writes the catalog before the org reconcile, so an
        # empty tmp ORGS_GENERATED_PATH does not change catalog output.
        monkeypatch.setattr(mod, "CATALOG_OUT_PATH", catalog_path)
        monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / (catalog_path.stem + "_orgs.yaml"))
        mod.regenerate_catalog(copy.deepcopy(full))
        return catalog_path.read_text()

    text1 = _regen(tmp_path / "cat1.yaml")
    text2 = _regen(tmp_path / "cat2.yaml")
    assert text1 == text2, "catalog regen is NON-DETERMINISTIC (would oscillate across cron runs)"

    regen_ids = {e["id"] for e in (_y.safe_load(text1) or []) if isinstance(e, dict) and e.get("id")}
    committed = _y.safe_load((REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml").read_text())
    committed_ids = {e["id"] for e in (committed or []) if isinstance(e, dict) and e.get("id")}

    # A drop is legitimate when another source still covers the id (a stale
    # catalog standalone of an HF-present model folds to an enrichment).
    other_source_ids = set()
    for f in ("hf_oracle", "hub_stats", "models_dev"):
        for e in _y.safe_load((REPO_ROOT / "seed" / "models" / "sources" / f"{f}.generated.yaml").read_text()) or []:
            if isinstance(e, dict) and e.get("id"):
                other_source_ids.add(e["id"])
    _core = _y.safe_load((REPO_ROOT / "seed" / "models" / "core.yaml").read_text()) or {}
    for e in (_core.get("entries") if isinstance(_core, dict) else _core) or []:
        if isinstance(e, dict) and e.get("id"):
            other_source_ids.add(e["id"])

    uncovered = (committed_ids - regen_ids) - other_source_ids
    assert not uncovered, (
        f"catalog regen DROPS {len(uncovered)} committed canonical id(s) NOT covered by "
        f"any other source — the daily --catalog cron would regress coverage: "
        f"{sorted(uncovered)[:12]}"
    )


# ---------------------------------------------------------------------------
# Non-catalog regen is deterministic from the pinned snapshot. The committed
# source is refreshed from live models.dev, so a byte comparison against it
# would not be a valid pin; the pipeline is run twice instead.
# ---------------------------------------------------------------------------
def test_noncatalog_regen_is_deterministic_from_pinned_snapshot(mod, api):
    def run() -> str:
        # An earlier test may have left a monkeypatched authority in the cache.
        mod._HF_AUTHORITY = None
        gen, skipped = mod._generate_models(api, mod._load_known_org_ids())
        assert skipped == []
        gen = mod._tag_openrouter_key_ids(
            mod._finalize_entries([dict(e) for e in gen]), api
        )
        committed = mod._catalog_load_list(mod.SEED_PATH)
        gen, claims = mod._carry_forward_committed(gen, committed)
        gen = mod.reconcile_generated_against_existing(gen, committed_claims=claims)
        canonicalize_org = mod._build_org_canonicalizer()
        for entry in gen:
            for field in ("org_id", "lineage_origin_model_org_id"):
                if entry.get(field):
                    entry[field] = canonicalize_org(entry[field])
        return mod._write_yaml(gen, mod.SEED_PATH)

    assert run() == run(), "non-catalog regen is non-deterministic for the pinned snapshot"


# ---------------------------------------------------------------------------
# A new upstream model whose org exists under HF-true casing but arrives
# lowercased must FK to the existing org row, not mint a case-variant twin.
# ---------------------------------------------------------------------------
def test_catalog_new_model_snaps_to_existing_community_org_casing(mod, tmp_path, monkeypatch):
    import yaml as _y

    gen_orgs = _y.safe_load((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text()) or []
    # An existing community org with non-lowercase casing.
    existing = next(
        e["id"] for e in gen_orgs
        if isinstance(e, dict) and e.get("id") and "/" not in e["id"]
        and e["id"].lower() != e["id"]
    )
    variant = existing.lower()
    assert variant != existing

    # Off-HF id (stays a fresh mint) with the lowercased org casing.
    synthetic = {
        "id": f"{variant}/Zzz-Synthetic-New-Model-9000",
        "display_name": "Zzz-Synthetic-New-Model-9000",
        "org_id": variant,
        "resolution_source": "models_dev",
        "review_status": "auto",
        "aliases": [],
        "metadata": "{}",
    }

    cat_out = tmp_path / "cat.yaml"
    orgs_out = tmp_path / "orgs.yaml"
    orgs_out.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat_out)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs_out)
    mod.regenerate_catalog([dict(synthetic)])

    cat = _y.safe_load(cat_out.read_text()) or []
    rec = next((e for e in cat if isinstance(e, dict) and e.get("id", "").startswith(f"{variant}/")), None)
    assert rec is not None, "synthetic fresh mint missing from catalog output"
    assert rec["org_id"] == existing, (
        f"org_id not snapped to existing community casing: {rec['org_id']!r} != {existing!r} "
        f"— a new upstream model would split-identity its org"
    )

    after = _y.safe_load(orgs_out.read_text()) or []
    after_ids = {e["id"] for e in after if isinstance(e, dict) and e.get("id")}
    assert variant not in after_ids, (
        f"minted a case-variant twin org {variant!r} alongside existing {existing!r}"
    )


# ---------------------------------------------------------------------------
# Carry-forward: a wholesale rewrite never deletes committed surface forms.
# ---------------------------------------------------------------------------

@pytest.fixture
def cf_mod(tmp_path):
    """Fresh module instance with CORE_PATH at a tmp file, so each test
    controls the core skip lists."""
    m = load_script_module("refresh_from_modelsdev", "refresh_modelsdev_carryforward")
    m.CORE_PATH = tmp_path / "core.yaml"
    return m


def _cf_entry(cid: str, aliases=(), metadata: str = "{}") -> dict:
    return {
        "id": cid,
        "display_name": cid.split("/")[-1],
        "org_id": cid.split("/")[0] if "/" in cid else None,
        "aliases": list(aliases),
        "metadata": metadata,
        "review_status": "draft",
    }


def test_carry_forward_retains_upstream_removed_entry(cf_mod):
    """A committed entry missing upstream is retained verbatim, tagged
    `upstream_status: removed`, with its forms in the claims map."""
    committed = [_cf_entry("groq/llama3-9b-9192", ["llama3-9b-9192"])]
    fresh = [_cf_entry("openai/gpt-4o-test", ["gpt-4o-test"])]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in batch}
    kept = by_id["groq/llama3-9b-9192"]
    assert kept["aliases"] == ["llama3-9b-9192"]
    assert json.loads(kept["metadata"])["upstream_status"] == "removed"
    assert claims["llama3-9b-9192"] == "groq/llama3-9b-9192"
    assert by_id["openai/gpt-4o-test"]["aliases"] == ["gpt-4o-test"]


def test_carry_forward_respects_core_skip(cf_mod):
    """An id core.yaml suppresses (skip_ids / skip_source_ids) is not carried
    forward."""
    cf_mod.CORE_PATH.write_text(
        yaml.safe_dump({"skip_source_ids": ["groq/llama3-9b-9192"], "entries": []})
    )
    committed = [_cf_entry("groq/llama3-9b-9192", ["llama3-9b-9192"])]
    batch, _claims = cf_mod._carry_forward_committed([], committed)
    assert batch == []


def test_carry_forward_unions_committed_aliases_back(cf_mod):
    """An alias dropped upstream is unioned back onto the surviving entry with
    its platform provenance."""
    committed = [_cf_entry(
        "meta/llama3.1-8b-cftest",
        ["llama3.1-8b-cftest"],
        metadata=json.dumps(
            {"alias_platforms": {"llama3.1-8b-cftest": "cerebras"}}, sort_keys=True
        ),
    )]
    fresh = [_cf_entry("meta/llama3.1-8b-cftest", ["llama-3.1-8b-cftest-new"])]
    batch, _claims = cf_mod._carry_forward_committed(fresh, committed)
    (e,) = batch
    assert e["aliases"] == ["llama-3.1-8b-cftest-new", "llama3.1-8b-cftest"]
    assert json.loads(e["metadata"])["alias_platforms"]["llama3.1-8b-cftest"] == "cerebras"


def test_new_mint_cannot_steal_retained_alias(cf_mod, tmp_path):
    """When upstream re-spells a model under a serving-org prefix, the fresh
    mint must not capture the carried-forward entry's alias claims."""
    core = tmp_path / "sources-core.yaml"
    core.write_text(yaml.safe_dump([]))
    committed = [_cf_entry("google/gemma-9-31B", ["gemma9-31b", "gemma9:31b"])]
    fresh = [_cf_entry("TEE/gemma9-31b", ["gemma9-31b", "gemma9:31b"])]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    out = cf_mod.reconcile_generated_against_existing(
        batch, sources=(core,), committed_claims=claims
    )
    by_id = {e["id"]: e for e in out}
    assert set(by_id["google/gemma-9-31B"]["aliases"]) >= {"gemma9-31b", "gemma9:31b"}
    assert not set(by_id["TEE/gemma9-31b"].get("aliases") or []) & {"gemma9-31b", "gemma9:31b"}


def test_carry_forward_absorbs_serving_prefixed_committed_mint(cf_mod):
    """A committed mint under a serving-host prefix is absorbed onto the fresh
    entry that aliases it, never retained as a `TEE/` canonical."""
    committed = [_cf_entry("TEE/zmodel9-70b", ["zmodel9-70b"])]
    fresh = [_cf_entry("meta/zmodel9-70b", ["TEE/zmodel9-70b", "zmodel9-70b"])]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    assert [e["id"] for e in batch] == ["meta/zmodel9-70b"]
    (e,) = batch
    assert {"TEE/zmodel9-70b", "zmodel9-70b"} <= set(e["aliases"])
    assert claims["TEE/zmodel9-70b"] == "meta/zmodel9-70b"
    assert claims["zmodel9-70b"] == "meta/zmodel9-70b"


def test_tee_stripped_mint_folds_onto_claims_owner_with_weak_donation(cf_mod, tmp_path):
    """After the serving-prefix strip the mint folds onto the canonical that
    claims the form; the raw `TEE/` form becomes an alias and the scalars
    donate weakly."""
    core = tmp_path / "sources-core.yaml"
    core.write_text(yaml.safe_dump([{
        "id": "google/gemma-9X-31B", "display_name": "Gemma 9X 31B",
        "aliases": ["google/gemma9x-31b"],
    }]))
    fresh = [{
        "id": "google/gemma9x-31b", "display_name": "Gemma 9X 31B",
        "org_id": "google", "aliases": ["TEE/gemma9x-31b", "gemma9x-31b"],
        "release_date": "2026-04-04", "open_weights": True,
        "metadata": "{}", "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = cf_mod.reconcile_generated_against_existing(fresh, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "google/gemma9x-31b" not in by_id
    rec = by_id["google/gemma-9X-31B"]
    assert {"TEE/gemma9x-31b", "gemma9x-31b", "google/gemma9x-31b"} <= set(rec["aliases"])
    assert rec["weak"] == {"release_date": "2026-04-04", "open_weights": True}


def test_carry_forward_unions_replaced_display_name_back(cf_mod):
    """A committed display_name is a loader-promoted alias: when upstream
    re-spells it, the old display survives as an alias."""
    committed = [dict(_cf_entry("ibm/granite-cftest-micro", ["granite-cftest-micro"]),
                      display_name="IBM: Granite CFTest Micro")]
    fresh = [dict(_cf_entry("ibm/granite-cftest-micro", ["granite-cftest-micro"]),
                  display_name="Granite CFTest Micro")]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    (e,) = batch
    assert "IBM: Granite CFTest Micro" in e["aliases"]
    assert claims["IBM: Granite CFTest Micro"] == "ibm/granite-cftest-micro"


def test_carry_forward_reappearing_id_clears_removed_tag(cf_mod):
    """A retained (`upstream_status: removed`) entry whose id reappears is
    replaced by the fresh entry: one record, tag cleared, aliases unioned."""
    committed = [dict(
        _cf_entry("groq/llama3-9b-9192", ["llama3-9b-9192"]),
        metadata=json.dumps({"upstream_status": "removed"}, sort_keys=True),
    )]
    fresh = [_cf_entry("groq/llama3-9b-9192")]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    assert [e["id"] for e in batch] == ["groq/llama3-9b-9192"]
    (e,) = batch
    assert "llama3-9b-9192" in e["aliases"]
    assert "upstream_status" not in json.loads(e["metadata"])
    assert claims["llama3-9b-9192"] == "groq/llama3-9b-9192"


def test_carry_forward_respelled_reappearance_committed_id_wins(cf_mod, tmp_path):
    """Stability rule: a model dropped and re-emitted respelled keeps the
    committed id; the fresh spelling becomes an alias, so upstream respells
    never thrash canonical ids. Deterministic."""
    core = tmp_path / "sources-core.yaml"
    core.write_text("entries: []\n")
    committed = [dict(_cf_entry("zorgvid/Zeo-3-1", ["zeo-3-1-alias"]),
                      display_name="Zeo-3.1-Fast")]
    fresh = [_cf_entry("zorgvid/zeo3-1")]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert [e["id"] for e in batch] == ["zorgvid/Zeo-3-1"], "expected a single survivor"
    (e,) = batch
    assert {"zorgvid/zeo3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(e["aliases"])
    assert "upstream_status" not in json.loads(e["metadata"])
    # Claims point at the survivor, so reconcile hygiene keeps the donated forms.
    assert claims["zorgvid/Zeo-3-1"] == "zorgvid/Zeo-3-1"
    assert claims["Zeo-3.1-Fast"] == "zorgvid/Zeo-3-1"
    out = cf_mod.reconcile_generated_against_existing(
        batch, sources=(core,), committed_claims=claims
    )
    (surv,) = [x for x in out if x["id"] == "zorgvid/Zeo-3-1"]
    assert {"zorgvid/zeo3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(surv["aliases"])
    batch2, claims2 = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert batch2 == batch and claims2 == claims


def test_carry_forward_respelled_reappearance_migration_flag_inverts(cf_mod, tmp_path):
    """With `--adopt-openrouter-ids-migration` the fresh spelling wins and the
    committed id, display_name and aliases become aliases on it, surviving
    the reconcile hygiene pass."""
    core = tmp_path / "sources-core.yaml"
    core.write_text("entries: []\n")
    committed = [dict(_cf_entry("zorgvid/Zeo-3-1", ["zeo-3-1-alias"]),
                      display_name="Zeo-3.1-Fast")]
    fresh = [_cf_entry("zorgvid/zeo3-1")]
    batch, claims = cf_mod._carry_forward_committed(
        [dict(e) for e in fresh], committed, adopt_migration=True
    )
    assert [e["id"] for e in batch] == ["zorgvid/zeo3-1"], "expected a single survivor"
    (e,) = batch
    assert {"zorgvid/Zeo-3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(e["aliases"])
    assert claims["zorgvid/Zeo-3-1"] == "zorgvid/zeo3-1"
    assert claims["Zeo-3.1-Fast"] == "zorgvid/zeo3-1"
    out = cf_mod.reconcile_generated_against_existing(
        batch, sources=(core,), committed_claims=claims
    )
    (surv,) = [x for x in out if x["id"] == "zorgvid/zeo3-1"]
    assert {"zorgvid/Zeo-3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(surv["aliases"]), (
        f"absorbed committed forms must survive migration hygiene: {surv['aliases']}"
    )


def test_carry_forward_stability_repoints_parent_edges(cf_mod):
    """When a fresh twin is renamed back to the committed id, parent edges
    pointing at the fresh spelling are repointed (no dangling edge)."""
    committed = [dict(_cf_entry("zorgvid/Zeo-3-1"), display_name="Zeo 3.1")]
    child = dict(_cf_entry("zorgvid/zeo3-1-mini"))
    child["parents"] = [{"id": "zorgvid/zeo3-1", "relationship": "variant", "axis": "tier"}]
    fresh = [_cf_entry("zorgvid/zeo3-1"), child]
    batch, _claims = cf_mod._carry_forward_committed(
        [dict(e) for e in fresh], committed
    )
    by_id = {e["id"]: e for e in batch}
    assert "zorgvid/Zeo-3-1" in by_id and "zorgvid/zeo3-1" not in by_id
    assert by_id["zorgvid/zeo3-1-mini"]["parents"][0]["id"] == "zorgvid/Zeo-3-1"


def test_carry_forward_size_guard_blocks_false_absorb(cf_mod):
    """The twin key strips separators, so `opt-1.3b`/`opt-13b` collide; the
    b-size guard keeps them apart and the committed entry is retained."""
    committed = [_cf_entry("acme/opt-1.3b")]
    fresh = [_cf_entry("acme/opt-13b")]
    batch, _claims = cf_mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in batch}
    assert set(by_id) == {"acme/opt-13b", "acme/opt-1.3b"}
    assert json.loads(by_id["acme/opt-1.3b"]["metadata"])["upstream_status"] == "removed"


def test_twin_picker_prefers_variant_preserving_identity(cf_mod):
    """When a collision bucket holds both ``qwen3.8`` and ``qwen-3-8``, the
    candidate matching the committed identity wins, not the lexical first.
    """
    candidates = [
        _cf_entry("alibaba/qwen-3-8-max"),
        _cf_entry("qwen/qwen3.8-max"),
    ]
    picked = cf_mod._pick_twin_candidate("alibaba/qwen3.8-max", candidates)
    assert picked is not None
    assert picked["id"] == "qwen/qwen3.8-max"


def test_carry_forward_does_not_absorb_enrich_record_onto_mint(cf_mod):
    """An enrich record's id is another canonical: a twin-matching mint must
    not absorb it (that would steal the canonical). Carried verbatim."""
    committed = [{"id": "Realorg/Real-Canon-9X", "aliases": ["real-canon-9x-form"]}]
    fresh = [_cf_entry("realorg/real-canon-9x")]
    batch, _claims = cf_mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in batch}
    assert set(by_id) == {"realorg/real-canon-9x", "Realorg/Real-Canon-9X"}
    assert "Realorg/Real-Canon-9X" not in (by_id["realorg/real-canon-9x"].get("aliases") or [])


# ---------------------------------------------------------------------------
# An enrich record whose owner vanished is dropped loudly, never kept as a
# bare canonical.
# ---------------------------------------------------------------------------
def test_reconcile_drops_orphaned_enrich_record_loudly(mod, tmp_path, capsys):
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({"entries": [{
        "id": "Kept/Owner-X", "display_name": "Owner X", "aliases": [],
    }]}))
    generated = [
        # Owner exists nowhere: drop + warn.
        {"id": "ghostorg/vanished-owner-xyz", "aliases": ["ghost-form-1"]},
        # Owner in core: keeps donating.
        {"id": "Kept/Owner-X", "aliases": ["kept-form-1"]},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated], sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "ghostorg/vanished-owner-xyz" not in by_id, "bare canonical must not survive"
    assert "kept-form-1" in by_id["Kept/Owner-X"]["aliases"]
    err = capsys.readouterr().err
    assert "vanished-owner-xyz" in err and "ghost-form-1" in err, (
        "dropping an orphaned enrich record must be loud (forms listed)"
    )


# ---------------------------------------------------------------------------
# Catalog carry-forward: donation on fold, retention / union-back / dup
# consolidation, and idempotence over its own output.
# ---------------------------------------------------------------------------
def test_catalog_carry_forward_donates_display_name_on_respelled_fold(mod, tmp_path, monkeypatch):
    """A committed catalog mint that reappears respelled keeps the committed
    id; the fresh spelling stays resolvable as an alias and no twin survives."""
    committed = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo-3.1-Fast", "org_id": None,
        "aliases": ["zeo-3-1"],
        "metadata": json.dumps({"alias_platforms": {"zeo-3-1": "poe"}}, sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/zeo3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh])
    out = yaml.safe_load(cat.read_text()) or []
    assert "zorgvid/zeo3-1" not in {e["id"] for e in out}, "respelled twin must not survive"
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo-3-1"]
    assert {"zorgvid/zeo3-1", "zeo-3-1"} <= set(surv["aliases"]), (
        f"surface forms lost on fold: {surv['aliases']}"
    )
    assert json.loads(surv["metadata"])["alias_platforms"]["zeo-3-1"] == "poe"


def test_catalog_carry_forward_respelled_fold_migration_flag_inverts(mod, tmp_path, monkeypatch):
    """With `adopt_migration=True` the fresh spelling wins and the committed
    id + display_name become aliases."""
    committed = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo-3.1-Fast", "org_id": None,
        "aliases": ["zeo-3-1"],
        "metadata": json.dumps({"alias_platforms": {"zeo-3-1": "poe"}}, sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/zeo3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh], adopt_migration=True)
    out = yaml.safe_load(cat.read_text()) or []
    assert "zorgvid/zeo-3-1" not in {e["id"] for e in out}, "respelled twin must not survive"
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo3-1"]
    assert {"zorgvid/zeo-3-1", "Zeo-3.1-Fast", "zeo-3-1"} <= set(surv["aliases"]), (
        f"committed surface forms lost on fold: {surv['aliases']}"
    )
    assert json.loads(surv["metadata"])["alias_platforms"]["zeo-3-1"] == "poe"


def test_catalog_carry_forward_retention_unionback_and_dup_consolidation(mod, tmp_path, monkeypatch):
    """Two committed records for one id consolidate onto the single fresh mint
    (aliases unioned); a committed mint with no fresh target is retained
    tagged `upstream_status: removed`."""
    committed = [
        {"id": "zorg/zmodel-cf", "display_name": "ZModel CF", "org_id": None,
         "aliases": ["zmodel-cf-a1"], "metadata": "{}",
         "review_status": "draft", "resolution_source": "models_dev"},
        {"id": "zorg/zmodel-cf", "aliases": ["zmodel-cf-a2"]},  # dup record, same id
        {"id": "zorg/zgone-cf", "display_name": "ZGone CF", "org_id": None,
         "aliases": ["zgone-cf-alias"], "metadata": "{}",
         "review_status": "draft", "resolution_source": "models_dev"},
    ]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorg/zmodel-cf", "display_name": "ZModel CF", "org_id": None,
        "aliases": ["zmodel-cf-a3"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog(fresh)
    out = yaml.safe_load(cat.read_text()) or []
    recs = [e for e in out if e["id"] == "zorg/zmodel-cf"]
    assert len(recs) == 1, f"dup committed records must consolidate, got {len(recs)}"
    assert {"zmodel-cf-a1", "zmodel-cf-a2", "zmodel-cf-a3"} <= set(recs[0]["aliases"])
    (gone,) = [e for e in out if e["id"] == "zorg/zgone-cf"]
    assert json.loads(gone["metadata"])["upstream_status"] == "removed"
    assert "zgone-cf-alias" in gone["aliases"]


def test_catalog_consolidates_enrich_records_per_owner(mod, tmp_path, monkeypatch):
    """Two donors folding onto one owner emit one enrich record: aliases
    unioned, alias_platforms and weak per-key first-wins."""
    src = tmp_path / "src-core.yaml"
    src.write_text(yaml.safe_dump([{
        "id": "zorg/ZModel-9B", "display_name": "ZModel 9B",
        "aliases": ["zorg/zmodel-9b", "zmodel-9b-v0"],
    }]))
    monkeypatch.setattr(mod, "_CATALOG_EXISTING_SOURCES", (src,))
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump([]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    full = [
        {"id": "zorg/zmodel-9b", "display_name": "ZModel 9B A", "org_id": "zorg",
         "aliases": ["zmodel-9b-alias1"],
         "metadata": json.dumps({"alias_platforms": {"zmodel-9b-alias1": "poe"}}),
         "release_date": "2024-01-01", "open_weights": True,
         "review_status": "draft", "resolution_source": "models_dev"},
        {"id": "zmodel-9b-v0", "display_name": "ZModel 9B B", "org_id": "zorg",
         "aliases": ["zmodel-9b-alias2"],
         "metadata": json.dumps({"alias_platforms": {"zmodel-9b-alias2": "novita-ai"}}),
         "release_date": "2024-02-02",
         "review_status": "draft", "resolution_source": "models_dev"},
    ]
    mod.regenerate_catalog(full)
    out = yaml.safe_load(cat.read_text()) or []
    recs = [e for e in out if e["id"] == "zorg/ZModel-9B"]
    assert len(recs) == 1, f"expected one consolidated enrich record, got {len(recs)}"
    rec = recs[0]
    assert {"zmodel-9b-alias1", "zmodel-9b-alias2"} <= set(rec["aliases"])
    ap = json.loads(rec["metadata"])["alias_platforms"]
    assert ap == {"zmodel-9b-alias1": "poe", "zmodel-9b-alias2": "novita-ai"}
    assert rec["weak"]["release_date"] == "2024-01-01"  # first donor wins per field
    assert rec["weak"]["open_weights"] is True


def test_catalog_pipeline_idempotent_over_committed_output(mod, api, tmp_path, monkeypatch):
    """The --catalog step over the committed catalog, then over its own output,
    is byte-identical (no oscillation across cron runs)."""
    import copy

    mod._HF_AUTHORITY = None
    full = mod._finalize_entries(mod._generate_models(api, mod._load_known_org_ids())[0])
    cat = tmp_path / "cat.yaml"
    cat.write_text((REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml").read_text())
    orgs = tmp_path / "orgs.yaml"
    orgs.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs)
    mod.regenerate_catalog(copy.deepcopy(full))
    text1 = cat.read_text()
    mod.regenerate_catalog(copy.deepcopy(full))
    assert cat.read_text() == text1, "catalog regen not idempotent over its own output"


def test_noncatalog_pipeline_idempotent_over_own_output(mod, api):
    """The non-catalog pipeline over the committed file, then over its own
    output, is byte-identical."""
    mod._HF_AUTHORITY = None

    def run(committed):
        gen, skipped = mod._generate_models(api, mod._load_known_org_ids())
        assert skipped == []
        gen = mod._finalize_entries([dict(e) for e in gen])
        batch, claims = mod._carry_forward_committed(gen, committed)
        out = mod.reconcile_generated_against_existing(batch, committed_claims=claims)
        _canon = mod._build_org_canonicalizer()
        for e in out:
            for f in ("org_id", "lineage_origin_model_org_id"):
                if e.get(f):
                    e[f] = _canon(e[f])
        return mod._write_yaml(out, mod.SEED_PATH)  # returns text; does not write

    text1 = run(mod._catalog_load_list(mod.SEED_PATH))
    text2 = run(yaml.safe_load(text1))
    assert text1 == text2, "non-catalog regen not idempotent over its own output"


# ---------------------------------------------------------------------------
# --reconcile-orgs never writes core.yaml (a re-dump hoists its inline
# comments); a needed core fix is warned loudly instead.
# ---------------------------------------------------------------------------
def test_reconcile_orgs_never_writes_core_by_default(mod, tmp_path, monkeypatch, capsys):
    gen_orgs = yaml.safe_load((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text()) or []
    _canon = mod._build_org_canonicalizer()
    # An org whose lowercased spelling canonicalizes back to it, so a core
    # entry carrying the variant genuinely needs the rewrite.
    existing = next(
        e["id"] for e in gen_orgs
        if isinstance(e, dict) and e.get("id") and "/" not in e["id"]
        and e["id"].lower() != e["id"] and _canon(e["id"].lower()) == e["id"]
    )
    variant = existing.lower()
    core = tmp_path / "core.yaml"
    core_text = (
        "# hand-curated header comment that a re-dump would hoist\n"
        + yaml.safe_dump(
            {"skip_ids": [], "entries": [
                {"id": f"{variant}/zzz-core-org-write-test", "org_id": variant}
            ]},
            sort_keys=False,
        )
    )
    core.write_text(core_text)
    monkeypatch.setattr(mod, "CORE_PATH", core)
    monkeypatch.setattr(mod, "_ALL_MODEL_SOURCES", (core,))

    n = mod.canonicalize_model_org_ids()  # cron default: no core write
    assert core.read_text() == core_text, "the cron must never rewrite core.yaml"
    assert n == 0
    out = capsys.readouterr().out
    assert "::warning::" in out and variant in out, "needed core fix must be loud"

    n2 = mod.canonicalize_model_org_ids(write_core=True)  # one-shot opt-in
    assert n2 == 1
    rewritten = yaml.safe_load(core.read_text())
    assert rewritten["entries"][0]["org_id"] == existing


# ---------------------------------------------------------------------------
# Weak scalar donation: a suppressed/folded/skipped mint's scalars travel onto
# the owner's enrich record under `weak:`; parents and metadata stay behind.
# ---------------------------------------------------------------------------
def test_reconciliation_donates_suppressed_mint_scalars(mod, tmp_path):
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump([
        {"id": "Foo/Bar-7B", "display_name": "Foo Bar 7B", "aliases": ["foo-bar-7b"]},
    ]))
    generated = [
        {
            "id": "foo/bar-7b",
            "display_name": "foo bar 7b",
            "aliases": ["foo-bar-7b-extra-alias"],
            "release_date": "2025-12-01",
            "open_weights": True,
            "params_billions": None,
            "input_modalities": ["text"],
            "output_modalities": ["video"],
            "parents": [{"id": "foo/bar", "relationship": "variant", "axis": "size"}],
            "metadata": json.dumps({"providers": ["fastrouter"], "underlying_key": "bar-7b"}),
        },
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "foo/bar-7b" not in by_id
    rec = by_id["Foo/Bar-7B"]
    assert rec["weak"] == {
        "release_date": "2025-12-01",
        "open_weights": True,
        "input_modalities": ["text"],
        "output_modalities": ["video"],
    }
    assert "display_name" not in rec, "enrich records must stay display_name-less"
    assert "parents" not in rec, "parents edges must NOT be donated (dangling-edge risk)"
    assert "metadata" not in rec, "mint-specific metadata must NOT be donated"


def test_carry_forward_skip_donates_scalars_onto_claiming_owner(cf_mod, tmp_path):
    """A core-skipped committed entry whose form an existing canonical claims
    donates its scalars weakly onto that owner; its aliases stay suppressed."""
    cf_mod.CORE_PATH.write_text(
        yaml.safe_dump({"skip_source_ids": ["wanx/wan-v2-6"], "entries": []})
    )
    existing = tmp_path / "existing.yaml"
    existing.write_text(yaml.safe_dump([
        {"id": "alibaba/wan-v2-6", "display_name": "Wan 2.6",
         "aliases": ["wanx/wan-v2-6", "wan-v2-6"]},
    ]))
    cf_mod._NONCATALOG_EXISTING_SOURCES = (existing,)
    committed = [{
        "id": "wanx/wan-v2-6",
        "display_name": "Wan 2.6",
        "aliases": ["wan-v2-6"],
        "release_date": "2025-12-01",
        "open_weights": True,
        "metadata": "{}",
    }]
    batch, _claims = cf_mod._carry_forward_committed([], committed)
    assert batch == [
        {"id": "alibaba/wan-v2-6",
         "weak": {"release_date": "2025-12-01", "open_weights": True}}
    ]


def test_carry_forward_skip_unclaimed_stays_pure_skip(cf_mod, tmp_path):
    """A core-skipped entry whose form nothing claims is a pure skip, even
    with scalars."""
    cf_mod.CORE_PATH.write_text(
        yaml.safe_dump({"skip_source_ids": ["lonely/mint-1b"], "entries": []})
    )
    empty = tmp_path / "existing.yaml"
    empty.write_text("[]\n")
    cf_mod._NONCATALOG_EXISTING_SOURCES = (empty,)
    committed = [{
        "id": "lonely/mint-1b", "display_name": "Lonely Mint 1B",
        "release_date": "2024-01-01", "metadata": "{}",
    }]
    batch, _claims = cf_mod._carry_forward_committed([], committed)
    assert batch == []


def test_catalog_enrich_record_carries_donor_scalars(mod, tmp_path, monkeypatch):
    """The --catalog HF-present fold (`_enrich_target`) donates the folded
    record's scalars under `weak:`."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({
        "skip_source_ids": [],
        "entries": [
            {"id": "alibaba/zzz-wan-v9", "display_name": "Zzz Wan V9",
             "aliases": ["wanx/zzz-wan-v9", "zzz-wan-v9"], "open_weights": None},
        ],
    }))
    cat_out = tmp_path / "cat.yaml"
    orgs_out = tmp_path / "orgs.yaml"
    orgs_out.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CORE_PATH", core)
    monkeypatch.setattr(mod, "_CATALOG_EXISTING_SOURCES", (core,))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat_out)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs_out)
    full = [{
        "id": "wanx/zzz-wan-v9",
        "display_name": "Zzz Wan V9",
        "org_id": "wanx",
        "aliases": ["zzz-wan-v9"],
        "release_date": "2025-12-01",
        "open_weights": True,
        "metadata": "{}",
        "review_status": "draft",
    }]
    mod.regenerate_catalog([dict(e) for e in full])
    cat = yaml.safe_load(cat_out.read_text()) or []
    rec = next(e for e in cat if e.get("id") == "alibaba/zzz-wan-v9")
    assert rec["weak"] == {"release_date": "2025-12-01", "open_weights": True}
    assert "display_name" not in rec


def test_catalog_committed_skip_donates_scalars(mod, tmp_path, monkeypatch):
    """The --catalog carry-forward applies the same skip_source_ids rule:
    scalars donate weakly, aliases stay suppressed."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({
        "skip_source_ids": ["wanx/zzz-wan-v9"],
        "entries": [
            {"id": "alibaba/zzz-wan-v9", "display_name": "Zzz Wan V9",
             "aliases": ["wanx/zzz-wan-v9", "zzz-wan-v9"], "open_weights": None},
        ],
    }))
    cat_out = tmp_path / "cat.yaml"
    cat_out.write_text(yaml.safe_dump([
        {"id": "wanx/zzz-wan-v9", "display_name": "Zzz Wan V9",
         "aliases": ["zzz-wan-v9"], "release_date": "2025-11-30",
         "open_weights": True, "metadata": "{}"},
    ]))
    orgs_out = tmp_path / "orgs.yaml"
    orgs_out.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CORE_PATH", core)
    monkeypatch.setattr(mod, "_CATALOG_EXISTING_SOURCES", (core,))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat_out)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs_out)
    mod.regenerate_catalog([])
    cat = yaml.safe_load(cat_out.read_text()) or []
    assert cat == [
        {"id": "alibaba/zzz-wan-v9",
         "weak": {"release_date": "2025-11-30", "open_weights": True}}
    ], "skipped committed mint's scalars must flow weakly onto the owner, aliases suppressed"


# ---------------------------------------------------------------------------
# Tiered ownership evidence: independent sources (hf_oracle / hub_stats / core
# / enrichments) are authoritative; a claim whose only evidence is a derived
# file (catalog / tier3) echoing the suppressed mint itself, or an id
# committed_claims assigns to the owner, never vetoes a donation.
# ---------------------------------------------------------------------------

def test_hygiene_strips_metadata_alias_platforms_with_foreign_alias(mod, tmp_path, monkeypatch):
    """When hygiene strips a foreign-owned alias, its alias_platforms key goes
    with it (`ap keys SUBSET-OF aliases`); otherwise the next run differs."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zother/ZOwner-1B", "display_name": "ZOwner 1B",
         "aliases": ["zshared-form"]},
    ]))
    batch = [{
        "id": "zorg/zmine-model", "display_name": "ZMine Model", "org_id": "zorg",
        "aliases": ["zmine-a", "zshared-form"],
        "metadata": json.dumps(
            {"alias_platforms": {"zmine-a": "kilo", "zshared-form": "poe"}},
            sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(batch, sources=(src,))
    (surv,) = [e for e in out if e["id"] == "zorg/zmine-model"]
    assert surv["aliases"] == ["zmine-a"], "foreign alias must be stripped"
    ap = json.loads(surv["metadata"])["alias_platforms"]
    assert ap == {"zmine-a": "kilo"}, (
        f"metadata.alias_platforms must stay in lockstep with aliases: {ap}"
    )
    assert set(ap) <= set(surv["aliases"])


def test_suppressed_mint_donates_forms_despite_stale_catalog_echo(mod, tmp_path, monkeypatch):
    """A mint suppressed onto an independent-source canonical donates its forms
    even when the stale committed catalog still echoes the mint under its own
    id; otherwise the forms vanish from every source."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3.1", "display_name": "ZChat v3.1"},
    ]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3-1", "aliases": ["zchat-v3-1"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    batch = [{
        "id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1", "org_id": "zorg",
        "aliases": ["zchat-v3-1"], "metadata": "{}",
        "release_date": "2026-05-28", "open_weights": False,
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(batch, sources=(src,))
    ids = {e["id"] for e in out}
    assert "zorg/zchat-v3-1" not in ids, "normalized-colliding mint must be suppressed"
    (rec,) = [e for e in out if e["id"] == "zorg/zchat-v3.1"]
    assert "zchat-v3-1" in (rec.get("aliases") or []), (
        "suppressed mint's forms must be donated onto the owner despite the "
        f"stale catalog echo: {rec}"
    )
    assert (rec.get("weak") or {}).get("release_date") == "2026-05-28", (
        "suppressed mint's scalars must flow weakly onto the owner"
    )


def test_suppressed_mint_donates_despite_full_stale_catalog_echo(mod, tmp_path, monkeypatch):
    """Same with a full (display_name-bearing) stale echo, which lands in
    `owner_defining` and must still be ignored."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3.1", "display_name": "ZChat v3.1"},
    ]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1",
         "aliases": ["zchat-v3-1"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    batch = [{
        "id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1", "org_id": "zorg",
        "aliases": ["zchat-v3-1"], "metadata": "{}",
        "release_date": "2026-05-28", "open_weights": False,
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(batch, sources=(src,))
    ids = {e["id"] for e in out}
    assert "zorg/zchat-v3-1" not in ids, "normalized-colliding mint must be suppressed"
    (rec,) = [e for e in out if e["id"] == "zorg/zchat-v3.1"]
    assert "zchat-v3-1" in (rec.get("aliases") or []), (
        f"donation must survive a FULL stale echo: {rec}"
    )


def test_suppressed_mint_donates_with_claims_escape(mod, tmp_path, monkeypatch):
    """When committed_claims maps the echoed id onto the owner, the claims
    escape hatch must not be short-circuited by `owner_defining`."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3.1", "display_name": "ZChat v3.1"},
    ]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3-1", "aliases": ["zchat-v3-1"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    batch = [{
        "id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1", "org_id": "zorg",
        "aliases": ["zchat-v3-1"], "metadata": "{}",
        "release_date": "2026-05-28", "open_weights": False,
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    claims = {"zorg/zchat-v3-1": "zorg/zchat-v3.1", "zchat-v3-1": "zorg/zchat-v3.1"}
    out = mod.reconcile_generated_against_existing(
        batch, sources=(src,), committed_claims=claims
    )
    ids = {e["id"] for e in out}
    assert "zorg/zchat-v3-1" not in ids
    (rec,) = [e for e in out if e["id"] == "zorg/zchat-v3.1"]
    assert "zchat-v3-1" in (rec.get("aliases") or []), (
        f"claims escape hatch must allow donation: {rec}"
    )


def test_steady_state_absorbed_fresh_forms_survive_hygiene(mod, tmp_path, monkeypatch):
    """Steady state (committed id wins) with the stale catalog echoing the
    committed id: the fresh spelling folds in as an alias and every form
    survives hygiene."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorgvid/zeo3-1", "aliases": ["zeo3-1", "zeo3-1-x"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    committed = [{
        "id": "zorgvid/zeo3-1", "display_name": "zeo3-1", "org_id": None,
        "aliases": ["zeo3-1-x"],
        "metadata": json.dumps({"alias_platforms": {"zeo3-1-x": "fastrouter"}},
                               sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    fresh = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": ["zeo-3-1"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    batch, claims = mod._carry_forward_committed(
        [dict(e) for e in fresh], committed
    )
    assert claims.get("zorgvid/zeo3-1") == "zorgvid/zeo3-1", (
        f"steady state: committed id must win: {claims}"
    )
    out = mod.reconcile_generated_against_existing(
        batch, sources=(src,), committed_claims=claims
    )
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo3-1"]
    assert {"zorgvid/zeo-3-1", "zeo-3-1", "zeo3-1-x"} <= set(surv["aliases"]), (
        f"fresh spelling + committed aliases must survive: {surv['aliases']}"
    )
    ap = json.loads(surv["metadata"])["alias_platforms"]
    assert ap.get("zeo3-1-x") == "fastrouter", "platform provenance must survive"
    assert set(ap) <= set(surv["aliases"])


def test_migration_absorbed_committed_forms_survive_hygiene(mod, tmp_path, monkeypatch):
    """Migration (fresh id wins) with a full stale echo of the committed id:
    the absorbed committed forms survive hygiene."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorgvid/zeo3-1", "display_name": "zeo3-1",
         "aliases": ["zeo3-1-x"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    committed = [{
        "id": "zorgvid/zeo3-1", "display_name": "zeo3-1", "org_id": None,
        "aliases": ["zeo3-1-x"],
        "metadata": json.dumps({"alias_platforms": {"zeo3-1-x": "fastrouter"}},
                               sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    fresh = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": ["zeo-3-1"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    batch, claims = mod._carry_forward_committed(
        [dict(e) for e in fresh], committed, adopt_migration=True
    )
    assert claims.get("zorgvid/zeo3-1") == "zorgvid/zeo-3-1", (
        f"migration: fresh id must win: {claims}"
    )
    out = mod.reconcile_generated_against_existing(
        batch, sources=(src,), committed_claims=claims
    )
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo-3-1"]
    assert {"zorgvid/zeo3-1", "zeo3-1-x"} <= set(surv["aliases"]), (
        f"absorbed committed forms must survive migration hygiene: {surv['aliases']}"
    )
    ap = json.loads(surv["metadata"])["alias_platforms"]
    assert ap.get("zeo3-1-x") == "fastrouter", "platform provenance must survive"


# ---------------------------------------------------------------------------
# Alias emission without carry-forward in the loop (committed aliases would
# otherwise self-ratify and mask an emission regression).
# ---------------------------------------------------------------------------

def test_generator_emits_provider_alias_forms_from_snapshot(mod):
    api = {
        "mistral": {"id": "mistral", "name": "Mistral", "models": {
            "mistral-7b-instruct-v0.2": {
                "id": "mistral-7b-instruct-v0.2", "name": "Mistral 7B Instruct v0.2",
                "release_date": "2023-12-11", "open_weights": True, "family": "mistral",
            },
        }},
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "mistralai/mistral-7b-instruct-v0.2": {
                "id": "mistralai/mistral-7b-instruct-v0.2",
                "name": "Mistral 7B Instruct v0.2",
                "release_date": "2023-12-11", "open_weights": True,
            },
        }},
    }
    out, missing = mod._generate_models(api, {"mistralai"})
    assert missing == []
    out = mod._finalize_entries(out)
    by_id = {e["id"]: e for e in out}
    # Oracle hit: every provider spelling survives as an alias on the HF id.
    leaf = by_id["mistralai/Mistral-7B-Instruct-v0.2"]
    assert leaf["aliases"] == [
        "Mistral 7B Instruct v0.2",
        "mistral-7b-instruct-v0-2",
        "mistral-7b-instruct-v0.2",
        "mistralai/mistral-7b-instruct-v0-2",
        "mistralai/mistral-7b-instruct-v0.2",
    ]
    meta = json.loads(leaf["metadata"])
    assert meta["hf_deferred"] is True
    assert "openrouter_adopted" not in meta
    ap = meta["alias_platforms"]
    assert ap == {
        "mistral-7b-instruct-v0-2": "mistral",
        "mistral-7b-instruct-v0.2": "mistral",
        "mistralai/mistral-7b-instruct-v0-2": "mistral",
        "mistralai/mistral-7b-instruct-v0.2": "openrouter",
    }


# ---------------------------------------------------------------------------
# Rung-monotone stability: a fresh hf_deferred twin (rung 1) is never renamed
# back to a committed non-HF id; same-rung twins keep the committed id; rung 2
# over 3 (OpenRouter key over invented) stays migration-gated with a counter.
# ---------------------------------------------------------------------------

def test_carry_forward_hf_deferred_twin_never_renamed_back(cf_mod):
    committed = [
        dict(_cf_entry("zorgx/zmod-3-1"), display_name="ZMod 3.1"),
        # Child pointing at the absorbed id: edge must be repointed.
        dict(_cf_entry("zorgx/zmod-3-1-mini"),
             parents=[{"id": "zorgx/zmod-3-1", "relationship": "variant",
                       "axis": "tier"}]),
    ]
    fresh = [dict(
        _cf_entry("zorgx/ZMod3-1"),
        aliases=["zmod3-1"],
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "zorgx/ZMod3-1" in by_id, "HF-deferred id must survive as canonical"
    assert "zorgx/zmod-3-1" not in by_id, "committed invented twin must be absorbed"
    surv = by_id["zorgx/ZMod3-1"]
    assert "zorgx/zmod-3-1" in surv["aliases"], "absorbed committed id stays resolvable"
    assert json.loads(surv["metadata"])["hf_deferred"] is True
    assert claims["zorgx/zmod-3-1"] == "zorgx/ZMod3-1"
    assert by_id["zorgx/zmod-3-1-mini"]["parents"][0]["id"] == "zorgx/ZMod3-1"


def test_carry_forward_promote_then_rename_chain_composes(cf_mod):
    """Two committed entries twin-match one fresh entry (promote, then rename
    back): the repoint map composes the chain so edges and claims land on the
    final survivor, never the intermediate."""
    committed = [
        dict(_cf_entry("zorgx/zmod-3-1"), display_name="ZMod 3.1"),
        dict(
            _cf_entry("zorgx/ZMod-3.1"),
            metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
        ),
        dict(_cf_entry("zorgx/zmod-3-1-mini"),
             parents=[{"id": "zorgx/zmod-3-1", "relationship": "variant",
                       "axis": "tier"}]),
    ]
    fresh = [dict(
        _cf_entry("zorgx/ZMod3-1"),
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "zorgx/ZMod-3.1" in by_id, "committed HF id wins the same-rung twin"
    assert "zorgx/ZMod3-1" not in by_id
    assert "zorgx/zmod-3-1" not in by_id
    edge = by_id["zorgx/zmod-3-1-mini"]["parents"][0]["id"]
    assert edge == "zorgx/ZMod-3.1", f"edge landed on renamed-away id: {edge}"
    # A renamed-away claim value would poison the committed_claims escape.
    assert claims["zorgx/zmod-3-1"] == "zorgx/ZMod-3.1"


def test_carry_forward_same_rung_hf_twins_keep_committed_id(cf_mod):
    """Two hf_deferred twins (same rung): the committed id wins."""
    committed = [dict(
        _cf_entry("zorgx/ZMod-1B"),
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    fresh = [dict(
        _cf_entry("zorgx/Z-Mod-1B"),
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert [e["id"] for e in batch] == ["zorgx/ZMod-1B"]
    assert "zorgx/Z-Mod-1B" in batch[0]["aliases"]
    assert claims["zorgx/ZMod-1B"] == "zorgx/ZMod-1B"


def test_carry_forward_pending_rung2_counter_logged(cf_mod, capsys):
    """A rung-2 twin deferred by the stability rule is counted and the debt
    line prints on a non-migration run."""
    committed = [dict(_cf_entry("zorgx/zmod-large"), display_name="ZMod Large")]
    fresh = [dict(
        _cf_entry("zorgx/zmodlarge"),
        metadata=json.dumps({"openrouter_adopted": True}, sort_keys=True),
    )]
    batch, _claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "zorgx/zmod-large" in by_id and "zorgx/zmodlarge" not in by_id, (
        "rung-2 promotion must stay migration-gated (committed id wins)"
    )
    err = capsys.readouterr().err
    assert "1 pending rung-2 id promotion(s)" in err, (
        f"pending rung-2 debt must be visible in the cron log: {err!r}"
    )


def test_catalog_carry_forward_hf_deferred_twin_not_renamed_back(mod, tmp_path, monkeypatch):
    """Catalog path: a committed invented mint reappearing as a fresh
    hf_deferred entry is absorbed onto the HF id, never the reverse."""
    committed = [{
        "id": "zorgvid/zeo-9", "display_name": "Zeo 9", "org_id": None,
        "aliases": ["zeo-9"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/Zeo9", "display_name": "Zeo 9", "org_id": None,
        "aliases": [], "metadata": json.dumps({"hf_deferred": True}, sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh])
    out = yaml.safe_load(cat.read_text()) or []
    by_id = {e["id"]: e for e in out}
    assert "zorgvid/Zeo9" in by_id, "HF-deferred id must survive as canonical"
    assert "zorgvid/zeo-9" not in by_id, "committed invented twin must be absorbed"
    surv = by_id["zorgvid/Zeo9"]
    assert {"zorgvid/zeo-9", "zeo-9"} <= set(surv.get("aliases") or []), (
        f"absorbed committed forms must stay resolvable: {surv.get('aliases')}"
    )


def test_catalog_carry_forward_case_only_respell_keeps_committed_casing(mod, tmp_path, monkeypatch):
    """Case-only respell at the same rung: the committed casing wins in the
    catalog path too; the fresh casing becomes an alias."""
    committed = [{
        "id": "zorgvid/zeo-x1", "display_name": "Zeo X1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/Zeo-X1", "display_name": "Zeo X1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh])
    out = yaml.safe_load(cat.read_text()) or []
    by_id = {e["id"]: e for e in out}
    assert "zorgvid/zeo-x1" in by_id, "committed casing must win among equals"
    assert "zorgvid/Zeo-X1" not in by_id
    assert "zorgvid/Zeo-X1" in (by_id["zorgvid/zeo-x1"].get("aliases") or [])


# ---------------------------------------------------------------------------
# HF promotion: the hub_stats index joins the oracle as the on-HF authority;
# a mint the index can name (canonical_model_id / provider id / name) becomes
# the HF-true id with every models.dev spelling as an alias.
# ---------------------------------------------------------------------------

def _tiny_index(tmp_path, rows):
    import pandas as pd

    p = tmp_path / "hub_stats_index.parquet"
    pd.DataFrame(rows, columns=["id", "downloads"]).to_parquet(p, index=False)
    return p


def _rec(raw, name, cmid=None, provider="nano-gpt"):
    record = {"id": raw, "name": name}
    if cmid:
        record["canonical_model_id"] = cmid
    return {"provider": provider, "raw": raw, "name": name, "record": record}


_PROMO_AUTH = {
    "mistralai": {
        "ministral 3 14b instruct 2512": "mistralai/Ministral-3-14B-Instruct-2512",
        "ministral 3 14b reasoning 2512": "mistralai/Ministral-3-14B-Reasoning-2512",
    },
    "thedrummer": {
        "unslopnemo 12b v4": "TheDrummer/UnslopNemo-12B-v4",
        "unslopnemo 12b v4 1": "TheDrummer/UnslopNemo-12B-v4.1",
    },
    "deepseek": {
        "deepseek v4 flash 0731": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "deepseek v4 1 flash": "deepseek-ai/DeepSeek-V4.1-Flash",
        "deepseek v4 pro": "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek v4 pro 0813": "deepseek-ai/DeepSeek-V4-Pro-0813",
    },
    "ibm": {
        "granite 4 0 micro": "ibm-granite/granite-4.0-micro",
        "granite 4 0 h micro": "ibm-granite/granite-4.0-h-micro",
    },
    "moonshotai": {"kimi k3": "moonshotai/Kimi-K3"},
    "alibaba": {"qwen3 vl 30b a3b thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking"},
}


def test_index_authority_buckets_by_dev_org_and_oracle_wins(mod, tmp_path):
    """The index is bucketed by curated dev org (uncurated orgs by lowercased
    slug) and keyed by normalized name; a shared name-norm resolves to the
    higher download count; the oracle wins a conflicting key."""
    p = _tiny_index(tmp_path, [
        ("Qwen/Qwen3-4B", 10),
        ("Qwen/qwen3_4b", 50),
        ("deepseek-ai/DeepSeek-OCR", 1),
        ("TheDrummer/Anubis-70B-v1", 2),
        ("mistralai/Ministral-3-3B-Instruct-2512", 3),
    ])
    ai = mod._build_org_alias_index()
    auth = mod._build_index_authority(mod._load_hub_stats_index(p), ai)
    assert auth["alibaba"]["qwen3 4b"] == "Qwen/qwen3_4b"
    assert auth["deepseek"]["deepseek ocr"] == "deepseek-ai/DeepSeek-OCR"
    assert auth[ai.get("thedrummer", "thedrummer")]["anubis 70b v1"] == "TheDrummer/Anubis-70B-v1"
    assert auth["mistralai"]["ministral 3 3b instruct 2512"] == "mistralai/Ministral-3-3B-Instruct-2512"
    merged = mod._merge_authorities({"alibaba": {"qwen3 4b": "Qwen/Qwen3-4B"}}, auth)
    assert merged["alibaba"]["qwen3 4b"] == "Qwen/Qwen3-4B"
    assert merged["deepseek"]["deepseek ocr"] == "deepseek-ai/DeepSeek-OCR"
    # A `mistral/...` key resolves inside the `mistralai` bucket.
    assert mod._group_hf_target(
        "mistralai/ministral-3b-2512", "mistralai",
        [_rec("ministral-3b-2512", "Ministral 3 3B", "mistral/ministral-3-3b-instruct-2512", provider="mistral")],
        ["ministral-3b-2512", "Ministral 3 3B"], ai, auth,
    ) == "mistralai/Ministral-3-3B-Instruct-2512"


def test_hub_stats_index_missing_fails_loudly(mod, tmp_path, monkeypatch):
    """With no index available the refresh fails rather than silently minting
    what is on HF."""
    import huggingface_hub

    def _offline(**_kw):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _offline)
    with pytest.raises(RuntimeError, match="hub_stats_index unavailable"):
        mod._load_hub_stats_index(tmp_path / "missing.parquet")


def test_group_hf_target_key_precedence(mod):
    ai = mod._build_org_alias_index()
    auth = _PROMO_AUTH
    # Key 1: canonical_model_id names the repo.
    assert mod._group_hf_target(
        "mistralai/ministral-14b-2512", "mistralai",
        [_rec("ministral-14b-2512", "Ministral 3 14B", "mistral/ministral-3-14b-instruct-2512", provider="mistral")],
        ["ministral-14b-2512", "Ministral 3 14B"], ai, auth,
    ) == "mistralai/Ministral-3-14B-Instruct-2512"
    # Key 2: a verbatim `org/name` provider id beats a sloppier record name;
    # a cased uncurated org falls back to the lowercased bucket.
    assert mod._group_hf_target(
        "TheDrummer/unslopnemo-12b-v4-1", "TheDrummer",
        [_rec("TheDrummer/UnslopNemo-12B-v4.1", "UnslopNemo 12b v4")],
        ["TheDrummer/unslopnemo-12b-v4-1"], ai, auth,
    ) == "TheDrummer/UnslopNemo-12B-v4.1"
    # Key 3: a record name names the repo.
    assert mod._group_hf_target(
        "mistralai/ministral-14b-instruct-2512", "mistralai",
        [_rec("mistralai/ministral-14b-instruct-2512", "Ministral 3 14B"),
         _rec("mistralai/ministral-14b-instruct-2512", "Ministral 3 14B Instruct 2512", provider="nvidia")],
        ["mistralai/ministral-14b-instruct-2512", "Ministral 3 14B"], ai, auth,
    ) == "mistralai/Ministral-3-14B-Instruct-2512"
    # Within key 1 the majority canonical key wins over a shorter minority one.
    assert mod._group_hf_target(
        "deepseek/deepseek-v4-flash", "deepseek",
        [_rec("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", "deepseek/deepseek-v4-flash-0731"),
         _rec("deepseek-v4-flash", "DeepSeek V4 Flash", "deepseek/deepseek-v4-flash-0731", provider="vivgrid"),
         _rec("deepseek-v4-flash", "DeepSeek V4 Flash", "deepseek/deepseek-v4.1-flash", provider="deepseek")],
        ["deepseek-v4-flash", "DeepSeek V4 Flash"], ai, auth,
    ) == "deepseek-ai/DeepSeek-V4-Flash-0731"
    # The would-be mint id is tried first: a rolling key never pulls a group
    # off its own repo.
    assert mod._group_hf_target(
        "deepseek/deepseek-v4-pro", "deepseek",
        [_rec("deepseek-v4-pro", "DeepSeek V4 Pro", "deepseek/deepseek-v4-pro-0813", provider="deepseek"),
         _rec("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", "deepseek/deepseek-v4-pro-0813")],
        ["deepseek-v4-pro", "DeepSeek V4 Pro"], ai, auth,
    ) == "deepseek-ai/DeepSeek-V4-Pro"
    # No hit / no org: mint.
    assert mod._group_hf_target(
        "anthropic/claude-opus-4-5", "anthropic",
        [_rec("claude-opus-4-5", "Claude Opus 4.5", provider="anthropic")],
        ["claude-opus-4-5", "Claude Opus 4.5"], ai, auth,
    ) is None
    assert mod._group_hf_target(
        "unslopnemo-12b-v4-1", None, [_rec("TheDrummer/UnslopNemo-12B-v4.1", "x")], [], ai, auth,
    ) is None


def test_group_hf_target_never_drops_mint_tokens_or_adds_variants(mod):
    ai = mod._build_org_alias_index()
    auth = _PROMO_AUTH
    granite = [_rec("@cf/ibm-granite/granite-4.0-h-micro", "Granite 4.0 Micro", "ibm/granite-4-h-micro",
                    provider="cloudflare-workers-ai")]
    assert mod._group_hf_target(
        "ibm/granite-4-0-h-micro", "ibm", granite, ["Granite 4.0 Micro"], ai, auth,
    ) == "ibm-granite/granite-4.0-h-micro"
    only_micro = {"ibm": {"granite 4 0 micro": "ibm-granite/granite-4.0-micro"}}
    assert mod._group_hf_target(
        "ibm/granite-4-0-h-micro", "ibm", granite, ["Granite 4.0 Micro"], ai, only_micro,
    ) is None
    # A rolling `-latest` key pointing at the current release stays a mint.
    assert mod._group_hf_target(
        "moonshotai/kimi-latest", "moonshotai",
        [_rec("kimi-latest", "Kimi Latest", "moonshotai/kimi-k3", provider="moonshotai")],
        ["kimi-latest", "Kimi Latest"], ai, auth,
    ) is None
    # A bare record never promotes to a `-thinking` repo its variant-stripped
    # group also holds; the thinking record itself does.
    thinking = [_rec("Qwen/Qwen3-VL-30B-A3B-Thinking", "Qwen3-VL-30B-A3B-Thinking", provider="siliconflow")]
    assert mod._group_hf_target(
        "alibaba/qwen3-vl-30b-a3b", "alibaba", thinking, ["qwen3-vl-30b-a3b"], ai, auth,
    ) is None
    assert mod._group_hf_target(
        "alibaba/qwen3-vl-30b-a3b-thinking", "alibaba", thinking, ["qwen3-vl-30b-a3b-thinking"], ai, auth,
    ) == "Qwen/Qwen3-VL-30B-A3B-Thinking"


def test_hf_deferred_entry_aliases_only_the_matching_canonical_key(mod):
    head = {"display_name": "DeepSeek-V3", "open_weights": True, "release_date": None, "root_key": "deepseek"}
    recs = [
        _rec("deepseek-ai/DeepSeek-V3", "DeepSeek-V3", "deepseek/deepseek-v3", provider="deepinfra"),
        _rec("deepseek-reasoner", "DeepSeek Reasoner", "deepseek/deepseek-reasoner"),
    ]
    e = mod._hf_deferred_entry("deepseek-ai/DeepSeek-V3", "deepseek", head, recs, "deepseek/deepseek", "DeepSeek-V3")
    assert json.loads(e["metadata"])["hf_deferred"] is True
    assert {"deepseek/deepseek", "deepseek/deepseek-v3"} <= set(e["aliases"])
    assert "deepseek/deepseek-reasoner" not in e["aliases"]


def test_attach_provider_aliases_hf_deferred_target_takes_models_dev_forms(mod):
    """An HF-deferred target attaches raws matching one of its id-shaped
    aliases, never a variant raw, and never one matching only through the
    display name (`_identity` drops a parenthetical)."""
    def _entry(deferred):
        meta = {"hf_deferred": True} if deferred else {}
        return {
            "id": "mistralai/Ministral-3-14B-Instruct-2512", "display_name": "Ministral 3 14B",
            "org_id": "mistralai", "parents": [],
            "aliases": ["mistralai/ministral-14b-2512", "Ministral 3 14B"],
            "metadata": json.dumps(meta, sort_keys=True),
        }
    recs = [
        {"provider": "nano-gpt", "raw": "mistralai/ministral-14b-2512"},
        {"provider": "mistral", "raw": "ministral-14b-2512"},
        {"provider": "nano-gpt", "raw": "mistralai/ministral-14b-2512-fp8"},
        {"provider": "ollama-cloud", "raw": "ministral-3:14b"},
    ]
    deferred = _entry(True)
    mod._attach_provider_aliases([deferred], recs, "mistralai")
    ap = deferred["alias_platforms"]
    assert {"ministral-14b-2512", "mistralai/ministral-14b-2512"} <= set(ap)
    assert not any("fp8" in k for k in ap)
    assert not any(k in ("ministral-3:14b", "ministral-3-14b", "mistralai/ministral-3-14b") for k in ap)
    plain = _entry(False)
    mod._attach_provider_aliases([plain], recs, "mistralai")
    assert not plain.get("alias_platforms"), "a non-deferred target keeps the strict identity guard"


def test_generation_promotes_ministral_group_to_hf_true_id(mod, monkeypatch):
    """First-party record and re-host group both land on the HF-true id with
    every spelling aliased; no `{org}/{slug}` mint or family root is minted
    for them. A record the authority cannot place still builds its family."""
    for prov in ("mistral", "nano-gpt", "nvidia", "clarifai"):
        assert prov in mod.PROVIDER_TO_INFERENCE_PLATFORM
    assert "mistral" in mod.STRICT_AUTHOR

    def _m(mid, name, cmid=None):
        d = {"id": mid, "name": name, "open_weights": True, "release_date": "2025-12-02"}
        if cmid:
            d["canonical_model_id"] = cmid
        return d

    api = {
        "mistral": {"models": {
            "ministral-14b-2512": _m("ministral-14b-2512", "Ministral 3 14B", "mistral/ministral-3-14b-instruct-2512"),
            "magistral-medium-2509": _m("magistral-medium-2509", "Magistral Medium 1.2"),
        }},
        "nano-gpt": {"models": {
            "mistralai/ministral-14b-2512": _m("mistralai/ministral-14b-2512", "Ministral 14B"),
            "mistralai/ministral-14b-instruct-2512": _m("mistralai/ministral-14b-instruct-2512", "Ministral 3 14B"),
        }},
        "nvidia": {"models": {
            "mistralai/ministral-14b-instruct-2512": _m("mistralai/ministral-14b-instruct-2512", "Ministral 3 14B Instruct 2512"),
        }},
        "clarifai": {"models": {
            "mistralai/completion/models/Ministral-3-14B-Reasoning-2512": _m(
                "mistralai/completion/models/Ministral-3-14B-Reasoning-2512", "Ministral 3 14B Reasoning 2512"),
        }},
    }
    monkeypatch.setattr(mod, "_HF_AUTHORITY", _PROMO_AUTH)
    out, skipped = mod._generate_models(api, mod._load_known_org_ids())
    assert skipped == []
    out = mod._finalize_entries(out)
    by_id = {e["id"]: e for e in out}
    assert set(by_id) == {
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/Ministral-3-14B-Reasoning-2512",
        "mistralai/magistral-medium-1.2",
    }, sorted(by_id)
    inst = by_id["mistralai/Ministral-3-14B-Instruct-2512"]
    assert json.loads(inst["metadata"])["hf_deferred"] is True
    assert inst["review_status"] == "reviewed"
    assert {
        "mistralai/ministral-14b-2512", "mistralai/ministral-14b-instruct-2512",
        "ministral-14b-2512", "ministral-14b-instruct-2512", "Ministral 3 14B",
        "mistral/ministral-3-14b-instruct-2512",
    } <= set(inst["aliases"]), inst["aliases"]
    assert json.loads(by_id["mistralai/Ministral-3-14B-Reasoning-2512"]["metadata"])["hf_deferred"] is True
    assert "magistral-medium-2509" in json.loads(by_id["mistralai/magistral-medium-1.2"]["metadata"])["snapshots"]


def test_carry_forward_absorbs_committed_mint_aliased_on_hf_deferred_twin(cf_mod):
    """When twin keys cannot link a promoted id to the HF-true id, the fresh
    hf_deferred entry aliasing the committed id absorbs it. A non-deferred
    fresh entry does not absorb that way; a committed hf_deferred id never
    is absorbed."""
    committed = [
        dict(_cf_entry("mistralai/ministral-14b-2512", ["ministral-14b-2512"]), display_name="Ministral 14B"),
        _cf_entry("mistralai/ministral-3-14b", ["ministral-3:14b"]),
        _cf_entry("zorgx/zmod-old", ["zmod-old"]),
        dict(_cf_entry("zorgx/ZMod-HF"), metadata=json.dumps({"hf_deferred": True}, sort_keys=True)),
    ]
    fresh = [
        dict(_cf_entry("mistralai/Ministral-3-14B-Instruct-2512"), display_name="Ministral 3 14B",
             aliases=["mistralai/ministral-14b-2512"],
             metadata=json.dumps({"hf_deferred": True}, sort_keys=True)),
        dict(_cf_entry("zorgx/zmod-new"), aliases=["zorgx/zmod-old"]),
        dict(_cf_entry("zorgx/ZMod-Other"), aliases=["zorgx/ZMod-HF"],
             metadata=json.dumps({"hf_deferred": True}, sort_keys=True)),
    ]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "mistralai/ministral-14b-2512" not in by_id
    surv = by_id["mistralai/Ministral-3-14B-Instruct-2512"]
    assert {"mistralai/ministral-14b-2512", "ministral-14b-2512", "Ministral 14B"} <= set(surv["aliases"])
    assert json.loads(surv["metadata"])["hf_deferred"] is True
    assert claims["mistralai/ministral-14b-2512"] == "mistralai/Ministral-3-14B-Instruct-2512"
    assert claims["ministral-14b-2512"] == "mistralai/Ministral-3-14B-Instruct-2512"
    kept = by_id["mistralai/ministral-3-14b"]
    assert json.loads(kept["metadata"])["upstream_status"] == "removed"
    assert "zorgx/zmod-old" in by_id, "alias-equality twin applies to hf_deferred fresh entries only"
    assert "zorgx/ZMod-HF" in by_id, "a committed HF id is never demoted to another HF id's alias"


def test_reconcile_keeps_absorbed_mint_alias_on_promoted_entry(cf_mod, tmp_path):
    """Carry-forward absorbs the committed id before reconcile, so the
    sibling-id strip leaves the alias in place."""
    core = tmp_path / "core.yaml"
    core.write_text("entries: []\n")
    committed = [_cf_entry("zorgx/zmod-3-2512", ["zmod-3-2512"])]
    fresh = [dict(_cf_entry("zorgx/ZMod-3-Instruct-2512"), display_name="ZMod 3",
                  aliases=["zorgx/zmod-3-2512"],
                  metadata=json.dumps({"hf_deferred": True}, sort_keys=True))]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    out = cf_mod.reconcile_generated_against_existing(batch, sources=(core,), committed_claims=claims)
    by_id = {e["id"]: e for e in out}
    assert "zorgx/zmod-3-2512" not in by_id
    assert {"zorgx/zmod-3-2512", "zmod-3-2512"} <= set(by_id["zorgx/ZMod-3-Instruct-2512"]["aliases"])


# ---------------------------------------------------------------------------
# Protected roots: a curated umbrella or multi-child family root is never
# deferred to an HF repo.
# ---------------------------------------------------------------------------


def test_multi_child_roots_counts_distinct_children(mod, tmp_path, monkeypatch):
    parents = tmp_path / "parents.yaml"
    parents.write_text(yaml.safe_dump([
        {"id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
         "parents": [{"id": "alibaba/qwen3-vl-235b-a22b", "relationship": "variant", "axis": "training_stage"}]},
        {"id": "Qwen/Qwen3-VL-235B-A22B-Thinking",
         "parents": [{"id": "alibaba/qwen3-vl-235b-a22b", "relationship": "variant", "axis": "mode"}]},
        {"id": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
         "parents": [{"id": "alibaba/qwen3-coder-480b-a35b", "relationship": "variant"}]},
        {"id": "lab/self", "parents": [{"id": "lab/self", "relationship": "variant"}]},
    ]))
    for name in ("CORE_PATH", "HUB_STATS_PATH", "TIER3_PATH", "SEED_PATH"):
        monkeypatch.setattr(mod, name, tmp_path / f"missing-{name}.yaml")
    monkeypatch.setattr(mod, "PARENTS_ENRICH_PATH", parents)
    assert mod._multi_child_roots() == frozenset({"alibaba/qwen3-vl-235b-a22b"})


def test_group_hf_target_never_defers_a_protected_root(mod, monkeypatch):
    authority = {"alibaba": {"qwen3 vl 235b a22b instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct"}}
    recs = [{"raw": "qwen3-vl-235b-a22b", "name": "Qwen3 VL 235B A22B",
             "record": {"canonical_model_id": "qwen/qwen3-vl-235b-a22b-instruct"}}]
    ai = mod._dev_alias_index()
    monkeypatch.setattr(mod, "_PROTECTED_ROOTS", frozenset())
    assert mod._group_hf_target(
        "alibaba/qwen3-vl-235b-a22b", "alibaba", recs, ["qwen3-vl-235b-a22b"], ai, authority
    ) == "Qwen/Qwen3-VL-235B-A22B-Instruct"
    monkeypatch.setattr(mod, "_PROTECTED_ROOTS", frozenset({"alibaba/qwen3-vl-235b-a22b"}))
    assert mod._group_hf_target(
        "alibaba/qwen3-vl-235b-a22b", "alibaba", recs, ["qwen3-vl-235b-a22b"], ai, authority
    ) is None


def test_group_hf_target_rejects_size_and_format_drift(mod, monkeypatch):
    ai = mod._dev_alias_index()
    monkeypatch.setattr(mod, "_PROTECTED_ROOTS", frozenset())
    auth = {"lab": {
        "model 70b": "lab/Model-70B",
        "model 7b gguf": "lab/Model-7B-GGUF",
        "model 7b": "lab/Model-7B",
    }}
    rec = lambda raw, name: {"raw": raw, "name": name, "record": {}}
    # `model7b` is a character subsequence of `model70b`: the size must match.
    assert mod._group_hf_target("lab/model-7b", "lab", [rec("model-7b", "Model-70B")], ["model-7b", "Model-70B"], ai, auth) == "lab/Model-7B"
    # A weight-format repo is a different upload.
    assert mod._group_hf_target("lab/model-7b", "lab", [rec("model-7b", "Model-7B-GGUF")], ["model-7b", "Model-7B-GGUF"], ai, auth) == "lab/Model-7B"
    assert mod._group_hf_target("lab/model-7b", "lab", [rec("model-7b", "Model-7B-GGUF")], ["model-7b", "Model-7B-GGUF"], ai, {"lab": {"model 7b gguf": "lab/Model-7B-GGUF"}}) is None


def test_orgless_mint_folds_only_on_a_unique_open_weight_name(mod):
    from eval_entity_resolver.normalization import normalize as _norm
    hf = lambda i, name: {"id": i, "display_name": name, "aliases": [], "metadata": '{"hf_deferred": true}'}
    bare = lambda i, name, ow: {"id": i, "display_name": name, "aliases": [], "open_weights": ow, "metadata": "{}"}
    # Drives the post-pass the way _generate_models does.
    def fold(entries):
        hf_by_name = {}
        for e in entries:
            if mod._entry_meta(e).get("hf_deferred") is True:
                hf_by_name.setdefault(_norm(e["display_name"]), []).append(e)
        kept = []
        for e in entries:
            cands = hf_by_name.get(_norm(e["display_name"]), []) if "/" not in e["id"] and e.get("open_weights") is True and mod._entry_meta(e).get("hf_deferred") is not True else []
            if len(cands) == 1:
                cands[0]["aliases"].append(e["id"])
                continue
            kept.append(e)
        return kept
    unique = fold([hf("lab/Flash-Pro", "Flash"), bare("flash", "Flash", True)])
    assert [e["id"] for e in unique] == ["lab/Flash-Pro"]
    ambiguous = fold([hf("lab/Flash-Pro", "Flash"), hf("other/Flash-Max", "Flash"), bare("flash", "Flash", True)])
    assert "flash" in {e["id"] for e in ambiguous}
    closed = fold([hf("lab/Flash-Pro", "Flash"), bare("private-flash", "Flash", False)])
    assert "private-flash" in {e["id"] for e in closed}
