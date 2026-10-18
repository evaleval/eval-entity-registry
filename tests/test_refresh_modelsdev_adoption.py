"""OpenRouter id adoption in the models.dev refresh: adoption on the mint and
family paths, the variant-identity guard, tag and `~` cleanup, the tie-break,
`openrouter/*` exclusion, old-id-as-alias, and core-collision suppression.
Synthetic data, no network."""
import json

import pytest
import yaml

from conftest import load_script_module


@pytest.fixture
def mod():
    return load_script_module("refresh_from_modelsdev")


KNOWN_ORGS = {"anthropic", "openai", "google", "meta", "deepseek", "perplexity"}


def _or_rec(raw, name=None, **extra):
    return {
        "id": raw, "name": name or raw.split("/")[-1],
        "family": None, "open_weights": False, **extra,
    }


# ---------------------------------------------------------------------------
# Mint path: eligible adoption
# ---------------------------------------------------------------------------
def test_mint_path_adopts_openrouter_key_and_keeps_old_id_as_alias(mod):
    """A models.dev-only mint whose group carries an identity-matching
    OpenRouter key adopts the key verbatim; the invented id becomes an alias
    and the entry is tagged `metadata.openrouter_adopted`."""
    api = {
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "zorg-ai/zmodel-2.5": _or_rec("zorg-ai/zmodel-2.5", "ZModel 2.5"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    by_id = {e["id"]: e for e in out}
    assert "zorg-ai/zmodel-2.5" in by_id, f"adopted id missing; got {sorted(by_id)}"
    e = by_id["zorg-ai/zmodel-2.5"]
    assert "zorg-ai/zmodel-2-5" in e["aliases"], "invented id must survive as an alias"
    assert json.loads(e["metadata"])["openrouter_adopted"] is True
    assert e["org_id"] == "zorg-ai"


def test_mint_path_order_swapped_key_adopts_via_identity_sig(mod):
    """Token order differs between the invented id and the key; the
    order-insensitive identity sig still matches."""
    api = {
        "anthropic": {"id": "anthropic", "name": "Anthropic", "models": {
            "claude-zzz-3": _or_rec("claude-zzz-3", "Claude Zzz 3"),
        }},
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "anthropic/claude-3-zzz": _or_rec("anthropic/claude-3-zzz", "Claude Zzz 3"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    by_id = {e["id"]: e for e in out}
    assert "anthropic/claude-3-zzz" in by_id
    assert "anthropic/claude-zzz-3" not in by_id
    assert "anthropic/claude-zzz-3" in by_id["anthropic/claude-3-zzz"]["aliases"]


# ---------------------------------------------------------------------------
# Family path: adoption + child edge repoint
# ---------------------------------------------------------------------------
def test_family_path_adopts_root_and_repoints_child_edges(mod):
    """An author-lab family root renames to the OpenRouter spelling; the dated
    child keeps its invented id but its parent edge repoints to the adopted
    root."""
    api = {
        "anthropic": {"id": "anthropic", "name": "Anthropic", "models": {
            "claude-zzz-3": _or_rec("claude-zzz-3", "Claude Zzz 3"),
            "claude-zzz-3-20250101": _or_rec("claude-zzz-3-20250101", "Claude Zzz 3"),
        }},
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "anthropic/claude-3-zzz": _or_rec("anthropic/claude-3-zzz", "Claude Zzz 3"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    by_id = {e["id"]: e for e in out}
    root = by_id["anthropic/claude-3-zzz"]
    assert root["parents"] == []
    assert "anthropic/claude-zzz-3" in root["aliases"]
    child = by_id["anthropic/claude-zzz-3-20250101"]
    assert child["parents"] == [{
        "id": "anthropic/claude-3-zzz", "relationship": "variant", "axis": "version",
    }]


# ---------------------------------------------------------------------------
# Eligibility guards
# ---------------------------------------------------------------------------
def test_variant_only_key_does_not_rename_base(mod):
    """A variant key never attaches to the base: the base keeps its invented id."""
    api = {
        "kilo": {"id": "kilo", "name": "Kilo", "models": {
            "zorg-ai/zmodel-2.5": _or_rec("zorg-ai/zmodel-2.5", "ZModel 2.5"),
        }},
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "zorg-ai/zmodel-2.5-thinking": _or_rec(
                "zorg-ai/zmodel-2.5-thinking", "ZModel 2.5 Thinking"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    by_id = {e["id"]: e for e in out}
    assert "zorg-ai/zmodel-2-5" in by_id, f"base must keep invented id; got {sorted(by_id)}"
    assert "zorg-ai/zmodel-2.5-thinking" not in by_id
    # The variant key must not contaminate the base's aliases either.
    assert not any("thinking" in a for a in by_id["zorg-ai/zmodel-2-5"]["aliases"])


def test_tagged_key_adopts_tag_stripped_never_with_tag(mod):
    """A tagged key is adopted tag-stripped; the raw tagged spelling stays
    an alias."""
    api = {
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "zorg-ai/zmodel-2.5:free": _or_rec("zorg-ai/zmodel-2.5:free", "ZModel 2.5"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    out = mod._finalize_entries(out)
    by_id = {e["id"]: e for e in out}
    assert "zorg-ai/zmodel-2.5" in by_id
    e = by_id["zorg-ai/zmodel-2.5"]
    assert "zorg-ai/zmodel-2.5:free" in e["aliases"], "raw tagged key must alias in"
    assert not any(":" in i for i in by_id), "no canonical id may carry a serving tag"


def test_latest_pointer_key_is_never_adopted(mod):
    """A `-latest` moving pointer names no fixed release."""
    assert mod._clean_openrouter_key("~anthropic/claude-zzz-latest") is None
    assert mod._clean_openrouter_key("anthropic/claude-zzz-latest") is None


def test_adoption_tie_break_shortest_then_alpha(mod):
    """Shortest tag-stripped key wins, then alphabetical."""
    alias_index = mod._build_org_alias_index()
    entries = [{
        "id": "perplexity/perplexity-zonar-1", "org_id": "perplexity",
        "aliases": [], "parents": [], "metadata": "{}",
    }]
    recs = [
        {"provider": "openrouter", "raw": "perplexity/perplexity-zonar-1x"},
        {"provider": "openrouter", "raw": "perplexity/zonar-1"},
    ]
    mod._adopt_openrouter_ids(entries, recs, "perplexity", alias_index)
    assert entries[0]["id"] == "perplexity/zonar-1"
    entries = [{
        "id": "zorg/z-a-9", "org_id": "zorg",
        "aliases": [], "parents": [], "metadata": "{}",
    }]
    recs = [
        {"provider": "openrouter", "raw": "zorg/a-z-9"},
        {"provider": "openrouter", "raw": "zorg/9-a-z"},
    ]
    mod._adopt_openrouter_ids(entries, recs, "zorg", alias_index)
    assert entries[0]["id"] == "zorg/9-a-z"


def test_org_disagreeing_key_is_not_adopted(mod):
    """A key under another developer's namespace never re-attributes the entry."""
    alias_index = mod._build_org_alias_index()
    entries = [{
        "id": "unsloth/zemma-3-4b-it", "org_id": "unsloth",
        "aliases": [], "parents": [], "metadata": "{}",
    }]
    recs = [{"provider": "openrouter", "raw": "google/zemma-3-4b-it"}]
    mod._adopt_openrouter_ids(entries, recs, "unsloth", alias_index)
    assert entries[0]["id"] == "unsloth/zemma-3-4b-it", "cross-dev adoption forbidden"


def test_hf_deferred_entry_is_never_renamed(mod):
    """An HF-deferred entry keeps the real HF id."""
    alias_index = mod._build_org_alias_index()
    entries = [{
        "id": "ZorgLabs/ZModel-2.5", "org_id": "zorglabs",
        "aliases": [], "parents": [],
        "metadata": json.dumps({"hf_deferred": True}),
    }]
    recs = [{"provider": "openrouter", "raw": "zorglabs/zmodel-2.5"}]
    mod._adopt_openrouter_ids(entries, recs, "zorglabs", alias_index)
    assert entries[0]["id"] == "ZorgLabs/ZModel-2.5"


def test_rehost_junk_key_is_never_adopted(mod, monkeypatch):
    """A key curation ruled junk (curation/rehost_repoint.json) is never adopted."""
    monkeypatch.setattr(mod, "_REHOST_JUNK_IDS", frozenset({"zorg-ai/zmodel-2.5"}))
    alias_index = mod._build_org_alias_index()
    entries = [{
        "id": "zorg-ai/zmodel-2-5", "org_id": "zorg-ai",
        "aliases": [], "parents": [], "metadata": "{}",
    }]
    recs = [{"provider": "openrouter", "raw": "zorg-ai/zmodel-2.5"}]
    mod._adopt_openrouter_ids(entries, recs, "zorg-ai", alias_index)
    assert entries[0]["id"] == "zorg-ai/zmodel-2-5"


# ---------------------------------------------------------------------------
# openrouter/* router pseudo-endpoints
# ---------------------------------------------------------------------------
def test_router_pseudo_endpoints_never_adopted_as_ids(mod):
    """`openrouter/*` keys are routing products: never a canonical id, never
    a verbatim alias. The bare mints stay (nothing is removed)."""
    api = {
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "openrouter/auto": _or_rec("openrouter/auto", "Auto Router"),
            "openrouter/bodybuilder": _or_rec("openrouter/bodybuilder", "Bodybuilder"),
        }},
        "kilo": {"id": "kilo", "name": "Kilo", "models": {
            "openrouter/bodybuilder": _or_rec("openrouter/bodybuilder", "Bodybuilder"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    out = mod._finalize_entries(out)
    assert all(not e["id"].lower().startswith("openrouter/") for e in out)
    for e in out:
        assert not any(a.lower().startswith("openrouter/") for a in e.get("aliases") or []), (
            f"router key aliased verbatim on {e['id']}: {e['aliases']}"
        )
    assert mod._clean_openrouter_key("openrouter/auto") is None, "router keys are never adoptable"


# ---------------------------------------------------------------------------
# Core collision suppresses adoption (curated id wins)
# ---------------------------------------------------------------------------
def test_core_collision_suppresses_adopted_mint(mod, tmp_path):
    """An adopted mint colliding with core folds onto the core id; the key
    survives only as an alias."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({"entries": [
        {"id": "zorg-ai/zmodel-2-5", "display_name": "ZModel 2.5", "aliases": []},
    ]}))
    adopted = [{
        "id": "zorg-ai/zmodel-2.5", "display_name": "ZModel 2.5 OR",
        "org_id": "zorg-ai", "aliases": ["zorg-ai/zmodel-2-5-old"],
        "metadata": json.dumps({"openrouter_adopted": True}),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(adopted, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "zorg-ai/zmodel-2.5" not in by_id, "core-colliding adoption must fold"
    rec = by_id["zorg-ai/zmodel-2-5"]
    assert "zorg-ai/zmodel-2-5-old" in rec["aliases"]


# ---------------------------------------------------------------------------
# Verbatim OpenRouter keys as aliases
# ---------------------------------------------------------------------------
def test_provider_alias_forms_emit_openrouter_key_verbatim(mod):
    forms = mod._provider_alias_forms(
        "meta-llama/llama-3.3-70b-instruct", "meta", "openrouter"
    )
    assert "meta-llama/llama-3.3-70b-instruct" in forms


def test_provider_alias_forms_slash_raw_still_dropped_for_other_providers(mod):
    forms = mod._provider_alias_forms(
        "meta-llama/llama-3.3-70b-instruct", "meta", "bedrock"
    )
    assert "meta-llama/llama-3.3-70b-instruct" not in forms


def test_provider_alias_forms_never_emit_router_keys_verbatim(mod):
    """A router key is never emitted verbatim; its bare form still flows."""
    forms = mod._provider_alias_forms("openrouter/auto", None, "openrouter")
    assert "openrouter/auto" not in forms


def test_openrouter_key_aliases_variant_entity_not_base(mod):
    """A variant key aliases the variant entity, never the base."""
    api = {
        "anthropic": {"id": "anthropic", "name": "Anthropic", "models": {
            "claude-zzz-3": _or_rec("claude-zzz-3", "Claude Zzz 3"),
            "claude-zzz-3-20250101": _or_rec("claude-zzz-3-20250101", "Claude Zzz 3"),
        }},
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "anthropic/claude-zzz-3-20250101": _or_rec(
                "anthropic/claude-zzz-3-20250101", "Claude Zzz 3 (2025-01-01)"),
        }},
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    out = mod._finalize_entries(out)
    by_id = {e["id"]: e for e in out}
    root = by_id["anthropic/claude-zzz-3"]
    snap = by_id["anthropic/claude-zzz-3-20250101"]
    assert "anthropic/claude-zzz-3-20250101" not in root["aliases"]
    assert snap["parents"], "dated child must keep its typed edge"
