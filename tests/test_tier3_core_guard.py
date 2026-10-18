"""The Tier-3 generator never mints an id that normalized-collides with a
curated core canonical under a different id: the curated id wins and the
mint is skipped, never twinned. Drives the guard helpers offline.
"""
from __future__ import annotations

import pytest

from conftest import load_script_module
from eval_entity_resolver.normalization import normalize as _rnz


@pytest.fixture(scope="module")
def mod():
    return load_script_module("generate_tier3_inferred_seed")


CORE_DOC = {
    "skip_ids": [],
    "skip_source_ids": [],
    "entries": [
        {
            "id": "Foo/Bar-7B",
            "display_name": "Foo Bar 7B",
            "aliases": ["foo-bar-7b", "Foo/Bar-7B"],
            "metadata": '{"hf_id": "Foo/Bar-7B"}',
        }
    ],
}


def test_build_core_norm_index_maps_forms_to_curated_owner(mod):
    idx = mod.build_core_norm_index(CORE_DOC)
    owner = "Foo/Bar-7B"
    assert idx[_rnz("Foo/Bar-7B")] == owner
    assert idx[_rnz("foo/bar-7b")] == owner
    assert idx[_rnz("Foo Bar 7B")] == owner
    assert idx[_rnz("foo-bar-7b")] == owner


def test_build_core_norm_index_handles_flat_list_shape(mod):
    flat = [{"id": "Baz/Qux-3B", "aliases": []}]
    idx = mod.build_core_norm_index(flat)
    assert idx[_rnz("baz/qux-3b")] == "Baz/Qux-3B"


def test_core_steals_flags_normalized_twin_under_different_id(mod):
    idx = mod.build_core_norm_index(CORE_DOC)
    assert mod.core_steals("foo/bar-7b", idx) is True
    assert mod.core_steals("Foo_Bar_7B", idx) is True
    assert mod.core_steals("foo-bar-7b", idx) is True


def test_core_steals_allows_curated_id_itself_and_novel_mints(mod):
    idx = mod.build_core_norm_index(CORE_DOC)
    assert mod.core_steals("Foo/Bar-7B", idx) is False
    assert mod.core_steals("openai/gpt-4o", idx) is False
    assert mod.core_steals("acme/widget-13b", idx) is False


def test_core_collision_is_SKIPPED_not_minted_as_inferred_twin(mod):
    """A core collision is the same model: `mint_collision_decision` skips it
    rather than minting a `-inferred` twin that splits one model into two."""
    idx = mod.build_core_norm_index(CORE_DOC)

    raw = "foo/bar-7b"
    cid, _org = mod.canon_id_for_org_present(raw, {})
    assert cid == "foo/bar-7b"
    assert mod.core_steals(cid, idx) is True

    decision = mod.mint_collision_decision(
        cid, core_skip_ids=set(), core_norm_index=idx, resolver_hit=False
    )
    assert decision == "skip", (
        "a same-model core collision must be skipped (raw resolves to curated), "
        "never minted as a -inferred twin"
    )


def test_genuine_cross_entity_clash_is_disambiguated_not_skipped(mod):
    """A clash with a skip_ids entry or a different existing canonical is
    cross-entity: suffix-disambiguated ('inferred'), never dropped."""
    idx = mod.build_core_norm_index(CORE_DOC)
    novel = "acme/widget-13b"
    assert mod.core_steals(novel, idx) is False

    assert mod.mint_collision_decision(novel, {novel}, idx, resolver_hit=False) == "inferred"
    assert mod.mint_collision_decision(novel, set(), idx, resolver_hit=True) == "inferred"
    assert mod.mint_collision_decision(novel, set(), idx, resolver_hit=False) == "mint"


def test_guard_never_rewrites_id_equal_to_its_hf_id(mod):
    """The curated id itself is never a steal, so it is never suffix-disambiguated."""
    idx = mod.build_core_norm_index(CORE_DOC)
    assert mod.core_steals("Foo/Bar-7B", idx) is False
