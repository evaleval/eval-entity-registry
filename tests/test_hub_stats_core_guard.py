"""Core guard at the hub-stats emit point (`build_entry`, driven offline):
no emitted id may normalized-collide with a curated core canonical under a
different id. A twin defers to the curated id and casing; a novel id is
suppressed; an emitted alias never steals another canonical's resolution.
"""
from __future__ import annotations

import pytest

from conftest import load_script_module


@pytest.fixture
def mod():
    return load_script_module("refresh_from_hub_stats")


# Synthetic curated-core canonical; normalizes to `foo-bar-7b`.
CORE_CANONICAL_ID = "Foo/Bar-7B"
CORE_ORG_ID = "foo"

# Normalized form -> exact curated id, as `load_existing_canonical_aliases` builds it.
ALIASES_TO_CANONICAL = {"foo-bar-7b": CORE_CANONICAL_ID}

HF_TO_DEV = {"foo": CORE_ORG_ID}

ORG_ALIAS_MAP = {"foo": CORE_ORG_ID}


def _build(mod, row, **overrides):
    kw = dict(
        org_alias_map=ORG_ALIAS_MAP,
        aliases_to_canonical=ALIASES_TO_CANONICAL,
        hf_to_dev=HF_TO_DEV,
        canonical_id_norms={mod._normalize(CORE_CANONICAL_ID): CORE_CANONICAL_ID},
        ambiguous_canonicals=set(),
    )
    kw.update(overrides)
    return mod.build_entry(row, **kw)


def test_emit_defers_to_curated_core_id_never_normalized_colliding_twin(mod):
    """A different-casing twin of a core canonical emits under the curated id
    and casing, merging onto core instead of minting a colliding twin."""
    row = {"id": "foo/bar-7b"}

    entry = _build(mod, row)

    assert entry is not None, "a row whose normalized form matches core must enrich it"
    assert entry["id"] == CORE_CANONICAL_ID
    assert entry["id"] != "foo/bar-7b"
    assert mod._normalize(entry["id"]) == mod._normalize(CORE_CANONICAL_ID)


def test_emit_never_produces_a_different_id_with_colliding_norm(mod):
    """Exact, separator-drift and case-drift spellings all emit the core id."""
    core_norm = mod._normalize(CORE_CANONICAL_ID)
    for hf_id in ("Foo/Bar-7B", "foo/bar-7b", "foo/bar_7b", "FOO/BAR-7B"):
        entry = _build(mod, {"id": hf_id})
        assert entry is not None, f"{hf_id} should match core (normalized)"
        if mod._normalize(entry["id"]) == core_norm:
            assert entry["id"] == CORE_CANONICAL_ID, (
                f"row {hf_id!r} emitted id {entry['id']!r} which normalized-collides "
                f"with curated core {CORE_CANONICAL_ID!r} under a DIFFERENT id"
            )


def test_novel_hf_id_is_suppressed_not_minted_as_canonical(mod):
    """The backfill-only script never mints a fresh canonical."""
    entry = _build(mod, {"id": "brand-new-org/totally-novel-model-99b"})
    assert entry is None


def test_emitted_aliases_never_steal_a_different_core_canonical(mod):
    """An emitted alias whose normalized form another canonical owns is
    dropped (`_alias_ok`), so no enrichment alias intercepts its resolution."""
    other_id = "Other/Thing-13B"
    aliases_to_canonical = dict(ALIASES_TO_CANONICAL)
    aliases_to_canonical[mod._normalize(other_id)] = other_id
    cid_norms = {
        mod._normalize(CORE_CANONICAL_ID): CORE_CANONICAL_ID,
        mod._normalize(other_id): other_id,
        # Bare-name claim, as load_canonical_id_norms emits it.
        mod._normalize("Thing-13B"): other_id,
    }
    # A row forced to resolve to Foo/Bar-7B whose raw id shares the OTHER
    # canonical's bare name.
    row = {"id": "foo/Thing-13B"}
    aliases_to_canonical[mod._normalize("foo/Thing-13B")] = CORE_CANONICAL_ID

    entry = _build(
        mod, row,
        aliases_to_canonical=aliases_to_canonical,
        canonical_id_norms=cid_norms,
    )
    assert entry is not None
    assert entry["id"] == CORE_CANONICAL_ID
    for a in entry.get("aliases", []):
        name_norm = mod._normalize(a.split("/", 1)[1]) if "/" in a else mod._normalize(a)
        assert cid_norms.get(name_norm, CORE_CANONICAL_ID) == CORE_CANONICAL_ID, (
            f"emitted alias {a!r} steals canonical {cid_norms.get(name_norm)!r}"
        )
