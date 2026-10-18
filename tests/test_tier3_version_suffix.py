"""Tier-3 generator edge classification: a dated/version snapshot of a
confirmed base is a `variant/version` edge (folding into the base's group),
never a blanket `finetune`; a date plus a trailing effort token chains onto
the dated snapshot under `axis: mode`. The o-series pass runs only where the
family-token pass found nothing, so it can add an edge but never move one.
Drives the pure helpers offline.
"""
from __future__ import annotations

import pytest

from conftest import load_script_module


@pytest.fixture(scope="module")
def mod():
    return load_script_module("generate_tier3_inferred_seed")


@pytest.mark.parametrize(
    "tokens",
    [
        ["2026", "03", "05"],   # ISO date split on separators
        ["2024", "05", "13"],
        ["20250929"],           # compact 8-digit date
        ["2025", "08"],         # year-month snapshot
        ["202608"],             # compact year-month
        ["v0", "3"],            # slugified v0.3
        ["v2"],                 # bare vN
    ],
)
def test_pure_version_suffix_folds(mod, tokens):
    assert mod._is_pure_version_suffix(tokens) is True


@pytest.mark.parametrize(
    "tokens",
    [
        [],                       # no delta -> not a snapshot edge
        ["instruct"],             # training stage, not a version
        ["dpo"],                  # finetune marker
        ["thinking", "20250929"], # mode token present alongside a date
        ["mini"],                 # tier token
        ["turbo"],                # named variant
        ["8b"],                   # disclosed size
        ["2025", "04", "16", "high"],  # date + effort is a mode chain, not a version
    ],
)
def test_non_version_suffix_stays_finetune(mod, tokens):
    assert mod._is_pure_version_suffix(tokens) is False


# ---------------------------------------------------------------------------
# Date-then-mode tails: `o3-2025-04-16-high` chains onto the dated sibling
# under `axis: mode` (itself a version child of the base). A bare effort token
# with no date in front stays parentless.
# ---------------------------------------------------------------------------
HF_TO_DEV = {"openai": "openai", "kimi": "moonshotai"}


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["2025", "04", "16", "high"], (["2025", "04", "16"], ["high"])),
        (["2025", "01", "31", "low"], (["2025", "01", "31"], ["low"])),
        (["20250416", "medium"], (["20250416"], ["medium"])),
        (["2025", "04", "16", "fc"], (["2025", "04", "16"], ["fc"])),
        (["v0", "3", "thinking"], (["v0", "3"], ["thinking"])),
    ],
)
def test_date_then_mode_splits(mod, tokens, expected):
    assert mod.split_date_mode_suffix(tokens) == expected


@pytest.mark.parametrize(
    "tokens",
    [
        ["high"],                     # no date in front: ambiguous on its own
        ["medium"],                   # Mistral Medium is a product tier
        ["fast"],                     # not a recognised effort token
        ["preview"],                  # pointer/product label, not a mode
        ["thinking", "20250929"],     # mode BEFORE the date: order not evidenced
        ["2025", "04", "16", "dpo"],  # date + finetune marker
        ["8b", "high"],               # size in front of the effort token
        ["april", "2025", "high"],    # named month is not a validated date
    ],
)
def test_ambiguous_suffix_has_no_mode_split(mod, tokens):
    assert mod.split_date_mode_suffix(tokens) is None


@pytest.mark.parametrize(
    "tokens",
    [
        ["2025", "99", "99", "high"],  # month/day out of range
        ["20250230", "high"],          # February 30th
        ["999999", "high"],            # neither YYYYMM nor YYMMDD
        ["4096", "high"],              # param count, not a vendor date tag
        ["20251301", "fc"],            # month 13
    ],
)
def test_date_shaped_but_invalid_calendar_has_no_mode_split(mod, tokens):
    """Date shape is not enough: a tail that only looks like a date must not
    assert two models are the same release."""
    assert mod.split_date_mode_suffix(tokens) is None


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["2025", "04", "16"], True),
        (["20250416"], True),      # compact YYYYMMDD
        (["250416"], True),        # compact YYMMDD
        (["202608"], True),        # compact YYYYMM
        (["2025", "08"], True),
        (["2411"], True),          # Mistral YYMM tag
        (["0613"], True),          # MMDD tag
        (["1106"], True),          # MMDD tag
        (["v0", "3"], True),
        (["2025", "02", "30"], False),
        (["20250230"], False),
        (["999999"], False),
        (["4096"], False),
        (["april", "2025"], False),
        ([], False),
    ],
)
def test_validated_date_suffix(mod, tokens, expected):
    assert mod.is_validated_date_suffix(tokens) is expected


# ---------------------------------------------------------------------------
# The o-series parent decision emits a validated version edge or a deferred
# date-then-mode edge, and nothing else.
# ---------------------------------------------------------------------------
KNOWN_BASES = {
    "openai/o1", "openai/o1-mini", "openai/o1-preview",
    "openai/o3", "openai/o3-mini", "openai/o3-pro",
    "openai/o4-mini", "openai/o4-mini-high",
}


def _confirm(candidate):
    return candidate if candidate in KNOWN_BASES else None


@pytest.mark.parametrize(
    "raw,base",
    [
        ("openai/o3-2025-04-16", "openai/o3"),
        ("openai/o3-mini-2025-01-31", "openai/o3-mini"),
        ("openai/o1-2024-12-17", "openai/o1"),
        ("openai/o1-mini-2024-09-12", "openai/o1-mini"),
        ("openai/o1-preview-2024-09-12", "openai/o1-preview"),
        ("openai/o4-mini-2025-04-16", "openai/o4-mini"),
    ],
)
def test_o_series_dated_row_gets_version_edge(mod, raw, base):
    edge, sibling, confirmed = mod.o_series_edge(raw, HF_TO_DEV, _confirm)
    assert edge == {"id": base, "relationship": "variant", "axis": "version"}
    assert sibling is None
    assert confirmed == base


@pytest.mark.parametrize(
    "raw,sibling",
    [
        ("openai/o3-2025-04-16-high", "openai/o3-2025-04-16"),
        ("openai/o3-2025-04-16-fc", "openai/o3-2025-04-16"),
        ("openai/o3-2025-04-16-prompt", "openai/o3-2025-04-16"),
        ("openai/o3-mini-2025-01-31-low", "openai/o3-mini-2025-01-31"),
        ("openai/o4-mini-2025-04-16-medium", "openai/o4-mini-2025-04-16"),
    ],
)
def test_o_series_effort_row_defers_to_dated_sibling(mod, raw, sibling):
    edge, dated, _confirmed = mod.o_series_edge(raw, HF_TO_DEV, _confirm)
    assert edge is None, "the mode edge settles after the mint loop, not here"
    assert dated == sibling


@pytest.mark.parametrize(
    "raw",
    [
        # Named month: evidences neither a date nor retraining.
        "openai/o1-december-2024",
        "openai/o3-high-april-2025",
        "openai/o3-medium-april-2025",
        "openai/o4-mini-high-april-2025",
        "openai/o4-mini-medium-april-2025",
        # Bare effort token with no dated release line in front of it.
        "openai/o3-high",
        "openai/o3-mini-medium",
    ],
)
def test_o_series_ambiguous_tail_stays_parentless(mod, raw):
    """No finetune fallback: an o-series model publishes no weights, so an
    unexplained tail is not evidence of derivation."""
    edge, dated, _confirmed = mod.o_series_edge(raw, HF_TO_DEV, _confirm)
    assert edge is None
    assert dated is None


def test_o_series_rejects_sentinel_org(mod):
    """A placeholder prefix is not a namespace: openness inheritance walks
    `variant` edges without an org check."""
    assert mod.o_series_edge("unknown/o4-mini-2025-04-16", HF_TO_DEV, _confirm) == (
        None, None, None,
    )


def test_o_series_rejects_other_developer(mod):
    """A non-OpenAI `o1-*` name never confirms against OpenAI's release line."""
    assert mod.o_series_edge(
        "otherlab/o1-2024-12-17", HF_TO_DEV, lambda _c: "openai/o1",
    ) == (None, None, None)


@pytest.mark.parametrize(
    "raw",
    [
        # Owned by the family-token pass.
        "openai/gpt-o3-high",
        "openai/gpt-5-mini-2025-08-07-high",
        "openai/gpt-4-1-2025-04-14-fc",
        "anthropic/claude-3-7-sonnet-20250219-thinking",
        "mistralai/mistral-large-2411-fc",
    ],
)
def test_o_series_pass_ignores_rows_the_family_pass_owns(mod, raw):
    """Rows the family-token pass classifies are invisible to this pass."""
    assert mod.o_series_edge(raw, HF_TO_DEV, _confirm) == (None, None, None)
    name = raw.split("/", 1)[1]
    assert mod.detect_base_token(name) is not None


@pytest.mark.parametrize("name", ["o3-2025-04-16", "o3-2025-04-16-high", "o1-preview"])
def test_family_token_pass_does_not_claim_o_series(mod, name):
    """The o-series tokens stay out of `BASE_FAMILY_TOKENS`, which is what
    keeps that pass's output fixed."""
    assert mod.detect_base_token(name) is None


def test_resolve_mode_edges_uses_same_run_mint(mod):
    """The edge settles against the same-run mint (and its final id), not
    the pre-run registry."""
    dated = {"id": "openai/o3-2025-04-16", "aliases": []}
    child = {"id": "openai/o3-2025-04-16-high", "aliases": []}
    mod.resolve_mode_edges(
        [(child, "openai/o3-2025-04-16")],
        {"openai/o3-2025-04-16": dated},
        lambda _c: None,
    )
    assert child["parents"] == [
        {"id": "openai/o3-2025-04-16", "relationship": "variant", "axis": "mode"}
    ]


def test_resolve_mode_edges_matches_same_run_mint_across_separators(mod):
    """Separator disagreement between child and sibling must not lose the edge."""
    dated = {"id": "openai/gpt-5.4-2026-03-05", "aliases": []}
    child = {"id": "openai/gpt-5-4-2026-03-05-high", "aliases": []}
    mod.resolve_mode_edges(
        [(child, "openai/gpt-5-4-2026-03-05")],
        {"openai/gpt-5.4-2026-03-05": dated},
        lambda _c: None,
    )
    assert child["parents"][0]["id"] == "openai/gpt-5.4-2026-03-05"


def test_resolve_mode_edges_rejects_ambiguous_normalized_sibling(mod):
    """Two same-run mints sharing a normalized form: no edge, or the output
    becomes order-dependent."""
    a = {"id": "acme/gpt-5.4-2026-03-05", "aliases": []}
    b = {"id": "acme/gpt-5-4-2026-03-05", "aliases": []}
    child = {"id": "acme/gpt_5_4_2026_03_05-high", "aliases": []}
    mod.resolve_mode_edges(
        [(child, "acme/gpt_5_4_2026_03_05")],
        {a["id"].lower(): a, b["id"].lower(): b},
        lambda _c: None,
    )
    assert "parents" not in child


def test_resolve_mode_edges_falls_back_to_registry(mod):
    child = {"id": "openai/o3-2025-04-16-high", "aliases": []}
    mod.resolve_mode_edges(
        [(child, "openai/o3-2025-04-16")], {},
        lambda c: "openai/o3-2025-04-16" if c == "openai/o3-2025-04-16" else None,
    )
    assert child["parents"][0]["id"] == "openai/o3-2025-04-16"


def test_resolve_mode_edges_leaves_unmintable_sibling_parentless(mod):
    """No sibling anywhere: parentless, never a dangling edge."""
    child = {"id": "openai/o3-2025-04-16-high", "aliases": []}
    mod.resolve_mode_edges([(child, "openai/o3-2025-04-16")], {}, lambda _c: None)
    assert "parents" not in child
