"""A reasoning-effort tier welded onto a raw model id is reported whichever
strategy matched the id, and only when the id without it names the same
model."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eval_entity_resolver import Resolver

_ALIAS_COLUMNS = (
    "id", "raw_value", "entity_type", "canonical_id", "source_config",
    "source_field", "status", "strategy", "confidence", "notes",
    "created_at", "updated_at",
)

_MODELS = [
    {"id": "openai/o1-2024-12-17", "org_id": "openai"},
    {"id": "anthropic/claude-sonnet-5", "org_id": "anthropic"},
    {"id": "mistralai/mistral-medium", "org_id": "mistralai"},
    {"id": "mistralai/mistral", "org_id": "mistralai"},
    {"id": "qwen/qwen3-max", "org_id": "qwen"},
    # An effort tier the registry holds as a mode leaf of its base.
    {"id": "openai/o3-mini-high", "org_id": "openai", "parents": json.dumps(
        [{"id": "openai/o3-mini", "relationship": "variant", "axis": "mode"}])},
    {"id": "openai/o3-mini", "org_id": "openai"},
    # A tier-named model with no link to the model its stem names.
    {"id": "openai/o4-mini-high", "org_id": "openai"},
    {"id": "openai/o4-mini", "org_id": "openai"},
    {"id": "anthropic/claude-opus-4", "org_id": "anthropic"},
    {"id": "anthropic/claude-opus-4-thinking", "org_id": "anthropic"},
]

_ALIASES = [
    ("openai/o1-2024-12-17", "openai/o1-2024-12-17"),
    # The tier spelling baked as an alias, so it matches before fuzzy.
    ("openai/o1-2024-12-17-low", "openai/o1-2024-12-17"),
    ("anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5"),
    ("mistralai/mistral-medium", "mistralai/mistral-medium"),
    ("mistralai/mistral", "mistralai/mistral"),
    ("qwen/qwen3-max", "qwen/qwen3-max"),
    ("openai/o3-mini-high", "openai/o3-mini-high"),
    ("openai/o3-mini", "openai/o3-mini"),
    ("openai/o4-mini-high", "openai/o4-mini-high"),
    ("openai/o4-mini", "openai/o4-mini"),
    ("anthropic/claude-opus-4", "anthropic/claude-opus-4"),
    ("anthropic/claude-opus-4-thinking", "anthropic/claude-opus-4-thinking"),
]


@pytest.fixture(scope="module")
def resolver(tmp_path_factory) -> Resolver:
    root: Path = tmp_path_factory.mktemp("snapshot")
    rows = []
    for i, (raw, canonical_id) in enumerate(_ALIASES):
        row = dict.fromkeys(_ALIAS_COLUMNS)
        row.update(
            id=f"a{i}", raw_value=raw, entity_type="model",
            canonical_id=canonical_id, status="confirmed",
            strategy="seed", confidence=1.0,
        )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "aliases.parquet")
    pd.DataFrame(_MODELS).to_parquet(root / "canonical_models.parquet")
    pd.DataFrame([{
        "id": "openrouter", "aliases": json.dumps(["OpenRouter", "-openrouter"]),
        "canonical_org": None,
    }]).to_parquet(root / "canonical_inference_platforms.parquet")
    return Resolver.from_parquet(root)


@pytest.mark.parametrize(
    "raw, strategy, effort",
    [
        ("openai/o1-2024-12-17-low", "exact", "low"),
        ("openai/O1_2024_12_17-low", "normalized", "low"),
        ("openai/o1-2024-12-17-medium", "fuzzy", "medium"),
        ("anthropic/claude-sonnet-5-xhigh", "fuzzy", "xhigh"),
        ("anthropic/claude-sonnet-5-high-effort", "fuzzy", "high"),
        ("anthropic/claude-sonnet-5-max", "fuzzy", "max"),
        # A host suffix after the tier does not hide it.
        ("anthropic/claude-sonnet-5-low-openrouter", "fuzzy", "low"),
    ],
)
def test_effort_tier_is_reported_at_every_strategy(resolver, raw, strategy, effort):
    result = resolver.resolve(raw, "model")
    assert result.strategy == strategy
    assert result.canonical_id in ("openai/o1-2024-12-17", "anthropic/claude-sonnet-5")
    assert result.reasoning_effort == effort


@pytest.mark.parametrize(
    "raw, canonical_id",
    [
        # The suffix is part of the product name: the stem is another model.
        ("mistralai/mistral-medium", "mistralai/mistral-medium"),
        # The stem names no model at all.
        ("qwen/qwen3-max", "qwen/qwen3-max"),
        # The registry holds a tier-named model unlinked to the stem's model.
        ("openai/o4-mini-high", "openai/o4-mini-high"),
        # No tier in the name.
        ("openai/o1-2024-12-17", "openai/o1-2024-12-17"),
    ],
)
def test_a_suffix_that_is_part_of_the_model_is_not_an_effort(resolver, raw, canonical_id):
    result = resolver.resolve(raw, "model")
    assert result.canonical_id == canonical_id
    assert result.reasoning_effort is None


def test_an_unresolved_name_reports_no_effort(resolver):
    result = resolver.resolve("nobody/unknown-model-high", "model")
    assert result.canonical_id is None
    assert result.reasoning_effort is None


@pytest.mark.parametrize(
    "raw, canonical_id, effort, tokens",
    [
        # The registry holds a thinking-mode model: the budget is run on it.
        ("anthropic/claude-opus-4-thinking-16k", "anthropic/claude-opus-4-thinking", None, 16000),
        # A tier after the budget: the fuzzy step lands on the plain model,
        # and both conditions are still reported.
        ("anthropic/claude-opus-4-thinking-120k-high", "anthropic/claude-opus-4", "high", 120000),
        # No thinking-mode model: the budget is run on the plain model.
        ("anthropic/claude-sonnet-5-thinking-8k", "anthropic/claude-sonnet-5", None, 8000),
    ],
)
def test_thinking_budget_is_reported_with_the_model_it_ran_on(
    resolver, raw, canonical_id, effort, tokens
):
    result = resolver.resolve(raw, "model")
    assert result.canonical_id == canonical_id
    assert (result.reasoning_effort, result.reasoning_tokens) == (effort, tokens)


def test_effort_tier_held_as_a_mode_leaf_keeps_the_leaf_and_reports_the_tier(resolver):
    result = resolver.resolve("openai/o3-mini-high", "model")
    assert result.canonical_id == "openai/o3-mini-high"
    assert result.parent_canonical_id == "openai/o3-mini"
    assert result.reasoning_effort == "high"
