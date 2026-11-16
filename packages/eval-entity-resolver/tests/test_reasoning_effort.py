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
    # A source's effort tier the registry holds as its own model.
    {"id": "openai/o3-mini-high", "org_id": "openai"},
    {"id": "openai/o3-mini", "org_id": "openai"},
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
        # The registry holds the tier as its own model, apart from the base.
        ("openai/o3-mini-high", "openai/o3-mini-high"),
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
