"""A host token in a raw model id names the serving platform whichever
strategy matched the id, unless the token is the matched model's developer."""
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
    {"id": "deepseek-ai/DeepSeek-R1", "org_id": "deepseek"},
    {"id": "microsoft/phi-4", "org_id": "microsoft"},
    {"id": "openai/gpt-4o", "org_id": "openai"},
    {"id": "Groq/Llama-3-Groq-8B-Tool-Use", "org_id": "Groq"},
    {"id": "fireworks/ember-1", "org_id": "fireworks"},
]

# (raw_value, canonical_id). The hosted spellings are aliases the way the
# seed's generated layer bakes them, so they match before the fuzzy step.
_ALIASES = [
    ("deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1"),
    ("fireworks/deepseek-r1", "deepseek-ai/DeepSeek-R1"),
    ("microsoft/phi-4", "microsoft/phi-4"),
    ("azure/phi-4", "microsoft/phi-4"),
    ("openai/gpt-4o", "openai/gpt-4o"),
    ("openai/gpt-4o-openrouter", "openai/gpt-4o"),
    ("Groq/Llama-3-Groq-8B-Tool-Use", "Groq/Llama-3-Groq-8B-Tool-Use"),
    ("fireworks/ember-1", "fireworks/ember-1"),
    # No canonical_models row: the id's namespace is the developer on record.
    ("groq/unlisted", "Groq/Unlisted"),
]

_PLATFORMS = [
    {"id": "fireworks-ai", "aliases": ["Fireworks AI", "fireworks/"]},
    {"id": "azure", "aliases": ["Azure", "azure/"]},
    {"id": "groq", "aliases": ["Groq", "groq/"]},
    {"id": "openrouter", "aliases": ["OpenRouter", "-openrouter"]},
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
    pd.DataFrame(
        [{"id": p["id"], "aliases": json.dumps(p["aliases"])} for p in _PLATFORMS]
    ).to_parquet(root / "canonical_inference_platforms.parquet")
    return Resolver.from_parquet(root)


@pytest.mark.parametrize(
    "raw, strategy",
    [
        ("fireworks/deepseek-r1", "exact"),
        ("fireworks/DeepSeek_R1", "normalized"),
        ("fireworks/deepseek-ai/DeepSeek-R1", "fuzzy"),
    ],
)
def test_host_prefix_names_the_platform_at_every_strategy(resolver, raw, strategy):
    result = resolver.resolve(raw, "model")
    assert (result.canonical_id, result.strategy) == ("deepseek-ai/DeepSeek-R1", strategy)
    assert result.inference_platform == "fireworks-ai"


def test_host_prefix_on_another_developers_model_is_a_host(resolver):
    result = resolver.resolve("azure/phi-4", "model")
    assert (result.canonical_id, result.strategy) == ("microsoft/phi-4", "exact")
    assert result.inference_platform == "azure"


def test_host_suffix_names_the_platform_on_an_exact_alias(resolver):
    result = resolver.resolve("openai/gpt-4o-openrouter", "model")
    assert (result.canonical_id, result.strategy) == ("openai/gpt-4o", "exact")
    assert result.inference_platform == "openrouter"


@pytest.mark.parametrize(
    "raw, canonical_id, strategy",
    [
        ("Groq/Llama-3-Groq-8B-Tool-Use", "Groq/Llama-3-Groq-8B-Tool-Use", "exact"),
        ("groq/llama_3_groq_8b_tool_use", "Groq/Llama-3-Groq-8B-Tool-Use", "normalized"),
        ("fireworks/ember-1", "fireworks/ember-1", "exact"),
        ("fireworks/ember-1-fc", "fireworks/ember-1", "fuzzy"),
        ("groq/unlisted", "Groq/Unlisted", "exact"),
    ],
)
def test_prefix_that_is_the_models_developer_is_not_a_host(
    resolver, raw, canonical_id, strategy
):
    result = resolver.resolve(raw, "model")
    assert (result.canonical_id, result.strategy) == (canonical_id, strategy)
    assert result.inference_platform is None


def test_plain_id_has_no_platform(resolver):
    assert resolver.resolve("openai/gpt-4o", "model").inference_platform is None


def test_unresolved_id_has_no_platform(resolver):
    result = resolver.resolve("fireworks/no-such-model", "model")
    assert result.canonical_id is None
    assert result.inference_platform is None
