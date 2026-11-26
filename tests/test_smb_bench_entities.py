"""The SMB Bench raw ids resolve to their seeded canonicals, the models its
datastore records name resolve to the ids those records carry, and a generic
"profit" never resolves to the namespaced metrics. Skips without fixtures.
"""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO_ROOT / "fixtures"

ENTITY_CANONICALS = {
    "smb-bench": [("benchmark", "smb-bench"), ("harness", "smb-bench")],
    "SMB Bench": [("benchmark", "smb-bench")],
    "smb-bench-v1": [("benchmark", "smb-bench")],
    "oro-agents/smb-bench": [("benchmark", "smb-bench")],
    # The per-kind cuts the datastore records name.
    "smb-bench-v1.normal-quarters": [("benchmark", "smb-bench-normal-quarters")],
    "smb-bench-v1.customer-quarters": [("benchmark", "smb-bench-customer-quarters")],
    "smb-bench-v1.supplier-quarters": [("benchmark", "smb-bench-supplier-quarters")],
    "smb-bench.mean-profit-usd": [("metric", "smb-bench.mean-profit-usd")],
    "smb-bench.mean-profit-vs-fixed-rule-usd": [
        ("metric", "smb-bench.mean-profit-vs-fixed-rule-usd")
    ],
    # The cost result uses the existing canonical.
    "cost-per-task": [("metric", "cost-per-task")],
}

# Raw model spelling in the SMB Bench source -> the id its records carry.
MODEL_CANONICALS = {
    "claude-fable-5-1": "anthropic/claude-fable-5.1",
    "claude-fable-5-2": "anthropic/claude-fable-5.2",
    "claude-haiku-5-5": "anthropic/claude-haiku-5.5",
    "claude-opus-5-5": "anthropic/claude-opus-5.5",
    "deepseek-v4.1-flash-1017": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "gemini-4-argon": "google/gemini-4-argon",
    "gemini-4-carbon": "google/gemini-4-carbon",
    "gemini-4-hydrogen": "google/gemini-4-hydrogen",
    "zai-org/GLM-5.3": "zai-org/GLM-5.3",
    "gpt-6-astra": "openai/gpt-6-astra",
    "gpt-6-luna": "openai/gpt-6-luna",
    "gpt-6.1-sol": "openai/gpt-6.1-sol",
    "grok-4.7": "x-ai/grok-4.7",
    "meta/muse-ember-1.0": "meta/muse-ember-1-0",
    "meta/muse-spark-1.3": "meta/muse-spark-1.3",
    "meta/muse-wonder-1.0": "meta/muse-wonder-1-0",
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize(
    "raw,entity_type,expected",
    [(raw, et, cid) for raw, pairs in ENTITY_CANONICALS.items() for et, cid in pairs],
)
def test_smb_bench_entity_resolves(resolver, raw, entity_type, expected):
    res = resolver.resolve(raw, entity_type=entity_type)
    assert res.canonical_id == expected, (
        f"{raw!r} ({entity_type}) -> {res.canonical_id!r} (expected {expected!r})"
    )


@pytest.mark.parametrize("raw,expected", sorted(MODEL_CANONICALS.items()))
def test_smb_bench_model_resolves(resolver, raw, expected):
    res = resolver.resolve(raw, entity_type="model")
    assert res.canonical_id == expected, (
        f"{raw!r} -> {res.canonical_id!r} (expected {expected!r})"
    )


def test_generic_profit_does_not_hijack(resolver):
    # Other sources' raw profit fields are not SMB Bench's dollar metrics.
    for raw in ("profit", "Mean profit", "mean_profit", "Profit"):
        res = resolver.resolve(raw, entity_type="metric")
        assert res.canonical_id not in (
            "smb-bench.mean-profit-usd",
            "smb-bench.mean-profit-vs-fixed-rule-usd",
        ), (
            f"generic form {raw!r} resolved to {res.canonical_id!r}; "
            "display_name must stay benchmark-scoped"
        )
