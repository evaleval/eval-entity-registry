"""WILD / openeval benchmark and metric slugs resolve to the right canonical,
including the ARC disambiguation (AI2 Reasoning Challenge, not ARC-AGI) and
the scoped openeval metric channels. Skips without fixtures."""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver
from eval_entity_resolver.eee import extract_metric

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "fixtures"

WILD_BENCHMARKS = {
    "arc_easy": "ai2-reasoning-challenge-arc-easy",
    "arc_challenge": "ai2-reasoning-challenge-arc-challenge",
    "race_h": "race",
    "squad": "squad",
    "paws": "paws",
    "chembench": "chembench",
    "finance_fundamentals": "finance-fundamentals",
    "pre_flight": "pre-flight",
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize("slug,expected", WILD_BENCHMARKS.items())
def test_wild_benchmark_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="benchmark")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


def test_arc_not_arc_agi(resolver):
    # WILD's arc_* is the AI2 Reasoning Challenge, never ARC-AGI or bare `arc`.
    for slug, expected in (
        ("arc_easy", "ai2-reasoning-challenge-arc-easy"),
        ("arc_challenge", "ai2-reasoning-challenge-arc-challenge"),
    ):
        cid = resolver.resolve(slug, entity_type="benchmark").canonical_id
        assert cid == expected
        assert cid not in ("arc-agi", "arc")


def test_openeval_wildbench_rescaled_metric_is_scoped(resolver):
    raw = "openeval.wildbench.wildbench-score-rescaled"
    assert resolver.resolve_structured_metric_id(
        raw, source_config="openeval", catch_all_ids=frozenset({"score"})
    ) == "wb-score"
    assert resolver.resolve_structured_metric_id(
        raw, catch_all_ids=frozenset({"score"})
    ) is None
    assert resolver.resolve_structured_metric_id(
        raw, source_config="helm_capabilities", catch_all_ids=frozenset({"score"})
    ) is None
    assert resolver.resolve("WB Score", entity_type="metric").canonical_id == "wb-score"


def _resolve_openeval_metric(resolver, raw):
    resolved = resolver.resolve_structured_metric_id(
        raw, source_config="openeval", catch_all_ids=frozenset({"score"})
    )
    if resolved is not None:
        return resolved
    return resolver.resolve(extract_metric(raw), entity_type="metric").canonical_id


@pytest.mark.parametrize(
    "raw", [
        "openeval.wildbench.gpt-score",
        "openeval.wildbench.claude-score",
        "openeval.wildbench.llama-score",
    ],
)
def test_openeval_wildbench_judge_metrics_stay_score(resolver, raw):
    assert _resolve_openeval_metric(resolver, raw) == "score"


def test_lm_evaluated_safety_score_extracts_to_catch_all(resolver):
    extracted = extract_metric("LM Evaluated Safety score")
    assert extracted == "score"
    assert resolver.resolve(extracted, entity_type="metric").canonical_id == "score"


def test_openeval_anthropic_red_teaming_resolves_structurally(resolver):
    result = resolver.resolve_structured_benchmark(
        "openeval.anthropic-red-teaming.safety-gpt-score", source_config="openeval"
    )
    assert result is not None
    assert result.canonical_id == "anthropic-red-team"


@pytest.mark.parametrize(
    "raw,expected", [
        ("openeval.harmbench.haiku-llm-judge", "attack-success-rate"),
        ("openeval.harmbench.refusal-strings", "refusal-rate"),
    ],
)
def test_openeval_harmbench_channels_have_own_metrics(resolver, raw, expected):
    # Each channel has its own meaning; falling through to the catch-all would
    # land haiku-llm-judge on harmbench-refusal-score with inverted direction.
    assert _resolve_openeval_metric(resolver, raw) == expected


@pytest.mark.parametrize(
    "raw", [
        "openeval.harmbench.haiku-llm-judge",
        "openeval.harmbench.refusal-strings",
    ],
)
def test_openeval_harmbench_channels_are_scoped(resolver, raw):
    assert resolver.resolve_structured_metric_id(
        raw, catch_all_ids=frozenset({"score"})
    ) is None


def test_openeval_harmbench_judge_metric_stays_score(resolver):
    assert _resolve_openeval_metric(resolver, "openeval.harmbench.safety-gpt-score") == "score"


# BERTScore / SummaC / BLEURT name the metric itself, not a benchmark channel,
# so their aliases are global and the structured id resolves off the metric
# segment alone (unlike the scoped WildBench / HarmBench channels above).
@pytest.mark.parametrize(
    "raw,expected", [
        ("openeval.cnndm.bertscore-f", "bertscore-f1"),
        ("openeval.xsum.bertscore-f", "bertscore-f1"),
        ("openeval.cnndm.bertscore-p", "bertscore-precision"),
        ("openeval.xsum.bertscore-r", "bertscore-recall"),
        ("openeval.cnndm.summac", "summac"),
        ("openeval.xsum.summac", "summac"),
        ("openeval.truthfulqa.bleurt-acc", "bleurt-accuracy"),
        ("openeval.truthfulqa.bleurt-diff", "bleurt-diff"),
        ("openeval.truthfulqa.bleurt-max", "bleurt-max"),
    ],
)
def test_openeval_model_scored_generation_metrics(resolver, raw, expected):
    assert _resolve_openeval_metric(resolver, raw) == expected
    assert resolver.resolve_structured_metric_id(
        raw, source_config="openeval", catch_all_ids=frozenset({"score"})
    ) == expected


@pytest.mark.parametrize(
    "raw,expected", [
        ("BERTScore-F", "bertscore-f1"),
        ("BERTScore-P", "bertscore-precision"),
        ("BERTScore-R", "bertscore-recall"),
        ("summac", "summac"),
        ("BLEURT_acc", "bleurt-accuracy"),
        ("BLEURT_diff", "bleurt-diff"),
        ("BLEURT_max", "bleurt-max"),
    ],
)
def test_generation_metric_names_resolve(resolver, raw, expected):
    # The metric_name path: the published label, not a structured id.
    assert resolver.resolve(raw, entity_type="metric").canonical_id == expected


def test_bleurt_20_is_the_bleurt_canonical(resolver):
    # BLEURT-20 is a checkpoint of BLEURT, not a separate metric, and is
    # distinct from the unbounded bleurt-max regression output.
    for raw in ("bleurt-20", "BLEURT-20", "BLEURT"):
        assert resolver.resolve(raw, entity_type="metric").canonical_id == "bleurt"
    assert resolver.resolve("bleurt-max", entity_type="metric").canonical_id == "bleurt-max"


def test_rouge_1_still_resolves(resolver):
    # The BLEURT/BERTScore aliases must not disturb the n-gram metrics on the
    # same summarization benchmarks.
    assert resolver.resolve("rouge_1", entity_type="metric").canonical_id == "rouge-1"


def test_generation_metric_bounds(resolver):
    # Rescaled BERTScore has a definitional max of 1.0 but no principled floor
    # (null); BLEURT diff/max are unbounded regression outputs.
    import math

    bounds = {
        row["id"]: (row["min_score"], row["max_score"])
        for row in _metric_rows()
    }
    assert bounds["bertscore-f1"] == (None, 1.0)
    assert bounds["summac"] == (-1.0, 1.0)
    assert bounds["bleurt-accuracy"] == (0.0, 1.0)
    assert bounds["bleurt-max"] == (-math.inf, math.inf)


def _metric_rows():
    import yaml

    return yaml.safe_load((_ROOT / "seed" / "metrics.yaml").read_text())


@pytest.mark.parametrize(
    "raw,subset",
    [
        # Two-letter language tails are slices, not case-folded metric aliases
        # (`mr` is Marathi here, not mean-recall `mR`).
        ("arc.arc_multilingual.mr", "mr"),
        ("arc.arc_multilingual.de", "de"),
    ],
)
def test_two_letter_language_tail_stays_a_slice(resolver, raw, subset):
    result = resolver.resolve_structured_benchmark(raw, "swissai_apertus_evals")
    assert result is not None
    assert result.canonical_id == "arc-multilingual"
    assert result.subset == subset


def test_byte_exact_short_metric_tail_is_still_a_metric(resolver):
    # Exact-alias tier still claims a byte-exact short tail (`F1`); only the
    # case-insensitive tiers need >= 3 chars.
    assert resolver._segment_is_metric("F1", None)
    assert not resolver._segment_is_metric("mr", None)
