"""Artificial Analysis price, speed and latency fields and the four Vectara
leaderboard measurements each resolve to their own metric instead of sharing
the catch-all `score`. Skips without fixtures."""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)

_CATCH_ALL = frozenset({"score", "mean-score", "overall"})


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


_AA = {
    "price_1m_input_tokens": "price-per-1m-input-tokens",
    "price_1m_output_tokens": "price-per-1m-output-tokens",
    "price_1m_blended_3_to_1": "blended-price-per-1m-tokens",
    "median_output_tokens_per_second": "median-output-tokens-per-second",
    "median_time_to_first_token_seconds": "median-time-to-first-token",
    "median_time_to_first_answer_token": "median-time-to-first-answer-token",
}


@pytest.mark.parametrize("field,expected", _AA.items())
def test_artificial_analysis_field_resolves(resolver, field, expected):
    match = resolver.resolve_structured_metric(
        "artificial_analysis." + field, "artificial-analysis-llms", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == expected


def test_the_two_first_token_latencies_stay_apart(resolver):
    assert resolver.resolve("Median time to first token", "metric", None).canonical_id \
        == "median-time-to-first-token"
    assert resolver.resolve("Median time to first answer token", "metric", None).canonical_id \
        == "median-time-to-first-answer-token"


@pytest.mark.parametrize("field", ["artificial_analysis_intelligence_index", "scicode", "tau2"])
def test_artificial_analysis_benchmark_scores_are_not_claimed(resolver, field):
    assert resolver.resolve_structured_metric(
        "artificial_analysis." + field, "artificial-analysis-llms", catch_all_ids=_CATCH_ALL) is None


_VECTARA = {
    "Answer Rate": "answer-rate",
    "Hallucination Rate": "hallucination-rate",
    "Factual Consistency Rate": "factual-consistency-rate",
    "Average Summary Length": "average-summary-length",
}


@pytest.mark.parametrize("name,expected", _VECTARA.items())
def test_vectara_name_resolves(resolver, name, expected):
    assert resolver.resolve(name, "metric", "vectara-hallucination-leaderboard").canonical_id == expected


@pytest.mark.parametrize(
    "raw_id,expected",
    [
        ("vectara-hallucination-leaderboard.answer-rate", "answer-rate"),
        ("vectara-hallucination-leaderboard.answer_rate", "answer-rate"),
        ("vectara-hallucination-leaderboard.hallucination_rate", "hallucination-rate"),
        ("vectara-hallucination-leaderboard.factual-consistency-rate", "factual-consistency-rate"),
        ("vectara-hallucination-leaderboard.average-summary-length", "average-summary-length"),
    ],
)
def test_vectara_metric_id_resolves(resolver, raw_id, expected):
    match = resolver.resolve_structured_metric(
        raw_id, "vectara-hallucination-leaderboard", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == expected


def test_vectara_measurements_are_four_metrics(resolver):
    got = {resolver.resolve(n, "metric", "vectara-hallucination-leaderboard").canonical_id for n in _VECTARA}
    assert len(got) == 4
