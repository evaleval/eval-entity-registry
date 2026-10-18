"""AlpacaEval converter slugs resolve to the intended canonicals.

A miss does not error: the converter falls back to a local `alpaca_eval.*`
id, fragmenting AlpacaEval results across two ids. Skips without fixtures.
"""
from pathlib import Path

import pandas as pd
import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

BENCHMARKS = {
    "AlpacaEval 1.0": "alpacaeval-1-0",
    "AlpacaEval 2.0": "alpacaeval-2-0",
    "AlpacaEval v1": "alpacaeval-1-0",
    "AlpacaEval 1": "alpacaeval-1-0",
    "alpaca_eval_v1": "alpacaeval-1-0",
    "alpaca_eval_v2": "alpacaeval-2-0",
}

# The converter looks metrics up by the raw leaderboard-CSV column name,
# not by its own `metric_name`.
METRICS = {
    "win_rate": "win-rate",
    "length_controlled_winrate": "length-controlled-win-rate",
    "discrete_win_rate": "discrete-win-rate",
    "avg_length": "average-response-length",
    # The converter's emitted spelling, and the leaderboard page's label.
    "length_controlled_win_rate": "length-controlled-win-rate",
    "LC Win Rate": "length-controlled-win-rate",
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def metrics_df():
    return pd.read_parquet(_FIXTURES / "canonical_metrics.parquet")


@pytest.mark.parametrize("slug,expected", BENCHMARKS.items())
def test_benchmark_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="benchmark")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


@pytest.mark.parametrize("slug,expected", METRICS.items())
def test_metric_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="metric")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


def test_harness_resolves(resolver):
    """The converter's literal lookup key reaches `alpaca-eval` via normalized match."""
    assert resolver.resolve("alpaca_eval", entity_type="harness").canonical_id == "alpaca-eval"


def test_bare_alpacaeval_is_not_a_benchmark_alias(resolver):
    """Bare `AlpacaEval` names the project; binding it to either leaderboard
    would silently mix two non-comparable score scales."""
    for raw in ("AlpacaEval", "alpaca_eval"):
        got = resolver.resolve(raw, entity_type="benchmark").canonical_id
        assert got is None, f"{raw} -> {got}"


@pytest.mark.parametrize(
    "metric_id,lower_is_better,min_score,max_score",
    [
        # Upstream computes all three win rates as `mean * 100`.
        ("win-rate", False, 0.0, 1.0),
        ("length-controlled-win-rate", False, 0.0, 1.0),
        ("discrete-win-rate", False, 0.0, 100.0),
        # Character count: unbounded by definition (`.inf`, not "not stated")
        # and not a quality score in either direction.
        ("average-response-length", None, 0.0, float("inf")),
    ],
)
def test_metric_bounds(metrics_df, metric_id, lower_is_better, min_score, max_score):
    row = metrics_df[metrics_df["id"] == metric_id]
    assert len(row) == 1, f"{metric_id}: expected exactly one row, got {len(row)}"
    row = row.iloc[0]
    assert row["lower_is_better"] is lower_is_better or (
        lower_is_better is None and pd.isna(row["lower_is_better"])
    ), f"{metric_id}: lower_is_better={row['lower_is_better']!r}"
    assert row["min_score"] == min_score
    if max_score is None:
        assert pd.isna(row["max_score"]), f"{metric_id}: max_score={row['max_score']!r}"
    else:
        assert row["max_score"] == max_score, f"{metric_id}: max_score={row['max_score']!r}"
