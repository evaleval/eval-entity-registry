"""BenchPress free-text metric labels resolve to the metric they name, only
under the `benchpress` source: after normalisation most are bare words
(`% correct` is `correct`). Bare units and the generic score forms are never
claimed, and the "resolved" family stays on the catch-all so its per-benchmark
folds to `accuracy` keep matching. Skips without fixtures."""
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


_BENCHPRESS = {
    "% correct": "accuracy",
    "% correct (pass@1)": "accuracy",
    "five-choice accuracy (%)": "accuracy",
    "% success": "task-success-rate",
    "% tasks passed": "task-success-rate",
    "% passed (8-trial average)": "task-success-rate",
    "% exact-match accuracy": "exact-match",
    "% normalized multiple-choice accuracy": "normalized-accuracy",
    "mean pass@1 (%)": "pass-at-1",
    "CorpusBLEU": "bleu",
    "normalized edit distance (lower=better)": "edit-distance",
    "harmful/dual-use compliance rate (%)": "fulfillment-rate",
    "severity-5 refusal rate (%)": "refusal-rate",
    "strict task pass rate (%)": "all-rules-pass-rate",
    "all-pass rate (%)": "all-rules-pass-rate",
    "compliance rate (%)": "fulfillment-rate",
    "benign refusal rate (%)": "over-refusal-rate",
    "fully resolved tasks (%)": "resolution-rate",
    "Spearman correlation": "correlation",
    "claim-level precision (%)": "precision",
    "validation recall (%)": "recall",
    "all-three-pass rate (%)": "pass-hat-3",
    "Pass^3 average (%)": "pass-hat-3",
    "rubric score (%)": "judge-rubric-score",
    "% rubric criteria passed": "rubric-criteria-pass-rate",
    "length-adjusted score": "healthbench-length-adjusted-score",
    "% non-hallucination": "non-hallucination-rate",
    "% clear pushback": "clear-pushback-rate",
    "reward (%)": "mean-reward",
    "average speedup ratio": "speedup",
    "macro-average behavioral tests passed (%)": "average-test-pass-rate",
    "tasks with at least 95% pass rate (%)": "almost-resolved-rate",
}


@pytest.mark.parametrize("label,expected", _BENCHPRESS.items())
def test_label_resolves_under_benchpress_only(resolver, label, expected):
    assert resolver.resolve(label, "metric", "benchpress").canonical_id == expected
    assert resolver.resolve(label, "metric", None).canonical_id != expected
    assert resolver.resolve(label, "metric", "llm-stats").canonical_id != expected


@pytest.mark.parametrize("label", ["%", "pct", "dollars", "score (0-1)", "reported score (%)"])
def test_bare_units_and_generic_scores_are_not_claimed(resolver, label):
    assert resolver.resolve(label, "metric", "benchpress").canonical_id in (None, *_CATCH_ALL)


@pytest.mark.parametrize(
    "label",
    ["% resolved (pass@1)", "% solved", "task resolve rate (%)", "resolution rate (pass@1, %)",
     "% resolved", "% tasks solved"],
)
def test_resolved_family_keeps_its_folds(resolver, label):
    # These publish as `accuracy` through `from_metric: score` folds; an
    # alias would move them off `score` before the fold runs.
    assert resolver.resolve(label, "metric", "benchpress").canonical_id in (None, *_CATCH_ALL)


def test_connectivity_score_is_not_claimed(resolver):
    # BenchPress publishes it x100 beside fractions from other sources, and
    # its target has no lower bound, so no fold row can state the conversion.
    assert resolver.resolve(
        "normalized connectivity score (%)", "metric", "benchpress"
    ).canonical_id in (None, *_CATCH_ALL)


@pytest.mark.parametrize(
    "slug,expected",
    [
        ("arc-agi-1", "arc-agi"),
        ("critpt", "critpt"),
        ("program-bench-fully-resolved", "programbench"),
        ("strongreject-compliance", "strongreject"),
        ("xai-self-harm-compliance", "xai-self-harm-compliance"),
        ("hackerbench-v0-2-benign-refusal", "hackerbench-v0-2-benign-refusal"),
        ("legal-agent-benchmark-harvey-held-out", "legal-agent-benchmark-harvey-held-out"),
    ],
)
def test_benchmark_slug_resolves(resolver, slug, expected):
    assert resolver.resolve(slug, "benchmark", "benchpress").canonical_id == expected


def test_benchpress_only_slugs_stay_in_benchpress(resolver):
    assert resolver.resolve("program-bench-fully-resolved", "benchmark", None).canonical_id != "programbench"
    assert resolver.resolve("strongreject-compliance", "benchmark", None).canonical_id != "strongreject"
