"""Papers with Code stated metric names that used to land on the catch-all
`score` resolve to their own metric. One-word names (`Tables`, `Binary`,
`Resolved`) are scoped to `paperswithcode`; `Pass^3` is scoped because it
normalises like `Pass@3`. Skips without fixtures."""
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


_SCOPED = {
    "Rerank": "mmteb-rerank-score",
    "Retrieval": "mmteb-retrieval-score",
    "Tables": "parsebench-tables",
    "Charts": "parsebench-charts",
    "Content Faithfulness": "parsebench-content-faithfulness",
    "Semantic Formatting": "parsebench-semantic-formatting",
    "Visual Grounding": "parsebench-visual-grounding",
    "Resolved": "resolution-rate",
    "Almost Resolved": "almost-resolved-rate",
    "Binary": "task-success-rate",
    "Partial": "partial-credit-score",
    "AJ": "average-jaccard",
    "Similarity Score": "blueprint-bench-similarity-score",
    "Pass^3": "pass-hat-3",
    "MTE": "median-trajectory-error",
    "Survival": "survival-rate",
    "APD": "average-pairwise-distance",
    "NE": "navigation-error",
    "Contact": "contact-rate",
    "Goal Dist": "goal-distance",
    "PC": "videophy-physical-commonsense",
    "SA": "videophy-semantic-adherence",
    "Quality": "qwen-image-bench-quality",
    "Aesthetics": "qwen-image-bench-aesthetics",
}


@pytest.mark.parametrize("name,expected", _SCOPED.items())
def test_scoped_name_resolves_under_paperswithcode_only(resolver, name, expected):
    assert resolver.resolve(name, "metric", "paperswithcode").canonical_id == expected
    assert resolver.resolve(name, "metric", None).canonical_id != expected
    assert resolver.resolve(name, "metric", "benchpress").canonical_id != expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Overall (ParseBench)", "parsebench-overall"),
        ("Overall (OmniParseBench)", "omniparsebench-overall"),
        ("Tables (OmniParseBench)", "omniparsebench-tables"),
        ("Average Test Pass Rate", "average-test-pass-rate"),
        ("Bank Balance ($)", "bank-balance"),
        ("delta_avg", "delta-avg"),
        ("Recall@1", "recall-at-1"),
        ("ATE RMSE (m)", "ate"),
        ("DER", "der"),
        ("EER", "eer"),
        ("Balanced Accuracy", "balanced-accuracy"),
        ("Unweighted Average Recall (UAR)", "balanced-accuracy"),
        ("Skill Score (%)", "skill-score"),
        ("Word Grounding F1", "word-grounding-f1"),
        ("Page Grounding F1", "page-grounding-f1"),
        ("PLCC", "correlation"),
        ("BD-PSNR", "bd-psnr"),
        ("DNSMOS-OVRL", "dnsmos-ovrl"),
        ("VBench-2.0 Total Score", "vbench-2-0-total-score"),
        ("Task Success Rate (TSR)", "task-success-rate"),
        ("FD (Inception-v3)", "fid"),
        ("AP50^n", "ap50-novel"),
        ("Rubric Score (%)", "judge-rubric-score"),
    ],
)
def test_stated_name_resolves(resolver, name, expected):
    assert resolver.resolve(name, "metric", "paperswithcode").canonical_id == expected


@pytest.mark.parametrize(
    "raw_id,expected",
    [
        ("paperswithcode.tables", "parsebench-tables"),
        ("paperswithcode.tables_omniparsebench", "omniparsebench-tables"),
        ("paperswithcode.overall_parsebench", "parsebench-overall"),
        ("paperswithcode.resolved", "resolution-rate"),
        ("paperswithcode.almost_resolved", "almost-resolved-rate"),
        ("paperswithcode.binary", "task-success-rate"),
        ("paperswithcode.partial", "partial-credit-score"),
        ("paperswithcode.pass_3", "pass-hat-3"),
        ("paperswithcode.bank_balance", "bank-balance"),
        ("paperswithcode.rubric_score", "judge-rubric-score"),
        ("paperswithcode.recall_1", "recall-at-1"),
    ],
)
def test_metric_id_resolves(resolver, raw_id, expected):
    match = resolver.resolve_structured_metric(raw_id, "paperswithcode", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == expected


def test_pass_hat_three_does_not_claim_pass_at_three(resolver):
    assert resolver.resolve("Pass@3", "metric", None).canonical_id != "pass-hat-3"
    assert resolver.resolve("Pass^3", "metric", None).canonical_id is None
    assert resolver.resolve("Pass^3", "metric", "tau-bench").canonical_id == "pass-hat-3"


def test_the_two_mmteb_categories_stay_apart(resolver):
    a = resolver.resolve("Rerank", "metric", "paperswithcode").canonical_id
    b = resolver.resolve("Retrieval", "metric", "paperswithcode").canonical_id
    assert a != b and "mmteb-score" not in (a, b)


@pytest.mark.parametrize("name", ["Score", "Overall Score", "MMEB-v2 Score"])
def test_benchmark_own_scores_are_not_claimed(resolver, name):
    assert resolver.resolve(name, "metric", "paperswithcode").canonical_id in (None, *_CATCH_ALL)
