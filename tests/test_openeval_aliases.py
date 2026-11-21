"""OpenEval channels that used to fall through to the catch-all `score`
resolve to the metric they report: the TruthfulQA BLEU / ROUGE family, the
compliance classifiers, HELM's disinformation and BOLD diagnostics. The
plain `bleu` / `rouge-*` metrics keep their own forms. Skips without
fixtures."""
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


_TRUTHFULQA = {
    "bleu_acc": "bleu-accuracy", "bleu_diff": "bleu-diff", "bleu_max": "bleu-max",
    "rouge1_acc": "rouge-1-accuracy", "rouge1_diff": "rouge-1-diff", "rouge1_max": "rouge-1-max",
    "rouge2_acc": "rouge-2-accuracy", "rouge2_diff": "rouge-2-diff", "rouge2_max": "rouge-2-max",
    "rougeL_acc": "rouge-l-accuracy", "rougeL_diff": "rouge-l-diff", "rougeL_max": "rouge-l-max",
}


@pytest.mark.parametrize("name,expected", _TRUTHFULQA.items())
def test_truthfulqa_metric_name_resolves(resolver, name, expected):
    assert resolver.resolve(name, "metric", "openeval").canonical_id == expected


@pytest.mark.parametrize("name,expected", _TRUTHFULQA.items())
def test_truthfulqa_metric_id_resolves(resolver, name, expected):
    raw_id = "openeval.truthfulqa." + name.lower().replace("_", "-")
    match = resolver.resolve_structured_metric(raw_id, "openeval", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == expected


@pytest.mark.parametrize(
    "raw_id,expected",
    [
        ("openeval.sorry-bench.ft-mistral-7b-sorrybench", "fulfillment-rate"),
        ("openeval.do-not-answer.longformer-action-ro", "fulfillment-rate"),
        ("openeval.disinformation.self-bleu", "self-bleu"),
        ("openeval.disinformation.monte-carlo-entropy", "monte-carlo-entropy"),
        ("openeval.bold.logprob", "logprob"),
        ("openeval.moralbench.human-weighted-score", "moralbench-human-weighted-score"),
        ("openeval.salad-bench.attack-success", "attack-success-rate"),
        ("openeval.harmbench.haiku-llm-judge", "attack-success-rate"),
        ("openeval.visfactor.correct", "accuracy"),
    ],
)
def test_scoped_channel_resolves(resolver, raw_id, expected):
    match = resolver.resolve_structured_metric(raw_id, "openeval", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == expected


def test_scoped_channels_stay_in_openeval(resolver):
    assert resolver.resolve("openeval.bold.logprob", "metric", "helm_lite").canonical_id is None
    assert resolver.resolve("human_weighted_score", "metric", None).canonical_id is None
    assert resolver.resolve("correct", "metric", "openeval").canonical_id is None


def test_quasi_postfix_exact_match(resolver):
    assert resolver.resolve(
        "quasi_postfix_exact_match", "metric", "openeval"
    ).canonical_id == "quasi-postfix-exact-match"


@pytest.mark.parametrize(
    "raw,expected",
    [("bleu", "bleu"), ("BLEU", "bleu"), ("rouge-1", "rouge-1"), ("ROUGE-1", "rouge-1"),
     ("rouge-2", "rouge-2"), ("rouge-l", "rouge-l")],
)
def test_plain_bleu_rouge_forms_unchanged(resolver, raw, expected):
    assert resolver.resolve(raw, "metric", None).canonical_id == expected


@pytest.mark.parametrize(
    "raw_id,expected",
    [
        ("openeval.emobench.emotion-quasi-exact-match", "emobench-emotion-quasi-exact-match"),
        ("openeval.emobench.cause-quasi-exact-match", "emobench-cause-quasi-exact-match"),
        ("openeval.opentom.fine-postfix-exact-match", "opentom-fine-postfix-exact-match"),
        ("openeval.opentom.coarse-postfix-exact-match", "opentom-coarse-postfix-exact-match"),
        ("openeval.do-not-answer.do-not-answer-annotation", "do-not-answer-action-class"),
        ("openeval.last-translation-benchmark.passes-all-rules", "all-rules-pass-rate"),
        ("openeval.dochop.heuristic-accuracy", "heuristic-accuracy"),
        ("openeval.or-bench.orbench-refusal", "over-refusal-rate"),
        ("openeval.health-orsc-bench.refusal-strings", "over-refusal-rate"),
    ],
)
def test_part_and_judge_channels_resolve(resolver, raw_id, expected):
    match = resolver.resolve_structured_metric(raw_id, "openeval", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == expected


def test_emobench_parts_stay_apart_from_plain_quasi_exact_match(resolver):
    assert resolver.resolve("quasi_exact_match", "metric", "openeval").canonical_id == "quasi-exact-match"
    assert resolver.resolve("emotion_quasi_exact_match", "metric", None).canonical_id is None
    assert resolver.resolve("cause_quasi_exact_match", "metric", None).canonical_id is None


def test_the_two_do_not_answer_channels_stay_apart(resolver):
    a = resolver.resolve_structured_metric(
        "openeval.do-not-answer.do-not-answer-annotation", "openeval", catch_all_ids=_CATCH_ALL)
    b = resolver.resolve_structured_metric(
        "openeval.do-not-answer.longformer-action-ro", "openeval", catch_all_ids=_CATCH_ALL)
    assert a.canonical_id != b.canonical_id


def test_dochop_benchmark_resolves(resolver):
    assert resolver.resolve("dochop", "benchmark", "openeval").canonical_id == "dochop"


def test_emobench_prefers_its_main_measurement():
    import pandas as pd

    benchmarks = pd.read_parquet(_FIXTURES / "canonical_benchmarks.parquet").set_index("id")
    assert benchmarks.loc["emobench", "preferred_metric_id"] == "quasi-exact-match"
