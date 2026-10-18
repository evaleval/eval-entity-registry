"""Every raw form the `lexam` adapter emits (dotted evaluation names, metric
names, harness name, leaderboard model labels) resolves to the intended
canonical; a miss fragments LEXam results across two ids. Skips without
fixtures.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

BENCHMARKS = {
    "lexam": "lexam",
    "LEXam": "lexam",
    "lexam.open_question": "lexam-open-question",
    "lexam.mcq_4_choices": "lexam-mcq-4-choices",
}

METRICS = {
    "Multiple-Choice Accuracy": "accuracy",
    "Accuracy on Multiple-Choice Questions": "accuracy",
    "Open Question Judge Score": "lexam-open-question-judge-score",
    "lexam-open-question-judge-score": "lexam-open-question-judge-score",
}

HARNESSES = {
    "lighteval": "lighteval",
    "Lighteval": "lighteval",
    "hf-lighteval": "lighteval",
}

MODELS = {
    "DeepSeek-V3.2-chat": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V3.2-reasoner": "deepseek/deepseek-v3-2-reasoning",
    "DeepSeek-V3.2-Exp": "deepseek-ai/DeepSeek-V3.2-Exp",
    "Llama-3.1-8B-it": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-it": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-3.1-405B-it": "meta-llama/Llama-3.1-405B-Instruct",
    "Qwen-2.5-7B-it": "Qwen/Qwen2.5-7B-Instruct",
    "EuroLLM-9B-it": "utter-project/EuroLLM-9B-Instruct",
    "EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
    "utter-project/EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
    # The base must stay distinct from the instruct checkpoint.
    "utter-project/EuroLLM-9B": "utter-project/EuroLLM-9B",
}

FAMILY_MEMBERS = {"lexam", "lexam-open-question", "lexam-mcq-4-choices"}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def benchmarks_df():
    return pd.read_parquet(_FIXTURES / "canonical_benchmarks.parquet")


@pytest.fixture(scope="module")
def families_df():
    return pd.read_parquet(_FIXTURES / "canonical_families.parquet")


@pytest.mark.parametrize("raw,expected", BENCHMARKS.items())
def test_benchmark_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="benchmark").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("raw,expected", METRICS.items())
def test_metric_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="metric").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("raw,expected", HARNESSES.items())
def test_harness_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="harness").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("raw,expected", MODELS.items())
def test_model_label_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="model").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


def test_lexam_is_a_family(families_df):
    row = families_df[families_df["id"] == "lexam"]
    assert len(row) == 1, "lexam missing from canonical_families"
    assert set(json.loads(row.iloc[0]["benchmark_ids"])) == FAMILY_MEMBERS


@pytest.mark.parametrize("member", sorted(FAMILY_MEMBERS))
def test_sub_tasks_are_standalone(benchmarks_df, member):
    """The sub-tasks are separately reported datasets, so neither may carry a
    parent edge: a parented benchmark is a slice to the producer and loses its
    own merged page."""
    row = benchmarks_df[benchmarks_df["id"] == member]
    assert len(row) == 1, f"{member} missing from canonical_benchmarks"
    parent = row.iloc[0]["parent_benchmark_id"]
    assert parent is None or pd.isna(parent), f"{member} has parent_benchmark_id={parent!r}"


# Column headers are generic surface forms any source could publish, so they
# are scoped to the lexam source_config. The metric-side "Accuracy on
# Multiple-Choice Questions" stays global on `accuracy`, right for every source.
SCOPED = {
    ("benchmark", "Judge Scores on Open Questions"): "lexam-open-question",
    ("benchmark", "Accuracy on Multiple-Choice Questions"): "lexam-mcq-4-choices",
    ("metric", "Judge Scores on Open Questions"): "lexam-open-question-judge-score",
}


@pytest.mark.parametrize("entity_type,raw", sorted(SCOPED))
def test_column_headers_resolve_only_for_the_lexam_source(resolver, entity_type, raw):
    scoped = resolver.resolve(raw, entity_type=entity_type, source_config="lexam")
    assert scoped.canonical_id == SCOPED[(entity_type, raw)]
    unscoped = resolver.resolve(raw, entity_type=entity_type)
    assert unscoped.canonical_id is None, f"{raw!r} must not be a global {entity_type} alias"


def test_bare_hf_config_names_are_not_global_aliases(resolver):
    """Generic HF config names mean different things per source; the adapter
    emits only dotted forms, so neither may be a global alias."""
    for raw in ("open_question", "mcq_4_choices"):
        assert resolver.resolve(raw, entity_type="benchmark").canonical_id is None, raw


def test_judge_score_scale_is_zero_to_one_hundred():
    """The judge score is published on a 0-100 scale and accuracy on 0-1;
    folding them would silently rescale one."""
    metrics = pd.read_parquet(_FIXTURES / "canonical_metrics.parquet")
    judge = metrics[metrics["id"] == "lexam-open-question-judge-score"].iloc[0]
    assert (judge["min_score"], judge["max_score"]) == (0.0, 100.0)
    acc = metrics[metrics["id"] == "accuracy"].iloc[0]
    assert (acc["min_score"], acc["max_score"]) == (0.0, 1.0)
