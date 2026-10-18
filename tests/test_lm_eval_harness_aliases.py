"""lm-evaluation-harness task and metric slugs resolve to the canonicals the
harness-derived converter joins on; a miss fragments one benchmark's results
across two ids. Skips without fixtures.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

HARNESS_BENCHMARKS = {
    # GLUE (nine tasks; the harness ships eight plus the mismatched MNLI split)
    "glue": "glue",
    "cola": "cola",
    "mnli": "mnli",
    "mnli_matched": "mnli",
    "mnli_mismatch": "mnli-mismatched",
    "mrpc": "mrpc",
    "qnli": "qnli",
    "qqp": "qqp",
    "rte": "rte",
    "sst2": "sst2",
    "stsb": "stsb",
    "wnli": "wnli",
    # SuperGLUE
    "super_glue": "superglue",
    "super-glue-lm-eval-v1": "superglue",
    "boolq": "boolq",
    "super_glue_boolq": "boolq",
    "cb": "cb",
    "copa": "copa",
    "multirc": "multirc",
    "record": "record",
    "sglue_rte": "rte",
    "wic": "wic",
    "wsc": "wsc",
    # ANLI rounds
    "anli": "anli",
    "anli_r1": "anli-r1",
    "anli_r2": "anli-r2",
    "anli_r3": "anli-r3",
    # HEAD-QA languages
    "headqa": "head-qa",
    "headqa_en": "head-qa-en",
    "headqa_es": "head-qa-es",
    # LAMBADA corpora
    "lambada": "lambada",
    "lambada_standard": "lambada-standard",
    "lambada_openai": "lambada-openai",
    # Classic LM benchmarks the harness names directly
    "medmcqa": "medmcqa",
    "nq_open": "nq-open",
    "sciq": "sciq",
    # Natural Questions spellings all land on one entity
    "nq": "naturalquestions",
    "llm_stats_nq": "naturalquestions",
    "natural_questions": "naturalquestions",
}

HARNESS_METRICS = {
    "acc": "accuracy",
    "acc_norm": "normalized-accuracy",
    "exact_match": "exact-match",
    "em": "exact-match",
    "f1": "f1",
    "mcc": "matthews-correlation",
    "matthews_corrcoef": "matthews-correlation",
    "perplexity": "perplexity",
    "ppl": "perplexity",
    "word_perplexity": "word-perplexity",
    "byte_perplexity": "byte-perplexity",
    "bits_per_byte": "bits-per-byte",
}

# Suites whose parts are separately reported datasets are families, never
# parent_benchmark_id edges (a parented benchmark is a slice to the producer).
FAMILIES = {
    "glue": {"glue", "cola", "mnli", "mnli-mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"},
    "superglue": {"superglue", "boolq", "cb", "copa", "multirc", "record", "wic", "wsc"},
    "anli": {"anli", "anli-r1", "anli-r2", "anli-r3"},
    "head-qa": {"head-qa", "head-qa-es", "head-qa-en"},
    "lambada": {"lambada", "lambada-standard", "lambada-openai"},
    "naturalquestions": {"naturalquestions", "nq-open"},
}

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


@pytest.mark.parametrize("slug,expected", HARNESS_BENCHMARKS.items())
def test_harness_task_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="benchmark")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


@pytest.mark.parametrize("slug,expected", HARNESS_METRICS.items())
def test_harness_metric_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="metric")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


@pytest.fixture(scope="module")
def families_df():
    return pd.read_parquet(_FIXTURES / "canonical_families.parquet")


@pytest.mark.parametrize("family,members", FAMILIES.items())
def test_harness_suite_is_a_family(families_df, family, members):
    row = families_df[families_df["id"] == family]
    assert len(row) == 1, f"family {family} missing from canonical_families"
    got = set(json.loads(row.iloc[0]["benchmark_ids"]))
    assert got == members, f"{family}: {sorted(got ^ members)} differ"


@pytest.mark.parametrize("member", sorted(set().union(*FAMILIES.values())))
def test_harness_suite_members_are_standalone(benchmarks_df, member):
    """No member carries a parent edge: a separately reported dataset must
    not render as a slice."""
    row = benchmarks_df[benchmarks_df["id"] == member]
    assert len(row) == 1, f"{member} missing from canonical_benchmarks"
    parent = row.iloc[0]["parent_benchmark_id"]
    assert parent is None or pd.isna(parent), f"{member} has parent_benchmark_id={parent!r}"


def test_perplexity_variants_are_distinct(resolver):
    """word / byte / token perplexity are different numbers; none may fold
    into the unqualified id."""
    ids = {
        resolver.resolve(k, entity_type="metric").canonical_id
        for k in ("perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte")
    }
    assert len(ids) == 4, ids


def test_no_bare_nq_canonical(benchmarks_df):
    """`nq` is an alias, not a second canonical: one entity carries Natural
    Questions."""
    assert "nq" not in set(benchmarks_df["id"])


def test_harness_compound_keys_are_not_aliases(resolver):
    """Splitting the harness `metric,filter` key is the converter's job; the
    registry carries no compound alias."""
    for key in ("acc,none", "acc_norm,none", "exact_match,flexible-extract"):
        assert resolver.resolve(key, entity_type="metric").canonical_id is None, key
