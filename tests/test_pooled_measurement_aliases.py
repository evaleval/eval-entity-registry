"""Measurements that shared one cell resolve to separate metrics: Terminal-Bench
pass@k for k = 2..5, CocoaBench total cost beside cost per task, The Last
Ones' full completion beside partial progress, and ExploitBench catalog
coverage. Every form is scoped to its source. Skips without fixtures."""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)

_CATCH_ALL = frozenset({"score", "mean-score", "overall"})
_BUDGETS = ("1M", "3M", "10M", "30M", "100M")


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize("config", ["terminal-bench-2.1", "terminal-bench-4.0"])
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_terminal_bench_pass_at_k_resolves(resolver, config, k):
    assert resolver.resolve(f"Pass@{k}", "metric", config).canonical_id == f"pass-at-{k}"


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_pass_at_k_and_pass_hat_k_stay_apart(resolver, k):
    assert resolver.resolve(f"Pass@{k}", "metric", None).canonical_id is None
    assert resolver.resolve(f"Pass^{k}", "metric", None).canonical_id is None
    assert resolver.resolve(f"Pass@{k}", "metric", "tau-bench").canonical_id != f"pass-at-{k}"
    if k < 5:
        assert resolver.resolve(f"Pass^{k}", "metric", "tau-bench").canonical_id == f"pass-hat-{k}"


def test_cocoabench_costs_are_two_metrics(resolver):
    total = resolver.resolve_structured_metric(
        "cocoabench.overall.total_cost_usd", "cocoabench", catch_all_ids=_CATCH_ALL)
    assert total is not None and total.canonical_id == "total-cost"
    assert resolver.resolve("Total evaluation cost", "metric", "cocoabench").canonical_id == "total-cost"
    assert resolver.resolve("Total evaluation cost", "metric", None).canonical_id is None
    assert resolver.resolve("cost", "metric", "cocoabench").canonical_id == "cost-per-task"


@pytest.mark.parametrize("budget", _BUDGETS)
def test_the_last_ones_measurements_resolve_apart(resolver, budget):
    scope = "aisi-the-last-ones"
    full = f"Full Completion Rate @ {budget} tokens"
    partial = f"Mean Partial Progress @ {budget} tokens"
    assert resolver.resolve(full, "metric", scope).canonical_id == "accuracy"
    assert resolver.resolve(partial, "metric", scope).canonical_id == "aisi-mean-progress-score"
    assert resolver.resolve(full, "metric", None).canonical_id is None
    assert resolver.resolve(partial, "metric", None).canonical_id is None


def test_exploitbench_coverage_resolves_under_paperswithcode_only(resolver):
    match = resolver.resolve_structured_metric(
        "paperswithcode.coverage", "paperswithcode", catch_all_ids=_CATCH_ALL)
    assert match is not None and match.canonical_id == "catalog-coverage"
    assert resolver.resolve("Coverage", "metric", "paperswithcode").canonical_id == "catalog-coverage"
    assert resolver.resolve("Coverage", "metric", None).canonical_id is None
