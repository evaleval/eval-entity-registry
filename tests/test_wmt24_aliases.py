"""llm-stats' `wmt24` is WMT24++ (the adapter strips the plus signs). Skips
without fixtures."""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize("raw", ["WMT24++", "wmt24pp", "Wmt24", "llm_stats_wmt24", "wmt24"])
def test_wmt24pp_resolves(resolver, raw):
    assert resolver.resolve(raw, "benchmark", "llm-stats").canonical_id == "wmt24"
