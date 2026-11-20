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


@pytest.mark.parametrize(
    "raw_name",
    ("EC²-Bench", "EC2-Bench", "EC2Bench", "Edge–Cloud Collaborative Agent Benchmark"),
)
def test_ec2_bench_resolves(resolver, raw_name):
    assert resolver.resolve(raw_name, entity_type="benchmark").canonical_id == "ec2-bench"
