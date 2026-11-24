"""A record's own "Standard Deviation" resolves to the diagnostic
`standard-deviation`; standard error and the bare `std` key stay unresolved.
Skips without fixtures."""
import json
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


def test_standard_deviation_resolves(resolver):
    assert resolver.resolve(
        "Standard Deviation", "metric", "swe-bench-verified-mini"
    ).canonical_id == "standard-deviation"


@pytest.mark.parametrize("raw", ["Standard Error", "std", "final_stderr"])
def test_other_dispersion_forms_stay_unresolved(resolver, raw):
    assert resolver.resolve(raw, "metric", None).canonical_id is None


def test_standard_deviation_is_diagnostic():
    import pandas as pd

    metrics = pd.read_parquet(_FIXTURES / "canonical_metrics.parquet").set_index("id")
    assert json.loads(metrics.loc["standard-deviation", "metadata"])["role"] == "diagnostic"
