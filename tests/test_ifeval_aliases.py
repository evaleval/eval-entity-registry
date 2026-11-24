"""inspect_evals IFEval states its four accuracy variants only in the
description (`prompt_strict_acc` ...) under metric_name "Accuracy". Each
lands on its own IFEval metric, scoped to the `IFEval` config; the
`<variant>_stderr` rows stay unresolved. Skips without fixtures."""
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
    "raw,expected",
    [
        ("prompt_strict_acc", "prompt-level-strict-accuracy"),
        ("prompt_loose_acc", "prompt-level-loose-accuracy"),
        ("inst_strict_acc", "inst-level-strict-accuracy"),
        ("inst_loose_acc", "inst-level-loose-accuracy"),
        ("final_acc", "final-acc"),
    ],
)
def test_variant_resolves(resolver, raw, expected):
    assert resolver.resolve(raw, "metric", "IFEval").canonical_id == expected


@pytest.mark.parametrize("raw", ["prompt_strict_acc", "inst_loose_acc"])
def test_variant_names_stay_in_ifeval(resolver, raw):
    assert resolver.resolve(raw, "metric", None).canonical_id is None


@pytest.mark.parametrize("raw", ["prompt_strict_stderr", "final_stderr", "Standard Error"])
def test_stderr_rows_stay_unresolved(resolver, raw):
    assert resolver.resolve(raw, "metric", "IFEval").canonical_id is None
