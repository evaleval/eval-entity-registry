"""Typed-decisions canonicals (parent + en/zh slice children), the Phocinae org,
the Phocinae-Largha-150M-v1 model and the laya harness resolve exactly.

Guards the entities minted for the Phocinae-Largha-150M-v1 typed-decisions
EEE_datastore submission (en + zh records): the raw ids the records carry must
resolve to the seeded canonicals, including the bare model name the model card
and the records' `model_info.name` use. Skips if fixtures aren't built (run
`eval-entity-registry seed --local` first).
"""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO_ROOT / "fixtures"

# Raw string carried by the EEE records -> (entity_type, expected canonical).
ENTITY_CANONICALS = {
    "typed-decisions": [("benchmark", "typed-decisions")],
    "typed-decisions-en": [("benchmark", "typed-decisions-en")],
    "typed-decisions-zh": [("benchmark", "typed-decisions-zh")],
    "Phocinae": [("org", "phocinae")],
    "Phocinae/Phocinae-Largha-150M-v1": [("model", "Phocinae/Phocinae-Largha-150M-v1")],
    "Phocinae-Largha-150M-v1": [("model", "Phocinae/Phocinae-Largha-150M-v1")],
    "laya": [("harness", "laya")],
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize(
    "raw,entity_type,expected",
    [(raw, et, cid) for raw, pairs in ENTITY_CANONICALS.items() for et, cid in pairs],
)
def test_typed_decisions_entity_resolves(resolver, raw, entity_type, expected):
    res = resolver.resolve(raw, entity_type=entity_type)
    assert res.canonical_id == expected, (
        f"{raw!r} ({entity_type}) -> {res.canonical_id!r} (expected {expected!r})"
    )
