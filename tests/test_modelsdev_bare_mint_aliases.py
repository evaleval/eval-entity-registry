"""models.dev keys that arrive without a developer namespace resolve to the
curated model: Microsoft's MAI Code models (listed only under github-copilot),
Cloudflare's Clairic Flash, and Neuralwatt's product names for models it also
lists under their real names. Skips without fixtures.
"""
from pathlib import Path

import pandas as pd
import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

MODELS = {
    "mai-code-1.1-flash": "microsoft/mai-code-1.1-flash",
    "mai-code-1-1-flash": "microsoft/mai-code-1.1-flash",
    "mai-code-2": "microsoft/mai-code-2",
    "mai-code-2-flash": "microsoft/mai-code-2-flash",
    "github-copilot/mai-code-2": "microsoft/mai-code-2",
    "clairic-flash": "Cloudflare/clairic-flash",
    "cloudflare/clairic-flash": "Cloudflare/clairic-flash",
    "nw-small": "Qwen/Qwen3.8-27B",
    "nw-small-flex": "alibaba/qwen-3-8-27b-flex",
    "nw-flash": "deepseek-ai/DeepSeek-V4.1-Flash",
    "nw-flash-flex": "deepseek/deepseek-v4-1-flash-flex",
    "nw-large": "moonshotai/Kimi-K3.1-Pro",
    "nw-large-flex": "moonshotai/kimi-k3-1-pro-flex",
}

ORGS = {
    "microsoft/mai-code-1.1-flash": "microsoft",
    "microsoft/mai-code-2": "microsoft",
    "microsoft/mai-code-2-flash": "microsoft",
    "Cloudflare/clairic-flash": "cloudflare",
}

BARE_MINTS = [
    "mai-code-1-1-flash", "mai-code-2", "mai-code-2-flash", "clairic-flash",
    "nw-small", "nw-small-flex", "nw-flash", "nw-flash-flex", "nw-large",
    "nw-large-flex",
]

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def models_df():
    return pd.read_parquet(_FIXTURES / "canonical_models.parquet")


@pytest.mark.parametrize("raw,expected", MODELS.items())
def test_raw_form_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="model").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("model_id,org_id", ORGS.items())
def test_curated_model_has_its_developer(models_df, model_id, org_id):
    row = models_df[models_df["id"] == model_id]
    assert len(row) == 1, f"{model_id} missing from canonical_models"
    assert row.iloc[0]["org_id"] == org_id


@pytest.mark.parametrize("bare", BARE_MINTS)
def test_bare_mint_is_not_a_canonical(models_df, bare):
    assert bare not in set(models_df["id"])
