"""The inference_platforms seed YAML, the seeded table, and the host-token
map are mutually consistent. Runs a real `seed --local` into a tmp dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from eval_entity_registry.cli import app
from eval_entity_registry.store import hf_store
from eval_entity_resolver import CanonicalStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed"
PROPOSAL_JSON = (
    REPO_ROOT
     / "curation"
    / "inference_platforms.proposed.json"
)

VALID_KINDS = {
    "author_lab",
    "inference_platform",
    "aggregator_gateway",
    "regional_variant",
    "coding_plan",
}


def _strip_marker(form: str) -> str:
    for marker in ("prefix:", "suffix:"):
        if form.startswith(marker):
            return form[len(marker):]
    return form


@pytest.fixture(scope="module")
def seeded_fixtures_dir(tmp_path_factory):
    """One real `seed --local` for the whole module. Every test here only reads
    the seeded tables, so they share it instead of reseeding (about 25s each)."""
    fixtures_dir = tmp_path_factory.mktemp("inference_platforms") / "fixtures"
    fixtures_dir.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCAL_MODE", "true")
        mp.setenv("FIXTURES_PATH", str(fixtures_dir))
        hf_store._store = None
        result = CliRunner().invoke(
            app, ["seed", "--local", "--seed-dir", str(SEED_DIR)]
        )
        hf_store._store = None
    assert result.exit_code == 0, result.output
    return fixtures_dir


@pytest.fixture
def seeded_store(seeded_fixtures_dir, monkeypatch):
    """A fresh store loaded from the shared seed, so no test sees another's state."""
    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("FIXTURES_PATH", str(seeded_fixtures_dir))
    hf_store._store = None
    store = hf_store.get_store()
    store.load()
    yield store
    hf_store._store = None


def test_seed_produces_138_rows(seeded_store):
    df = seeded_store.table("canonical_inference_platforms")
    assert len(df) == 138


def test_kind_enum(seeded_store):
    df = seeded_store.table("canonical_inference_platforms")
    bad = set(df["kind"].dropna().unique()) - VALID_KINDS
    assert not bad, f"unexpected kind values: {bad}"
    assert df["kind"].notna().all()


def test_author_lab_canonical_org_fk(seeded_store):
    import pandas as pd

    plat = seeded_store.table("canonical_inference_platforms")
    org_ids = set(seeded_store.table("canonical_orgs")["id"])
    authors = plat[plat["kind"] == "author_lab"]
    for _, row in authors.iterrows():
        co = row["canonical_org"]
        # `canonical_org` is set only when the org is curated; FK must hold then.
        if pd.isna(co) or co == "":
            continue
        assert co in org_ids, (
            f"author_lab {row['id']!r} canonical_org {co!r} not in canonical_orgs"
        )


def test_variant_of_points_at_platform(seeded_store):
    import pandas as pd

    df = seeded_store.table("canonical_inference_platforms")
    all_ids = set(df["id"])
    variants = df[df["kind"].isin(["regional_variant", "coding_plan"])]
    for _, row in variants.iterrows():
        vo = row["variant_of"]
        # `variant_of` is set only when the base platform is curated; FK must hold then.
        if pd.isna(vo):
            continue
        assert vo in all_ids, (
            f"{row['kind']} {row['id']!r} variant_of {vo!r} not an existing platform id"
        )


def test_eee_host_tokens_resolve(seeded_store):
    data = json.loads(PROPOSAL_JSON.read_text())
    eee = data["eee_host_tokens"]
    assert len(eee) == 8, f"expected 8 EEE host tokens, got {len(eee)}"
    platforms = seeded_store.table("canonical_inference_platforms")
    seeded_ids = set(platforms["id"])
    host_token_platforms = CanonicalStore(
        inference_platforms_df=platforms
    ).host_token_platforms()
    for token_name, info in eee.items():
        expected_platform = info["platform"]
        for form in info["forms"]:
            raw = _strip_marker(form)
            resolved = host_token_platforms.get(raw.lower())
            assert resolved == expected_platform, (
                f"host token {raw!r} ({token_name}) resolved to {resolved!r}, "
                f"expected {expected_platform!r}"
            )
            assert resolved in seeded_ids, (
                f"resolved platform {resolved!r} not in seeded table"
            )


def test_seed_yaml_author_lab_org_consistency():
    rows = yaml.safe_load((SEED_DIR / "inference_platforms.yaml").read_text())
    org_ids = {
        o["id"] for o in yaml.safe_load((SEED_DIR / "orgs.yaml").read_text())
    }
    for row in rows:
        if row.get("kind") == "author_lab" and row.get("canonical_org"):
            assert row["canonical_org"] in org_ids, (
                f"{row['id']}: canonical_org {row['canonical_org']!r} "
                "not in seed/orgs.yaml"
            )


def test_every_host_suffix_the_fuzzy_strategy_strips_is_attributed(seeded_store):
    """A host suffix is stripped only to capture its platform, so each one
    needs a spelling in the seeded table."""
    from eval_entity_resolver.strategies.fuzzy import _HOST_SUFFIXES

    host_token_platforms = CanonicalStore(
        inference_platforms_df=seeded_store.table("canonical_inference_platforms")
    ).host_token_platforms()
    missing = [s for s in _HOST_SUFFIXES if s not in host_token_platforms]
    assert not missing, f"host suffixes with no platform in the seed: {missing}"


# --- An inference provider and a model developer are separate entities ------


@pytest.fixture(scope="module")
def seeded_snapshot(seeded_fixtures_dir):
    """`(resolver, tables)` from the module's shared seed."""
    import pandas as pd

    tables = {
        name: pd.read_parquet(seeded_fixtures_dir / f"canonical_{name}.parquet")
        for name in ("models", "inference_platforms", "orgs")
    }
    from eval_entity_resolver import Resolver

    return Resolver.from_parquet(str(seeded_fixtures_dir)), tables


def _prefix_token_platforms(platforms) -> dict[str, dict]:
    """Prefix host token (`together`) -> its platform row."""
    out: dict[str, dict] = {}
    for row in platforms.to_dict("records"):
        aliases = row["aliases"]
        for alias in json.loads(aliases) if isinstance(aliases, str) else aliases:
            if alias.endswith("/"):
                out[alias[:-1].lower()] = row
    return out


def provider_tokens_used_as_developer(models, platforms) -> list[tuple[str, str]]:
    """`(model id, org_id)` for every model whose developer is a provider's
    host token while that provider's row declares no `canonical_org`. Such a
    provider only serves other developers' models, so it developed nothing."""
    import pandas as pd

    serving_only = {
        token
        for token, row in _prefix_token_platforms(platforms).items()
        if pd.isna(row.get("canonical_org")) or not row.get("canonical_org")
    }
    return [
        (m["id"], m["org_id"])
        for m in models[["id", "org_id"]].dropna().to_dict("records")
        if m["org_id"].lower() in serving_only
    ]


def test_provider_token_check_refuses_a_serving_only_provider_as_developer():
    import pandas as pd

    platforms = pd.DataFrame(
        [
            {"id": "azure", "aliases": '["Azure", "azure/"]', "canonical_org": None},
            {"id": "groq", "aliases": '["Groq", "groq/"]', "canonical_org": "Groq"},
        ]
    )
    models = pd.DataFrame(
        [
            {"id": "azure/phi-4", "org_id": "azure"},
            {"id": "microsoft/phi-4", "org_id": "microsoft"},
            {"id": "Groq/Llama-3-Groq-8B-Tool-Use", "org_id": "Groq"},
        ]
    )
    assert provider_tokens_used_as_developer(models, platforms) == [
        ("azure/phi-4", "azure")
    ]


def test_no_serving_only_provider_is_a_model_developer(seeded_snapshot):
    _, tables = seeded_snapshot
    bad = provider_tokens_used_as_developer(
        tables["models"], tables["inference_platforms"]
    )
    assert not bad, f"provider host token recorded as the developer: {bad}"


def test_every_canonical_org_is_an_org(seeded_snapshot):
    """`canonical_org` is the link from a platform to the developer that
    operates it, whatever the platform's kind."""
    _, tables = seeded_snapshot
    plat = tables["inference_platforms"]
    org_ids = set(tables["orgs"]["id"])
    linked = plat[plat["canonical_org"].notna() & (plat["canonical_org"] != "")]
    missing = {
        r["id"]: r["canonical_org"]
        for r in linked.to_dict("records")
        if r["canonical_org"] not in org_ids
    }
    assert not missing, f"canonical_org not in canonical_orgs: {missing}"


@pytest.mark.parametrize(
    "raw, developer, platform",
    [
        # Fireworks serving other developers' models: the `fireworks/` prefix
        # is the platform, and the developer is the model's own.
        ("fireworks/deepseek-v3p1", "deepseek", "fireworks-ai"),
        ("fireworks/deepseek-v3p2", "deepseek", "fireworks-ai"),
        ("fireworks/deepseek-v3p2-thinking", "deepseek", "fireworks-ai"),
        ("fireworks/llama4-maverick-instruct-basic", "meta", "fireworks-ai"),
        ("fireworks/deepseek-r1", "deepseek", "fireworks-ai"),
        ("together/llama-2-7b", "meta", "togetherai"),
        ("azure/phi-4", "microsoft", "azure"),
        # The provider's own model: the prefix may be a namespace.
        ("fireworks/ember-1", "fireworks", None),
        ("together/redpajama-incite-base-7b", "togethercomputer", None),
        ("together/RedPajama-INCITE-Instruct-7B", "togethercomputer", None),
        ("together/RedPajama-INCITE-7B-Base", "togethercomputer", None),
        ("together/redpajama-incite-instruct-3b-v1", "togethercomputer", None),
        ("together/RedPajama-INCITE-Base-v1-3B", "togethercomputer", None),
        ("Groq/Llama-3-Groq-8B-Tool-Use", "Groq", None),
    ],
)
def test_seeded_registry_separates_provider_from_developer(
    seeded_snapshot, raw, developer, platform
):
    resolver, tables = seeded_snapshot
    result = resolver.resolve(raw, "model")
    assert result.canonical_id is not None, raw
    models = tables["models"].set_index("id")
    assert models.loc[result.canonical_id, "org_id"] == developer
    assert result.inference_platform == platform
