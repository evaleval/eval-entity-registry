"""The seed CLI against a tmp seed dir: alias rename repointing, cross-source
merge semantics (parents, metadata, weak scalars), org collision and
--prune-stale FK guards, and the benchmark / metric-fold validators.

An alias moved between canonicals in YAML must be repointed in place; the
failure mode is swallowing add_alias's uniqueness error and then letting
stale-removal delete the row.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from eval_entity_registry.cli import app
from eval_entity_registry.store import hf_store


@pytest.fixture
def fresh_seed_env(tmp_path, monkeypatch):
    """Isolated LOCAL_MODE registry under tmp_path with empty seed stubs."""
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()

    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("FIXTURES_PATH", str(fixtures_dir))

    hf_store._store = None

    (seed_dir / "metrics.yaml").write_text("[]\n")
    (seed_dir / "harnesses.yaml").write_text("[]\n")
    (seed_dir / "models").mkdir()
    (seed_dir / "models" / "sources").mkdir()
    (seed_dir / "models" / "enrichments").mkdir()
    (seed_dir / "models" / "core.yaml").write_text("[]\n")

    yield seed_dir, fixtures_dir

    hf_store._store = None


def _write_benchmarks(seed_dir: Path, entries: list[dict]) -> None:
    (seed_dir / "benchmarks.yaml").write_text(yaml.safe_dump(entries))


def _read_aliases(fixtures_dir: Path):
    import pandas as pd
    return pd.read_parquet(fixtures_dir / "aliases.parquet")


def _seed(seed_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir)])
    assert result.exit_code == 0, f"seed failed: {result.output}"


def test_alias_canonical_rename_doesnt_drop(fresh_seed_env):
    """An alias moved to another canonical is repointed in place (same row
    id), never deleted by stale-removal."""
    seed_dir, fixtures_dir = fresh_seed_env

    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
            "aliases": ["special-token"],
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "special-token")
                     & (aliases_df["entity_type"] == "benchmark")]
    assert len(row) == 1, "alias should exist after first seed"
    assert row.iloc[0]["canonical_id"] == "old-bench"
    original_alias_uuid = row.iloc[0]["id"]

    # The old canonical is kept, so only the rename path is exercised.
    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
        },
        {
            "id": "new-bench",
            "display_name": "New Bench",
            "review_status": "reviewed",
            "aliases": ["special-token"],
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "special-token")
                     & (aliases_df["entity_type"] == "benchmark")]
    assert len(row) == 1, (
        f"alias 'special-token' was deleted by the rename "
        f"(this is the bug). Aliases now: "
        f"{aliases_df[['raw_value', 'canonical_id']].to_dict('records')}"
    )
    assert row.iloc[0]["canonical_id"] == "new-bench", (
        f"alias should have been repointed to 'new-bench', got "
        f"{row.iloc[0]['canonical_id']!r}"
    )
    # Same UUID: updated, not deleted and recreated (downstream joins carry it).
    assert row.iloc[0]["id"] == original_alias_uuid


def test_repeat_seed_is_idempotent(fresh_seed_env):
    """Re-seeding identical YAML neither multiplies nor churns rows."""
    seed_dir, fixtures_dir = fresh_seed_env

    _write_benchmarks(seed_dir, [
        {
            "id": "bench-a",
            "display_name": "Bench A",
            "review_status": "reviewed",
            "aliases": ["tok-a", "tok-b"],
        },
    ])
    _seed(seed_dir)
    first = _read_aliases(fixtures_dir)
    first_count = len(first[first["entity_type"] == "benchmark"])

    _seed(seed_dir)
    second = _read_aliases(fixtures_dir)
    second_count = len(second[second["entity_type"] == "benchmark"])

    assert first_count == second_count, (
        f"alias count changed across identical re-seed: "
        f"{first_count} -> {second_count}"
    )


def _read_models(fixtures_dir: Path):
    import pandas as pd
    return pd.read_parquet(fixtures_dir / "canonical_models.parquet")


def _read_orgs(fixtures_dir: Path):
    import pandas as pd
    return pd.read_parquet(fixtures_dir / "canonical_orgs.parquet")


def test_parents_union_by_id_across_sources(fresh_seed_env):
    """Parent edges from two sources union by id, keeping the axis whichever
    side supplied it."""
    import json
    seed_dir, fixtures_dir = fresh_seed_env

    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "test.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "lab/model-a",
            "display_name": "Model A",
            "parents": [
                {"id": "lab/parent-shared", "relationship": "variant", "axis": "size"},
                {"id": "lab/parent-source-only", "relationship": "variant"},
            ],
            "review_status": "reviewed",
        },
    ]))
    (seed_dir / "models" / "core.yaml").write_text(yaml.safe_dump([
        {
            "id": "lab/model-a",
            "display_name": "Model A",
            "parents": [
                # Same id as the source edge, without its axis.
                {"id": "lab/parent-shared", "relationship": "variant"},
                {"id": "lab/parent-core-only", "relationship": "finetune"},
            ],
            "review_status": "reviewed",
        },
    ]))
    _seed(seed_dir)

    df = _read_models(fixtures_dir)
    row = df[df["id"] == "lab/model-a"]
    assert len(row) == 1
    parents = json.loads(row.iloc[0]["parents"])
    by_id = {p["id"]: p for p in parents}
    assert set(by_id) == {"lab/parent-shared", "lab/parent-source-only", "lab/parent-core-only"}
    assert by_id["lab/parent-shared"]["axis"] == "size"
    assert by_id["lab/parent-shared"]["relationship"] == "variant"
    assert by_id["lab/parent-source-only"]["relationship"] == "variant"
    assert by_id["lab/parent-core-only"]["relationship"] == "finetune"


def test_side_record_keyed_by_promoted_alias_folds_onto_hf_entry(fresh_seed_env):
    """An enrichment or parent edge still keyed by a replaced invented id
    folds onto the HF entry that aliases it, instead of minting a twin."""
    import json
    seed_dir, fixtures_dir = fresh_seed_env
    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "models_dev.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "Lab/Model-A-Instruct",
            "display_name": "Model A Instruct",
            "org_id": "lab",
            "aliases": ["lab/model-a", "model-a"],
            "metadata": json.dumps({"hf_deferred": True}),
            "review_status": "reviewed",
            "resolution_source": "models_dev",
        },
    ]))
    (seed_dir / "models" / "enrichments" / "aliases.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-a", "aliases": ["Model A (old key)"]},
    ]))
    (seed_dir / "models" / "enrichments" / "parents.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-a-reasoning", "display_name": "Model A Reasoning",
         "review_status": "reviewed",
         "parents": [{"id": "lab/model-a", "relationship": "variant", "axis": "mode"}]},
    ]))
    _seed(seed_dir)

    df = _read_models(fixtures_dir)
    assert "lab/model-a" not in set(df["id"])
    aliases = _read_aliases(fixtures_dir)
    owner = dict(zip(aliases["raw_value"], aliases["canonical_id"]))
    assert owner["lab/model-a"] == "Lab/Model-A-Instruct"
    assert owner["Model A (old key)"] == "Lab/Model-A-Instruct"
    child = df[df["id"] == "lab/model-a-reasoning"].iloc[0]
    assert [p["id"] for p in json.loads(child["parents"])] == ["Lab/Model-A-Instruct"]


def test_metadata_merges_per_key_across_sources(fresh_seed_env):
    """metadata merges per key (later source wins on overlap); a side that is
    not a JSON object falls back to last-non-empty-wins."""
    import json
    seed_dir, fixtures_dir = fresh_seed_env

    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "a.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "lab/model-m",
            "display_name": "Model M",
            "metadata": json.dumps(
                {"hf_id": "lab/model-m", "downloads_all_time": 42, "license": "mit"}
            ),
            "review_status": "reviewed",
        },
        {
            "id": "lab/model-n",
            "display_name": "Model N",
            "metadata": "{not-valid-json",
            "review_status": "reviewed",
        },
    ]))
    (sources_dir / "b.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "lab/model-m",
            "display_name": "Model M",
            "metadata": json.dumps(
                {"alias_platforms": {"model-m": "poe"}, "license": "apache-2.0"}
            ),
            "review_status": "reviewed",
        },
        {
            "id": "lab/model-n",
            "display_name": "Model N",
            "metadata": json.dumps({"source": "b"}),
            "review_status": "reviewed",
        },
    ]))
    _seed(seed_dir)

    df = _read_models(fixtures_dir)
    meta = json.loads(df[df["id"] == "lab/model-m"].iloc[0]["metadata"])
    assert meta["hf_id"] == "lab/model-m"
    assert meta["downloads_all_time"] == 42
    assert meta["alias_platforms"] == {"model-m": "poe"}
    assert meta["license"] == "apache-2.0"
    meta_n = json.loads(df[df["id"] == "lab/model-n"].iloc[0]["metadata"])
    assert meta_n == {"source": "b"}


def test_orgs_generated_curated_wins_on_collision(fresh_seed_env):
    """Curated orgs.yaml wins over orgs.generated.yaml on an id collision."""
    seed_dir, fixtures_dir = fresh_seed_env

    (seed_dir / "orgs.yaml").write_text(yaml.safe_dump([
        {
            "id": "anthropic",
            "display_name": "Anthropic",
            "kind": "lab",
            "review_status": "reviewed",
        },
    ]))
    (seed_dir / "orgs.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "anthropic",
            "display_name": "Anthropic Auto",
            "kind": "unknown",
            "review_status": "auto",
        },
        {
            "id": "some-uploader",
            "display_name": "Some Uploader",
            "kind": "individual",
            "review_status": "auto",
        },
    ]))
    _seed(seed_dir)

    orgs = _read_orgs(fixtures_dir)
    anthropic = orgs[orgs["id"] == "anthropic"].iloc[0]
    assert anthropic["display_name"] == "Anthropic", "curated entry must win on collision"
    assert anthropic["kind"] == "lab"
    assert anthropic["review_status"] == "reviewed"

    auto = orgs[orgs["id"] == "some-uploader"].iloc[0]
    assert auto["display_name"] == "Some Uploader"
    assert auto["kind"] == "individual"


def test_prune_stale_keeps_org_referenced_by_model(fresh_seed_env):
    """--prune-stale keeps an auto-created org still referenced by a model's
    org_id; pruning it would leave a dangling FK."""
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "orgs.yaml").write_text(yaml.safe_dump([
        {"id": "acme", "display_name": "Acme", "kind": "lab", "review_status": "reviewed"},
    ]))
    (seed_dir / "models" / "core.yaml").write_text(yaml.safe_dump([
        {
            "id": "novelorg/some-model",
            "display_name": "Some Model",
            "review_status": "reviewed",
        },
    ]))
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir), "--prune-stale"])
    assert result.exit_code == 0, f"seed --prune-stale failed: {result.output}"

    models = _read_models(fixtures_dir)
    orgs = _read_orgs(fixtures_dir)
    row = models[models["id"] == "novelorg/some-model"]
    assert len(row) == 1
    org_id = row.iloc[0]["org_id"]
    assert org_id and str(org_id) != "nan", "model should have a derived org_id"
    assert str(org_id) in set(orgs["id"].astype(str)), (
        f"--prune-stale dropped org {org_id!r} still referenced by a surviving "
        f"model (dangling org_id FK). Org ids: {sorted(orgs['id'].astype(str))}"
    )


def test_prune_stale_keeps_org_referenced_by_parent_org_id(fresh_seed_env):
    """--prune-stale keeps an org dropped from YAML while another org's
    parent_org_id still references it."""
    seed_dir, fixtures_dir = fresh_seed_env

    orgs_path = seed_dir / "orgs.yaml"
    orgs_path.write_text(yaml.safe_dump([
        {"id": "parent-lab", "display_name": "Parent Lab",
         "kind": "lab", "review_status": "reviewed"},
        {"id": "child-lab", "display_name": "Child Lab", "kind": "lab",
         "review_status": "reviewed", "parent_org_id": "parent-lab"},
    ]))
    _seed(seed_dir)
    assert "parent-lab" in set(_read_orgs(fixtures_dir)["id"].astype(str))

    orgs_path.write_text(yaml.safe_dump([
        {"id": "child-lab", "display_name": "Child Lab", "kind": "lab",
         "review_status": "reviewed", "parent_org_id": "parent-lab"},
    ]))
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir), "--prune-stale"])
    assert result.exit_code == 0, f"seed --prune-stale failed: {result.output}"

    orgs = _read_orgs(fixtures_dir)
    assert "parent-lab" in set(orgs["id"].astype(str)), (
        "--prune-stale dropped an org still referenced by a surviving org's "
        f"parent_org_id FK. Org ids: {sorted(orgs['id'].astype(str))}"
    )


def test_orgs_generated_alone_loads(fresh_seed_env):
    """orgs.generated.yaml loads without an orgs.yaml."""
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "orgs.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "lone-uploader",
            "display_name": "Lone",
            "kind": "individual",
            "review_status": "auto",
        },
    ]))
    _seed(seed_dir)

    orgs = _read_orgs(fixtures_dir)
    assert "lone-uploader" in set(orgs["id"])


def test_scoped_alias_rename_doesnt_drop(fresh_seed_env):
    """The rename repoint also holds for scoped aliases."""
    seed_dir, fixtures_dir = fresh_seed_env

    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
            "scoped_aliases": {"some_config": ["Overall"]},
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "Overall")
                     & (aliases_df["source_config"] == "some_config")]
    assert len(row) == 1
    assert row.iloc[0]["canonical_id"] == "old-bench"

    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
        },
        {
            "id": "new-bench",
            "display_name": "New Bench",
            "review_status": "reviewed",
            "scoped_aliases": {"some_config": ["Overall"]},
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "Overall")
                     & (aliases_df["source_config"] == "some_config")]
    assert len(row) == 1, (
        "scoped alias 'Overall' was deleted by the rename "
        "(this is the bug, scoped variant)"
    )
    assert row.iloc[0]["canonical_id"] == "new-bench"


# ----------------------------------------------------------------------
# Weak scalar fill: `weak:` values apply after all strong merges and fill
# empty fields only; an explicit core key (even null) blocks them; ties
# break to the first contribution (sorted filename, then record order).
# ----------------------------------------------------------------------

def test_weak_scalar_fills_empty_but_never_beats_full_entry(fresh_seed_env):
    """A full entry's value beats a weak one whether the weak file loads
    before or after it."""
    seed_dir, fixtures_dir = fresh_seed_env
    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "a.generated.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-a", "aliases": [],
         "weak": {"release_date": "2024-01-01", "open_weights": True}},
    ]))
    (sources_dir / "z.generated.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-a", "display_name": "Model A",
         "release_date": "2023-05-05", "review_status": "reviewed"},
        {"id": "lab/model-b", "display_name": "Model B",
         "release_date": "2023-06-06", "review_status": "reviewed"},
    ]))
    (sources_dir / "zz.generated.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-b", "aliases": [],
         "weak": {"release_date": "2024-02-02", "open_weights": True}},
    ]))
    _seed(seed_dir)

    df = _read_models(fixtures_dir)
    a = df[df["id"] == "lab/model-a"].iloc[0]
    b = df[df["id"] == "lab/model-b"].iloc[0]
    assert a["release_date"] == "2023-05-05", "weak (loaded first) must not beat full entry"
    assert b["release_date"] == "2023-06-06", "weak (loaded last) must not beat full entry"
    assert bool(a["open_weights"]) and bool(b["open_weights"]), "weak must fill the empty field"


def test_core_explicit_null_blocks_weak_fill(fresh_seed_env):
    """An explicit core key, even null, is a curated claim that blocks weak
    fill; an absent key stays open."""
    import pandas as pd
    seed_dir, fixtures_dir = fresh_seed_env
    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "a.generated.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-c", "aliases": [],
         "weak": {"open_weights": True, "release_date": "2025-12-01"}},
    ]))
    (seed_dir / "models" / "core.yaml").write_text(yaml.safe_dump({
        "skip_ids": [],
        "skip_source_ids": [],
        "entries": [
            {"id": "lab/model-c", "display_name": "Model C",
             "open_weights": None, "review_status": "reviewed"},
        ],
    }))
    _seed(seed_dir)

    row = _read_models(fixtures_dir)
    row = row[row["id"] == "lab/model-c"].iloc[0]
    assert pd.isna(row["open_weights"]), "explicit core null must block weak fill"
    assert row["release_date"] == "2025-12-01", "absent core key must accept weak fill"


def test_two_weak_values_tiebreak_first_file_first_record(fresh_seed_env):
    """Two weak values for one field: the first contribution wins."""
    seed_dir, fixtures_dir = fresh_seed_env
    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "a.generated.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-d", "aliases": [], "weak": {"release_date": "2024-03-03"}},
        {"id": "lab/model-d", "aliases": [], "weak": {"release_date": "2024-04-04"}},
        {"id": "lab/model-d", "display_name": "Model D", "review_status": "reviewed"},
    ]))
    (sources_dir / "b.generated.yaml").write_text(yaml.safe_dump([
        {"id": "lab/model-d", "aliases": [], "weak": {"release_date": "2024-05-05"}},
    ]))
    _seed(seed_dir)

    df = _read_models(fixtures_dir)
    row = df[df["id"] == "lab/model-d"].iloc[0]
    assert row["release_date"] == "2024-03-03"


def _seed_expect_fail(seed_dir: Path) -> str:
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir)])
    assert result.exit_code != 0, "seed unexpectedly succeeded"
    return result.output


def test_benchmark_separator_collision_rejected(fresh_seed_env):
    """Two benchmark ids differing only by separators fail the seed with
    both ids named."""
    seed_dir, _ = fresh_seed_env
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
        {"id": "foobench", "display_name": "FooBench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "foo-bench" in out and "foobench" in out


def test_benchmark_alias_collision_rejected(fresh_seed_env):
    """A global alias normalized-colliding with another benchmark's id fails too."""
    seed_dir, _ = fresh_seed_env
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
        {"id": "bar-bench", "display_name": "Bar Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed",
         "aliases": ["Foo_Bench"]},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "bar-bench" in out and "foo-bench" in out


def test_benchmark_collision_allowlist(fresh_seed_env):
    """A group listed in seed/benchmarks_distinct_allowlist.yaml passes."""
    seed_dir, _ = fresh_seed_env
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
        {"id": "foobench", "display_name": "FooBench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
    ])
    (seed_dir / "benchmarks_distinct_allowlist.yaml").write_text(
        yaml.safe_dump([["foo-bench", "foobench"]])
    )
    _seed(seed_dir)


def test_benchmark_id_with_slash_rejected(fresh_seed_env):
    """'/' is reserved for merged-view routing."""
    seed_dir, _ = fresh_seed_env
    _write_benchmarks(seed_dir, [
        {"id": "foo/bar", "display_name": "Foo Bar", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "foo/bar" in out


def test_preferred_metric_validated_and_renamed(fresh_seed_env):
    """`preferred_metric` lands as `preferred_metric_id`; an unknown id fails."""
    import pandas as pd
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "metrics.yaml").write_text(yaml.safe_dump([
        {"id": "accuracy", "display_name": "Accuracy", "score_type": "continuous",
         "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
         "metadata": "{}", "review_status": "reviewed"},
    ]))
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed", "preferred_metric": "accuracy"},
    ])
    _seed(seed_dir)
    df = pd.read_parquet(fixtures_dir / "canonical_benchmarks.parquet")
    assert df.set_index("id").loc["foo-bench", "preferred_metric_id"] == "accuracy"

    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed", "preferred_metric": "acuracy"},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "acuracy" in out


def test_preferred_metric_llm_judged_validated(fresh_seed_env):
    import pandas as pd

    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "metrics.yaml").write_text(yaml.safe_dump([
        {"id": "accuracy", "display_name": "Accuracy", "score_type": "continuous",
         "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
         "metadata": "{}", "review_status": "reviewed"},
    ]))
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed", "preferred_metric": "accuracy",
         "preferred_metric_llm_judged": True},
    ])
    _seed(seed_dir)
    df = pd.read_parquet(fixtures_dir / "canonical_benchmarks.parquet")
    assert bool(df.set_index("id").loc["foo-bench", "preferred_metric_llm_judged"]) is True

    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed", "preferred_metric": "accuracy",
         "preferred_metric_llm_judged": "true"},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "preferred_metric_llm_judged" in out

    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed", "preferred_metric_llm_judged": True},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "preferred_metric" in out

    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed", "preferred_metric_llm_judged": False},
    ])
    out = _seed_expect_fail(seed_dir)
    assert "preferred_metric" in out

    for spelling in ("yes", "on", "off"):
        (seed_dir / "benchmarks.yaml").write_text(
            "- id: foo-bench\n"
            "  display_name: Foo Bench\n"
            "  metadata: '{}'\n"
            "  tags: '[]'\n"
            "  review_status: reviewed\n"
            "  preferred_metric: accuracy\n"
            f"  preferred_metric_llm_judged: {spelling}\n"
        )
        out = _seed_expect_fail(seed_dir)
        assert spelling in out


def test_metric_folds_loaded_and_validated(fresh_seed_env):
    """metric_folds.yaml lands as benchmark_metric_folds; malformed scales,
    unknown benchmarks, contradictory scoped rows and chained folds fail."""
    import pandas as pd
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "metrics.yaml").write_text(yaml.safe_dump([
        {"id": "accuracy", "display_name": "Accuracy", "score_type": "continuous",
         "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
         "metadata": "{}", "review_status": "reviewed"},
        {"id": "score", "display_name": "Score", "score_type": "continuous",
         "lower_is_better": False, "min_score": None, "max_score": None,
         "metadata": "{}", "review_status": "reviewed"},
        {"id": "f1", "display_name": "F1", "score_type": "continuous",
         "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
         "metadata": "{}", "review_status": "reviewed"},
    ]))
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
    ])
    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "source_config": "openeval", "scale_factor": 0.5, "scale_offset": -0.25},
    ]))
    _seed(seed_dir)
    df = pd.read_parquet(fixtures_dir / "benchmark_metric_folds.parquet")
    row = df[df["source_config"] == "openeval"].iloc[0].to_dict()
    assert row["benchmark_id"] == "foo-bench"
    assert row["from_metric_id"] == "score"
    assert row["to_metric_id"] == "accuracy"
    assert row["source_config"] == "openeval"
    assert row["note"] is None
    assert row["scale_factor"] == 0.5
    assert row["scale_offset"] == -0.25

    for key, value in [
        ("scale_factor", True), ("scale_factor", float("nan")),
        ("scale_factor", float("inf")), ("scale_offset", True),
        ("scale_offset", float("nan")), ("scale_offset", float("inf")),
    ]:
        (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
            {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
            {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
             "source_config": "openeval", key: value},
        ]))
        out = _seed_expect_fail(seed_dir)
        assert key in out

    for key in ("scale_factor", "scale_offset"):
        for value in ("1e400", ".inf"):
            (seed_dir / "metric_folds.yaml").write_text(
                "- benchmark: foo-bench\n"
                "  from_metric: score\n"
                "  to_metric: accuracy\n"
                "- benchmark: foo-bench\n"
                "  from_metric: score\n"
                "  to_metric: accuracy\n"
                "  source_config: openeval\n"
                f"  {key}: {value}\n"
            )
            out = _seed_expect_fail(seed_dir)
            assert key in out

    for value in (0, -0.5):
        (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
            {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
            {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
             "source_config": "openeval", "scale_factor": value},
        ]))
        out = _seed_expect_fail(seed_dir)
        assert "positive" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "accuracy", "to_metric": "score"},
        {"benchmark": "foo-bench", "from_metric": "accuracy", "to_metric": "score",
         "source_config": "openeval", "scale_offset": 0.1},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "must have finite" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "scale_factor": 0.5},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "require source_config" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "source_config": "openeval"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "scoped row" in out and "unscoped" in out

    # A scoped row with an invalid config name is a typo, not a scope.
    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "source_config": "data/openeval"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "dataset config name" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "f1",
         "source_config": "openeval"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "to_metric" in out and "unscoped" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "source_config": "openeval"},
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "source_config": "openeval"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "duplicate fold" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "nope-bench", "from_metric": "score", "to_metric": "accuracy"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "nope-bench" in out

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
        {"benchmark": "foo-bench", "from_metric": "accuracy", "to_metric": "score"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "chained" in out


def test_metric_folds_pure_published_scale_conversion(fresh_seed_env):
    """A scoped self-fold with a conversion may stand alone: the right metric
    on the wrong scale needs no naming row."""
    import pandas as pd
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "metrics.yaml").write_text(yaml.safe_dump([
        {"id": "accuracy", "display_name": "Accuracy", "score_type": "continuous",
         "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
         "metadata": "{}", "review_status": "reviewed"},
        {"id": "score", "display_name": "Score", "score_type": "continuous",
         "lower_is_better": False, "min_score": None, "max_score": None,
         "metadata": "{}", "review_status": "reviewed"},
        {"id": "f1", "display_name": "F1", "score_type": "continuous",
         "lower_is_better": False, "min_score": 0.0, "max_score": 1.0,
         "metadata": "{}", "review_status": "reviewed"},
    ]))
    _write_benchmarks(seed_dir, [
        {"id": "foo-bench", "display_name": "Foo Bench", "metadata": "{}",
         "tags": "[]", "review_status": "reviewed"},
    ])

    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "accuracy", "to_metric": "accuracy",
         "source_config": "openeval", "scale_factor": 0.01},
    ]))
    _seed(seed_dir)
    df = pd.read_parquet(fixtures_dir / "benchmark_metric_folds.parquet")
    assert len(df) == 1
    row = df.iloc[0].to_dict()
    assert row["from_metric_id"] == "accuracy"
    assert row["to_metric_id"] == "accuracy"
    assert row["source_config"] == "openeval"
    assert row["scale_factor"] == 0.01

    # Still rejected: an unscoped self-fold says nothing.
    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "accuracy", "to_metric": "accuracy"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "self-fold" in out

    # Still rejected: no conversion and no unscoped row does nothing.
    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "accuracy", "to_metric": "accuracy",
         "source_config": "openeval"},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "pure published-scale conversion" in out

    # Still rejected: a scoped row may not contradict the benchmark-wide row.
    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy"},
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "f1",
         "source_config": "openeval", "scale_factor": 0.01},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "to_metric" in out and "unscoped" in out

    # Still rejected: an unscoped row may not carry a conversion.
    (seed_dir / "metric_folds.yaml").write_text(yaml.safe_dump([
        {"benchmark": "foo-bench", "from_metric": "score", "to_metric": "accuracy",
         "scale_factor": 0.01},
    ]))
    out = _seed_expect_fail(seed_dir)
    assert "require source_config" in out


def test_composites_scoped_members_json_encode(fresh_seed_env):
    """Scoped config members survive as JSON objects in `source_configs`;
    str() coercion would ship Python repr the producer cannot parse."""
    import json

    import pandas as pd

    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "composites.yaml").write_text(yaml.safe_dump({
        "aisi-study": {
            "display": "AISI study",
            "configs": [
                "plain-config",
                {"config": "hle", "org": "uk-aisi"},
                {"config": "healthbench", "org": "uk-aisi",
                 "source": "some-study"},
            ],
        },
    }))
    _seed(seed_dir)
    df = pd.read_parquet(fixtures_dir / "canonical_composites.parquet")
    row = df[df["id"] == "aisi-study"].iloc[0]
    decoded = json.loads(row["source_configs"])
    assert decoded == [
        "plain-config",
        {"config": "hle", "org": "uk-aisi"},
        {"config": "healthbench", "org": "uk-aisi", "source": "some-study"},
    ]


def test_composites_scoped_member_validation(fresh_seed_env):
    seed_dir, _ = fresh_seed_env
    (seed_dir / "composites.yaml").write_text(yaml.safe_dump({
        "bad": {"configs": [{"config": "hle", "source": "no-org"}]},
    }))
    out = _seed_expect_fail(seed_dir)
    assert "without `org`" in out

    (seed_dir / "composites.yaml").write_text(yaml.safe_dump({
        "bad": {"configs": [{"organization": "uk-aisi"}]},
    }))
    out = _seed_expect_fail(seed_dir)
    assert "config, org" in out


def test_seeded_wildbench_folds_carry_both_source_conversions():
    """The real seed's wildbench folds: one naming row plus per-source scale
    conversions (OpenEval 1-10, BenchPress -100..100) onto `wb-score`."""
    import pandas as pd

    repo_root = Path(__file__).resolve().parent.parent
    folds_parquet = repo_root / "fixtures" / "benchmark_metric_folds.parquet"
    if not folds_parquet.exists():
        pytest.skip("fixtures not built; run `eval-entity-registry seed --local`")

    df = pd.read_parquet(folds_parquet)
    wb = df[(df["benchmark_id"] == "wildbench") & (df["from_metric_id"] == "score")]
    assert set(wb["to_metric_id"]) == {"wb-score"}

    unscoped = wb[wb["source_config"].isna()]
    assert len(unscoped) == 1
    assert pd.isna(unscoped.iloc[0]["scale_factor"])
    assert pd.isna(unscoped.iloc[0]["scale_offset"])

    scoped = wb[wb["source_config"].notna()]
    assert sorted(scoped["source_config"]) == ["benchpress", "openeval"]

    bp = scoped[scoped["source_config"] == "benchpress"].iloc[0]

    def convert(published):
        return published * bp["scale_factor"] + bp["scale_offset"]

    assert convert(68.5) == pytest.approx(0.8250, abs=1e-4)
    assert convert(33.1) == pytest.approx(0.6283, abs=1e-4)
    # The published scale is (rating-5)*20: endpoints -80 and 100 map to the bounds.
    assert convert(-80.0) == pytest.approx(0.0, abs=1e-9)
    assert convert(100.0) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------
# Seed as a pure function of the YAML: row replacement, prune by claim,
# deterministic alias ids and order.
# ------------------------------------------------------------------

def _write_models(seed_dir: Path, entries: list[dict]) -> None:
    (seed_dir / "models" / "core.yaml").write_text(yaml.safe_dump(entries))


def _seed_prune(seed_dir: Path) -> str:
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir), "--prune-stale"])
    assert result.exit_code == 0, f"seed --prune-stale failed: {result.output}"
    return result.output


def _write_table(fixtures_dir: Path, table: str, df) -> None:
    hf_store._store = None
    df.to_parquet(fixtures_dir / f"{table}.parquet", index=False)


def test_reseed_nulls_field_dropped_from_yaml(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    _write_models(seed_dir, [
        {"id": "acme/thing", "display_name": "Thing", "review_status": "reviewed",
         "params_billions": 7.0},
    ])
    _seed(seed_dir)
    first = _read_models(fixtures_dir).set_index("id").loc["acme/thing"]
    assert float(first["params_billions"]) == 7.0

    _write_models(seed_dir, [
        {"id": "acme/thing", "display_name": "Thing", "review_status": "reviewed"},
    ])
    _seed(seed_dir)
    second = _read_models(fixtures_dir).set_index("id").loc["acme/thing"]
    import pandas as pd
    assert pd.isna(second["params_billions"]), (
        f"field dropped from YAML survived the reseed: {second['params_billions']!r}"
    )


def test_prune_removes_stale_draft(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    _write_models(seed_dir, [
        {"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"},
        {"id": "acme/old-draft", "display_name": "Old Draft", "review_status": "draft"},
        {"id": "acme/no-status", "display_name": "No Status"},
    ])
    _seed(seed_dir)
    assert set(_read_models(fixtures_dir)["id"]) == {"acme/keep", "acme/old-draft", "acme/no-status"}

    _write_models(seed_dir, [
        {"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"},
    ])
    _seed_prune(seed_dir)
    models = _read_models(fixtures_dir)
    aliases = _read_aliases(fixtures_dir)
    assert set(models["id"]) == {"acme/keep"}
    assert not aliases["canonical_id"].isin(["acme/old-draft", "acme/no-status"]).any()


def test_prune_keeps_entity_referenced_by_eval_results(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    from eval_entity_registry.store import schemas
    import pandas as pd
    _write_models(seed_dir, [
        {"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"},
        {"id": "acme/scored-draft", "display_name": "Scored Draft", "review_status": "draft"},
    ])
    _seed(seed_dir)
    results = pd.concat([schemas.empty("eval_results"), pd.DataFrame([{
        "id": "r1", "evaluation_id": "e1", "result_index": 0, "source_config": "cfg",
        "model_id": "acme/scored-draft", "benchmark_id": "b", "score": 1.0,
    }])], ignore_index=True)
    _write_table(fixtures_dir, "eval_results", results)

    _write_models(seed_dir, [
        {"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"},
    ])
    out = _seed_prune(seed_dir)
    assert "acme/scored-draft" in set(_read_models(fixtures_dir)["id"])
    assert "referenced by eval_results" in out


def test_prune_removes_sync_draft_once_seed_claims_all_its_aliases(fresh_seed_env):
    """A sync-created draft whose every alias was repointed into YAML has no
    claimant left and goes, regardless of review_status."""
    seed_dir, fixtures_dir = fresh_seed_env
    _write_models(seed_dir, [
        {"id": "acme/thing", "display_name": "Thing", "review_status": "draft"},
    ])
    _seed(seed_dir)
    aliases = _read_aliases(fixtures_dir)
    mine = aliases["canonical_id"] == "acme/thing"
    aliases.loc[mine, "strategy"] = "auto_draft"
    aliases.loc[mine, "status"] = "auto"
    _write_table(fixtures_dir, "aliases", aliases)

    _write_models(seed_dir, [
        {"id": "acme/thing-v2", "display_name": "Thing v2", "review_status": "reviewed",
         "aliases": ["Thing", "acme/thing"]},
    ])
    _seed_prune(seed_dir)
    models = _read_models(fixtures_dir)
    aliases = _read_aliases(fixtures_dir)
    assert set(models["id"]) == {"acme/thing-v2"}
    assert set(aliases.loc[aliases["raw_value"].isin(["Thing", "acme/thing"]), "canonical_id"]) == {"acme/thing-v2"}


def test_prune_keeps_sync_draft_with_a_non_seed_alias(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    _write_models(seed_dir, [
        {"id": "acme/thing", "display_name": "Thing", "review_status": "draft"},
    ])
    _seed(seed_dir)
    aliases = _read_aliases(fixtures_dir)
    mine = (aliases["canonical_id"] == "acme/thing") & (aliases["raw_value"] == "Thing")
    aliases.loc[mine, "strategy"] = "auto_draft"
    _write_table(fixtures_dir, "aliases", aliases)

    _write_models(seed_dir, [{"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"}])
    _seed_prune(seed_dir)
    assert "acme/thing" in set(_read_models(fixtures_dir)["id"])


def test_prune_drops_prefix_org_with_its_model_in_one_run(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "orgs.yaml").write_text(yaml.safe_dump([
        {"id": "acme", "display_name": "Acme", "kind": "lab", "review_status": "reviewed"},
    ]))
    _write_models(seed_dir, [
        {"id": "novelorg/some-model", "display_name": "Some Model", "review_status": "reviewed"},
    ])
    _seed(seed_dir)
    org_id = str(_read_models(fixtures_dir).set_index("id").loc["novelorg/some-model", "org_id"])
    assert org_id in set(_read_orgs(fixtures_dir)["id"].astype(str))

    _write_models(seed_dir, [{"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"}])
    _seed_prune(seed_dir)
    assert "novelorg/some-model" not in set(_read_models(fixtures_dir)["id"])
    assert org_id not in set(_read_orgs(fixtures_dir)["id"].astype(str))


def test_local_flag_prunes_by_default(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    runner = CliRunner()
    _write_models(seed_dir, [
        {"id": "acme/old-draft", "display_name": "Old Draft", "review_status": "draft"},
    ])
    assert runner.invoke(app, ["seed", "--local", "--seed-dir", str(seed_dir)]).exit_code == 0
    _write_models(seed_dir, [{"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"}])
    assert runner.invoke(app, ["seed", "--local", "--seed-dir", str(seed_dir)]).exit_code == 0
    assert "acme/old-draft" not in set(_read_models(fixtures_dir)["id"])
    _write_models(seed_dir, [
        {"id": "acme/old-draft", "display_name": "Old Draft", "review_status": "draft"},
    ])
    assert runner.invoke(app, ["seed", "--local", "--seed-dir", str(seed_dir)]).exit_code == 0
    _write_models(seed_dir, [{"id": "acme/keep", "display_name": "Keep", "review_status": "reviewed"}])
    r = runner.invoke(app, ["seed", "--local", "--no-prune-stale", "--seed-dir", str(seed_dir)])
    assert r.exit_code == 0, r.output
    assert "acme/old-draft" in set(_read_models(fixtures_dir)["id"])


def test_seed_alias_rows_are_deterministic(fresh_seed_env, tmp_path, monkeypatch):
    from eval_entity_registry.store import queries
    seed_dir, fixtures_dir = fresh_seed_env
    _write_models(seed_dir, [
        {"id": "acme/thing", "display_name": "Thing", "review_status": "reviewed",
         "aliases": ["zeta", "alpha", "Mid"], "scoped_aliases": {"cfg_a": ["s2", "s1"]}},
        {"id": "acme/other", "display_name": "Other", "review_status": "reviewed",
         "aliases": ["omega"]},
    ])
    _seed(seed_dir)
    first = _read_aliases(fixtures_dir)

    other_dir = tmp_path / "fixtures2"
    other_dir.mkdir()
    monkeypatch.setenv("FIXTURES_PATH", str(other_dir))
    hf_store._store = None
    _seed(seed_dir)
    second = _read_aliases(other_dir)

    volatile = ["created_at", "updated_at"]
    import pandas as pd
    pd.testing.assert_frame_equal(first.drop(columns=volatile), second.drop(columns=volatile))
    row = first[(first["raw_value"] == "s1") & (first["source_config"] == "cfg_a")].iloc[0]
    assert row["id"] == queries.seed_alias_id("model", "s1", "cfg_a")
    assert list(first.loc[first["canonical_id"] == "acme/thing", "raw_value"]) == [
        "Mid", "Thing", "acme/thing", "alpha", "zeta", "s2", "s1",
    ]


def test_derived_alias_id_falls_back_when_a_rejected_row_holds_it(fresh_seed_env):
    seed_dir, fixtures_dir = fresh_seed_env
    _write_models(seed_dir, [
        {"id": "acme/thing", "display_name": "Thing", "review_status": "reviewed",
         "aliases": ["nickname"]},
    ])
    _seed(seed_dir)
    aliases = _read_aliases(fixtures_dir)
    mine = aliases["raw_value"] == "nickname"
    derived = aliases.loc[mine, "id"].iloc[0]
    aliases.loc[mine, "status"] = "rejected"
    _write_table(fixtures_dir, "aliases", aliases)

    _seed(seed_dir)
    aliases = _read_aliases(fixtures_dir)
    rows = aliases[aliases["raw_value"] == "nickname"]
    assert sorted(rows["status"]) == ["confirmed", "rejected"]
    assert rows["id"].nunique() == 2
    assert derived in set(rows["id"])
