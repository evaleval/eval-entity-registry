"""core.yaml `skip_ids` drops an id from core and sources alike;
`skip_source_ids` drops it from sources only, so a source's bad aliases
cannot re-merge onto the curated entry. Runs the real seed CLI into a tmp dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from eval_entity_registry.cli import app


def _write(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def test_skip_ids_and_skip_source_ids(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    monkeypatch.setenv("FIXTURES_PATH", str(fixtures))

    _write(seed / "orgs.yaml", [{"id": "testorg", "display_name": "Test Org", "kind": "lab"}])

    # One id to drop entirely, one whose bad alias core suppresses, one to keep.
    _write(seed / "models" / "sources" / "z.generated.yaml", [
        {"id": "testorg/dropme", "org_id": "testorg", "aliases": ["dropme-alias"]},
        {"id": "testorg/curated", "org_id": "testorg", "aliases": ["bad-source-alias"]},
        {"id": "testorg/keep", "org_id": "testorg", "aliases": ["keep-alias"]},
    ])
    _write(seed / "models" / "core.yaml", {
        "skip_ids": ["testorg/dropme"],
        "skip_source_ids": ["testorg/curated"],
        "entries": [
            {"id": "testorg/curated", "org_id": "testorg", "aliases": ["good-alias"]},
        ],
    })

    res = CliRunner().invoke(app, ["seed", "--local", "--seed-dir", str(seed)])
    assert res.exit_code == 0, res.output

    m = pd.read_parquet(fixtures / "canonical_models.parquet")
    ids = set(m["id"])
    assert "testorg/dropme" not in ids, "skip_ids did not drop the id"
    assert "testorg/keep" in ids and "testorg/curated" in ids

    aliases = pd.read_parquet(fixtures / "aliases.parquet")
    curated_aliases = set(
        aliases[aliases["canonical_id"] == "testorg/curated"]["raw_value"]
    )
    assert "bad-source-alias" not in curated_aliases, "skip_source_ids did not suppress source alias"
    assert "good-alias" in curated_aliases, "curated alias missing"
    assert "dropme-alias" not in set(aliases["raw_value"])
