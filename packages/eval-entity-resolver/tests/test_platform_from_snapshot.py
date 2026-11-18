"""Host-token platform attribution reads the registry snapshot the resolver
is built over, so it cannot depend on where the package is installed."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import eval_entity_resolver
from eval_entity_resolver import Resolver

_ALIAS_COLUMNS = (
    "id", "raw_value", "entity_type", "canonical_id", "source_config",
    "source_field", "status", "strategy", "confidence", "notes",
    "created_at", "updated_at",
)


def _write_snapshot(root: Path, platforms: list[dict] | None) -> Path:
    """A registry snapshot holding one model alias and, unless `platforms` is
    None, a `canonical_inference_platforms` table."""
    root.mkdir(parents=True, exist_ok=True)
    alias = dict.fromkeys(_ALIAS_COLUMNS)
    alias.update(
        id="a1", raw_value="meta/llama-3-8b", entity_type="model",
        canonical_id="meta/llama-3-8b", status="confirmed",
        strategy="confirmed", confidence=1.0,
    )
    pd.DataFrame([alias]).to_parquet(root / "aliases.parquet")
    if platforms is not None:
        pd.DataFrame(
            [{"id": p["id"], "aliases": json.dumps(p["aliases"])} for p in platforms],
            columns=["id", "aliases"],
        ).to_parquet(root / "canonical_inference_platforms.parquet")
    return root


# An id the seed YAML does not use, so only the snapshot can produce it.
_SNAPSHOT_PLATFORMS = [{"id": "snapshot-together", "aliases": ["Together", "together/"]}]


def test_platform_comes_from_the_snapshot_table(tmp_path):
    resolver = Resolver.from_parquet(_write_snapshot(tmp_path, _SNAPSHOT_PLATFORMS))
    result = resolver.resolve("together/meta/llama-3-8b", "model")
    assert result.canonical_id == "meta/llama-3-8b"
    assert result.inference_platform == "snapshot-together"


def test_platform_resolves_when_installed_outside_the_repo(tmp_path):
    """A wheel install holds only the package directory. Copy it out of the
    repo and resolve from there."""
    site = tmp_path / "site"
    shutil.copytree(
        Path(eval_entity_resolver.__file__).parent, site / "eval_entity_resolver"
    )
    snapshot = _write_snapshot(tmp_path / "snapshot", _SNAPSHOT_PLATFORMS)
    script = (
        "import sys, eval_entity_resolver as m\n"
        "from eval_entity_resolver import Resolver\n"
        "r = Resolver.from_parquet(sys.argv[1])\n"
        "print(m.__file__)\n"
        "print(r.resolve('together/meta/llama-3-8b', 'model').inference_platform)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(snapshot)],
        env={"PYTHONPATH": str(site), "PATH": ""},
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert Path(out[0]).is_relative_to(site), out[0]
    assert out[1] == "snapshot-together"


@pytest.mark.parametrize(
    "platforms",
    [None, [], [{"id": "togetherai", "aliases": []}]],
    ids=["table missing", "table empty", "no host tokens"],
)
def test_from_parquet_refuses_a_snapshot_without_host_tokens(tmp_path, platforms):
    snapshot = _write_snapshot(tmp_path, platforms)
    with pytest.raises(ValueError, match="canonical_inference_platforms"):
        Resolver.from_parquet(snapshot)
