"""Tests for the hub-stats refresh script — synthetic rows, no network."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from conftest import load_script_module


@pytest.fixture
def mod():
    return load_script_module("refresh_from_hub_stats")


# Mini fixtures mimicking the registry's shape — bypass loading from disk
# so the tests don't depend on the live seed contents.
ORG_ALIAS_MAP = {
    "meta": "meta",
    "meta-llama": "meta",
    "alibaba": "alibaba",
    "qwen": "alibaba",
    "anthropic": "anthropic",
    "huggingface": "huggingface",
    "huggingfaceh4": "huggingface",
    "nous-research": "nous-research",
    "nousresearch": "nous-research",
}

# Existing canonicals our script knows about. Keys are *normalized* forms
# (the `_normalize` helper collapses ./-/_/:/ all to single dashes).
ALIASES_TO_CANONICAL = {
    # Registry canonicals are the real HF repo ids (org never folded
    # into the id). Keys are the normalized form of those real ids.
    "meta-llama-llama-3-1-70b": "meta-llama/Llama-3.1-70B",
    "meta-llama-llama-3-1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-qwen2-5-7b": "Qwen/Qwen2.5-7B",
}


# Two-tier HF-org -> developer slug map (lowercase keys), mirroring the
# `hf_to_dev` the refresh script builds from `_ORG_ALIASES` + `orgs.yaml.hf_org`.
HF_TO_DEV = {
    "meta-llama": "meta",
    "qwen": "alibaba",
}


def test_hf_id_to_canonical_cased_maps_known_org_alias(mod):
    """canonical_id is the real HF repo id verbatim (org never folded
    into the id); only org_id re-maps to the developer slug `meta`."""
    cid, org = mod.hf_id_to_canonical_cased("meta-llama/Llama-3.1-70B", HF_TO_DEV)
    assert cid == "meta-llama/Llama-3.1-70B"
    assert org == "meta"


def test_hf_id_to_canonical_cased_unknown_org_keeps_hf_casing(mod):
    """An author not in the two-tier map stays as its own HF-cased org id —
    case preserved on BOTH segments (the community default)."""
    cid, org = mod.hf_id_to_canonical_cased("Random-Uploader/Some-Model", HF_TO_DEV)
    assert cid == "Random-Uploader/Some-Model"
    assert org == "Random-Uploader"


def test_hf_id_to_canonical_cased_no_slash_uses_unknown_placeholder(mod):
    """No org prefix → `unknown/...` placeholder, name casing preserved."""
    cid, org = mod.hf_id_to_canonical_cased("Standalone-Model-Name", HF_TO_DEV)
    assert cid == "unknown/Standalone-Model-Name"
    assert org == "unknown"


def test_coerce_date_handles_datetime_str_none(mod):
    assert mod._coerce_date(datetime(2025, 11, 1, 10, 30)) == "2025-11-01"
    assert mod._coerce_date("2025-11-01T10:30:00") == "2025-11-01"
    assert mod._coerce_date(None) is None


def test_extract_license_from_dict(mod):
    assert mod._extract_license({"license": "apache-2.0", "tags": []}) == "apache-2.0"


def test_extract_license_from_json_string(mod):
    assert mod._extract_license('{"license": "mit"}') == "mit"


def test_extract_license_returns_none_when_missing(mod):
    assert mod._extract_license({"tags": []}) is None
    assert mod._extract_license(None) is None


def test_filter_useful_tags(mod):
    tags = [
        "transformers", "safetensors", "deepseek_v4", "text-generation",
        "conversational", "license:mit", "eval-results", "endpoints_compatible",
        "8-bit", "fp8", "region:us", "en", "zh",
    ]
    out = mod._filter_useful_tags(tags)
    # Useful: license:mit, eval-results, safetensors (format), en, zh
    assert "license:mit" in out
    assert "eval-results" in out
    assert "safetensors" in out
    assert "en" in out
    assert "zh" in out
    # Filtered out: noisy library/region markers
    assert "transformers" not in out
    assert "region:us" not in out
    assert "conversational" not in out


def test_build_entry_job_a_existing_canonical(mod):
    """Job A: hub-stats id matches an existing canonical → emit
    enrichment entry. With `baseModels: None`, no parents are extracted
    (vacuous); the `baseModels`-populated case is covered separately."""
    row = {
        "id": "meta-llama/Llama-3.1-70B",
        "author": "meta-llama",
        "createdAt": datetime(2024, 7, 14),
        "tags": ["text-generation", "eval-results", "en"],
        "cardData": {"license": "llama3.1", "library_name": "transformers"},
        "safetensors": {"total": 70_500_000_000},  # total = param count
        "baseModels": None,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": 5_000_000,
        "likes": 12_345,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert e["id"] == "meta-llama/Llama-3.1-70B"
    # No org_id stamped: the seed-time org-from-prefix rule derives (and
    # auto-creates) the developer; stamping it here would override curated
    # org-less canonicals and leave dangling FKs for unseeded prefixes.
    assert "org_id" not in e
    assert e["release_date"] == "2024-07-14"
    assert e["params_billions"] == 70.5  # safetensors.total is the param count
    assert "parents" not in e, "no baseModels in row → no parents in entry"
    assert "eval-results" in e["tags"]
    metadata = json.loads(e["metadata"])
    assert metadata["source"] == "hub_stats"
    assert metadata["license"] == "llama3.1"
    # canonical IS the real repo id -> the hf_id claim is kept.
    assert metadata["hf_id"] == "meta-llama/Llama-3.1-70B"


def test_build_entry_propagates_resolvable_parents(mod):
    """When hub-stats `baseModels` references an HF id that resolves to
    one of our canonicals, the entry carries `parents` and
    `lineage_origin_model_org_id`. The seed loader's `_merge_into` unions
    these with any curated parents in core.yaml — see
    test_parents_union_by_id_across_sources for the cross-source merge
    semantics."""
    row = {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "author": "meta-llama",
        "createdAt": datetime(2024, 7, 23),
        "tags": ["text-generation"],
        "cardData": {"license": "llama3.1"},
        "safetensors": {"total": 141_000_000_000},
        "baseModels": {
            "relation": "finetune",
            "models": [{"id": "meta-llama/Llama-3.1-70B"}],
        },
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": 1_000_000,
        "likes": 999,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert e["id"] == "meta-llama/Llama-3.1-70B-Instruct"
    assert e["parents"] == [
        {"id": "meta-llama/Llama-3.1-70B", "relationship": "finetune"}
    ]
    assert e["lineage_origin_model_org_id"] == "meta"


@pytest.mark.parametrize(
    ("parent_id", "canonical_org"),
    [
        ("CohereLabs/Command-R", "cohere"),
        ("ibm-granite/Granite-3B", "ibm"),
    ],
)
def test_build_entry_uses_complete_org_map_for_lineage_origin(
    mod, parent_id, canonical_org
):
    """The bulk refresh uses the shared HF-prefix map, not only aliases that
    happen to be representable in the lightweight YAML normalizer. This keeps
    lineage org ids stable across the hub-stats and models.dev crons.
    """
    child_id = "example/Derived-Model"
    aliases = {
        mod._normalize(child_id): child_id,
        mod._normalize(parent_id): parent_id,
    }
    row = {
        "id": child_id,
        "author": "example",
        "createdAt": datetime(2026, 1, 1),
        "tags": [],
        "cardData": None,
        "safetensors": None,
        "baseModels": {
            "relation": "finetune",
            "models": [{"id": parent_id}],
        },
        "library_name": None,
        "pipeline_tag": None,
        "downloadsAllTime": None,
        "likes": None,
    }
    hf_to_dev = {
        "example": "example",
        "coherelabs": "cohere",
        "ibm-granite": "ibm",
    }

    entry = mod.build_entry(
        row,
        org_alias_map={},
        aliases_to_canonical=aliases,
        hf_to_dev=hf_to_dev,
    )

    assert entry is not None
    assert entry["lineage_origin_model_org_id"] == canonical_org


def test_build_entry_drops_unresolvable_parents(mod):
    """`baseModels` pointing at an HF id we don't track yields no
    parents — dangling edges would break the lineage graph."""
    row = {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "author": "meta-llama",
        "createdAt": datetime(2024, 7, 23),
        "tags": [],
        "cardData": None,
        "safetensors": None,
        "baseModels": {
            "relation": "finetune",
            "models": [{"id": "some-org/model-we-dont-track"}],
        },
        "library_name": None,
        "pipeline_tag": None,
        "downloadsAllTime": None,
        "likes": None,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert "parents" not in e
    assert "lineage_origin_model_org_id" not in e


def test_build_entry_skips_unknown_hf_ids(mod):
    """Backfill-only: hub-stats rows whose HF id isn't already an alias
    on one of our canonicals are skipped. The lineage-descendant
    pre-load (community quants/finetunes) is deferred."""
    row = {
        "id": "casperhansen/llama-3.3-70b-instruct-awq",  # not in our aliases
        "author": "casperhansen",
        "createdAt": datetime(2024, 12, 6),
        "tags": ["safetensors"],
        "cardData": None,
        "safetensors": None,
        "library_name": None,
        "pipeline_tag": None,
        "downloadsAllTime": None,
        "likes": None,
    }
    assert mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL) is None


def test_build_entry_uses_registry_canonical_id_not_slugified(mod):
    """When the registry's canonical id keeps dots (`alibaba/qwen2.5-7b`)
    but the HF id has dashes only (`Qwen/Qwen2.5-7B`), the emitted entry
    must use the registry's spelling so the seed merge lands on the
    right row."""
    row = {
        "id": "Qwen/Qwen2.5-7B",
        "author": "Qwen",
        "createdAt": datetime(2024, 9, 16),
        "tags": ["en"],
        "cardData": {"license": "apache-2.0"},
        "safetensors": None,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": 1000,
        "likes": 50,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    # Registry canonical = the real HF repo id; build_entry uses the
    # registry's exact spelling (from aliases_to_canonical), not a re-slugify.
    assert e["id"] == "Qwen/Qwen2.5-7B"
    assert "org_id" not in e


def test_build_entry_drops_hf_id_claim_when_canonical_is_not_the_repo(mod):
    """When the enrichment lands on a registry canonical that is NOT the real
    HF repo id (an org-slug mint, e.g. `deepseek/deepseek-vl2`) and the real
    id is no canonical anywhere, `metadata.hf_id` is dropped — keeping the
    claim would trip the canonical-id-equals-hf-id gate (the canonical id
    404s on HF). The rest of the metadata survives."""
    aliases = {
        "deepseek-deepseek-vl2": "deepseek/deepseek-vl2",
        "deepseek-ai-deepseek-vl2": "deepseek/deepseek-vl2",
    }
    row = {
        "id": "deepseek-ai/deepseek-vl2",
        "author": "deepseek-ai",
        "createdAt": datetime(2024, 12, 13),
        "tags": ["image-text-to-text"],
        "cardData": {"license": "other"},
        "safetensors": None,
        "library_name": "transformers",
        "pipeline_tag": "image-text-to-text",
        "downloadsAllTime": 174_382,
        "likes": 385,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, aliases)
    assert e is not None
    assert e["id"] == "deepseek/deepseek-vl2"
    meta = json.loads(e["metadata"])
    assert "hf_id" not in meta, "hf_id claim on a non-repo canonical trips the gate"
    assert meta["source"] == "hub_stats"
    assert meta["downloads_all_time"] == 174_382


# ---------------------------------------------------------------------------
# Carry-forward: a wholesale regen never deletes committed enrichment for a
# model that fell out of today's fetch.
# ---------------------------------------------------------------------------

def _committed_entry(cid: str, **extra) -> dict:
    e = {
        "id": cid,
        "org_id": cid.split("/")[0],  # pre-no-stamping committed shape
        "release_date": "2024-01-01",
        "tags": ["en"],
        "aliases": [],
        "metadata": json.dumps({"hf_id": cid, "source": "hub_stats"}, sort_keys=True),
        "review_status": "reviewed",
        "params_billions": 7.0,
        "open_weights": True,
    }
    e.update(extra)
    return e


def test_carry_forward_retains_entry_missing_from_fetch(mod):
    """A committed id absent from today's fetch is retained with its
    enrichment intact, tagged `upstream_status: removed`, and its legacy
    org_id dropped (the seed-time prefix rule recovers the org)."""
    committed = [_committed_entry("openai-community/gpt2-cftest")]
    fresh = [_committed_entry("dreamgen/other-model-cftest")]
    fresh[0].pop("org_id")
    out = mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in out}
    kept = by_id["openai-community/gpt2-cftest"]
    assert kept["release_date"] == "2024-01-01"
    assert kept["params_billions"] == 7.0
    assert kept["open_weights"] is True
    assert "org_id" not in kept
    meta = json.loads(kept["metadata"])
    assert meta["upstream_status"] == "removed"
    assert meta["hf_id"] == "openai-community/gpt2-cftest"  # keys preserved


def test_carry_forward_reappearance_replaces_and_clears_tag(mod):
    """An id that reappears in the fetch is replaced by the fresh entry —
    exactly one record, no `upstream_status` tag left behind."""
    committed = [_committed_entry(
        "openai-community/gpt2-cftest",
        metadata=json.dumps({"hf_id": "x", "upstream_status": "removed"}, sort_keys=True),
    )]
    fresh = [_committed_entry("openai-community/gpt2-cftest", release_date="2024-06-01")]
    fresh[0].pop("org_id")
    out = mod._carry_forward_committed(fresh, committed)
    assert len(out) == 1
    (e,) = out
    assert e["release_date"] == "2024-06-01"
    assert "upstream_status" not in json.loads(e["metadata"])


def test_carry_forward_deterministic_and_stable_over_own_output(mod):
    """Same inputs -> same output, and re-running over its own output is a
    fixed point (the daily-cron stability property)."""
    committed = [
        _committed_entry("zz-org/model-b-cftest"),
        _committed_entry("aa-org/model-a-cftest"),
    ]
    fresh = [_committed_entry("mm-org/model-m-cftest")]
    fresh[0].pop("org_id")
    first = mod._carry_forward_committed([dict(e) for e in fresh], committed)
    again = mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert first == again
    assert [e["id"] for e in first] == sorted(e["id"] for e in first)
    # Fixed point: carrying forward the merged output retains the same rows
    # (the removed tag is already present; re-tagging is byte-identical).
    second = mod._carry_forward_committed([dict(e) for e in fresh], first)
    assert second == first


def test_carry_forward_malformed_committed_metadata_does_not_crash(mod):
    """A malformed committed metadata string degrades to {} + the tag,
    never an exception (the cron must not crash on a bad committed row)."""
    committed = [_committed_entry("bad-org/model-x-cftest", metadata="{broken")]
    out = mod._carry_forward_committed([], committed)
    (e,) = out
    assert json.loads(e["metadata"]) == {"upstream_status": "removed"}


def test_approx_params_billions_from_safetensors_total(mod):
    """safetensors.total IS the parameter count → billions = total / 1e9."""
    assert mod._approx_params_billions({"total": 8_000_000_000}) == 8.0
    assert mod._approx_params_billions({"total": 0}) is None
    assert mod._approx_params_billions(None) is None


def test_build_entry_sets_open_weights_when_safetensors_present(mod):
    """A row with safetensors data is downloadable HF weights → open."""
    row = {
        "id": "Qwen/Qwen2.5-7B",
        "author": "Qwen",
        "createdAt": datetime(2024, 9, 16),
        "tags": ["en"],
        "cardData": None,
        "safetensors": {"total": 14_000_000_000},
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": None, "likes": None,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert e["open_weights"] is True


def test_build_entry_skips_open_weights_when_no_artifacts(mod):
    """A row with neither safetensors nor gguf data → don't infer
    open_weights (could be a metadata-only repo, deleted weights, etc.).
    Leaving it absent lets the seed loader default to NA."""
    row = {
        "id": "Qwen/Qwen2.5-7B",
        "author": "Qwen",
        "createdAt": datetime(2024, 9, 16),
        "tags": [],
        "cardData": None,
        "safetensors": None,
        "library_name": None, "pipeline_tag": None,
        "downloadsAllTime": None, "likes": None,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert "open_weights" not in e


# ---------------------------------------------------------------------------
# --live path: pinned snapshot download + sha256 watermark (offline; the
# snapshot helpers are monkeypatched to a tiny parquet cut from the frozen
# cache). Every file write is redirected to tmp_path; nothing touches the
# real seed or the process environment.
# ---------------------------------------------------------------------------

import duckdb  # noqa: E402

from eval_entity_registry.services.hub_stats_snapshot import HubStatsSnapshot  # noqa: E402

_FROZEN = __import__("pathlib").Path(__file__).resolve().parent.parent / "curation" / "hub_stats_frozen.parquet"
_SHA = "c2dc24e395df9df1052f067fcda6314e86ecd84c"
_ETAG = "7fd4aaf31072a5f502622a629749177c03e7d620f8488977d8003a403e9a4974"


class _RecordingCon:
    """Wrap a DuckDB connection and record every SQL statement."""

    def __init__(self, inner):
        self._inner = inner
        self.sql: list[str] = []

    def execute(self, sql, *a, **k):
        self.sql.append(sql)
        return self._inner.execute(sql, *a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def live_env(mod, monkeypatch, tmp_path):
    """Redirect outputs to tmp, cut a small parquet from the frozen cache and
    patch resolve/download to hand it back as the 'snapshot'."""
    small = tmp_path / "models.parquet"
    duckdb.connect().execute(
        f"COPY (SELECT * FROM read_parquet('{_FROZEN}') LIMIT 25) TO '{small}' (FORMAT PARQUET)"
    )
    snap = HubStatsSnapshot("cfahlgren1/hub-stats", _SHA, "models.parquet", _ETAG, small.stat().st_size)
    calls = {"resolve": 0, "download": 0}

    def fake_resolve(token=None, **_):
        calls["resolve"] += 1
        return snap

    def fake_download(s, token=None, **_):
        calls["download"] += 1
        return HubStatsSnapshot(s.repo_id, s.revision, s.filename, s.sha256, s.size, local_path=small)

    monkeypatch.setattr(mod, "resolve_snapshot", fake_resolve)
    monkeypatch.setattr(mod, "download_snapshot", fake_download)
    monkeypatch.setattr(mod, "MODELS_OUT_PATH", tmp_path / "hub_stats.generated.yaml")
    monkeypatch.setattr(mod, "STATE_PATH", tmp_path / "hub_stats.state.json")
    cons: list[_RecordingCon] = []
    real_connect = duckdb.connect

    def recording_connect(*a, **k):
        c = _RecordingCon(real_connect(*a, **k))
        cons.append(c)
        return c

    monkeypatch.setattr(mod.duckdb, "connect", recording_connect)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv(mod.HUB_STATS_LOCAL_PARQUET_ENV, raising=False)
    return dict(snap=snap, small=small, calls=calls, cons=cons, tmp=tmp_path)


def test_live_reads_downloaded_snapshot_locally_and_writes_sha_watermark(mod, live_env, monkeypatch):
    monkeypatch.setattr("sys.argv", ["refresh_from_hub_stats.py", "--live"])
    assert mod.main() == 0
    assert live_env["calls"] == {"resolve": 1, "download": 1}
    # The query ran against the downloaded local file; no httpfs, no secret.
    sql = "\n".join(s for c in live_env["cons"] for s in c.sql)
    assert str(live_env["small"]) in sql
    assert "httpfs" not in sql and "CREATE SECRET" not in sql
    assert "huggingface.co" not in sql
    # Outputs landed in tmp, keyed by the snapshot's LFS sha256.
    assert mod.MODELS_OUT_PATH.exists()
    state = json.loads(mod.STATE_PATH.read_text())
    assert state[mod.STATE_SHA_KEY] == _ETAG
    assert isinstance(state[mod.STATE_ROWS_KEY], list) and state[mod.STATE_ROWS_KEY]
    assert "parquet_etag" not in state
    # Process env untouched.
    import os
    assert mod.HUB_STATS_LOCAL_PARQUET_ENV not in os.environ


def test_live_short_circuits_on_unchanged_sha_before_download(mod, live_env, monkeypatch):
    monkeypatch.setattr("sys.argv", ["refresh_from_hub_stats.py", "--live"])
    assert mod.main() == 0  # first run: writes the watermark
    assert live_env["calls"]["download"] == 1

    def boom(*a, **k):
        raise AssertionError("download must not run on a no-op cycle")

    monkeypatch.setattr(mod, "download_snapshot", boom)
    n_cons = len(live_env["cons"])
    assert mod.main() == 0  # same sha + same candidates + YAML present → no-op
    assert live_env["calls"]["resolve"] == 2
    assert len(live_env["cons"]) == n_cons  # no DuckDB work either

    # --force bypasses the watermark (and so would hit the download).
    monkeypatch.setattr("sys.argv", ["refresh_from_hub_stats.py", "--live", "--force"])
    with pytest.raises(AssertionError, match="must not run"):
        mod.main()


def test_legacy_etag_state_is_not_a_watermark(mod, live_env, monkeypatch):
    mod.STATE_PATH.write_text(json.dumps({"parquet_etag": "c215dbd6", "rows_checked_at_etag": []}))
    assert mod.load_state() == {}
    mod.STATE_PATH.write_text(json.dumps({mod.STATE_SHA_KEY: _ETAG, mod.STATE_ROWS_KEY: ["a/b"]}))
    assert mod.load_state() == {mod.STATE_SHA_KEY: _ETAG, mod.STATE_ROWS_KEY: ["a/b"]}
    mod.STATE_PATH.write_text(json.dumps({mod.STATE_SHA_KEY: 5, mod.STATE_ROWS_KEY: ["a/b"]}))
    assert mod.load_state() == {}
    # A legacy-keyed file must NOT short-circuit a live run.
    mod.STATE_PATH.write_text(json.dumps({"parquet_etag": _ETAG, "rows_checked_at_etag": []}))
    monkeypatch.setattr("sys.argv", ["refresh_from_hub_stats.py", "--live"])
    assert mod.main() == 0
    assert live_env["calls"]["download"] == 1


def test_offline_default_reads_frozen_cache_and_writes_no_watermark(mod, live_env, monkeypatch):
    """No --live: the committed frozen cache is the source; no snapshot
    resolution, no download, no state file."""
    monkeypatch.setattr("sys.argv", ["refresh_from_hub_stats.py"])
    assert mod.main() == 0
    assert live_env["calls"] == {"resolve": 0, "download": 0}
    sql = "\n".join(s for c in live_env["cons"] for s in c.sql)
    assert str(mod.FROZEN_CACHE_PATH) in sql
    assert not mod.STATE_PATH.exists()
