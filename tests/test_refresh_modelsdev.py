"""The models.dev refresh script: family derivation, provider filtering,
snapshot and variant chains, open_weights and release_date, TEE prefixes.
Synthetic data, no network."""
import json

import pytest

from conftest import load_script_module


@pytest.fixture
def mod():
    return load_script_module("refresh_from_modelsdev")


SYNTHETIC_API = {
    # Author provider. `name` carries the lab's spelling; `id` is the slug.
    "anthropic": {
        "id": "anthropic", "name": "Anthropic", "doc": "...", "env": [], "npm": "",
        "models": {
            "claude-opus-4-5": {
                "id": "claude-opus-4-5", "name": "Claude Opus 4.5",
                "family": "claude-opus", "release_date": "2025-11-01",
                "open_weights": False, "knowledge": "2025-05",
            },
            "claude-opus-4-5-20251101": {
                "id": "claude-opus-4-5-20251101", "name": "Claude Opus 4.5",
                "family": "claude-opus", "release_date": "2025-11-01",
                "open_weights": False,
            },
            "claude-opus-4-7": {
                "id": "claude-opus-4-7", "name": "Claude Opus 4.7",
                "family": "claude-opus", "open_weights": False,
            },
            # Multi-segment mirror id: must be skipped.
            "anthropic/claude-opus-4-5": {
                "id": "anthropic/claude-opus-4-5", "name": "(mirror)", "family": "claude-opus",
            },
        },
    },
    # Not in PROVIDER_TO_ORG.
    "fireworks-ai": {
        "id": "fireworks-ai", "name": "Fireworks", "models": {
            "some-model": {"id": "some-model", "name": "X", "family": "x"},
        },
    },
}


KNOWN_ORGS = {"anthropic", "openai", "google", "meta", "deepseek"}


def test_family_for_uses_name_field_with_dots(mod):
    """`name` carries the lab's spelling; `id` has slug-mangled separators."""
    assert mod._family_for({
        "id": "qwen2-5-7b-instruct",
        "name": "Qwen2.5 7B Instruct",
    }) == "qwen2.5-7b"


def test_family_for_strips_date_suffix(mod):
    """A dated snapshot id folds to its family."""
    assert mod._family_for({"id": "claude-opus-4-5-20251101"}) == "claude-opus-4-5"


def test_family_for_strips_latest_suffix(mod):
    assert mod._family_for({"id": "claude-3-5-haiku-latest"}) == "claude-3-5-haiku"


def test_family_for_latest_pointer_keeps_id_stem_when_name_rotates(mod):
    """A rolling `-latest` pointer uses its `id` stem when the lab's marketing
    `name` diverges, while still keeping dotted spellings when stems agree."""
    assert mod._family_for({
        "id": "ministral-14b-latest",
        "name": "Ministral 3 14B (latest)",
    }) == "ministral-14b"
    assert mod._family_for({
        "id": "gpt-daybreak-blue-latest",
        "name": "Daybreak Blue",
    }) == "gpt-daybreak-blue"
    assert mod._family_for({
        "id": "claude-3-5-haiku-latest",
        "name": "Claude 3.5 Haiku (latest)",
    }) == "claude-3.5-haiku"


def test_generate_latest_pointer_with_rotated_name_stays_under_id_family(mod, monkeypatch):
    """Rotating a `-latest` record's display name must keep the pointer under
    its `id` family and never emit a childless root under the new name."""
    api = {
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "ministral-14b-2512": {
                    "id": "ministral-14b-2512", "name": "Ministral 3 14B",
                    "family": "ministral", "release_date": "2025-12-02",
                    "open_weights": True,
                    "canonical_model_id": "mistral/ministral-3-14b-instruct-2512",
                },
                "ministral-14b-latest": {
                    "id": "ministral-14b-latest", "name": "Ministral 3 14B (latest)",
                    "family": "ministral", "release_date": "2025-12-02",
                    "open_weights": True,
                    "canonical_model_id": "mistral/ministral-3-14b-instruct-2512",
                },
            },
        },
    }
    monkeypatch.setattr(
        mod, "_HF_AUTHORITY",
        {"mistralai": {"ministral 3 14b instruct 2512": "mistralai/Ministral-3-14B-Instruct-2512"}},
    )
    out, missing = mod._generate_models(api, {"mistralai"})
    assert missing == []
    out = mod._finalize_entries(out)
    by_id = {e["id"]: e for e in out}
    assert "mistralai/ministral-3-14b" not in by_id
    assert "mistralai/ministral-14b" in by_id
    latest = by_id["mistralai/ministral-14b-latest"]
    assert latest["parents"] == [{
        "id": "mistralai/ministral-14b",
        "relationship": "variant",
        "axis": "version",
    }]
    assert "ministral-14b-latest" in latest["aliases"]
    assert "Ministral 3 14B" in by_id["mistralai/Ministral-3-14B-Instruct-2512"]["aliases"]


def test_generate_dotted_id_latest_preserves_child_for_matching_and_divergent_names(mod):
    """Dotted snapshot/latest ids match their dotted family root during prefix
    extraction so the child canonical is emitted for both matching and rotated
    marketing names."""
    api = {
        "openai": {
            "id": "openai", "name": "OpenAI", "models": {
                "foo-1.2-latest": {
                    "id": "foo-1.2-latest", "name": "Bar 2",
                    "family": "foo", "open_weights": False,
                },
                "baz-3.4-latest": {
                    "id": "baz-3.4-latest", "name": "Baz 3.4 (latest)",
                    "family": "baz", "open_weights": False,
                },
            },
        },
    }
    out, missing = mod._generate_models(api, KNOWN_ORGS)
    assert missing == []
    by_id = {e["id"]: e for e in mod._finalize_entries(out)}
    assert "openai/foo-1.2" in by_id
    assert by_id["openai/foo-1.2-latest"]["parents"] == [{
        "id": "openai/foo-1.2", "relationship": "variant", "axis": "version",
    }]
    assert "foo-1.2-latest" in by_id["openai/foo-1.2-latest"]["aliases"]
    assert "openai/baz-3.4" in by_id
    assert by_id["openai/baz-3.4-latest"]["parents"] == [{
        "id": "openai/baz-3.4", "relationship": "variant", "axis": "version",
    }]
    assert "baz-3.4-latest" in by_id["openai/baz-3.4-latest"]["aliases"]


def test_seeded_ministral_and_daybreak_latest_resolutions():
    """Seeded fixtures preserve the retired tier3/name-root forms as aliases on
    the new -latest families and keep Ministral 3 14B on the HF Instruct repo."""
    from pathlib import Path
    from eval_entity_resolver import Resolver

    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    if not (fixtures / "aliases.parquet").exists():
        pytest.skip("fixtures not built; run `eval-entity-registry seed --local`")
    r = Resolver.from_parquet(str(fixtures))
    assert r.resolve("Ministral 3 14B", "model").canonical_id == "mistralai/Ministral-3-14B-Instruct-2512"
    assert r.resolve("ministral-14b-latest", "model").canonical_id == "mistralai/ministral-14b-latest"
    assert r.resolve("mistral/ministral-14b-latest", "model").canonical_id == "mistralai/ministral-14b-latest"
    assert r.resolve("gpt-daybreak-blue-latest", "model").canonical_id == "openai/gpt-daybreak-blue-latest"
    assert r.resolve("Daybreak Blue", "model").canonical_id == "openai/gpt-daybreak-blue-latest"
    assert r.resolve("openai/daybreak-blue", "model").canonical_id == "openai/gpt-daybreak-blue"
    assert r.resolve("gpt-daybreak-red-latest", "model").canonical_id == "openai/gpt-daybreak-red-latest"
    assert r.resolve("Daybreak Red", "model").canonical_id == "openai/gpt-daybreak-red-latest"
    assert r.resolve("openai/daybreak-red", "model").canonical_id == "openai/gpt-daybreak-red"


def test_family_for_strips_training_stage_suffixes(mod):
    """Training-stage suffixes fold into the family."""
    assert mod._family_for({"id": "qwen3-30b-instruct"}) == "qwen3-30b"
    assert mod._family_for({"id": "yi-34b-chat"}) == "yi-34b"
    assert mod._family_for({"id": "gemma-2-9b-it"}) == "gemma-2-9b"
    assert mod._family_for({"id": "olmo-7b-base"}) == "olmo-7b"


def test_family_for_passes_through_when_no_date(mod):
    assert mod._family_for({"id": "gpt-5-mini"}) == "gpt-5-mini"


def test_generate_filters_to_known_providers(mod):
    """A provider not in PROVIDER_TO_ORG never becomes an org_id."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    org_ids = {e["org_id"] for e in out}
    assert "anthropic" in org_ids
    assert all(e["org_id"] != "fireworks-ai" for e in out)


def test_generate_skips_mirror_entries(mod):
    """Only true `org/a/b` mirrors are dropped; a model with no extractable
    org mints as a bare org-less slug tagged `org-unknown`."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    for e in out:
        assert e["id"].count("/") <= 1, f"expected <=1 slash, got {e['id']!r}"
    by_id = {e["id"]: e for e in out}
    assert "some-model" in by_id
    assert by_id["some-model"]["org_id"] is None
    assert "org-unknown" in by_id["some-model"]["tags"]


def test_generate_dated_snapshot_becomes_child_canonical(mod):
    """A dated snapshot is its own canonical with a version-axis edge to the
    family root; its id takes the family's dotted spelling and the dashed
    models.dev id stays as an alias."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    by_id = {e["id"]: e for e in out}
    root = by_id["anthropic/claude-opus-4.5"]
    assert root["parents"] == []
    snap_id = "anthropic/claude-opus-4.5-20251101"
    assert snap_id in by_id, f"missing snapshot canonical {snap_id!r} in output"
    snap = by_id[snap_id]
    assert snap["parents"] == [{
        "id": "anthropic/claude-opus-4.5",
        "relationship": "variant",
        "axis": "version",
    }]
    assert snap["release_date"] == "2025-11-01"
    assert "claude-opus-4-5-20251101" in snap["aliases"]
    assert "anthropic/claude-opus-4-5-20251101" in snap["aliases"]
    assert "claude-opus-4-5-20251101" in json.loads(root["metadata"]).get("snapshots", [])


def test_generate_emits_mode_variants_as_children(mod):
    """A dated -instruct snapshot yields root -> instruct (training_stage)
    -> dated leaf (version); `-instruct` is a training stage, not a mode."""
    api = {
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-9b-instruct-v0-3": {
                    "id": "mistral-9b-instruct-v0-3",
                    "name": "Mistral 9B Instruct v0.3",
                    "release_date": "2024-05-22",
                    "open_weights": True,
                },
            },
        },
    }
    known = {"mistralai"}
    out, _missing = mod._generate_models(api, known)
    by_id = {e["id"]: e for e in out}
    assert "mistralai/mistral-9b" in by_id, "family root missing"
    assert "mistralai/mistral-9b-instruct" in by_id, "instruct intermediate missing"
    assert "mistralai/mistral-9b-instruct-v0-3" in by_id, "leaf snapshot missing"
    assert by_id["mistralai/mistral-9b"]["parents"] == []
    assert by_id["mistralai/mistral-9b-instruct"]["parents"] == [{
        "id": "mistralai/mistral-9b", "relationship": "variant", "axis": "training_stage",
    }]
    assert by_id["mistralai/mistral-9b-instruct-v0-3"]["parents"] == [{
        "id": "mistralai/mistral-9b-instruct", "relationship": "variant", "axis": "version",
    }]
    # The intermediate is anchor-only: no release_date.
    assert by_id["mistralai/mistral-9b-instruct-v0-3"]["release_date"] == "2024-05-22"
    assert by_id["mistralai/mistral-9b-instruct"]["release_date"] is None


def test_generate_keeps_distinct_major_versions(mod):
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    ids = {e["id"] for e in out}
    assert "anthropic/claude-opus-4.5" in ids
    assert "anthropic/claude-opus-4.7" in ids


def test_generate_propagates_open_weights_flag(mod):
    """open_weights round-trips from models.dev to every level of the chain."""
    api = {
        "anthropic": {
            "id": "anthropic", "name": "Anthropic", "models": {
                "claude-foo-1": {
                    "id": "claude-foo-1", "name": "Claude Foo 1",
                    "release_date": "2025-06-15", "open_weights": False,
                },
            },
        },
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-9b-instruct-v0-3": {
                    "id": "mistral-9b-instruct-v0-3",
                    "name": "Mistral 9B Instruct v0.3",
                    "release_date": "2024-05-22", "open_weights": True,
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, {"anthropic", "mistralai"})
    by_id = {e["id"]: e for e in out}
    assert by_id["anthropic/claude-foo-1"]["open_weights"] is False
    assert by_id["mistralai/mistral-9b"]["open_weights"] is True
    assert by_id["mistralai/mistral-9b-instruct"]["open_weights"] is True
    assert by_id["mistralai/mistral-9b-instruct-v0-3"]["open_weights"] is True


def test_generate_promotes_earliest_release_date(mod):
    """The family's release_date is its earliest snapshot date; None when
    the source has none."""
    api = {
        "anthropic": {
            "id": "anthropic", "name": "Anthropic", "models": {
                "claude-foo-1": {
                    "id": "claude-foo-1", "name": "Claude Foo 1",
                    "release_date": "2025-06-15",
                },
                "claude-foo-1-20250901": {
                    "id": "claude-foo-1-20250901", "name": "Claude Foo 1",
                    "release_date": "2025-09-01",
                },
                "claude-bar-1": {
                    "id": "claude-bar-1", "name": "Claude Bar 1",
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, KNOWN_ORGS)
    by_id = {e["id"]: e for e in out}
    assert by_id["anthropic/claude-foo-1"]["release_date"] == "2025-06-15"
    assert by_id["anthropic/claude-bar-1"]["release_date"] is None


def test_tee_prefix_is_serving_marker_for_org_derivation(mod):
    """`TEE/` is a serving marker, not an uploader org; the developer comes
    from the name."""
    org, rehost = mod._derive_group_org(
        [{"raw": "TEE/gemma9x-31b", "name": "Gemma 9X 31B"}], mod._dev_alias_index()
    )
    assert org == "google"
    assert rehost is None


def test_provider_alias_forms_keep_tee_prefixed_raw(mod):
    """The raw `TEE/...` spelling stays resolvable as an alias."""
    forms = mod._provider_alias_forms("TEE/gemma9x-31b", "google")
    assert "TEE/gemma9x-31b" in forms
    assert "gemma9x-31b" in forms
    assert "google/gemma9x-31b" in forms


def test_generate_tee_group_mints_under_name_vendor_with_tee_alias(mod):
    """A TEE-only group mints under the name's developer, never org `TEE`,
    with the raw spellings as aliases."""
    api = {
        "nano-gpt": {
            "id": "nano-gpt", "name": "NanoGPT", "models": {
                "TEE/gemma9x-31b": {
                    "id": "TEE/gemma9x-31b", "name": "Gemma 9X 31B",
                    "release_date": "2026-04-04", "open_weights": False,
                },
                "TEE/gemma9x-26b-uncensored": {
                    "id": "TEE/gemma9x-26b-uncensored",
                    "name": "Gemma 9X 26B Uncensored TEE", "open_weights": False,
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, KNOWN_ORGS)
    out = mod._finalize_entries(out)
    assert all(not e["id"].startswith("TEE/") for e in out)
    assert all(e.get("org_id") != "TEE" for e in out)
    by_id = {e["id"]: e for e in out}
    assert "TEE/gemma9x-31b" in by_id["google/gemma9x-31b"]["aliases"]
    assert "TEE/gemma9x-26b-uncensored" in by_id["google/gemma9x-26b-uncensored"]["aliases"]


def test_generate_reports_missing_org_ids(mod):
    """An org_id missing from known_orgs is reported, not silently minted."""
    api = {
        "anthropic": {
            "id": "anthropic", "name": "Anthropic", "doc": "...", "env": [], "npm": "",
            "models": {"claude-x": {"id": "claude-x", "name": "X", "family": "claude"}},
        },
    }
    out, missing = mod._generate_models(api, set())
    assert out == []
    assert any("anthropic" in m for m in missing)


# Override merging happens in the seed loader (`_load_models_merged`), not in
# this script; it is covered in test_cli_seed.py.
