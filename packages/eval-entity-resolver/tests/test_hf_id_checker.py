"""The injected HF id checker step and the exact-only resolve mode.

Chain under test (models):
  1. exact alias
  2. HF check, VERBATIM hits — outranks any alias mapping
  3. normalized alias
  4. HF check, NORMALIZED hits — ranks below a curated normalized alias
  5. fuzzy (resolver mode only)
  6. no_match
"""
import uuid
from datetime import datetime, timezone

import pandas as pd

from eval_entity_resolver import AliasStore, Resolver
from eval_entity_resolver.alias_store import _empty_df
from eval_entity_resolver.models import HfIdHit


def _store_with_aliases(*rows) -> AliasStore:
    """Build an AliasStore from (raw_value, entity_type, canonical_id,
    source_config, status) tuples."""
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for raw_value, entity_type, canonical_id, source_config, status in rows:
        records.append({
            "id": str(uuid.uuid4()),
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": None,
            "status": status,
            "strategy": "confirmed",
            "confidence": 1.0,
            "notes": None,
            "created_at": now,
            "updated_at": now,
        })
    df = pd.DataFrame(records) if records else _empty_df()
    return AliasStore(df)


class CountingChecker:
    def __init__(self, hits: dict):
        """`hits` maps raw_value -> HfIdHit."""
        self.hits = hits
        self.calls: list[str] = []

    def __call__(self, raw_value: str):
        self.calls.append(raw_value)
        return self.hits.get(raw_value)


def test_no_checker_behaves_classically():
    store = _store_with_aliases(
        ("acme/widget-7b", "model", "acme/widget-7b", None, "confirmed"))
    r = Resolver(store).resolve("acme/widget-7b", "model")
    assert r.canonical_id == "acme/widget-7b"
    assert r.strategy == "exact"
    assert r.hf_attestation is None


def test_verbatim_hit_beats_disagreeing_exact_alias():
    # A curated alias folds the real HF repo id into a different canonical;
    # the verbatim HF confirmation must still win.
    store = _store_with_aliases(
        ("acme/Widget-7B-FP8", "model", "acme/widget-7b", None, "confirmed"))
    checker = CountingChecker({
        "acme/Widget-7B-FP8": HfIdHit(
            hf_id="acme/Widget-7B-FP8", verbatim=True, source="hub_stats_index"),
    })
    r = Resolver(store, hf_id_checker=checker).resolve("acme/Widget-7B-FP8", "model")
    assert r.canonical_id == "acme/Widget-7B-FP8"
    assert r.strategy == "exact"
    assert r.confidence == 1.0
    assert r.resolution_source == "hub_stats_index"
    assert r.hf_attested_unregistered is True
    assert r.hf_attestation == "acme/Widget-7B-FP8"
    assert r.ancestry == []
    assert r.resolution_detail == {
        "granularity": None, "hf_repo_id": "acme/Widget-7B-FP8"}


def test_verbatim_hit_recovers_true_casing_over_byte_equal_alias():
    # A lowercased draft alias exists byte-equal to the raw value, but the
    # row is NOT HF-attested (no canonical_store at all here), so the
    # checker still runs and its HF-true casing wins.
    store = _store_with_aliases(
        ("acme/widget-7b", "model", "acme/widget-7b", None, "auto"))
    checker = CountingChecker({
        "acme/widget-7b": HfIdHit(
            hf_id="acme/Widget-7B", verbatim=True, source="hf_live"),
    })
    r = Resolver(store, hf_id_checker=checker).resolve("acme/widget-7b", "model")
    assert checker.calls == ["acme/widget-7b"]
    assert r.canonical_id == "acme/Widget-7B"
    assert r.resolution_source == "hf_live"


def test_agreeing_exact_alias_keeps_registry_result_and_stamps_attestation():
    store = _store_with_aliases(
        ("acme/Widget-7B", "model", "acme/Widget-7B", None, "confirmed"))
    checker = CountingChecker({
        "acme/Widget-7B": HfIdHit(
            hf_id="acme/Widget-7B", verbatim=True, source="hub_stats_index"),
    })
    r = Resolver(store, hf_id_checker=checker).resolve("acme/Widget-7B", "model")
    assert r.canonical_id == "acme/Widget-7B"
    assert r.strategy == "exact"
    assert r.hf_attestation == "acme/Widget-7B"
    # No canonical_store attached -> bare result, but attestation is stamped.
    assert r.hf_attested_unregistered is False


def test_checker_not_called_for_non_models_or_non_hf_shapes():
    checker = CountingChecker({})
    resolver = Resolver(_store_with_aliases(), hf_id_checker=checker)
    resolver.resolve("IFEval", "benchmark")
    resolver.resolve("plain-name", "model")
    resolver.resolve("a/b/c", "model")
    assert checker.calls == []


def test_normalized_checker_hit_ranks_below_normalized_alias():
    # The alias table carries a curated separator-variant; the checker also
    # has a normalized-tier hit for the same raw. The alias must win.
    store = _store_with_aliases(
        ("acme/widget 7b", "model", "acme/curated-target", None, "confirmed"))
    checker = CountingChecker({
        "acme/widget_7b": HfIdHit(
            hf_id="acme/Widget-7B", verbatim=False, source="hub_stats_index"),
    })
    r = Resolver(store, hf_id_checker=checker).resolve("acme/widget_7b", "model")
    assert r.canonical_id == "acme/curated-target"
    assert r.strategy == "normalized"


def test_normalized_checker_hit_wins_when_no_alias_matches():
    checker = CountingChecker({
        "qwen/qwen2.5_7b": HfIdHit(
            hf_id="Qwen/Qwen2.5-7B", verbatim=False, source="hub_stats_index"),
    })
    r = Resolver(_store_with_aliases(), hf_id_checker=checker).resolve(
        "qwen/qwen2.5_7b", "model")
    assert r.canonical_id == "Qwen/Qwen2.5-7B"
    assert r.strategy == "normalized"
    assert r.confidence == 0.95
    assert r.hf_attested_unregistered is True


def test_checker_miss_falls_through_to_no_match():
    checker = CountingChecker({})
    r = Resolver(_store_with_aliases(), hf_id_checker=checker).resolve(
        "someorg/unknown", "model")
    assert checker.calls == ["someorg/unknown"]
    assert r.canonical_id is None
    assert r.strategy == "no_match"


def test_check_hf_false_disables_checker():
    checker = CountingChecker({
        "acme/Widget-7B": HfIdHit(
            hf_id="acme/Widget-7B", verbatim=True, source="hub_stats_index"),
    })
    resolver = Resolver(_store_with_aliases(), hf_id_checker=checker)
    r = resolver.resolve("acme/Widget-7B", "model", check_hf=False)
    assert checker.calls == []
    assert r.canonical_id is None


class TestExactMode:
    def test_exact_mode_skips_fuzzy(self):
        # Quant-suffix stem strip would fuzzy-match in resolver mode.
        store = _store_with_aliases(
            ("acme/widget-7b", "model", "acme/widget-7b", None, "confirmed"))
        resolver = Resolver(store)
        assert resolver.resolve("acme/widget-7b-fp8", "model").strategy == "fuzzy"
        r = resolver.resolve("acme/widget-7b-fp8", "model", mode="exact")
        assert r.canonical_id is None
        assert r.strategy == "no_match"

    def test_exact_mode_keeps_exact_and_normalized(self):
        store = _store_with_aliases(
            ("Widget 7B", "model", "acme/widget-7b", None, "confirmed"))
        resolver = Resolver(store)
        assert resolver.resolve("Widget 7B", "model", mode="exact").strategy == "exact"
        assert resolver.resolve(
            "widget-7b", "model", mode="exact").strategy == "normalized"

    def test_exact_mode_still_accepts_checker_hits(self):
        checker = CountingChecker({
            "acme/Widget-7B": HfIdHit(
                hf_id="acme/Widget-7B", verbatim=True, source="hub_stats_index"),
        })
        r = Resolver(_store_with_aliases(), hf_id_checker=checker).resolve(
            "acme/Widget-7B", "model", mode="exact")
        assert r.canonical_id == "acme/Widget-7B"
        assert r.strategy == "exact"
