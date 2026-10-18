"""HfIdVerifier: index tiers, live-tier caching, budget, breaker and
single-flight. The Hub API is mocked; the live flag (off by default in
conftest) is re-enabled per test."""
from unittest.mock import MagicMock, patch

import pytest

from eval_entity_registry.config import settings
from eval_entity_registry.services.hf_id_verifier import HfIdVerifier


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _verifier(index=None, **kwargs) -> tuple[HfIdVerifier, FakeClock]:
    clock = FakeClock()
    v = HfIdVerifier(index_provider=lambda: index or {}, clock=clock, **kwargs)
    return v, clock


class _FakeRepoNotFound(Exception):
    pass


class _FakeGated(_FakeRepoNotFound):
    # GatedRepoError subclasses RepositoryNotFoundError upstream; the
    # except-clause order in _live_lookup depends on it.
    pass


class _FakeValidation(Exception):
    pass


def _patch_live(verifier, side_effect=None, return_id=None):
    """Fake HfApi on the verifier, with the live path's error classes faked."""
    api = MagicMock()
    if side_effect is not None:
        api.model_info.side_effect = side_effect
    else:
        info = MagicMock()
        info.id = return_id
        api.model_info.return_value = info
    verifier._hf_api = api
    errors = MagicMock()
    errors.RepositoryNotFoundError = _FakeRepoNotFound
    errors.GatedRepoError = _FakeGated
    errors.HFValidationError = _FakeValidation
    patches = patch.dict(
        "sys.modules",
        {"huggingface_hub.errors": errors},
    )
    return api, patches


@pytest.fixture
def live_enabled():
    original = settings.hf_live_id_check_enabled
    settings.hf_live_id_check_enabled = True
    yield
    settings.hf_live_id_check_enabled = original


def test_non_hf_shaped_returns_none():
    v, _ = _verifier(index={"someindex": "Some/Index"})
    assert v.check("gpt-4o") is None
    assert v.check("a/b/c") is None
    assert v.check("/x") is None


def test_index_verbatim_hit():
    v, _ = _verifier(index={"qwen-qwen2-5-7b": "Qwen/Qwen2.5-7B"})
    hit = v.check("qwen/qwen2.5-7b")
    assert hit is not None
    assert hit.hf_id == "Qwen/Qwen2.5-7B"
    assert hit.verbatim is True
    assert hit.source == "hub_stats_index"


def test_index_normalized_hit_is_not_verbatim():
    v, _ = _verifier(index={"qwen-qwen2-5-7b": "Qwen/Qwen2.5-7B"})
    hit = v.check("qwen/qwen2_5_7b")
    assert hit is not None
    assert hit.verbatim is False


def test_live_disabled_and_index_miss_returns_none():
    v, _ = _verifier(index={})
    assert v.check("org/unknown-model") is None


def test_live_success_caches_positive(live_enabled):
    v, clock = _verifier(index={})
    api, patches = _patch_live(v, return_id="NousResearch/Hermes-4-70B")
    with patches:
        hit = v.check("nousresearch/hermes-4-70b")
    assert hit is not None
    assert hit.hf_id == "NousResearch/Hermes-4-70B"
    assert hit.verbatim is True
    assert hit.source == "hf_live"
    assert api.model_info.call_count == 1
    # Second call answers from cache — no new HTTP call.
    with patches:
        hit2 = v.check("nousresearch/hermes-4-70b")
    assert hit2 is not None and hit2.hf_id == "NousResearch/Hermes-4-70B"
    assert api.model_info.call_count == 1


def test_live_404_negative_cached_with_ttl(live_enabled):
    v, clock = _verifier(index={})
    api, patches = _patch_live(v, side_effect=_FakeRepoNotFound())
    with patches:
        assert v.check("org/nope") is None
        assert v.check("org/nope") is None
    assert api.model_info.call_count == 1
    # After the negative TTL, the live path is retried.
    clock.now += v._negative_ttl + 1
    with patches:
        v.check("org/nope")
    assert api.model_info.call_count == 2


def test_transport_error_not_cached_and_breaker_opens(live_enabled):
    v, clock = _verifier(index={}, breaker_threshold=3, breaker_cooldown=100.0)
    api, patches = _patch_live(v, side_effect=RuntimeError("boom"))
    with patches:
        for i in range(3):
            assert v.check(f"org/m{i}") is None
    assert api.model_info.call_count == 3
    # Breaker is open — no further calls.
    with patches:
        assert v.check("org/m9") is None
    assert api.model_info.call_count == 3
    # After cooldown the live path closes again.
    clock.now += 101.0
    with patches:
        v.check("org/m9")
    assert api.model_info.call_count == 4


def test_budget_exhaustion_answers_unknown_without_caching(live_enabled):
    v, clock = _verifier(index={}, budget_per_minute=2)
    api, patches = _patch_live(v, return_id="Org/M")
    with patches:
        assert v.check("org/m1") is not None
        assert v.check("org/m2") is not None
        # Budget spent — this answers unknown, uncached.
        assert v.check("org/m3") is None
    assert api.model_info.call_count == 2
    # A minute later the bucket refills.
    clock.now += 61.0
    with patches:
        assert v.check("org/m3") is not None
    assert api.model_info.call_count == 3


def test_single_flight_duplicate_answers_unknown(live_enabled):
    v, _ = _verifier(index={})
    v._in_flight.add("org/dup")
    api, patches = _patch_live(v, return_id="Org/Dup")
    with patches:
        assert v.check("org/dup") is None
    assert api.model_info.call_count == 0


def test_gated_repo_counts_as_existing(live_enabled):
    # Gated must be caught before not-found, else every gated repo is
    # negative-cached as a miss.
    v, _ = _verifier(index={})
    api, patches = _patch_live(v, side_effect=_FakeGated())
    with patches:
        hit = v.check("meta-llama/Llama-Gated")
    assert hit is not None
    assert hit.hf_id == "meta-llama/Llama-Gated"
    assert hit.source == "hf_live"
    # Positive-cached, NOT negative-cached.
    with patches:
        assert v.check("meta-llama/Llama-Gated") is not None
    assert api.model_info.call_count == 1


def test_validation_error_does_not_poison_breaker(live_enabled):
    v, _ = _verifier(index={}, breaker_threshold=3)
    api, patches = _patch_live(v, side_effect=_FakeValidation())
    with patches:
        for i in range(5):
            assert v.check(f"acme/bad value {i}") is None
    # Validation errors are misses, not failures: no breaker trip.
    assert api.model_info.call_count == 5
    assert v._breaker_open_until == 0.0
    with patches:
        assert v.check("acme/bad value 0") is None
    assert api.model_info.call_count == 5  # served from negative cache


def test_single_flight_duplicate_does_not_burn_budget(live_enabled):
    v, _ = _verifier(index={}, budget_per_minute=1)
    v._in_flight.add("org/dup")
    api, patches = _patch_live(v, return_id="Org/Other")
    with patches:
        assert v.check("org/dup") is None       # single-flight: unknown
        assert v.check("org/other") is not None  # budget still available
    assert api.model_info.call_count == 1


def test_index_collision_verbatim_only():
    # Two repos share an id_norm: only case-insensitive-exact raws answer.
    collided = {"acme-widget-7b": {
        "acme/widget-7b": "acme/Widget-7B",
        "acme/widget.7b": "acme/Widget.7B",
    }}
    v, _ = _verifier(index=collided)
    hit = v.check("acme/widget-7b")
    assert hit is not None and hit.hf_id == "acme/Widget-7B" and hit.verbatim
    hit2 = v.check("ACME/WIDGET.7B")
    assert hit2 is not None and hit2.hf_id == "acme/Widget.7B"
    assert v.check("acme/widget_7b") is None


def test_index_error_degrades_to_live_or_none():
    def boom():
        raise RuntimeError("index unavailable")

    clock = FakeClock()
    v = HfIdVerifier(index_provider=boom, clock=clock)
    assert v.check("org/model") is None
