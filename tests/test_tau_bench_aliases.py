"""tau-bench voice interaction channels resolve to their own metrics instead
of the catch-all `score`; the rate names are scoped to tau-bench. `Pass^1`
keeps today's resolution and terminal-bench's `Pass@2` is not claimed.
Skips without fixtures."""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Response latency (mean)", "voice-response-latency"),
        ("response_latency_mean", "voice-response-latency"),
        ("Yield latency (mean)", "voice-yield-latency"),
        ("yield_latency_mean", "voice-yield-latency"),
        ("Response rate", "voice-response-rate"),
        ("response_rate", "voice-response-rate"),
        ("Yield rate", "voice-yield-rate"),
        ("yield_rate", "voice-yield-rate"),
        ("Agent interruption rate", "agent-interruption-rate"),
        ("Backchannel selectivity", "backchannel-selectivity"),
        ("selectivity_backchannel", "backchannel-selectivity"),
        ("Non-directed-speech selectivity", "non-directed-speech-selectivity"),
        ("selectivity_non_directed", "non-directed-speech-selectivity"),
        ("Vocal-tic selectivity", "vocal-tic-selectivity"),
        ("selectivity_vocal_tic", "vocal-tic-selectivity"),
    ],
)
def test_voice_metric_resolves(resolver, name, expected):
    assert resolver.resolve(name, "metric", "tau-bench").canonical_id == expected


def test_rate_names_stay_in_tau_bench(resolver):
    assert resolver.resolve("Response rate", "metric", None).canonical_id is None
    assert resolver.resolve("Yield rate", "metric", None).canonical_id is None


def test_pass_hat_one_resolves_as_before(resolver):
    assert resolver.resolve("Pass^1", "metric", "tau-bench").canonical_id == "pass-at-1"


@pytest.mark.parametrize("source_config", ["terminal-bench-2.0", None])
def test_pass_at_two_is_not_claimed(resolver, source_config):
    assert resolver.resolve("Pass@2", "metric", source_config).canonical_id is None


@pytest.mark.parametrize("k", [2, 3, 4])
def test_pass_hat_k_resolves_in_tau_bench_only(resolver, k):
    assert resolver.resolve(f"Pass^{k}", "metric", "tau-bench").canonical_id == f"pass-hat-{k}"
    assert resolver.resolve(f"Pass^{k}", "metric", None).canonical_id is None
    assert resolver.resolve(f"Pass@{k}", "metric", "terminal-bench-2.0").canonical_id is None
