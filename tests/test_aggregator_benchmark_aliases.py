"""Benchmark spellings of the re-publisher sources (BenchPress, LLM Stats,
Papers with Code) resolve to the canonical they report, and BenchPress's
dotted names keep only the benchmark segment when unresolved. Skips without
fixtures."""
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


def _resolved(resolver, name, source_config):
    match = resolver.resolve_structured_benchmark(name, source_config)
    if match is not None:
        return match.canonical_id
    raw = resolver.fallback_benchmark_raw(name, source_config)
    return resolver.resolve(raw, "benchmark", source_config, check_hf=False).canonical_id


@pytest.mark.parametrize("name,source_config,expected", [
    ("benchpress.bigbench-hard", "benchpress", "bbh"),
    ("benchpress.bigbench-extra-hard", "benchpress", "big-bench-extra-hard"),
    ("benchpress.mcpatlas", "benchpress", "mcp-atlas"),
    ("benchpress.mrcr-v2-8needle", "benchpress", "mrcr-v2-8-needle"),
    ("benchpress.matharena-apex-2025", "benchpress", "matharena-apex"),
    ("benchpress.deepresearchbench", "benchpress", "deep-research-bench"),
    ("benchpress.korbench", "benchpress", "kor-bench"),
    ("benchpress.corpfin-v2", "benchpress", "corp-fin-v2"),
    ("benchpress.vstar", "benchpress", "v-star"),
    ("benchpress.videoholmes", "benchpress", "video-holmes"),
    ("benchpress.vlms-are-blind", "benchpress", "vlmsareblind"),
    ("benchpress.automation-bench-private-heldout", "benchpress", "automationbench"),
    # BenchPress labels this row SWE-Lancer IC Diamond, the subset canonical.
    ("benchpress.swelancer", "benchpress", "swe-lancer-ic-diamond-subset"),
    ("llm_stats.automationbench-aa", "llm-stats", "automationbench"),
    ("llm_stats.automationbench-1.0.6", "llm-stats", "automationbench"),
    ("llm_stats.arcagi2", "llm-stats", "arc-agi-2"),
    ("llm_stats.artifacts-bench", "llm-stats", "artifactsbench"),
    ("llm_stats.mmsibench", "llm-stats", "mmsi-bench"),
    ("llm_stats.omniscience", "llm-stats", "aa-omniscience"),
    ("llm_stats.videoholmes", "llm-stats", "video-holmes"),
    ("paperswithcode.agents.automationbench_aa", "paperswithcode", "automationbench"),
    ("paperswithcode.agents.mcpatlas_public", "paperswithcode", "mcp-atlas"),
    ("paperswithcode.agents.mcpatlas_full", "paperswithcode", "mcp-atlas"),
    ("paperswithcode.agents.taubench", "paperswithcode", "tau-bench"),
    ("paperswithcode.agents.taubench_airline", "paperswithcode", "tau-bench-airline"),
    ("paperswithcode.agents.taubench_retail", "paperswithcode", "tau-bench-retail"),
    ("paperswithcode.instruction_following.mtbench", "paperswithcode", "mt-bench"),
    ("paperswithcode.world_knowledge.c_simpleqa", "paperswithcode", "csimpleqa"),
    ("paperswithcode.image_understanding.cv_bench", "paperswithcode", "cvbench"),
    ("paperswithcode.image_understanding.vlms_are_blind", "paperswithcode", "vlmsareblind"),
    ("paperswithcode.image_understanding.olympiadbench_full", "paperswithcode", "olympiadbench"),
    ("paperswithcode.computer_use_agents.ground_ui_1k", "paperswithcode", "groundui-1k"),
    ("paperswithcode.computer_use_agents.windowsagentarena", "paperswithcode", "windows-agent-arena"),
    ("paperswithcode.audio_understanding.musicavqa", "paperswithcode", "music-avqa"),
])
def test_republished_benchmark_resolves(resolver, name, source_config, expected):
    assert _resolved(resolver, name, source_config) == expected


def test_benchpress_fallback_keeps_the_benchmark_segment(resolver):
    """BenchPress declares a `source.benchmark` layout, so an unresolved name
    whose benchmark segment reads as a metric keeps that segment instead of
    collapsing onto the bare source name."""
    assert resolver.fallback_benchmark_raw(
        "benchpress.aa-briefcase-elo", "benchpress"
    ) == "aa-briefcase-elo"
    assert resolver.fallback_benchmark_raw(
        "benchpress.hle-text", "benchpress"
    ) == "hle-text"


def test_swelancer_scope_does_not_leak(resolver):
    assert resolver.resolve("swelancer", "benchmark", check_hf=False).canonical_id is None


@pytest.mark.parametrize("name,source_config,not_expected", [
    # IF-Bench (casiatao/IF-Bench) is an infrared-image benchmark, not IFBench.
    ("paperswithcode.image_understanding.if_bench", "paperswithcode", "ifbench"),
    # These report FrontierSWE v1, whose scores are not comparable with v2.
    ("benchpress.frontier-swe", "benchpress", "frontierswe-v2"),
    ("llm_stats.frontierswe", "llm-stats", "frontierswe-v2"),
    ("paperswithcode.coding_agents.frontierswe", "paperswithcode", "frontierswe-v2"),
])
def test_look_alike_stays_off_the_other_benchmark(resolver, name, source_config, not_expected):
    assert _resolved(resolver, name, source_config) != not_expected
