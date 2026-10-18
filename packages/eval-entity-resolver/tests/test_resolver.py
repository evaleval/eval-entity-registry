"""Resolver strategy chain, structured metric/benchmark id resolution, and
the harness-version fallback: which tier answers, what must stay no_match,
and how a structured match reports subset/qualifier/observation_role."""

import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, Resolver, ResolverConfig


def _store_with_aliases(*rows) -> AliasStore:
    """Build an AliasStore from (raw_value, entity_type, canonical_id, source_config, status) tuples."""
    from datetime import datetime, timezone
    import uuid

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for raw_value, entity_type, canonical_id, source_config, status in rows:
        records.append(
            {
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
            }
        )
    from eval_entity_resolver.alias_store import _empty_df
    df = pd.DataFrame(records) if records else _empty_df()
    return AliasStore(df)


class TestExactStrategy:
    def test_exact_match(self):
        store = _store_with_aliases(("IFEval", "benchmark", "ifeval", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("IFEval", "benchmark")
        assert result.canonical_id == "ifeval"
        assert result.strategy == "exact"
        assert result.confidence == 1.0

    def test_config_scoped_before_global(self):
        store = _store_with_aliases(
            ("MATH", "benchmark", "math-global", None, "confirmed"),
            ("MATH", "benchmark", "math-helm", "helm_lite", "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("MATH", "benchmark", source_config="helm_lite")
        assert result.canonical_id == "math-helm"

    def test_falls_back_to_global(self):
        store = _store_with_aliases(("MATH", "benchmark", "math-global", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("MATH", "benchmark", source_config="some_other_config")
        assert result.canonical_id == "math-global"

    def test_rejected_alias_skipped(self):
        store = _store_with_aliases(("IFEval", "benchmark", "ifeval", None, "rejected"))
        resolver = Resolver(store)
        result = resolver.resolve("IFEval", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestScopedAliasIsolation:
    """Scoped aliases (source_config != None) must not leak into unrelated lookups."""

    def test_scoped_isolated_from_other_config(self):
        store = _store_with_aliases(
            ("Overall", "benchmark", "ace", "ace", "confirmed"),
            ("Overall", "benchmark", "apex-v1", "apex-v1", "confirmed"),
        )
        resolver = Resolver(store)
        # Different config — scoped alias must not match.
        result = resolver.resolve("Overall", "benchmark", source_config="hfopenllm_v2")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_scoped_isolated_without_source_config(self):
        store = _store_with_aliases(
            ("Arabic", "benchmark", "global-mmlu-lite", "global-mmlu-lite", "confirmed"),
        )
        resolver = Resolver(store)
        # No source_config provided — scoped alias must not match.
        result = resolver.resolve("Arabic", "benchmark")
        assert result.canonical_id is None

    def test_scoped_normalized_match_respects_scope(self):
        store = _store_with_aliases(
            ("Abstract Algebra", "benchmark", "mmlu", "helm_mmlu", "confirmed"),
        )
        resolver = Resolver(store)
        # Same scope, different casing — normalized strategy must match.
        result = resolver.resolve("abstract algebra", "benchmark", source_config="helm_mmlu")
        assert result.canonical_id == "mmlu"
        assert result.strategy == "normalized"
        # Different scope — must NOT match via normalized either.
        result = resolver.resolve("abstract algebra", "benchmark", source_config="helm_lite")
        assert result.canonical_id is None

    def test_global_alias_still_matches_any_scope(self):
        store = _store_with_aliases(("MMLU", "benchmark", "mmlu", None, "confirmed"))
        resolver = Resolver(store)
        for sc in [None, "helm_mmlu", "helm_lite", "some_other"]:
            result = resolver.resolve("MMLU", "benchmark", source_config=sc)
            assert result.canonical_id == "mmlu", f"failed for source_config={sc}"


class TestNormalizedStrategy:
    def test_normalized_match(self):
        store = _store_with_aliases(("MATH Level 5", "benchmark", "math-level-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("math level 5", "benchmark")
        assert result.canonical_id == "math-level-5"
        assert result.strategy == "normalized"

    def test_punctuation_stripped(self):
        store = _store_with_aliases(("GPQA!", "benchmark", "gpqa", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("gpqa", "benchmark")
        assert result.canonical_id == "gpqa"
        assert result.strategy in ("exact", "normalized")


class TestFuzzyStrategy:
    def test_suffix_strip_matches_base(self):
        """model-name-fc should match model-name via suffix stripping."""
        store = _store_with_aliases(("writer/palmyra-x-004", "model", "writer/palmyra-x-004", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("writer/palmyra-x-004-fc", "model")
        assert result.canonical_id == "writer/palmyra-x-004"
        assert result.strategy == "fuzzy"

    def test_org_normalization_matches(self):
        """deepseek-ai/model should match deepseek/model via org alias."""
        store = _store_with_aliases(("deepseek/deepseek-r1-0528", "model", "deepseek/deepseek-r1-0528", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("deepseek-ai/deepseek-r1-0528", "model")
        assert result.canonical_id == "deepseek/deepseek-r1-0528"
        assert result.strategy == "fuzzy"

    def test_distinct_versions_not_merged(self):
        """gpt-5-mini should NOT fuzzy-match to gpt-5 — they are different models."""
        store = _store_with_aliases(("openai/gpt-5-2025-08-07", "model", "openai/gpt-5-2025-08-07", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-mini-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_distinct_benchmarks_not_merged(self):
        """fibble2 should NOT fuzzy-match to fibble1."""
        store = _store_with_aliases(("fibble1_arena_win_rate", "benchmark", "fibble1-arena-win-rate", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("fibble2_arena_win_rate", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_no_match_on_unrelated_string(self):
        store = _store_with_aliases(("completely-different", "harness", "x", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("unrelated string xyz", "harness")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_thinking_budget_suffix_stripped(self):
        """`-thinking-16k` strips to the bare canonical when no thinking-mode canonical is aliased."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-thinking-16k", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_thinking_budget_prefers_thinking_canonical_when_aliased(self):
        """When a thinking-mode canonical is aliased, `-thinking-Nk` peels only the
        budget and lands on the thinking variant, never the bare base (that would
        lose the mode signal in eval results)."""
        store = _store_with_aliases(
            ("anthropic/claude-haiku-4-5-20251001",
             "model", "anthropic/claude-haiku-4-5-20251001", None, "confirmed"),
            ("anthropic/claude-haiku-4-5-20251001-thinking",
             "model", "anthropic/claude-haiku-4-5-20251001-thinking", None, "confirmed"),
        )
        resolver = Resolver(store)
        for budget in ("1k", "8k", "16k", "32k"):
            raw = f"anthropic/claude-haiku-4-5-20251001-thinking-{budget}"
            result = resolver.resolve(raw, "model")
            assert result.canonical_id == "anthropic/claude-haiku-4-5-20251001-thinking", (
                f"{raw!r} expected to peel just the budget and stay on the "
                f"thinking-mode canonical; got {result.canonical_id!r}"
            )
            assert result.strategy == "fuzzy"

    def test_thinking_none_suffix_stripped(self):
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-thinking-none", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_date_version_suffix_no_longer_strips_to_family(self):
        """A trailing YYYYMMDD must not collapse to the family pointer (that loses
        per-snapshot release_date); no_match lets auto-create produce a linked
        snapshot canonical."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_date_version_resolves_when_snapshot_canonical_exists(self):
        """An aliased snapshot canonical wins by exact/normalized match before
        fuzzy strips anything."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed"),
            ("anthropic/claude-opus-4-5-20251101", "model",
             "anthropic/claude-opus-4-5-20251101", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5-20251101"

    def test_date_plus_thinking_no_longer_strips_to_family(self):
        """A compound `-YYYYMMDD-thinking-Nk` tail must not strip through to the
        family; without an aliased mode-promoted snapshot canonical, no_match."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101-thinking-16k", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_dot_version_normalizes_to_hyphen(self):
        """claude-opus-4.5 should normalize the same as claude-opus-4-5."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4.5", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "normalized"

    def test_meta_llama_org_alias(self):
        """meta-llama/ → meta/ via the org alias map."""
        store = _store_with_aliases(("meta/llama-3-70b", "model", "meta/llama-3-70b", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("meta-llama/llama-3-70b", "model")
        assert result.canonical_id == "meta/llama-3-70b"
        assert result.strategy == "fuzzy"

    def test_qwen_org_alias_to_alibaba(self):
        """`Qwen/<model>` (HF upload namespace) → `alibaba/<model>` via the org
        alias map."""
        store = _store_with_aliases(
            ("alibaba/qwen2-vl-7b-instruct", "model", "alibaba/qwen2-vl-7b-instruct", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("Qwen/Qwen2-VL-7B-Instruct", "model")
        assert result.canonical_id == "alibaba/qwen2-vl-7b-instruct"

    def test_year_only_no_longer_strips_to_family(self):
        """A trailing bare year must not peel to the family; no_match (auto-create
        owns it)."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_does_not_apply_to_non_openai(self):
        """The OpenAI YYYY-MM-DD date peel is org-scoped: `meta/llama-3-2024` must
        not strip to `meta/llama-3`."""
        store = _store_with_aliases(("meta/llama-3", "model", "meta/llama-3", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-3-2024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_rejects_non_year_4digit_tail(self):
        """The year-range guard keeps a non-year 4-digit tail like `-1024` from
        triggering the OpenAI date peel."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-1024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_diverse_date_shapes_all_no_match_when_only_family_aliased(self):
        """Trailing MMDD, YYYYMM and YYYYMMDD tails all fall through fuzzy without
        collapsing to the family when only the family is aliased."""
        store = _store_with_aliases(
            ("google/gemini-exp", "model", "google/gemini-exp", None, "confirmed"),
            ("stepfun/step-2-16k", "model", "stepfun/step-2-16k", None, "confirmed"),
            ("tencent/hunyuan-turbos", "model", "tencent/hunyuan-turbos", None, "confirmed"),
        )
        resolver = Resolver(store)
        for raw in [
            "google/gemini-exp-1114",
            "stepfun/step-2-16k-202411",
            "tencent/hunyuan-turbos-20250313",
        ]:
            result = resolver.resolve(raw, "model")
            assert result.canonical_id is None, f"{raw!r} unexpectedly matched"
            assert result.strategy == "no_match"

    def test_iso_full_date_no_longer_collapses_to_family(self):
        """The OpenAI ISO-date peel tries the truncated-month and year forms as
        snapshot candidates but never the bare family; with only the family
        aliased, no_match."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_prefers_truncated_month_canonical(self):
        """The peel stops at the first aliased hit (truncated month) rather than
        over-collapsing to the family."""
        store = _store_with_aliases(
            ("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"),
            ("openai/gpt-5-2025-08", "model", "openai/gpt-5-2025-08", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2025-08-07", "model")
        assert result.canonical_id == "openai/gpt-5-2025-08"
        assert result.strategy == "fuzzy"

    def test_iso_date_with_unknown_host_prefix_no_longer_collapses(self):
        """Host-prefix strip plus the OpenAI date peel still never yields the bare
        family; with only the family aliased, no_match."""
        store = _store_with_aliases(
            ("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"),
            ("gpt-5",        "model", "openai/gpt-5", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("unknown/gpt-5-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestNoMatch:
    def test_empty_store(self):
        store = _store_with_aliases()
        resolver = Resolver(store)
        result = resolver.resolve("anything", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"
        assert result.confidence == 0.0


class TestPromotedVariantResolution:
    """Instruct/chat/quantized/snapshot variants are first-class canonicals;
    resolving one must not collapse to the base (scores on `Llama-3-8B-Instruct`
    are not comparable to the base). There is no resolver-side instruct/chat
    strip: unknown post-training suffixes are no_match."""

    def _registry_like_store(self):
        """Base + promoted instruct + promoted instruct-quant + promoted snapshot,
        each with its own aliases."""
        return _store_with_aliases(
            # Base
            ("Llama-3-8B", "model", "meta/llama-3-8b", None, "confirmed"),
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"),
            # Promoted instruct (variant/mode of the base)
            ("Llama-3-8B-Instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            ("meta-llama/Meta-Llama-3-8B-Instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            ("meta/llama-3-8b-instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            # Promoted instruct-turbo (quantized of the instruct)
            ("Llama-3-8B-Instruct-Turbo", "model", "meta/llama-3-8b-instruct-turbo", None, "confirmed"),
            ("meta/llama-3-8b-instruct-turbo", "model", "meta/llama-3-8b-instruct-turbo", None, "confirmed"),
            # Promoted snapshot (variant/version of a base)
            ("gpt-4-0613", "model", "openai/gpt-4-0613", None, "confirmed"),
            ("openai/gpt-4-0613", "model", "openai/gpt-4-0613", None, "confirmed"),
            ("openai/gpt-4", "model", "openai/gpt-4", None, "confirmed"),
        )

    def test_instruct_resolves_to_instruct_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"
        assert result.strategy == "exact"

    def test_base_still_resolves_to_base(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B", "model")
        assert result.canonical_id == "meta/llama-3-8b"
        assert result.strategy == "exact"

    def test_doubled_org_prefix_instruct_resolves_to_instruct(self):
        """An org form duplicated inside the bare model id still lands on the
        instruct canonical."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("meta-llama/Meta-Llama-3-8B-Instruct", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"

    def test_quantized_variant_resolves_to_quantized_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct-Turbo", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct-turbo"

    def test_snapshot_resolves_to_snapshot_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("gpt-4-0613", "model")
        assert result.canonical_id == "openai/gpt-4-0613"
        assert result.canonical_id != "openai/gpt-4"

    def test_quant_falls_through_to_nearest_parent(self):
        """A quantization with no canonical of its own strips to the nearest
        unquantized canonical (the accepted precision-loss fallback)."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("meta/llama-3-8b-instruct-fp8", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"
        assert result.strategy == "fuzzy"

    def test_unknown_finetune_suffix_does_not_collapse_to_base(self):
        """An unrecognised suffix must not be stripped to the base (that
        misattributes scores); no_match lets the caller auto-draft."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct-CommunityFinetune", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_unknown_instruct_does_not_collapse_to_base(self):
        """No resolver-side `-instruct` strip: an un-promoted instruct variant is
        no_match, not the base."""
        store = _store_with_aliases(
            ("Llama-3-8B", "model", "meta/llama-3-8b", None, "confirmed"),
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"),
            # NB: no -instruct alias / canonical seeded
        )
        resolver = Resolver(store)
        result = resolver.resolve("Llama-3-8B-Instruct", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestSnakeCaseEquivalence:
    """Snake_case forms of seeded display-form aliases resolve via normalized matcher,
    without requiring the snake_case alias to be listed explicitly."""

    @pytest.mark.parametrize(
        "raw,expected_canonical",
        [
            ("easy_problems", "livecodebench-pro"),
            ("medium_problems", "livecodebench-pro"),
            ("hard_problems", "livecodebench-pro"),
        ],
    )
    def test_lcb_pro_difficulty_tiers_normalize(self, raw, expected_canonical):
        store = _store_with_aliases(
            ("Easy Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
            ("Medium Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
            ("Hard Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve(raw, "benchmark", source_config="livecodebenchpro")
        assert result.canonical_id == expected_canonical
        assert result.strategy == "normalized"


class TestResolveStructuredMetricId:
    """Positionless, registry-driven segment resolution for namespaced
    metric_config ids, with catch-all deferral."""

    CATCH_ALL = frozenset({"score"})

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("elo", "metric", "elo", None, "confirmed"),
                ("accuracy", "metric", "accuracy", None, "confirmed"),
                ("exact-match", "metric", "exact-match", None, "confirmed"),
                ("score", "metric", "score", None, "confirmed"),
            )
        )

    def test_metric_in_last_segment(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "vals_ai.mgsm.mgsm_de.accuracy", catch_all_ids=self.CATCH_ALL
            )
            == "accuracy"
        )

    def test_metric_in_middle_segment(self):
        # Position must not matter: the metric is not the last segment.
        assert (
            self._resolver().resolve_structured_metric_id(
                "lmarena.elo.overall", catch_all_ids=self.CATCH_ALL
            )
            == "elo"
        )

    def test_slash_separator(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "openeval/bbq/exact-match", catch_all_ids=self.CATCH_ALL
            )
            == "exact-match"
        )

    def test_catch_all_hit_defers(self):
        # "score" is a raw field name, not a disclosure; flagged catch-all defers
        # to the prose path.
        assert (
            self._resolver().resolve_structured_metric_id(
                "llm_stats.gdpval-aa.score", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_catch_all_unflagged_wins(self):
        # Without the registry flag, "score" is an ordinary vocabulary hit.
        assert (
            self._resolver().resolve_structured_metric_id("llm_stats.gdpval-aa.score")
            == "score"
        )

    def test_no_segment_resolves(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "artificial_analysis.mmlu_pro", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_conflicting_specific_hits_defer(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "adapter.elo.accuracy", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_adapter_namespace_segment_excluded(self):
        # First segment never contributes, even when it matches the vocabulary.
        assert (
            self._resolver().resolve_structured_metric_id(
                "elo.some-benchmark", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_duplicate_hits_are_one_disclosure(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "adapter.accuracy.accuracy", catch_all_ids=self.CATCH_ALL
            )
            == "accuracy"
        )

    def test_bare_id_never_routes_through(self):
        assert self._resolver().resolve_structured_metric_id("accuracy") is None

    def test_whitespace_text_never_routes_through(self):
        # Prose containing a dot (a sentence) must not be parsed as an id.
        assert (
            self._resolver().resolve_structured_metric_id(
                "reports performance as an Elo score."
            )
            is None
        )

    def test_none_and_blank_inputs(self):
        r = self._resolver()
        assert r.resolve_structured_metric_id(None) is None
        assert r.resolve_structured_metric_id("  ") is None

    # The whole namespaced id may itself be a registered alias. Separate store
    # so the cases above stay segment-only.

    def _whole_id_resolver(self):
        return Resolver(
            _store_with_aliases(
                ("elo", "metric", "elo", None, "confirmed"),
                ("accuracy", "metric", "accuracy", None, "confirmed"),
                ("Accuracy", "metric", "accuracy", None, "confirmed"),
                ("score", "metric", "score", None, "confirmed"),
                ("polistemics.adherence", "metric", "polistemics.adherence", None, "confirmed"),
                ("adapter.rubric.score", "metric", "score", None, "confirmed"),
                ("adapter.elo.rank", "metric", "custom-elo-rank", None, "confirmed"),
                ("adapter.elo.accuracy", "metric", "custom-elo-acc", None, "confirmed"),
                ("cvebench.mean", "metric", "cvebench.mean", None, "confirmed"),
                ("vendor.metric", "metric", "vendor-metric", "vendorcfg", "confirmed"),
                ("Avg/Accuracy", "metric", "avg-accuracy", None, "confirmed"),
                ("STFT-Dist.", "metric", "stft-dist", None, "confirmed"),
            )
        )

    def test_whole_id_alias_recovers_after_segments_miss(self):
        assert (
            self._whole_id_resolver().resolve_structured_metric_id(
                "polistemics.adherence", catch_all_ids=self.CATCH_ALL
            )
            == "polistemics.adherence"
        )

    def test_whole_id_alias_is_config_scoped(self):
        r = self._whole_id_resolver()
        assert (
            r.resolve_structured_metric_id(
                "vendor.metric", source_config="vendorcfg", catch_all_ids=self.CATCH_ALL
            )
            == "vendor-metric"
        )
        assert r.resolve_structured_metric_id("vendor.metric", catch_all_ids=self.CATCH_ALL) is None

    def test_whole_id_catch_all_alias_still_defers(self):
        assert (
            self._whole_id_resolver().resolve_structured_metric_id(
                "adapter.rubric.score", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_whole_id_alias_outranks_segment_hit(self):
        r = self._whole_id_resolver()
        # The segment reading says `accuracy`; the whole-id alias's declared
        # canonical wins.
        assert (
            r.resolve_structured_metric_id("Avg/Accuracy", catch_all_ids=self.CATCH_ALL)
            == "avg-accuracy"
        )
        assert (
            r.resolve_structured_metric_id("adapter.elo.rank", catch_all_ids=self.CATCH_ALL)
            == "custom-elo-rank"
        )

    def test_single_segment_id_never_routes_through(self):
        # A single-segment id fails the precheck before the whole-id lookup runs.
        assert (
            self._whole_id_resolver().resolve_structured_metric_id(
                "STFT-Dist.", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_whole_id_alias_resolves_conflicting_segments(self):
        assert (
            self._whole_id_resolver().resolve_structured_metric_id(
                "adapter.elo.accuracy", catch_all_ids=self.CATCH_ALL
            )
            == "custom-elo-acc"
        )

    def test_whole_id_placeholder_alias_is_returned(self):
        assert (
            self._whole_id_resolver().resolve_structured_metric_id(
                "cvebench.mean", catch_all_ids=self.CATCH_ALL
            )
            == "cvebench.mean"
        )

    def test_whole_id_lookup_is_exact_not_normalized(self):
        r = self._whole_id_resolver()
        assert (
            r.resolve_structured_metric_id(
                "Polistemics.Adherence", catch_all_ids=self.CATCH_ALL
            )
            is None
        )
        assert (
            r.resolve_structured_metric_id(
                "polistemics/adherence", catch_all_ids=self.CATCH_ALL
            )
            is None
        )


class TestResolveStructuredMetricQualifier:
    """The same match, with the span reported: which segment named the
    metric and what tail narrows it to a scoring variant."""

    CATCH_ALL = frozenset({"score"})

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("accuracy", "metric", "accuracy", None, "confirmed"),
                ("f1", "metric", "f1", None, "confirmed"),
                ("elo", "metric", "elo", None, "confirmed"),
                ("live_accuracy", "metric", "lenient-accuracy", None, "confirmed"),
                ("score", "metric", "score", None, "confirmed"),
                (
                    "polyglotoxicityprompts.score",
                    "metric",
                    "unsafe-continuation-rate",
                    None,
                    "confirmed",
                ),
            )
        )

    def test_trailing_qualifier_is_reported(self):
        m = self._resolver().resolve_structured_metric(
            "gpqa.accuracy.strict", catch_all_ids=self.CATCH_ALL
        )
        assert (m.canonical_id, m.matched_segment, m.qualifier) == (
            "accuracy", "accuracy", "strict",
        )

    def test_squad_answerability_qualifier(self):
        m = self._resolver().resolve_structured_metric(
            "squadv2.f1.has_ans", catch_all_ids=self.CATCH_ALL
        )
        assert (m.canonical_id, m.matched_segment, m.qualifier) == (
            "f1", "f1", "has_ans",
        )

    def test_aggregate_marker_tail_is_not_a_qualifier(self):
        # An aggregate-marker tail must not become an `elo::overall` key.
        m = self._resolver().resolve_structured_metric(
            "lmarena.elo.overall", catch_all_ids=self.CATCH_ALL
        )
        assert (m.canonical_id, m.matched_segment, m.qualifier) == ("elo", "elo", None)

    def test_subgroup_aggregate_marker_tail_is_not_a_qualifier(self):
        m = self._resolver().resolve_structured_metric(
            "global_mmlu.accuracy.fr_overall", catch_all_ids=self.CATCH_ALL
        )
        assert m.qualifier is None

    def test_numeric_tail_is_a_qualifier(self):
        # A numeric tail is a scoring variant (k-shot readings reported
        # separately); pooling them on the bare metric id would median unrelated
        # numbers. Only aggregate markers are excluded.
        m = self._resolver().resolve_structured_metric(
            "bench.accuracy.5", catch_all_ids=self.CATCH_ALL
        )
        assert m.qualifier == "5"

    def test_numeric_qualifiers_stay_distinct(self):
        r = self._resolver()
        one = r.resolve_structured_metric(
            "bench.accuracy.1", catch_all_ids=self.CATCH_ALL
        )
        five = r.resolve_structured_metric(
            "bench.accuracy.5", catch_all_ids=self.CATCH_ALL
        )
        assert one.canonical_id == five.canonical_id == "accuracy"
        assert (one.qualifier, five.qualifier) == ("1", "5")

    def test_metric_in_last_segment_has_no_qualifier(self):
        m = self._resolver().resolve_structured_metric(
            "bfcl.live.live_accuracy", catch_all_ids=self.CATCH_ALL
        )
        assert (m.canonical_id, m.matched_segment, m.qualifier) == (
            "lenient-accuracy", "live_accuracy", None,
        )

    def test_leading_namespace_segments_are_never_the_qualifier(self):
        m = self._resolver().resolve_structured_metric(
            "vals_ai.mgsm.mgsm_de.accuracy", catch_all_ids=self.CATCH_ALL
        )
        assert (m.canonical_id, m.matched_segment, m.qualifier) == (
            "accuracy", "accuracy", None,
        )

    def test_whole_id_alias_yields_no_qualifier(self):
        m = self._resolver().resolve_structured_metric(
            "polyglotoxicityprompts.score", catch_all_ids=self.CATCH_ALL
        )
        assert (m.canonical_id, m.matched_segment, m.qualifier) == (
            "unsafe-continuation-rate", "polyglotoxicityprompts.score", None,
        )

    def test_no_metric_segment_returns_none(self):
        assert (
            self._resolver().resolve_structured_metric(
                "mmlu_pro.biology", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_multi_segment_tail_joins(self):
        m = self._resolver().resolve_structured_metric(
            "bench.accuracy.strict.has_ans", catch_all_ids=self.CATCH_ALL
        )
        assert m.qualifier == "strict.has_ans"

    def test_repeated_metric_segment_ends_the_identity(self):
        # A later restatement of the same metric is not a qualifier of itself.
        m = self._resolver().resolve_structured_metric(
            "adapter.accuracy.accuracy", catch_all_ids=self.CATCH_ALL
        )
        assert m.qualifier is None

    def test_id_resolution_is_unchanged_by_the_new_return(self):
        r = self._resolver()
        for raw in (
            "gpqa.accuracy.strict",
            "lmarena.elo.overall",
            "mmlu_pro.biology",
            "polyglotoxicityprompts.score",
        ):
            structured = r.resolve_structured_metric(raw, catch_all_ids=self.CATCH_ALL)
            assert r.resolve_structured_metric_id(raw, catch_all_ids=self.CATCH_ALL) == (
                structured.canonical_id if structured is not None else None
            )


class TestResolveStructuredBenchmark:
    """Segment-wise, registry-driven resolution for dotted evaluation_names."""

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("bbq", "benchmark", "bbq", None, "confirmed"),
                ("agieval_lsat_lr", "benchmark", "agieval", None, "confirmed"),
                ("MMLU", "benchmark", "mmlu", None, "confirmed"),
                ("MMLU-Pro", "benchmark", "mmlu-pro", None, "confirmed"),
                ("mmlu_pro", "benchmark", "mmlu-pro", None, "confirmed"),
                ("Mmlu pro math", "benchmark", "mmlu-pro", None, "confirmed"),
                ("math", "benchmark", "math", None, "confirmed"),
                ("bfcl", "benchmark", "bfcl", None, "confirmed"),
                ("live_accuracy", "metric", "accuracy", None, "confirmed"),
                ("mmlu_college_physics", "benchmark", "mmlu", None, "confirmed"),
            )
        )

    def test_doubled_segments_collapse(self):
        match = self._resolver().resolve_structured_benchmark("bbq.bbq.overall")
        assert (match.canonical_id, match.benchmark_raw, match.subset) == ("bbq", "bbq", None)


    def test_doubled_segments_collapse_with_underscores(self):
        match = self._resolver().resolve_structured_benchmark(
            "agieval_lsat_lr.agieval_lsat_lr.overall"
        )
        assert match.canonical_id == "agieval"
        assert match.benchmark_raw == "agieval lsat lr"

    def test_most_specific_segment_wins(self):
        # The family slot must not outrank the benchmark it qualifies.
        match = self._resolver().resolve_structured_benchmark("MMLU.MMLU-Pro.overall")
        assert match.canonical_id == "mmlu-pro"
        assert match.benchmark_raw == "MMLU-Pro"

    def test_subset_stays_with_its_parent(self):
        # `math` is a benchmark of its own, but here it is MMLU-Pro's subject.
        match = self._resolver().resolve_structured_benchmark(
            "vals_ai.mmlu_pro.math", source_config="vals-ai"
        )
        assert match.canonical_id == "mmlu-pro"
        assert match.subset == "math"

    def test_leading_source_namespace_loses_to_a_deeper_hit(self):
        """The folder is dropped when a deeper segment resolves; when nothing
        deeper resolves it is the fallback identity (see
        TestStructuredBenchmarkFolderFallback)."""
        match = self._resolver().resolve_structured_benchmark(
            "vals_ai.mmlu_pro.biology", source_config="vals-ai"
        )
        assert match.canonical_id == "mmlu-pro"
        assert match.benchmark_raw == "mmlu pro biology"
        assert match.subset == "biology"

    def test_trailing_metric_segment_is_dropped(self):
        match = self._resolver().resolve_structured_benchmark("bfcl.live.live_accuracy")
        assert match.canonical_id == "bfcl"
        assert match.subset == "live"

    def test_run_config_tokens_resolve_through_the_strip(self):
        match = self._resolver().resolve_structured_benchmark(
            "mmlu_flan_cot_zeroshot_college_physics."
            "mmlu_flan_cot_zeroshot_college_physics.overall"
        )
        assert match.canonical_id == "mmlu"

    def test_bare_name_never_routes_through(self):
        assert self._resolver().resolve_structured_benchmark("gsm8k") is None

    def test_no_segment_resolves_falls_back(self):
        assert self._resolver().resolve_structured_benchmark("nope.nothing.overall") is None

    def test_whitespace_text_never_routes_through(self):
        assert (
            self._resolver().resolve_structured_benchmark("NaturalQuestions (open-book).")
            is None
        )

    def test_none_and_blank_inputs(self):
        r = self._resolver()
        assert r.resolve_structured_benchmark(None) is None
        assert r.resolve_structured_benchmark("  ") is None


class TestStructuredBenchmarkFolderFallback:
    """When the source_config-named leading segment was dropped and nothing else
    resolved, the folder segment is probed (alias tiers only), so a folder that
    is itself a registered benchmark is the fallback identity for names under
    it."""

    def _resolver(self, *extra_rows):
        # `extra_rows` lets one test register an alias the others must not see
        # (a registered `open_question` makes lexam resolve in the first pass).
        return Resolver(
            _store_with_aliases(
                ("polistemics", "benchmark", "polistemics", None, "confirmed"),
                ("cocoabench", "benchmark", "cocoabench", None, "confirmed"),
                ("lexam", "benchmark", "lexam", None, "confirmed"),
                ("lexam open question", "benchmark", "lexam-open-question", None, "confirmed"),
                ("llm_stats", "benchmark", "llm-stats", None, "confirmed"),
                ("mcp-atlas", "benchmark", "mcp-atlas", None, "confirmed"),
                ("vals_ai", "benchmark", "vals-ai", None, "confirmed"),
                *extra_rows,
            )
        )

    def test_first_pass_wins_when_the_child_resolves(self):
        # Pins the `match is None` guard on the retry: with `open_question`
        # registered the first pass resolves the child alone; an unconditional
        # retry would let the joined `lexam open question` form win instead.
        match = self._resolver(
            ("open_question", "benchmark", "lexam-open-question", None, "confirmed")
        ).resolve_structured_benchmark("lexam.open_question", source_config="lexam")
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "lexam-open-question",
            "open question",
            None,
        )

    def test_folder_benchmark_recovers_when_nothing_else_resolves(self):
        match = self._resolver().resolve_structured_benchmark(
            "polistemics.nl_tk2025.en.faithfulness", source_config="polistemics"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "polistemics",
            "polistemics nl tk2025 en faithfulness",
            "nl tk2025 en faithfulness",
        )

    def test_folder_fallback_still_drops_trailing_metric(self):
        match = self._resolver().resolve_structured_benchmark(
            "cocoabench.overall.accuracy_percent", source_config="cocoabench"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "cocoabench",
            "cocoabench",
            None,
        )

    def test_folder_fallback_prefers_known_joined_form(self):
        # The joined surface form is a registered alias, so it outranks the
        # folder hit. `subset` is informational here and not asserted.
        match = self._resolver().resolve_structured_benchmark(
            "lexam.open_question", source_config="lexam"
        )
        assert match.canonical_id == "lexam-open-question"
        assert match.benchmark_raw == "lexam open question"

    def test_folder_fallback_keeps_unregistered_child_as_subset(self):
        match = self._resolver().resolve_structured_benchmark(
            "llm_stats.zzz-bench", source_config="llm-stats"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "llm-stats",
            "llm stats zzz-bench",
            "zzz-bench",
        )

    def test_folder_fallback_never_runs_when_dropped_pass_hits(self):
        match = self._resolver().resolve_structured_benchmark(
            "llm_stats.mcp-atlas", source_config="llm-stats"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "mcp-atlas",
            "mcp-atlas",
            None,
        )

    def test_collapsed_duplicate_folder_name_recovers(self):
        match = self._resolver().resolve_structured_benchmark(
            "polistemics.polistemics.baseline", source_config="polistemics"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "polistemics",
            "polistemics baseline",
            "baseline",
        )

    def test_folder_fallback_needs_a_registered_folder_benchmark(self):
        assert (
            self._resolver().resolve_structured_benchmark(
                "nope.nothing.overall", source_config="nope"
            )
            is None
        )

    def test_no_drop_means_first_pass_answers(self):
        # Segment 0 does not name the folder, so nothing is dropped and the
        # retry never runs.
        match = self._resolver().resolve_structured_benchmark(
            "polistemics.nl_tk2025.en.faithfulness", source_config="other-folder"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "polistemics",
            "polistemics nl tk2025 en faithfulness",
            "nl tk2025 en faithfulness",
        )

    def test_unregistered_child_under_registered_folder_reports_folder(self):
        # A subset of an unregistered benchmark under a registered aggregator is
        # reported as that aggregator with the subset kept, never the subset alone.
        match = self._resolver().resolve_structured_benchmark(
            "vals_ai.programbench.strict", source_config="vals-ai"
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "vals-ai",
            "vals ai programbench strict",
            "programbench strict",
        )

    def test_no_source_config_is_not_an_a_case(self):
        # Without a source_config nothing is dropped, so the first pass answers.
        match = self._resolver().resolve_structured_benchmark(
            "vals_ai.programbench.strict", source_config=None
        )
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "vals-ai",
            "vals ai programbench strict",
            "programbench strict",
        )

    def test_unregistered_folder_without_source_config_is_none(self):
        assert (
            self._resolver().resolve_structured_benchmark("nope.child.strict") is None
        )

    def test_folder_fallback_never_uses_fuzzy_on_the_folder(self):
        # `zzz-bench-cot` fuzzy-stems to `zzz-bench`, so the folder would match
        # through the full chain; the retry uses alias tiers only.
        store = _store_with_aliases(
            ("zzz-bench", "benchmark", "zzz-bench", None, "confirmed")
        )
        resolver = Resolver(store)
        assert resolver.resolve("zzz-bench-cot", "benchmark").canonical_id == "zzz-bench"
        assert (
            resolver.resolve_structured_benchmark(
                "zzz-bench-cot.child.strict", source_config="zzz-bench-cot"
            )
            is None
        )


class TestHarnessVersionFallback:
    """A versioned harness string whose bare name is a registered harness
    resolves to it, in resolve mode only, after every other strategy missed."""

    def _resolver(self, config=None):
        return Resolver(
            _store_with_aliases(
                ("lm_eval", "harness", "lm-evaluation-harness", None, "confirmed"),
                ("lm_eval 0.4.11.dev0", "harness", "lm-evaluation-harness", None, "confirmed"),
                ("helm", "harness", "helm", None, "confirmed"),
                ("Inspect AI", "harness", "inspect-ai", None, "confirmed"),
                ("mmlu", "benchmark", "mmlu", None, "confirmed"),
                ("myharness", "harness", "scoped-harness", "cfg", "confirmed"),
            ),
            config=config,
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("lm_eval 0.4.12", "lm_eval"),
            ("lm_eval v0.4.12", "lm_eval"),
            ("lm_eval 0.4.12-rc1", "lm_eval"),
            ("lm_eval 0.4.12.post1", "lm_eval"),
            ("lm_eval 0.4.11.dev0", "lm_eval"),
            ("inspect_ai 0.3.100 ", "inspect_ai"),
            ("inspect_ai inspect_ai:0.3.75", "inspect_ai"),
            ("inspect_ai inspect_ai:0.3.80.dev25+g91a50728.d20250331", "inspect_ai"),
            ("helm unknown", None),
            ("swe-bench v2", None),
            ("BFCL v4", None),
            ("kaggle kernel 4", None),
            ("terminal-bench-2.0", None),
            ("1.2.3", None),
            ("helm", None),
            ("inspect_ai foo:0.3.75", None),
            ("inspect_ai inspect_ai:0.3.202,inference_scaling_paper:0.1.0", None),
        ],
    )
    def test_strip_helper(self, raw, expected):
        from eval_entity_resolver.resolver import _strip_harness_version

        assert _strip_harness_version(raw) == expected

    def test_versioned_name_resolves_via_bare_exact_tier(self):
        result = self._resolver().resolve("lm_eval 0.4.12", "harness")
        assert result.canonical_id == "lm-evaluation-harness"
        assert result.strategy == "normalized"
        assert result.confidence == 0.95
        assert result.raw_value == "lm_eval 0.4.12"
        assert result.resolution_detail == {
            "harness_version_stripped": "0.4.12",
            "bare_name": "lm_eval",
            "bare_tier": "exact",
        }

    def test_versioned_name_resolves_via_bare_normalized_tier(self):
        result = self._resolver().resolve("inspect_ai inspect_ai:0.3.75", "harness")
        assert (result.canonical_id, result.strategy, result.confidence) == (
            "inspect-ai",
            "normalized",
            0.95,
        )
        assert result.resolution_detail["bare_tier"] == "normalized"
        assert result.resolution_detail["harness_version_stripped"] == "inspect_ai:0.3.75"

    @pytest.mark.parametrize(
        "raw", ["lm_eval v0.4.12", "lm_eval 0.4.12-rc1", "lm_eval 0.4.12.post1"]
    )
    def test_prerelease_tails(self, raw):
        assert self._resolver().resolve(raw, "harness").canonical_id == "lm-evaluation-harness"

    def test_existing_versioned_alias_still_wins_first(self):
        result = self._resolver().resolve("lm_eval 0.4.11.dev0", "harness")
        assert (result.canonical_id, result.strategy, result.confidence) == (
            "lm-evaluation-harness",
            "exact",
            1.0,
        )
        assert "harness_version_stripped" not in (result.resolution_detail or {})

    @pytest.mark.parametrize(
        "raw", ["helm unknown", "swe-bench v2", "BFCL v4", "terminal-bench-2.0", "1.2.3"]
    )
    def test_non_version_tails_stay_no_match(self, raw):
        result = self._resolver().resolve(raw, "harness")
        assert result.canonical_id is None
        assert result.strategy == "no_match"
        assert result.confidence == 0.0

    def test_unregistered_bare_name_stays_no_match(self):
        assert self._resolver().resolve("polistemics 1.1.0", "harness").canonical_id is None

    def test_exact_mode_stays_no_match(self):
        resolver = self._resolver()
        assert resolver.resolve("helm 1.2.3", "harness", mode="exact").canonical_id is None
        # control: the alias tiers still answer in exact mode
        assert resolver.resolve("helm", "harness", mode="exact").canonical_id == "helm"
        # the gate lives in the helper itself, not in the control flow
        direct = resolver._no_match_or_harness_retry("helm 1.2.3", "harness", None, "exact")
        assert direct.canonical_id is None and direct.strategy == "no_match"

    def test_harness_detail_is_empty_without_strip(self):
        # A non-stripped harness hit carries no strip keys (the HTTP projection
        # turns that into `{}`).
        result = self._resolver().resolve("helm", "harness")
        assert "harness_version_stripped" not in (result.resolution_detail or {})

    def test_other_entity_types_untouched(self):
        resolver = self._resolver()
        assert resolver.resolve("mmlu 1.0", "benchmark").canonical_id is None
        assert resolver.resolve("helm 1.2.3", "metric").canonical_id is None

    def test_threshold_gates_retry(self):
        resolver = self._resolver(config=ResolverConfig(threshold=0.99))
        assert resolver.resolve("inspect_ai 0.3.100", "harness").canonical_id is None
        assert resolver.resolve("helm 1.2.3", "harness").canonical_id is None

    def test_source_config_scoping_respected(self):
        resolver = self._resolver()
        assert (
            resolver.resolve("myharness 1.0", "harness", "cfg").canonical_id
            == "scoped-harness"
        )
        assert resolver.resolve("myharness 1.0", "harness").canonical_id is None


class TestStructuredBenchmarkSubsetIsIdentity:
    """Whether the subset names a PART of the benchmark or just spells the
    benchmark itself at greater length.

    The two are indistinguishable in `subset` and mean opposite things to the
    consumer deciding what a row measured, so the flag is what separates a
    whole-benchmark observation from one task inside it.
    """

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("mmlu", "benchmark", "mmlu", None, "confirmed"),
                # every MMLU subject reaches `mmlu` through the normalized tier
                ("mmlu_anatomy", "benchmark", "mmlu", None, "confirmed"),
                ("polyglotoxicityprompts", "benchmark",
                 "polyglotoxicityprompts", None, "confirmed"),
                # the curated alias that says "this spelling IS the benchmark"
                ("polyglotoxicityprompts small overall", "benchmark",
                 "polyglotoxicityprompts", None, "confirmed"),
                ("realtoxicityprompts", "benchmark",
                 "realtoxicityprompts", None, "confirmed"),
                ("realtoxicityprompts small", "benchmark",
                 "realtoxicityprompts", None, "confirmed"),
            )
        )

    def test_curated_whole_spelling_is_identity_not_a_part(self):
        m = self._resolver().resolve_structured_benchmark(
            "polyglotoxicityprompts.polyglotoxicityprompts.small_overall"
        )
        assert m.canonical_id == "polyglotoxicityprompts"
        assert m.subset == "small overall"
        assert m.subset_is_identity is True

    def test_a_sibling_without_that_alias_is_a_part(self):
        m = self._resolver().resolve_structured_benchmark(
            "polyglotoxicityprompts.polyglotoxicityprompts.small_english"
        )
        assert m.canonical_id == "polyglotoxicityprompts"
        assert m.subset == "small english"
        assert m.subset_is_identity is False

    def test_realtoxicityprompts_small_is_the_whole_benchmark(self):
        m = self._resolver().resolve_structured_benchmark(
            "realtoxicityprompts.realtoxicityprompts.small"
        )
        assert (m.canonical_id, m.subset, m.subset_is_identity) == (
            "realtoxicityprompts", "small", True,
        )

    def test_a_normalized_alias_hit_is_not_an_identity_claim(self):
        """A normalized-tier hit is not an identity claim: that tier absorbs every
        MMLU subject name, so treating it as identity would make every subject
        row a whole-benchmark observation."""
        m = self._resolver().resolve_structured_benchmark("mmlu.mmlu.anatomy")
        assert m.canonical_id == "mmlu"
        assert m.subset == "anatomy"
        assert m.subset_is_identity is False

    def test_an_aggregate_marker_leaves_no_subset_at_all(self):
        m = self._resolver().resolve_structured_benchmark("mmlu.mmlu.overall")
        assert (m.canonical_id, m.subset, m.subset_is_identity) == (
            "mmlu", None, False,
        )

    def test_a_real_member_name_is_still_a_part(self):
        """Only a subset that names the benchmark itself is identity; `anatomy`
        names something inside MMLU."""
        r = Resolver(
            _store_with_aliases(
                ("mmlu", "benchmark", "mmlu", None, "confirmed"),
                ("mmlu_anatomy", "benchmark", "mmlu", None, "confirmed"),
            )
        )
        m = r.resolve_structured_benchmark("mmlu.mmlu.anatomy")
        assert (m.subset, m.subset_is_identity) == ("anatomy", False)

    def test_subset_naming_the_benchmark_is_detected_space_insensitively(self):
        """`normalize` maps separators to spaces (`swe-bench` → `swe bench`) while
        a source may spell the subset `swebench`; comparing on alphanumerics
        makes them the same benchmark."""
        r = Resolver(_store_with_aliases(("mmlu", "benchmark", "mmlu", None,
                                          "confirmed")))
        assert r._subset_names_the_benchmark("swebench", "swe-bench") is True
        assert r._subset_names_the_benchmark("SWE_Bench", "swe-bench") is True
        assert r._subset_names_the_benchmark("anatomy", "mmlu") is False
        assert r._subset_names_the_benchmark(None, "mmlu") is False

    def test_resolving_to_the_same_canonical_is_not_identity(self):
        """The predicate asks whether the subset SPELLS the benchmark, not whether
        it resolves to it: aliases naming a benchmark's PARTS also resolve to it,
        and treating those as identity would make category aggregates
        whole-benchmark readings."""
        r = Resolver(
            _store_with_aliases(
                ("legalbench", "benchmark", "legalbench", None, "confirmed"),
                ("conclusion_tasks", "benchmark", "legalbench", None, "confirmed"),
            )
        )
        assert r.resolve(
            "conclusion_tasks", "benchmark", None, check_hf=False
        ).canonical_id == "legalbench"
        assert r._subset_names_the_benchmark(
            "legal bench conclusion tasks", "legalbench"
        ) is False
        assert r._subset_names_the_benchmark("conclusion tasks", "legalbench") is False
        # the benchmark's own name, however it is spaced, still is identity
        assert r._subset_names_the_benchmark("legal bench", "legalbench") is True


class TestValsLegalBenchObservationRoles:
    """The six real Vals LegalBench names, end to end. The benchmark, its
    `.overall` total and its five category aggregates are all aliases of one
    canonical, so resolution cannot serve as the identity test; roles come from
    what each name says. Only the leading `vals_ai` segment resolves on its own,
    so the identity path lands on the folder and the joined spelling carries the
    benchmark."""

    ALIASES = (
        "vals_ai_legal_bench",
        "vals_ai_legal_bench_overall",
        "vals_ai_legal_bench_conclusion_tasks",
        "vals_ai_legal_bench_interpretation_tasks",
        "vals_ai_legal_bench_issue_tasks",
        "vals_ai_legal_bench_rhetoric_tasks",
        "vals_ai_legal_bench_rule_tasks",
        "Legal bench overall",
        "Legal bench conclusion tasks",
        "Legal bench interpretation tasks",
        "Legal bench issue tasks",
        "Legal bench rhetoric tasks",
        "Legal bench rule tasks",
    )
    GROUPS = ("conclusion_tasks", "interpretation_tasks", "issue_tasks",
              "rhetoric_tasks", "rule_tasks")

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("legalbench", "benchmark", "legalbench", None, "confirmed"),
                # the source folder is itself a registered benchmark, which is
                # why the segment scan stops on it
                ("vals_ai", "benchmark", "vals-ai", None, "confirmed"),
                *((a, "benchmark", "legalbench", None, "confirmed")
                  for a in self.ALIASES),
            )
        )

    def test_only_the_folder_segment_resolves_on_its_own(self):
        r = self._resolver()
        assert r.resolve(
            "vals_ai", "benchmark", "vals-ai", check_hf=False
        ).canonical_id == "vals-ai"
        for segment in ("legal_bench", *self.GROUPS):
            assert r.resolve(
                segment, "benchmark", "vals-ai", check_hf=False
            ).canonical_id is None, segment

    def test_the_submitted_total_is_a_whole(self):
        m = self._resolver().resolve_structured_benchmark(
            "vals_ai.legal_bench.overall", "vals-ai"
        )
        assert m.canonical_id == "legalbench"
        assert m.observation_role == "whole"

    def test_every_category_aggregate_is_a_part(self):
        r = self._resolver()
        roles = {
            g: r.resolve_structured_benchmark(
                f"vals_ai.legal_bench.{g}", "vals-ai"
            ).observation_role
            for g in self.GROUPS
        }
        assert roles == dict.fromkeys(self.GROUPS, "part"), roles


class TestBenchmarkObservationRole:
    """`observation_role`: a real subset is a `part`, anything else a `whole`.
    `unknown` is the absence of a match and belongs to the caller: an unread
    name must not be called a whole, or a benchmark's own subjects can outvote
    the total published beside them."""

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("mmlu", "benchmark", "mmlu", None, "confirmed"),
                ("mmlu_anatomy", "benchmark", "mmlu", None, "confirmed"),
                ("realtoxicityprompts", "benchmark",
                 "realtoxicityprompts", None, "confirmed"),
                ("realtoxicityprompts small", "benchmark",
                 "realtoxicityprompts", None, "confirmed"),
            )
        )

    def test_a_real_subset_is_a_part(self):
        m = self._resolver().resolve_structured_benchmark("mmlu.mmlu.anatomy")
        assert m.observation_role == "part"

    def test_no_subset_is_a_whole(self):
        m = self._resolver().resolve_structured_benchmark("mmlu.mmlu.overall")
        assert m.observation_role == "whole"

    def test_a_subset_that_names_the_benchmark_is_a_whole(self):
        m = self._resolver().resolve_structured_benchmark(
            "realtoxicityprompts.realtoxicityprompts.small"
        )
        assert (m.subset, m.subset_is_identity, m.observation_role) == (
            "small", True, "whole",
        )

    def test_a_flat_or_spaced_name_yields_no_match_at_all(self):
        """`prepare_eval_name_segments` declines names without a dot or with a
        space, so there is no match to carry a role; the caller must supply
        `unknown`, not `whole`."""
        r = self._resolver()
        assert r.resolve_structured_benchmark("MMLU All Subjects") is None
        assert r.resolve_structured_benchmark("mmlu") is None

