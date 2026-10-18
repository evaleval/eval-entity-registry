"""Fuzzy strategy primitives: duplicated-org-prefix collapse, host-prefix
stripping, org normalisation, quant/HF suffix strip, platform capture, and
the benchmark run-config strip.
"""
from __future__ import annotations

import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, Resolver
from eval_entity_resolver.strategies.fuzzy import (
    _drop_duplicated_org_prefix,
    _drop_host_prefix,
    _normalize_org,
    _strip_and_capture_platform_suffix,
    fuzzy_match,
)


# ---------------------------------------------------------------------------
# _drop_duplicated_org_prefix unit tests
# ---------------------------------------------------------------------------


class TestDropDuplicatedOrgPrefix:
    def test_slash_dash_form(self):
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai-kimi-k2-instruct")
            == "moonshotai/kimi-k2-instruct"
        )

    def test_slug_form_dash(self):
        # Slug input keeps its `__` separator shape.
        assert (
            _drop_duplicated_org_prefix("moonshotai__moonshotai-kimi-k2-instruct")
            == "moonshotai__kimi-k2-instruct"
        )

    def test_slash_double_slash_form(self):
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai/kimi-k2-instruct")
            == "moonshotai/kimi-k2-instruct"
        )

    def test_slug_double_underscore_form(self):
        assert (
            _drop_duplicated_org_prefix("moonshotai__moonshotai__kimi-k2-instruct")
            == "moonshotai__kimi-k2-instruct"
        )

    def test_does_not_collapse_distinct_tokens(self):
        # `gpt-4` prefixes `gpt-4-turbo` but is a different model path;
        # the hyphen-in-org guard must stop the collapse.
        assert _drop_duplicated_org_prefix("gpt-4/gpt-4-turbo") is None

    def test_real_case_openai_o1(self):
        assert _drop_duplicated_org_prefix("openai/openai-o1") == "openai/o1"

    def test_single_prefix_returns_none(self):
        assert _drop_duplicated_org_prefix("anthropic/claude") is None

    def test_no_separator_returns_none(self):
        assert _drop_duplicated_org_prefix("singleton") is None

    def test_empty_returns_none(self):
        assert _drop_duplicated_org_prefix("") is None

    def test_underscore_separator_in_slash_form(self):
        # `<org>/<org>_<rest>`
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai_kimi-k2")
            == "moonshotai/kimi-k2"
        )

    def test_case_insensitive_token_match(self):
        # Case-insensitive token match; original casing is preserved.
        assert (
            _drop_duplicated_org_prefix("Moonshotai/MOONSHOTAI-Kimi")
            == "Moonshotai/Kimi"
        )

    def test_substring_overlap_does_not_match(self):
        # `openai` is not `openaicommunity` followed by a separator.
        assert _drop_duplicated_org_prefix("openai/openaicommunity-x") is None


# ---------------------------------------------------------------------------
# _ORG_ALIASES integration tests
# ---------------------------------------------------------------------------


def _store_with_aliases(*rows) -> AliasStore:
    """Build an AliasStore from (raw_value, entity_type, canonical_id,
    source_config, status) tuples."""
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


class TestNewOrgAliases:
    @pytest.mark.parametrize(
        "raw_value,canonical",
        [
            ("zhipu/glm-4.5", "zai/glm-4.5"),
            ("zhipu-ai/glm-4.5", "zai/glm-4.5"),
            ("z-ai/glm-4.5", "zai/glm-4.5"),
            ("zai-org/glm-4.5", "zai/glm-4.5"),
        ],
    )
    def test_zhipu_family_to_zai(self, raw_value, canonical):
        store = _store_with_aliases(("zai/glm-4.5", "model", "zai/glm-4.5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve(raw_value, "model")
        assert result.canonical_id == canonical
        assert result.strategy == "fuzzy"

    @pytest.mark.parametrize(
        "raw_value,canonical",
        [
            ("moonshot/kimi-k2", "moonshotai/kimi-k2"),
            ("moonshot-ai/kimi-k2", "moonshotai/kimi-k2"),
        ],
    )
    def test_moonshot_family_to_moonshotai(self, raw_value, canonical):
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve(raw_value, "model")
        assert result.canonical_id == canonical
        assert result.strategy == "fuzzy"


class TestDuplicatedOrgIntegration:
    """The deduped candidate goes through alias lookup AND the org-alias pass."""

    def test_duplicated_prefix_resolves_via_alias(self):
        store = _store_with_aliases(
            ("moonshotai/kimi-k2-instruct", "model", "moonshotai/kimi-k2-instruct", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshotai/moonshotai-kimi-k2-instruct", "model")
        assert result.canonical_id == "moonshotai/kimi-k2-instruct"
        assert result.strategy == "fuzzy"

    def test_duplicated_prefix_plus_org_alias(self):
        """Dedup then org-alias: `moonshot/moonshot-kimi-k2` → `moonshotai/kimi-k2`."""
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshot/moonshot-kimi-k2", "model")
        assert result.canonical_id == "moonshotai/kimi-k2"
        assert result.strategy == "fuzzy"


# ---------------------------------------------------------------------------
# _drop_host_prefix unit tests
# ---------------------------------------------------------------------------


class TestDropHostPrefix:
    """`_drop_host_prefix` accepts ``host/model`` and ``host.model``; returns
    ``(bare_suffix, platform_id)`` (platform_id None when the host token is
    not in the seed map), or None when the prefix is not a known host.
    """

    # ------- slash-form positive cases -------

    def test_strips_unknown_slash_prefix(self):
        # `unknown/` is a leaderboard placeholder for a missing developer
        # field; the sentinel carries no platform.
        assert _drop_host_prefix("unknown/openai-o1") == ("openai-o1", None)

    def test_strips_bedrock_slash_prefix(self):
        assert _drop_host_prefix("bedrock/anthropic-claude")[0] == "anthropic-claude"

    def test_strips_amazon_bedrock_slash_prefix(self):
        assert _drop_host_prefix("amazon-bedrock/claude-3")[0] == "claude-3"

    def test_strips_aws_bedrock_slash_prefix(self):
        assert _drop_host_prefix("aws-bedrock/claude-3")[0] == "claude-3"

    def test_strips_azure_slash_prefix(self):
        assert _drop_host_prefix("azure/gpt-4")[0] == "gpt-4"

    def test_strips_azure_openai_slash_prefix(self):
        assert _drop_host_prefix("azure-openai/gpt-4")[0] == "gpt-4"

    def test_strips_azure_cognitive_services_slash_prefix(self):
        assert _drop_host_prefix("azure-cognitive-services/gpt-4")[0] == "gpt-4"

    def test_strips_vertex_slash_prefix(self):
        assert _drop_host_prefix("vertex/gemini-2.0")[0] == "gemini-2.0"

    def test_strips_google_vertex_slash_prefix(self):
        assert _drop_host_prefix("google-vertex/gemini-2.0")[0] == "gemini-2.0"

    def test_strips_vertex_anthropic_slash_prefix(self):
        assert _drop_host_prefix("vertex-anthropic/claude-3")[0] == "claude-3"

    def test_strips_fireworks_slash_prefix(self):
        assert _drop_host_prefix("fireworks/llama-3")[0] == "llama-3"

    def test_strips_fireworks_ai_slash_prefix(self):
        assert _drop_host_prefix("fireworks-ai/llama-3")[0] == "llama-3"

    def test_strips_groq_slash_prefix(self):
        assert _drop_host_prefix("groq/llama-3")[0] == "llama-3"

    def test_strips_together_slash_prefix(self):
        assert _drop_host_prefix("together/llama-3")[0] == "llama-3"

    def test_strips_togetherai_slash_prefix(self):
        assert _drop_host_prefix("togetherai/llama-3")[0] == "llama-3"

    def test_strips_together_ai_slash_prefix(self):
        assert _drop_host_prefix("together-ai/llama-3")[0] == "llama-3"

    def test_strips_openrouter_slash_prefix(self):
        assert _drop_host_prefix("openrouter/anthropic-claude")[0] == "anthropic-claude"

    def test_strips_perplexity_agent_slash_prefix(self):
        assert _drop_host_prefix("perplexity-agent/sonar")[0] == "sonar"

    def test_strips_deepinfra_slash_prefix(self):
        assert _drop_host_prefix("deepinfra/llama-3")[0] == "llama-3"

    def test_strips_anyscale_slash_prefix(self):
        assert _drop_host_prefix("anyscale/llama-3")[0] == "llama-3"

    def test_strips_novita_slash_prefix(self):
        assert _drop_host_prefix("novita/llama-3")[0] == "llama-3"

    def test_strips_novita_ai_slash_prefix(self):
        assert _drop_host_prefix("novita-ai/llama-3")[0] == "llama-3"

    def test_strips_replicate_slash_prefix(self):
        assert _drop_host_prefix("replicate/llama-3")[0] == "llama-3"

    def test_strips_ollama_slash_prefix(self):
        assert _drop_host_prefix("ollama/llama-3")[0] == "llama-3"

    def test_strips_ollama_cloud_slash_prefix(self):
        assert _drop_host_prefix("ollama-cloud/llama-3")[0] == "llama-3"

    def test_strips_github_models_slash_prefix(self):
        assert _drop_host_prefix("github-models/gpt-4")[0] == "gpt-4"

    def test_strips_github_copilot_slash_prefix(self):
        assert _drop_host_prefix("github-copilot/gpt-4")[0] == "gpt-4"

    def test_strips_lambda_slash_prefix(self):
        assert _drop_host_prefix("lambda/llama-3")[0] == "llama-3"

    def test_strips_baseten_slash_prefix(self):
        assert _drop_host_prefix("baseten/llama-3")[0] == "llama-3"

    def test_strips_modal_slash_prefix(self):
        assert _drop_host_prefix("modal/llama-3")[0] == "llama-3"

    def test_strips_runpod_slash_prefix(self):
        assert _drop_host_prefix("runpod/llama-3")[0] == "llama-3"

    def test_strips_cerebras_slash_prefix(self):
        assert _drop_host_prefix("cerebras/llama-3")[0] == "llama-3"

    def test_strips_sap_ai_core_slash_prefix(self):
        assert _drop_host_prefix("sap-ai-core/gpt-4")[0] == "gpt-4"

    def test_strips_cloudflare_ai_gateway_slash_prefix(self):
        assert _drop_host_prefix("cloudflare-ai-gateway/llama-3")[0] == "llama-3"

    def test_strips_aihubmix_slash_prefix(self):
        assert _drop_host_prefix("aihubmix/gpt-4")[0] == "gpt-4"

    def test_strips_kilo_slash_prefix(self):
        assert _drop_host_prefix("kilo/gpt-4")[0] == "gpt-4"

    def test_strips_vercel_slash_prefix(self):
        assert _drop_host_prefix("vercel/gpt-4")[0] == "gpt-4"

    def test_strips_llmgateway_slash_prefix(self):
        assert _drop_host_prefix("llmgateway/gpt-4")[0] == "gpt-4"

    def test_strips_poe_slash_prefix(self):
        assert _drop_host_prefix("poe/gpt-4")[0] == "gpt-4"

    # ------- platform-capture cases: tokens in the single-source map -------

    def test_captures_fireworks_platform(self):
        assert _drop_host_prefix("fireworks/llama-3") == ("llama-3", "fireworks-ai")

    def test_captures_together_platform(self):
        assert _drop_host_prefix("together/llama-3") == ("llama-3", "togetherai")

    def test_captures_groq_platform(self):
        assert _drop_host_prefix("groq/llama-3") == ("llama-3", "groq")

    def test_captures_azure_platform(self):
        assert _drop_host_prefix("azure/gpt-4") == ("gpt-4", "azure")

    def test_unmapped_host_token_captures_none(self):
        # `vertex` is stripped for matching but has no seed-map token.
        assert _drop_host_prefix("vertex/gemini-2.0") == ("gemini-2.0", None)

    # ------- dot-form positive cases -------

    def test_strips_bedrock_dot_prefix(self):
        # Bedrock model IDs use dots: `bedrock.anthropic.claude-3-5`.
        assert (
            _drop_host_prefix("bedrock.anthropic-claude-3-5")[0]
            == "anthropic-claude-3-5"
        )

    def test_strips_vertex_dot_prefix(self):
        assert _drop_host_prefix("vertex.google-gemini-2.0")[0] == "google-gemini-2.0"

    def test_dot_form_only_first_dot_consumed(self):
        # Only the first dot is the separator; the rest keeps its dots.
        assert (
            _drop_host_prefix("bedrock.anthropic.claude-3-5")[0]
            == "anthropic.claude-3-5"
        )

    # ------- case-insensitivity -------

    def test_uppercase_unknown_slash(self):
        assert _drop_host_prefix("UNKNOWN/openai-o1")[0] == "openai-o1"

    def test_mixed_case_bedrock_slash(self):
        assert _drop_host_prefix("Bedrock/Claude-3")[0] == "Claude-3"

    def test_mixed_case_dot_form(self):
        assert _drop_host_prefix("BEDROCK.claude-3")[0] == "claude-3"

    # ------- negative cases (still return None, not a tuple) -------

    def test_unknown_org_returns_none(self):
        assert _drop_host_prefix("random/openai-o1") is None

    def test_real_org_slash_returns_none(self):
        # A real org is not a host.
        assert _drop_host_prefix("openai/gpt-4") is None

    def test_anthropic_dot_form_returns_none(self):
        # A real org in Bedrock dot form is still not a host.
        assert _drop_host_prefix("anthropic.claude-3-5-sonnet") is None

    def test_no_separator_returns_none(self):
        assert _drop_host_prefix("unknown") is None

    def test_empty_string_returns_none(self):
        assert _drop_host_prefix("") is None

    def test_empty_suffix_after_slash_returns_none(self):
        assert _drop_host_prefix("unknown/") is None

    def test_empty_suffix_after_dot_returns_none(self):
        assert _drop_host_prefix("bedrock.") is None

    def test_slash_takes_precedence_over_dot(self):
        # With both separators present, the slash prefix is the one tested.
        assert (
            _drop_host_prefix("groq/anthropic.claude-3-5")[0]
            == "anthropic.claude-3-5"
        )

    def test_substring_of_host_name_does_not_match(self):
        # Full-token comparison: `bedrocky` is not `bedrock`.
        assert _drop_host_prefix("bedrocky/claude-3") is None


# ---------------------------------------------------------------------------
# _normalize_org direct unit tests
# ---------------------------------------------------------------------------


class TestNormalizeOrg:
    """`_normalize_org` contract, independent of the resolver wiring."""

    def test_deepseek_ai_to_deepseek(self):
        assert (
            _normalize_org("deepseek-ai/deepseek-v3")
            == "deepseek/deepseek-v3"
        )

    def test_meta_llama_to_meta(self):
        assert (
            _normalize_org("meta-llama/llama-3-8b")
            == "meta/llama-3-8b"
        )

    def test_zhipu_ai_to_zai(self):
        assert _normalize_org("zhipu-ai/glm-4.5") == "zai/glm-4.5"

    def test_moonshot_ai_to_moonshotai(self):
        assert (
            _normalize_org("moonshot-ai/kimi-k2")
            == "moonshotai/kimi-k2"
        )

    def test_unknown_org_returns_none(self):
        # Already-canonical org: not in `_ORG_ALIASES`.
        assert _normalize_org("openai/gpt-4") is None

    def test_no_slash_returns_none(self):
        assert _normalize_org("deepseek-ai") is None

    def test_empty_string_returns_none(self):
        assert _normalize_org("") is None

    def test_case_insensitive_org_match(self):
        # Org lookup is case-insensitive; the rest keeps its casing.
        assert (
            _normalize_org("DeepSeek-AI/DeepSeek-V3")
            == "deepseek/DeepSeek-V3"
        )


# ---------------------------------------------------------------------------
# Integration: suffix-strip after duplicated-org-prefix collapse
# ---------------------------------------------------------------------------


class TestDuplicatedOrgPlusSuffixIntegration:
    """Order of operations in `fuzzy_match`: suffix-strip, host-strip,
    dedup, org-alias. A value with both a duplicated prefix and a known
    suffix resolves because the suffix-stripped form still carries the
    duplicated prefix.
    """

    def test_dedup_alone_produces_expected_candidate(self):
        # Dedup alone leaves the `-fc` suffix in place.
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai-kimi-k2-fc")
            == "moonshotai/kimi-k2-fc"
        )

    def test_double_prefix_plus_fc_suffix_resolves(self):
        """Suffix-strip fires first (`moonshotai/moonshotai-kimi-k2`), then
        dedup collapses it to `moonshotai/kimi-k2`."""
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshotai/moonshotai-kimi-k2-fc", "model")
        assert result.canonical_id == "moonshotai/kimi-k2"
        assert result.strategy == "fuzzy"

    def test_suffix_first_then_dedup_path(self):
        """Suffix-strip yields `<org>/<org>-...`; dedup then runs on it."""
        store = _store_with_aliases(
            ("openai/o1", "model", "openai/o1", None, "confirmed")
        )
        resolver = Resolver(store)
        # `-prompt` is a known suffix → `openai/openai-o1` → dedup → `openai/o1`.
        result = resolver.resolve("openai/openai-o1-prompt", "model")
        assert result.canonical_id == "openai/o1"
        assert result.strategy == "fuzzy"

    def test_dedup_then_suffix_path_not_currently_covered(self):
        """Derived candidates are not re-suffix-stripped. Dedup only trims
        leading tokens, so it can never introduce a trailing suffix the
        suffix-first pass missed; a suffix on the original is enough.
        """
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshotai/moonshotai-kimi-k2-fc", "model")
        assert result.canonical_id == "moonshotai/kimi-k2"


class TestQuantizationAndHfStripping:
    """Quantization / HF-upload suffix stripping (see the PRECISION-LOSS
    POLICY in fuzzy.py): these collapses sacrifice precision deliberately."""

    def test_hf_suffix_strips(self):
        store = _store_with_aliases(
            ("meta/llama-2-7b", "model", "meta/llama-2-7b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-2-7b-hf", "model")
        assert result.canonical_id == "meta/llama-2-7b"
        assert result.strategy == "fuzzy"

    def test_fp8_suffix_strips(self):
        store = _store_with_aliases(
            ("deepseek/deepseek-v4-flash", "model", "deepseek/deepseek-v4-flash", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("deepseek/deepseek-v4-flash-fp8", "model")
        assert result.canonical_id == "deepseek/deepseek-v4-flash"

    def test_int4_awq_strips_as_one_unit(self):
        # Compound `-int4-awq` must strip in one step, not leave `-int4` behind.
        store = _store_with_aliases(
            ("meta/llama-3-70b", "model", "meta/llama-3-70b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-3-70b-int4-awq", "model")
        assert result.canonical_id == "meta/llama-3-70b"

    def test_gguf_suffix_strips(self):
        store = _store_with_aliases(
            ("alibaba/qwen2-72b", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen2-72b-gguf", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"


class TestLetterDigitDashCollapse:
    """Letter-digit boundary dashes: ``qwen-2-72b`` should match ``qwen2-72b``."""

    def test_qwen_dash_form_resolves(self):
        store = _store_with_aliases(
            ("alibaba/qwen2-72b", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen-2-72b", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"
        assert result.strategy == "fuzzy"

    def test_dash_form_with_instruct_suffix(self):
        # Digit-dash collapse must compose with the `-instruct` alias.
        store = _store_with_aliases(
            ("alibaba/qwen2-72b-instruct", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen-2-72b-instruct", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"

    def test_no_change_when_no_letter_digit_boundary(self):
        store = _store_with_aliases(
            ("alibaba/qwen2-72b", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen2-72b", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"
        assert result.strategy == "exact"

    def test_distinct_models_stay_distinct(self):
        # The collapse must not merge `gpt-4` into `gpt-4-mini`.
        store = _store_with_aliases(
            ("openai/gpt4", "model", "openai/gpt4", None, "confirmed"),
            ("openai/gpt4-mini", "model", "openai/gpt4-mini", None, "confirmed"),
        )
        resolver = Resolver(store)
        assert resolver.resolve("openai/gpt-4", "model").canonical_id == "openai/gpt4"
        assert resolver.resolve("openai/gpt-4-mini", "model").canonical_id == "openai/gpt4-mini"


# ---------------------------------------------------------------------------
# platform-capture suffix + fuzzy_match 3-tuple shape + threading
# ---------------------------------------------------------------------------


class TestStripAndCapturePlatformSuffix:
    """Host suffixes are captured as a platform, not silently dropped."""

    def test_together_suffix_captures_platform(self):
        assert _strip_and_capture_platform_suffix("llama-3-8b-together") == (
            "llama-3-8b",
            "togetherai",
        )

    def test_bedrock_suffix_captures_platform(self):
        assert _strip_and_capture_platform_suffix("claude-3-bedrock") == (
            "claude-3",
            "amazon-bedrock",
        )

    def test_openrouter_suffix_captures_platform(self):
        assert _strip_and_capture_platform_suffix("gpt-4o-openrouter") == (
            "gpt-4o",
            "openrouter",
        )

    def test_no_platform_suffix_returns_none(self):
        # A quant suffix is not a platform suffix.
        assert _strip_and_capture_platform_suffix("llama-3-8b-fp8") is None

    def test_case_insensitive_suffix(self):
        stem, platform = _strip_and_capture_platform_suffix("Claude-3-BEDROCK")
        assert stem == "Claude-3"
        assert platform == "amazon-bedrock"


class TestFuzzyMatchPlatformCapture:
    """`fuzzy_match` returns ``(canonical_id, confidence, inference_platform)``.
    Capture happens once, suffix beats prefix, and the platform never
    changes the resolved id."""

    def test_returns_three_tuple_on_match(self):
        # Bare alias so the suffix-stripped stem matches.
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        result = fuzzy_match("gpt-4o-openrouter", "model", 0.85, store)
        assert len(result) == 3
        canonical_id, confidence, platform = result
        assert canonical_id == "openai/gpt-4o"
        assert platform == "openrouter"

    def test_returns_three_tuple_on_no_match(self):
        store = _store_with_aliases()
        assert fuzzy_match("nonexistent-model-xyz", "model", 0.85, store) == (
            None,
            0.0,
            None,
        )

    def test_non_model_entity_returns_three_tuple(self):
        store = _store_with_aliases()
        assert fuzzy_match("anything", "benchmark", 0.85, store) == (None, 0.0, None)

    def test_prefix_host_captured_when_no_suffix(self):
        store = _store_with_aliases(
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed")
        )
        canonical_id, _, platform = fuzzy_match(
            "together/meta/llama-3-8b", "model", 0.85, store
        )
        assert canonical_id == "meta/llama-3-8b"
        assert platform == "togetherai"

    def test_suffix_beats_prefix_capture(self):
        # Prefix and suffix both present: the suffix wins.
        store = _store_with_aliases(
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed")
        )
        canonical_id, _, platform = fuzzy_match(
            "fireworks/meta/llama-3-8b-bedrock", "model", 0.85, store
        )
        assert canonical_id == "meta/llama-3-8b"
        assert platform == "amazon-bedrock"

    def test_capture_does_not_change_canonical(self):
        # Full resolver chain: the plain id hits exact match, which fuzzy
        # alone never sees.
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        resolver = Resolver(store)
        plain = resolver.resolve("gpt-4o", "model").canonical_id
        hosted = resolver.resolve("gpt-4o-openrouter", "model").canonical_id
        assert plain == hosted == "openai/gpt-4o"

    def test_no_host_token_captures_none(self):
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        _, _, platform = fuzzy_match("gpt-4o-fc", "model", 0.85, store)
        assert platform is None


class TestResolverThreadsPlatform:
    """`Resolver.resolve` threads the fuzzy-captured platform onto the result."""

    def test_resolve_sets_inference_platform_on_host_suffix(self):
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        result = Resolver(store).resolve("gpt-4o-openrouter", "model")
        assert result.canonical_id == "openai/gpt-4o"
        assert result.inference_platform == "openrouter"

    def test_resolve_leaves_platform_none_on_plain_id(self):
        store = _store_with_aliases(
            ("openai/gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        # Exact match runs no fuzzy capture.
        result = Resolver(store).resolve("openai/gpt-4o", "model")
        assert result.inference_platform is None


class TestBenchmarkRunConfigStrip:
    """Benchmarks get a run-configuration-only strip vocabulary. Eval
    methodology is not benchmark identity, but answer format and metric
    tokens are held back deliberately."""

    def _store(self):
        return _store_with_aliases(
            ("mmlu", "benchmark", "mmlu", None, "confirmed"),
            ("mmlu_college_physics", "benchmark", "mmlu", None, "confirmed"),
            ("bbh_snarks", "benchmark", "bbh", None, "confirmed"),
            ("global-mmlu", "benchmark", "global-mmlu", None, "confirmed"),
            ("truthfulqa", "benchmark", "truthfulqa", None, "confirmed"),
            ("acp_bench", "benchmark", "acpbench", None, "confirmed"),
        )

    def test_strips_cot_zeroshot_infix(self):
        result = Resolver(self._store()).resolve(
            "mmlu flan cot zeroshot college physics", "benchmark"
        )
        assert result.canonical_id == "mmlu"
        assert result.strategy == "fuzzy"

    def test_strips_cot_fewshot_infix(self):
        assert (
            Resolver(self._store()).resolve("bbh_cot_fewshot_snarks", "benchmark").canonical_id
            == "bbh"
        )

    def test_strips_gen_and_shot_suffixes(self):
        assert (
            Resolver(self._store()).resolve("global_mmlu_gen_0shot", "benchmark").canonical_id
            == "global-mmlu"
        )

    def test_holds_back_metric_tokens(self):
        # `_mc2` is metric identity; it must not strip onto the parent.
        assert (
            Resolver(self._store()).resolve("truthfulqa_de_mc2", "benchmark").canonical_id
            is None
        )

    def test_holds_back_answer_format_tokens(self):
        # `_bool` / `_mcq` are subsets the registry models separately.
        assert (
            Resolver(self._store()).resolve("acp_bench_bool", "benchmark").canonical_id is None
        )

    def test_no_run_config_token_is_a_no_op(self):
        assert (
            Resolver(self._store()).resolve("some_unknown_benchmark", "benchmark").canonical_id
            is None
        )
