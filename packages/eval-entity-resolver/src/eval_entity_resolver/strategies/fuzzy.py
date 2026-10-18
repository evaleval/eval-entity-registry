"""Fuzzy matching strategy: targeted stem rewrites (known suffix strips, host
and org prefix handling) checked against the exact/normalized alias index.
No generic string similarity, which would merge distinct versions, sizes and
variants (``gpt-5-mini`` must never fold into ``gpt-5``).

PRECISION-LOSS POLICY: the strip list collapses ``-hf`` re-uploads and
quantization suffixes (``-fp8``, ``-int4``, ``-awq``, ``-gguf``, ...) onto the
base canonical when no quantized canonical is aliased. Benchmark scores differ
measurably between quantization levels, so a resolved row may represent any
reported quantization; consumers needing that precision must read the original
``model_route_id`` and ``generation_args``, not the canonical_id alone.
"""
from __future__ import annotations

import re
from typing import Optional

from eval_entity_resolver.normalization import normalize
from eval_entity_resolver.strategies._platform_map import get_host_token_platform


# Stripped only from the end of the raw value; longer suffixes first so a
# shorter one cannot pre-empt a compound (`-int8` vs `-int8-awq`). Host
# suffixes (`-together`, `-bedrock`, `-openrouter`) are deliberately absent:
# they live in `_SUFFIX_PLATFORM_MAP` so the platform is captured, not lost.
_STRIP_SUFFIXES = [
    # Evaluation-mode suffixes (BFCL).
    "-fc",
    "-prompt",
    # Reasoning-effort tiers. `-max` IS an effort tier (arc-agi's low/high/max
    # ladder); genuine `-max` products (qwen3-max) are safe because
    # exact/normalized matching wins before fuzzy ever strips.
    "-xhigh-effort",
    "-high-effort",
    "-medium-effort",
    "-low-effort",
    "-minimal-effort",
    "-non-reasoning",
    "-xhigh",
    "-high",
    "-medium",
    "-low",
    "-minimal",
    "-max",
    # Thinking-style suffixes.
    "-highthinking",
    "-nothinking",
    "-nothink",
    "-thinking-none",
    # HF re-upload of an official release: same weights.
    "-hf",
    # Quantization variants: see the module docstring's precision-loss policy.
    "-int4-awq",
    "-int8-awq",
    "-int4-gptq",
    "-int8-gptq",
    "-fp4",
    "-fp8",
    "-fp16",
    "-bf16",
    "-int4",
    "-int8",
    "-q4",
    "-q8",
    "-awq",
    "-gptq",
    "-gguf",
    "-quant",
]

# Tail patterns applied after the literal suffixes; each must anchor on `$`.
_STRIP_SUFFIX_PATTERNS: list[re.Pattern[str]] = [
    # Drops the whole thinking-budget tail (`-thinking-16k` → base); the
    # `_THINKING_BUDGET_PRESERVE_RE` candidate is tried first so an aliased
    # thinking-mode canonical wins when it exists.
    re.compile(r"-thinking-\d+k$", re.IGNORECASE),
    # A trailing packed date (`-20251101`) is deliberately NOT stripped: that
    # always lands on the bare family and loses the snapshot's identity and
    # `release_date`. Snapshots are minted as their own canonicals with a
    # version edge instead, and an aliased snapshot wins before fuzzy runs.
]

# Keeps `-thinking`, drops only the `-Nk` budget.
_THINKING_BUDGET_PRESERVE_RE = re.compile(r"(.*-thinking)-\d+k$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Benchmark run-configuration vocabulary
# ---------------------------------------------------------------------------
# Harness task names weld the run configuration into the benchmark name
# (`mmlu_flan_cot_zeroshot_college_physics`); methodology is not benchmark
# identity, so these tokens are dropped anywhere in the name. Deliberately NOT
# here: `mc1`/`mc2` (metric identity) and `bool`/`mcq` (answer format, which the
# registry models as distinct subsets).
_BENCHMARK_RUN_CONFIG_TOKENS = frozenset({
    "cot", "zeroshot", "fewshot", "0shot", "flan", "gen",
})
_BENCHMARK_SHOT_TOKEN_RE = re.compile(r"^\d+shot$")


def _strip_run_config_tokens(value: str) -> str | None:
    """Remaining tokens space-joined after dropping run-configuration tokens, or
    None when nothing was dropped or nothing would survive."""
    tokens = re.split(r"[\s_\-]+", value.strip())
    kept = [
        t for t in tokens
        if t.lower() not in _BENCHMARK_RUN_CONFIG_TOKENS
        and not _BENCHMARK_SHOT_TOKEN_RE.match(t.lower())
    ]
    if not kept or len(kept) == len(tokens):
        return None
    return " ".join(kept)


def _benchmark_stem_match(
    raw_value: str, alias_store, source_config: Optional[str]
) -> tuple[Optional[str], float, None]:
    """One candidate, built by removing run-configuration tokens. Runs after
    exact and normalized, so a canonical whose name genuinely carries such a
    token still wins on its own spelling."""
    candidate = _strip_run_config_tokens(raw_value)
    if candidate is None:
        return None, 0.0, None
    canonical_id = (
        alias_store.lookup(candidate, "benchmark", source_config)
        or alias_store.get_normalized_lookup("benchmark", source_config).get(
            normalize(candidate)
        )
    )
    if canonical_id is None:
        return None, 0.0, None
    return canonical_id, _STEM_CONFIDENCE, None

from eval_entity_resolver.fold import _ORG_ALIASES, _norm_org_key, dev_org_of_prefix  # noqa: E402,F401


def _canon_org(prefix: str, org_map: dict) -> str:
    """Canonical developer org for an id-prefix (lowercase key, then
    separator-stripped key), else the prefix itself."""
    return org_map.get(prefix.lower()) or org_map.get(_norm_org_key(prefix)) or prefix


def _orgs_agree(
    raw_value: str,
    matched_id: str,
    org_map: Optional[dict] = None,
    known_orgs: Optional[frozenset] = None,
) -> bool:
    """False-merge guard: True unless `raw_value` and `matched_id` carry org
    prefixes that fold to DIFFERENT developers.

    A fuzzy stem match must never cross developers (`DevQuasar/...` must not
    match `DevQuasar-3/...`); same-developer namespaces (meta-llama vs facebook)
    fold equal under `org_map` and still match; org-less candidates have no org
    to disagree on. `org_map` is the store-backed namespace -> developer map
    (falls back to `_ORG_ALIASES`). Two prefixes resolving to DIFFERENT ids in
    `known_orgs` are never separator-merged (`Enno-Ai` vs `EnnoAi`, both real
    rows, stay distinct); the case/separator fold applies only to the
    unregistered tail (`aleph-alpha` whose only registered twin is `AlephAlpha`)."""
    if "/" not in raw_value or "/" not in matched_id:
        return True
    m = org_map if org_map is not None else _ORG_ALIASES
    ca = _canon_org(raw_value.split("/", 1)[0], m)
    cm = _canon_org(matched_id.split("/", 1)[0], m)
    if ca == cm:
        return True
    if known_orgs is not None and ca in known_orgs and cm in known_orgs:
        return False
    return _norm_org_key(ca) == _norm_org_key(cm)

# Hosting platforms, gateways and missing-developer placeholders: dropped as a
# prefix so the bare model body is tried, since they are not developers.
# Deliberately broader than the `inference_platforms` seed: stripping for
# matching is independent of platform attribution. Values come from the seed
# map; tokens absent there capture `None`.
_HOST_PREFIX_TOKENS: tuple[str, ...] = (
    "unknown",
    "bedrock", "amazon-bedrock", "aws-bedrock",
    "azure", "azure-openai", "azure-cognitive-services",
    "vertex", "google-vertex", "vertex-anthropic",
    "fireworks", "fireworks-ai",
    "groq",
    "together", "togetherai", "together-ai",
    "openrouter",
    "perplexity-agent",
    "deepinfra", "anyscale", "novita", "novita-ai", "replicate",
    "ollama", "ollama-cloud",
    "github-models", "github-copilot",
    "lambda", "baseten", "modal", "runpod", "cerebras",
    "sap-ai-core", "cloudflare-ai-gateway", "aihubmix",
    "kilo", "vercel", "llmgateway", "poe",
)

_HOST_PREFIXES_TO_STRIP: dict[str, Optional[str]] = {
    token: (None if token == "unknown" else get_host_token_platform(f"{token}/"))
    for token in _HOST_PREFIX_TOKENS
}

# Host suffixes: captured as the platform side-value rather than discarded.
_SUFFIX_PLATFORM_MAP: dict[str, Optional[str]] = {
    suffix: get_host_token_platform(suffix)
    for suffix in ("-together", "-bedrock", "-openrouter")
}


def _drop_duplicated_org_prefix(value: str) -> str | None:
    """Collapse a repeated-org-prefix typo (``openai/openai-o1`` →
    ``openai/o1``), or None when the prefix is not duplicated.

    Shapes: ``<org>/<org>[-_/]<rest>`` and the route_id slug forms
    ``<org>__<org>[-_]<rest>`` / ``<org>__<org>__<rest>``. Token equality is
    case-insensitive; the original casing is preserved so exact aliases still
    match. The two leading tokens must be equal in full: a substring overlap
    (``gpt-4/gpt-4-turbo``) is a real two-segment path, not a typo, and the
    heuristic only fires when the org token has no hyphen (model-family
    prefixes like ``gpt-4`` do; hyphenated orgs such as ``mistral-ai`` are
    rewritten by the org-alias pass instead).
    """
    if not value:
        return None

    if "/" in value:
        first_slash = value.index("/")
        org = value[:first_slash]
        rest = value[first_slash + 1:]
        if not org or not rest:
            return None
        if "-" in org:
            return None
        org_lower = org.lower()
        if "/" in rest:
            second, after = rest.split("/", 1)
            if second.lower() == org_lower and after:
                return f"{org}/{after}"
        for sep in ("-", "_"):
            prefix = org_lower + sep
            if rest.lower().startswith(prefix) and len(rest) > len(prefix):
                return f"{org}/{rest[len(prefix):]}"

    if "__" in value:
        first = value.index("__")
        org = value[:first]
        rest = value[first + 2:]
        if not org or not rest:
            return None
        if "-" in org:
            return None
        org_lower = org.lower()
        if "__" in rest:
            second, after = rest.split("__", 1)
            if second.lower() == org_lower and after:
                return f"{org}__{after}"
        for sep in ("-", "_"):
            prefix = org_lower + sep
            if rest.lower().startswith(prefix) and len(rest) > len(prefix):
                return f"{org}__{rest[len(prefix):]}"

    return None


def _drop_host_prefix(value: str) -> tuple[str, Optional[str]] | None:
    """``(rest, platform_id)`` when the developer prefix (``host/`` or
    ``host.``) is a known hosting platform, else None. ``platform_id`` is None
    when the host token has no attribution in the seed map; ``rest`` keeps the
    raw casing.
    """
    if "/" in value:
        org, rest = value.split("/", 1)
        org_lower = org.lower()
        if org_lower in _HOST_PREFIXES_TO_STRIP and rest:
            return (rest, _HOST_PREFIXES_TO_STRIP[org_lower])
    if "." in value:
        # Bedrock-style `bedrock.anthropic.claude-3-5-sonnet`: only the
        # segment before the first dot can be the host.
        first_dot = value.index(".")
        org = value[:first_dot]
        rest = value[first_dot + 1:]
        org_lower = org.lower()
        if org_lower in _HOST_PREFIXES_TO_STRIP and rest:
            return (rest, _HOST_PREFIXES_TO_STRIP[org_lower])
    return None

# Below exact (1.0) and normalized (0.95) so the tier is recoverable from the
# confidence alone.
_STEM_CONFIDENCE = 0.90


def _strip_suffix(value: str) -> str | None:
    """Stem after stripping one known suffix, or None when none matched."""
    lower = value.lower()
    for suffix in _STRIP_SUFFIXES:
        if lower.endswith(suffix):
            return value[: len(value) - len(suffix)]
    for pattern in _STRIP_SUFFIX_PATTERNS:
        m = pattern.search(value)
        if m:
            return value[: m.start()]
    return None


def _strip_and_capture_platform_suffix(value: str) -> tuple[str, Optional[str]] | None:
    """``(stem, platform_id)`` after stripping a trailing host suffix
    (`-together` / `-bedrock` / `-openrouter`), or None. The platform is a side
    value, never embedded in the candidate."""
    lower = value.lower()
    for suffix, platform in _SUFFIX_PLATFORM_MAP.items():
        if lower.endswith(suffix):
            stem = value[: len(value) - len(suffix)]
            return (stem, platform)
    return None


# ``qwen-2-72b`` ↔ ``qwen2-72b``. Safe because distinguishing tokens sit on the
# other side of a separator (``gpt-4`` vs ``gpt-4-mini`` keeps ``-mini``).
_LETTER_DIGIT_DASH = re.compile(r"([a-zA-Z])-(\d)")


def _collapse_letter_digit_dashes(value: str) -> str | None:
    """Value with letter-digit boundary dashes removed, or None if unchanged."""
    collapsed = _LETTER_DIGIT_DASH.sub(r"\1\2", value)
    if collapsed == value:
        return None
    return collapsed


def _normalize_org(value: str) -> str | None:
    """Value with a known org-alias prefix rewritten, or None."""
    if "/" not in value:
        return None
    org, rest = value.split("/", 1)
    canonical_org = _ORG_ALIASES.get(org.lower())
    if canonical_org is None:
        return None
    return f"{canonical_org}/{rest}"


# Anchored at a word boundary so `gptq-int4` and `o3rganization` do not match.
_OPENAI_MODEL_BODY_RE = re.compile(
    r"^(?:gpt|chatgpt|o\d|davinci|babbage|curie|ada)(?:-|$)",
    re.IGNORECASE,
)


def _is_openai_shaped(value: str) -> bool:
    """True if `value` looks like an OpenAI model handle.

    ISO-date peeling is scoped to OpenAI because only its cadence emits daily
    snapshots (`gpt-5-2025-08-07`) of an aliased month form (`gpt-5-2025-08`);
    other orgs use other date formats (Anthropic `YYYYMMDD`, Allen-AI `1124`),
    so peeling broadly would conflate distinct snapshots.
    """
    if value.lower().startswith("openai/"):
        return True
    body = value.split("/", 1)[1] if "/" in value else value
    return bool(_OPENAI_MODEL_BODY_RE.match(body))


# Strict component widths so a context length (`-32k`) is never peeled; the
# year range is checked at strip time so future snapshots need no regex bump.
_ISO_DATE_FULL_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})-(\d{2})$")
_ISO_DATE_MONTH_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})$")
_ISO_DATE_YEAR_RE = re.compile(r"^(.+)-(\d{4})$")


def _strip_openai_iso_date(value: str) -> list[str]:
    """Progressively truncated candidates for an OpenAI-shaped value ending in
    an ISO date, each still carrying at least one date component:
    `openai/gpt-5-2025-08-07` → `[openai/gpt-5-2025-08, openai/gpt-5-2025]`.

    The bare-family candidate is deliberately omitted: collapsing a snapshot
    onto its family pointer loses the snapshot's identity and `release_date`;
    an unknown snapshot is minted as its own canonical with a version edge
    instead. A year-only value therefore yields `[]`.
    """
    if not _is_openai_shaped(value):
        return []

    # Rejects 4-digit tokens that are not years (parameter sizes, batch numbers).
    def _is_release_year(s: str) -> bool:
        try:
            y = int(s)
        except ValueError:
            return False
        return 2015 <= y <= 2035

    candidates: list[str] = []
    m = _ISO_DATE_FULL_RE.match(value)
    if m:
        prefix, y, mo, d = m.groups()
        if _is_release_year(y) and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            candidates.append(f"{prefix}-{y}-{mo}")
            candidates.append(f"{prefix}-{y}")
            return candidates

    m = _ISO_DATE_MONTH_RE.match(value)
    if m:
        prefix, y, mo = m.groups()
        if _is_release_year(y) and 1 <= int(mo) <= 12:
            candidates.append(f"{prefix}-{y}")
            return candidates

    return candidates


def fuzzy_match(
    raw_value: str,
    entity_type: str,
    threshold: float,  # unused; kept for API compat
    alias_store,
    source_config: Optional[str] = None,
    org_dev_map: Optional[dict] = None,
    known_orgs: Optional[frozenset] = None,
) -> tuple[Optional[str], float, Optional[str]]:
    """Returns ``(canonical_id, confidence, inference_platform)``; canonical_id
    is None on no match.

    ``inference_platform`` is a SIDE VALUE captured from a host token and never
    embedded in the matched candidate, so capture cannot alter the canonical_id.
    A suffix capture (`-bedrock`) takes priority over a prefix capture
    (`together/`): a host token in the model name is the stronger per-run signal.
    """
    if entity_type == "benchmark":
        return _benchmark_stem_match(raw_value, alias_store, source_config)

    # The rewrites below are model-specific; on metrics/harnesses they would
    # merge unrelated entities that merely share a host-like prefix or tail.
    if entity_type != "model":
        return None, 0.0, None

    candidates_to_try: list[str] = []
    captured_platform: Optional[str] = None

    # Tried before the generic strip so an aliased thinking-mode canonical
    # (`model-thinking`) wins over the drop-thinking stem (`model`).
    preserve_match = _THINKING_BUDGET_PRESERVE_RE.match(raw_value)
    if preserve_match:
        candidates_to_try.append(preserve_match.group(1))

    # Before the generic strip so the host suffix is captured, not discarded;
    # the stem is re-stripped (`model-fc-together` → capture, then drop `-fc`).
    suffix_capture = _strip_and_capture_platform_suffix(raw_value)
    if suffix_capture:
        stem, platform = suffix_capture
        candidates_to_try.append(stem)
        if platform is not None:
            captured_platform = platform
        stem_stripped = _strip_suffix(stem)
        if stem_stripped:
            candidates_to_try.append(stem_stripped)

    stripped = _strip_suffix(raw_value)
    if stripped:
        candidates_to_try.append(stripped)
        double = _strip_suffix(stripped)
        if double:
            candidates_to_try.append(double)

    for val in [raw_value] + candidates_to_try[:]:
        host = _drop_host_prefix(val)
        if host:
            bare, platform = host
            candidates_to_try.append(bare)
            if captured_platform is None and platform is not None:
                captured_platform = platform
            stripped_bare = _strip_suffix(bare)
            if stripped_bare:
                candidates_to_try.append(stripped_bare)

    # Before org normalization so the collapsed string can still be rewritten.
    for val in [raw_value] + candidates_to_try[:]:
        deduped = _drop_duplicated_org_prefix(val)
        if deduped:
            candidates_to_try.append(deduped)

    for val in [raw_value] + candidates_to_try[:]:
        rewritten = _normalize_org(val)
        if rewritten:
            candidates_to_try.append(rewritten)

    # Last, so it composes with every earlier rewrite.
    for val in [raw_value] + candidates_to_try[:]:
        collapsed = _collapse_letter_digit_dashes(val)
        if collapsed:
            candidates_to_try.append(collapsed)

    for val in [raw_value] + candidates_to_try[:]:
        for peeled in _strip_openai_iso_date(val):
            candidates_to_try.append(peeled)

    norm_lookup = alias_store.get_normalized_lookup(entity_type, source_config)

    for candidate in candidates_to_try:
        # Org-agree against the candidate, not raw_value: a hosted raw
        # (`together/meta-llama/Llama-3-8B`) carries the host as its prefix.
        exact_id = alias_store.lookup(candidate, entity_type, source_config)
        if exact_id is not None and _orgs_agree(candidate, exact_id, org_dev_map, known_orgs):
            return exact_id, _STEM_CONFIDENCE, captured_platform

        norm = normalize(candidate)
        canonical_id = norm_lookup.get(norm)
        if canonical_id is not None and _orgs_agree(candidate, canonical_id, org_dev_map, known_orgs):
            return canonical_id, _STEM_CONFIDENCE, captured_platform

    return None, 0.0, None
