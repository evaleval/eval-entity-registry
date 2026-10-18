"""Display-name humanization for canonical model slugs (`gpt-4o-2024-05-13` ->
`GPT-4o (2024-05-13)`). Consumers read `canonical_models.display_name`; they
do not re-humanize.

Rules, in order: strip the org prefix; strip and parenthesize a trailing date
suffix; format tokens (acronyms upper, mixed-case and vendor overrides, param
sizes `7b` -> `7B`, version tags `4o` and o-series kept, else capitalize);
hyphen-glue an acronym to a following bare version number (`GPT 5 Mini` ->
`GPT-5 Mini`) but not to a param size (`GPT 7B`).
"""

from __future__ import annotations

import re

ACRONYMS: frozenset[str] = frozenset(
    {
        "gpt",
        "glm",
        "llm",
        "vl",
        "vlm",
        "qvq",
        "qwq",
        "mt",
        "vit",
        "clip",
        "dit",
        "hf",
        "ocr",
        "tts",
        "asr",
        "moe",
        "mlp",
        "rlhf",
    }
)

# Acronyms rendered mixed case rather than ALL CAPS.
CASE_OVERRIDES: dict[str, str] = {
    "moe": "MoE",
    "vit": "ViT",
    "dit": "DiT",
}

# Only tokens the capitalize() default mangles; not a general branding registry.
TOKEN_OVERRIDES: dict[str, str] = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "stepfun": "StepFun",
    "moonshotai": "MoonshotAI",
    "mistralai": "MistralAI",
}

# Parameter-count units, not version letters: `7b` is never hyphen-glued.
_SIZE_SUFFIXES: frozenset[str] = frozenset({"b", "m", "k"})


def humanize_model_slug(slug: str) -> str:
    """Human display name for a bare slug or a full canonical id (the org
    prefix is dropped)."""
    if not slug:
        return ""
    if "/" in slug:
        slug = slug.split("/", 1)[1]

    slug, suffix = _strip_date_suffix(slug)

    tokens = slug.split("-")
    formatted = [_format_token(t) for t in tokens]

    out: list[str] = []
    i = 0
    while i < len(formatted):
        cur_lower = tokens[i].lower()
        if (
            i + 1 < len(formatted)
            and cur_lower in ACRONYMS
            and _is_version_token(tokens[i + 1])
        ):
            out.append(f"{formatted[i]}-{formatted[i + 1]}")
            i += 2
        else:
            out.append(formatted[i])
            i += 1

    return " ".join(out) + suffix


def _strip_date_suffix(slug: str) -> tuple[str, str]:
    """Pop a trailing date or 4-digit code; return (slug_without, ' (suffix)').

    Most specific pattern first: the bare 4-digit rule would otherwise take
    the `2025` of `-2025-08` as a code.
    """
    m = re.search(r"-(20\d{2}-\d{2}-\d{2})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(1)})"
    m = re.search(r"-(20\d{6})$", slug)
    if m:
        d = m.group(1)
        return slug[: m.start()], f" ({d[:4]}-{d[4:6]}-{d[6:8]})"
    m = re.search(r"-(20\d{2})-(\d{2})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(1)}-{m.group(2)})"
    # Cohere's `-MM-YYYY` (`command-r-08-2024`), rendered ISO-ordered.
    m = re.search(r"-(\d{2})-(20\d{2})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(2)}-{m.group(1)})"
    # Bare 4-digit code (`grok-4-0709`).
    m = re.search(r"-(\d{4})$", slug)
    if m:
        return slug[: m.start()], f" ({m.group(1)})"
    return slug, ""


def _format_token(tok: str) -> str:
    if not tok:
        return tok
    low = tok.lower()
    if low in CASE_OVERRIDES:
        return CASE_OVERRIDES[low]
    if low in ACRONYMS:
        return low.upper()
    if low in TOKEN_OVERRIDES:
        return TOKEN_OVERRIDES[low]
    # Param sizes: 7b, 1.5b, 30m; MoE active-expert a22b; 8x7b.
    if re.fullmatch(r"\d+(?:\.\d+)?[bmk]", low):
        return low[:-1] + low[-1].upper()
    if re.fullmatch(r"a\d+(?:\.\d+)?b", low):
        return "A" + low[1:-1] + "B"
    if re.fullmatch(r"\d+x\d+(?:\.\d+)?b", low):
        return low[:-1] + "B"
    # Version tags (`4o`) and o-series (`o3`) keep their case.
    if re.fullmatch(r"\d+(?:\.\d+)?[a-z]", low) and low[-1] not in _SIZE_SUFFIXES:
        return low
    if re.fullmatch(r"o\d+", low):
        return low
    return tok[0].upper() + tok[1:] if tok[0].isalpha() else tok


def _is_version_token(tok: str) -> bool:
    """True for a version token (`5`, `4.5`, `4o`), not a param size (`7b`)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]?)", tok.lower())
    if not m:
        return False
    return m.group(2) not in _SIZE_SUFFIXES
