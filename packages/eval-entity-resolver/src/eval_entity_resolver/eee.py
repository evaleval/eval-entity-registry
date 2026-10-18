"""EEE-specific preprocessing: an EEE field often encodes several entity types
(``evaluation_name`` carries both benchmark and metric); these helpers extract
resolvable strings first.

    extract_metric("Accuracy on IFEval")       # → "Accuracy"
    clean_eval_name("bfcl.live.live_accuracy")  # → "bfcl live"
"""
from __future__ import annotations

import re


# ------------------------------------------------------------------
# Metric extraction
# ------------------------------------------------------------------

def extract_metric(metric_desc: str) -> str:
    """Extract a resolvable metric name from an EEE evaluation description.

    * "X on Y" — ``"Accuracy on IFEval"`` → ``"Accuracy"``
    * Dot notation — ``"bfcl.live.live_accuracy"`` → ``"accuracy"``
    * Verbose description — ``"Chat accuracy - includes easy subsets"`` →
      ``"accuracy"`` (keyword extraction)
    * No keyword — ``"Global MMLU Lite - Arabic"`` → ``"score"``
    """
    text = metric_desc.strip()
    if not text:
        return text

    from_dot = False

    if "." in text and " " not in text:
        text = text.rsplit(".", 1)[1].replace("_", " ").strip()
        from_dot = True

    # Only a short prefix is an "X on Y" metric: a prose sentence with a
    # mid-sentence " on " ("GDPval-AA evaluates AI agents on …") must keep the
    # keywords disclosed later in it ("… as an Elo score").
    if not from_dot:
        m = re.match(r"^(.+?)\s+on\s+\S+", text, re.IGNORECASE)
        if m and len(m.group(1).split()) <= 3:
            text = m.group(1).strip()

    # Single bare words ("Accuracy", "F1", "EM") go straight to the resolver.
    word_count = len(text.split())
    needs_extraction = from_dot or word_count > 1

    if needs_extraction:
        canonical = _keyword_extract(text)
        if canonical:
            return canonical
        # A 2-3 word phrase may still be an alias ("Equivalent (CoT)"), so only
        # longer verbose text falls back to the generic bucket.
        if not from_dot and word_count > 3:
            return "score"

    return text


# Most specific first; among matches, the earliest position in the text wins
# (see `_keyword_extract`).
_METRIC_KEYWORDS: list[tuple[str, str]] = [
    # The trailing (?!\d) is load-bearing: `pass@1` matches inside `pass@10`.
    (r"pass@10(?!\d)",                   "Pass@10"),
    (r"pass@8(?!\d)",                    "Pass@8"),
    (r"pass@1(?!\d)",                    "Pass@1"),
    (r"pass[\s_-]*at[\s_-]*10(?!\d)",    "Pass@10"),
    (r"pass[\s_-]*at[\s_-]*8(?!\d)",     "Pass@8"),
    (r"pass[\s_-]*at[\s_-]*1(?!\d)",     "Pass@1"),
    (r"mean[\s_-]*win[\s_-]*rate",       "Mean Win Rate"),
    # AlpacaEval's variants are different estimators from the plain win rate.
    # `lc` is anchored so "Calc Win Rate" stays a plain win rate.
    (r"(?:(?<![a-z])lc|length[\s_-]*controlled)[\s_-]*win[\s_-]*rate", "Length-Controlled Win Rate"),
    (r"discrete[\s_-]*win[\s_-]*rate",   "Discrete Win Rate"),
    (r"win[\s_-]*rate",                  "Win Rate"),
    (r"mean[\s_-]*response[\s_-]*time",  "Mean Response Time"),
    # A fraction of scenario steps completed, not a mean of scores.
    (r"mean[\s_-]*progress[\s_-]*score", "Mean Progress Score"),
    (r"mean[\s_-]*score",                "Mean Score"),
    # HELM's near-miss exact-match variants are separately computed stats.
    (r"prefix[\s_-]*quasi[\s_-]*exact[\s_-]*match", "Prefix Quasi-Exact Match"),
    (r"quasi[\s_-]*prefix[\s_-]*exact[\s_-]*match", "Prefix Quasi-Exact Match"),
    (r"quasi[\s_-]*exact[\s_-]*match",   "Quasi-Exact Match"),
    (r"prefix[\s_-]*exact[\s_-]*match",  "Prefix Exact Match"),
    (r"exact[\s_-]*match",               "Exact Match"),
    (r"equivalent[\s_-]*\(?chain[\s_-]*of[\s_-]*thought", "Equivalent (CoT)"),
    (r"\bbrier\b",                       "Brier Score"),
    # Its own canonical, not the generic score. The LEXam header "Judge Scores
    # on Open Questions" is deliberately NOT matched: it is a source-scoped
    # alias in the seed, and this extractor is source-agnostic.
    (r"open[\s_-]*questions?[\s_-]*judge[\s_-]*scores?", "Open Question Judge Score"),
    (r"bleu[\s_-]*4",                    "BLEU-4"),
    (r"cot[\s_-]*correct",              "COT correct"),
    (r"wb[\s_-]*score",                  "WB Score"),
    (r"avg[\s_-]*attempts",              "Average Attempts"),
    (r"latency[\s_-]*mean",              "mean-latency"),
    (r"latency.*(?:p95|95th)",            "p95-latency"),
    (r"latency.*(?:std|standard)",        "latency-stddev"),
    (r"max[\s_-]*delta",                 "max-delta"),
    (r"benchmark\s+evaluation",          "score"),
    (r"outperform",                      "rank"),
    # Compound accuracy types, before generic accuracy.
    (r"ast[\s_-]*accuracy",              "AST Accuracy"),
    (r"ifeval[\s_-]*strict",             "IFEval Strict Acc"),
    (r"normali[sz]ed[\s_-]*accuracy",    "Normalized Accuracy"),
    (r"overall[\s_-]*accuracy",          "Accuracy"),
    (r"(?:ir)?relevance[\s_-]*detection[\s_-]*accuracy", "Accuracy"),
    (r"no[\s_-]*snippet[\s_-]*accuracy", "Accuracy"),
    (r"long[\s_-]*context[\s_-]*accuracy", "Accuracy"),
    (r"kv[\s_-]*accuracy",               "Accuracy"),
    (r"vector[\s_-]*accuracy",           "Accuracy"),
    (r"recursive[\s_-]*summarization[\s_-]*accuracy", "Accuracy"),
    (r"total[\s_-]*cost",                "cost"),
    (r"cost[\s_-]*per[\s_-]*task",       "cost-per-task"),
    (r"macro[\s_-]*f1",                  "Macro F1"),
    (r"micro[\s_-]*f1",                  "Micro F1"),
    # Benchmark-scoped safety rubrics: each has its own rubric and direction,
    # so the generic tail must not fold them onto `score` / `harmlessness`.
    (r"harmbench[\s_-]*refusal[\s_-]*score",
     "HarmBench Refusal Score"),
    (r"xstest[\s_-]*appropriateness[\s_-]*score",
     "XSTest Appropriateness Score"),
    (r"simple[\s_-]*safety[\s_-]*tests?[\s_-]*safety[\s_-]*score",
     "SimpleSafetyTests Safety Score"),
    (r"anthropic[\s_-]*red[\s_-]*team(?:ing)?[\s_-]*harmlessness[\s_-]*score",
     "Anthropic Red Team Harmlessness Score"),
    # Model-scored metrics whose tail ("F1", "Accuracy") the single-word
    # patterns would otherwise claim.
    (r"bertscore[\s_-]*f1?\b",            "BERTScore-F"),
    (r"bertscore[\s_-]*p(?:recision)?\b", "BERTScore-P"),
    (r"bertscore[\s_-]*r(?:ecall)?\b",    "BERTScore-R"),
    (r"bleurt[\s_-]*acc(?:uracy)?\b",     "BLEURT_acc"),
    (r"bleurt[\s_-]*diff\b",              "BLEURT_diff"),
    (r"bleurt[\s_-]*max\b",               "BLEURT_max"),
    # Generic single-word patterns.
    (r"\baccuracy\b",                    "Accuracy"),
    (r"\bacc\b",                         "Accuracy"),
    (r"\bscores?\b",                     "score"),
    (r"\bf1\b",                          "F1"),
    (r"\bem\b",                          "Exact Match"),
    (r"\belo\b",                         "Elo Rating"),
    (r"\branks?\b",                      "rank"),
    (r"\bcosts?\b",                      "cost"),
    (r"\bharmlessness\b",                "harmlessness"),
    (r"\bstddev\b",                      "stddev"),
]


def _keyword_extract(text: str) -> str | None:
    """Canonical metric name for the earliest-positioned keyword in *text*."""
    lower = text.lower()
    best: str | None = None
    best_pos = len(lower) + 1
    for pattern, canonical in _METRIC_KEYWORDS:
        m = re.search(pattern, lower)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best = canonical
    return best


# ------------------------------------------------------------------
# Benchmark-name cleaning
# ------------------------------------------------------------------

# Trailing metric patterns for space-separated names ("Gaming Score" →
# "Gaming"); first match wins.
_TRAILING_METRIC_RE: list[str] = [
    r"mean\s+win\s+rate$",
    r"mean\s+response\s+time$",
    r"mean\s+score$",
    r"win\s+rate$",
    r"avg\s+attempts$",
    r"avg\s+latency\s+ms$",
    r"cost\s+per\s+\d+\s+calls\s+usd$",
    r"cost\s+per\s+task$",
    r"pass@\d+$",
    r"\b(?:score|accuracy|acc|elo|rank|f1|em)$",
]


# Aggregate markers never carry benchmark identity in a dotted ``evaluation_name``.
_AGGREGATE_SEGMENTS = {"overall"}


def prepare_eval_name_segments(eval_name: str | None) -> list[str] | None:
    """Split a dotted ``evaluation_name`` into candidate identity segments,
    collapsing identical adjacent segments (``bbq.bbq.overall`` →
    ``bbq.overall``) and dropping aggregate markers and numeric fragments
    (``terminal-bench-2.0`` → ``2``, ``0``). Returns None for names
    ``clean_eval_name`` would not dot-split, so both paths see the same names.
    """
    if not eval_name or not isinstance(eval_name, str):
        return None
    name = eval_name.strip()
    if "." not in name or " " in name:
        return None

    segments: list[str] = []
    for part in name.split("."):
        part = part.strip()
        if not part or part.isdigit():
            continue
        if part.lower() in _AGGREGATE_SEGMENTS:
            continue
        if segments and segments[-1].lower() == part.lower():
            continue
        segments.append(part)
    return segments or None


def clean_eval_name(eval_name: str) -> str:
    """Strip embedded metric information from an ``evaluation_name``, leaving
    the benchmark portion.

    * Dot notation — ``"bfcl.live.live_accuracy"`` → ``"bfcl live"`` (the last
      segment is the metric)
    * Underscore suffix — ``"fibble1_arena_win_rate"`` → ``"fibble1 arena"``
    * Trailing words — ``"Gaming Score"`` → ``"Gaming"``
    """
    name = eval_name.strip()
    if not name:
        return name

    if "." in name and " " not in name:
        parts = name.rsplit(".", 1)[0].split(".")
        return " ".join(p.replace("_", " ") for p in parts)

    has_underscores = "_" in name and " " not in name
    normalized = name.replace("_", " ") if has_underscores else name

    lower = normalized.lower()
    for pattern in _TRAILING_METRIC_RE:
        m = re.search(pattern, lower)
        if m:
            prefix = normalized[: m.start()].strip()
            if prefix:
                return prefix
            break  # the whole name is the metric; keep it as-is

    return name
