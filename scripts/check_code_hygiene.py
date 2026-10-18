#!/usr/bin/env python3
"""Code-hygiene gate for staged changes. Runs from hooks/pre-commit
(enable with `git config core.hooksPath hooks`).

The standard a comment is held to:

  A comment earns its place only by telling the reader something the code
  cannot: an invariant the code relies on but does not express; a
  non-obvious WHY (an outside constraint such as an API quirk or data shape,
  or a deliberate choice that would otherwise look like a mistake); or a
  temporary measure together with what makes it safe to remove.

  A comment must not narrate what the code does, restate history (what
  changed, when, which run or incident), carry working-session reasoning
  (alternatives considered, budgets, step-by-step justification), repeat a
  rationale stated elsewhere (state it once where it is enforced and point
  by file or function name), or lean on local-only documents for its
  explanation. Keep the rule, drop the story: one sentence is the norm, and
  a block beyond a few lines must be a real design note on a real invariant.

This script catches the mechanically detectable violations: references to
ephemeral working artifacts (spec sections, decision labels, phase/session
tags), tombstones describing what the code used to do, dated observations
and run ids, and long comment blocks. Judgement cases are for review.

Checks only lines ADDED in the staged diff, so existing occurrences don't
block unrelated commits. Bypass a false positive with `git commit
--no-verify` (and consider tightening the offending rule below).

To add a rule, append to RULES.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import NamedTuple


class Rule(NamedTuple):
    name: str
    pattern: re.Pattern
    hint: str


RULES = [
    Rule(
        "spec-section-ref",
        re.compile("§"),  # the section sign
        "Spec section references don't survive the session that wrote them. "
        "State the rule itself in the comment instead.",
    ),
    Rule(
        "spec-label-ref",
        re.compile(r"[Ss]pec[.,]?\s*[A-Z]\d+\b"),
        "Decision labels like 'spec D9' / 'spec P3' are one-time artifacts. "
        "Say what the decision was, not where it was numbered.",
    ),
    Rule(
        "spec-deixis",
        re.compile(r"\b(per|as per|see)\s+(the\s+)?spec\b|\bspec\s+section\b", re.I),
        "Don't point readers at 'the spec'. Inline the relevant rule.",
    ),
    Rule(
        "section-number-ref",
        re.compile(r"\bsection\s+(\d|\([a-z]\))", re.I),
        "Numbered-section references rot. Describe the rule in place.",
    ),
    Rule(
        "design-point-ref",
        re.compile(r"\bdesign\s+pt\.?\s*\d", re.I),
        "Design-point numbers are meaningless outside the design doc. "
        "State the design decision directly.",
    ),
    Rule(
        "phase-ref",
        re.compile(r"\bphase[-\s]?\d", re.I),
        "Phase numbers only make sense inside a plan. Name the feature or "
        "component, not the phase that built it.",
    ),
    Rule(
        "session-ref",
        re.compile(r"\bsession\s+\d+[a-z]?\b|\bthis (session|conversation|chat)\b", re.I),
        "Working-session references are unintelligible to later readers.",
    ),
    Rule(
        "tombstone-previously",
        re.compile(r"\b(was|were)\s+(previously|formerly)\b", re.I),
        "Tombstone: git history records what the code used to do; the "
        "comment should only explain what it does now.",
    ),
    Rule(
        "tombstone-used-to",
        re.compile(r"\bused\s+to\s+(be|do|have|return|live|contain)\b", re.I),
        "Tombstone: describe current behaviour; history lives in git.",
    ),
    Rule(
        "dated-observation",
        re.compile(r"\b(observed|seen|measured|as of|since)\s+20\d\d-\d\d(-\d\d)?\b", re.I),
        "Dates of observations belong in the commit message, not the code. "
        "State the rule the observation established.",
    ),
    Rule(
        "run-id-ref",
        re.compile(r"actions/runs/\d+|\brun\s+#?\d{8,}\b|\bissue\s+#\d+\b", re.I),
        "Run and issue numbers are session artifacts. Describe the condition, "
        "not the incident.",
    ),
    Rule(
        "session-deixis",
        re.compile(r"\b(yesterday|this (change|commit|PR|session)|we (saw|found|noticed|hit))\b", re.I),
        "Comments outlive the session. Say what is true, not what happened.",
    ),
    Rule(
        "tombstone-dated-change",
        re.compile(r"\b(renamed|removed|deleted|moved|dropped|changed)\s+(on|in)\s+20\d\d", re.I),
        "Dated change notes belong in the commit message, not the file.",
    ),
]

CHECKED_EXTENSIONS = {
    ".py", ".pyi", ".sql", ".sh", ".bash", ".zsh",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts",
    ".css", ".scss", ".yaml", ".yml", ".toml",
}

# Design docs may legitimately carry section numbering; frozen trees are
# not held to current hygiene.
EXCLUDED_DIR_PARTS = {"notes", "specs", "docs", "vendor", "node_modules", "deprecated_refs"}
EXCLUDED_PATHS = {"scripts/check_code_hygiene.py", "hooks/pre-commit"}


def is_checked(path: str) -> bool:
    if path in EXCLUDED_PATHS:
        return False
    parts = path.split("/")
    if any(p in EXCLUDED_DIR_PARTS or p.startswith("DEPRECATED") for p in parts[:-1]):
        return False
    dot = parts[-1].rfind(".")
    return dot != -1 and parts[-1][dot:] in CHECKED_EXTENSIONS


def staged_added_lines():
    """Yield (path, line_number, text) for every line added in the staged diff."""
    diff = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--no-color", "--diff-filter=ACMR"],
        capture_output=True, text=True, check=True,
    ).stdout
    path, lineno = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
        elif raw.startswith("+") and not raw.startswith("+++"):
            if path is not None:
                yield path, lineno, raw[1:]
            lineno += 1


# Consecutive ADDED comment lines beyond this count are flagged: a block that
# long is almost always narration or history rather than a rule.
MAX_COMMENT_BLOCK = 8
_COMMENT_LINE = re.compile(r"^\s*(#|//|\*|\"\"\")")
LONG_BLOCK_RULE = Rule(
    "long-comment-block",
    re.compile(r"(?!)"),  # never matches a single line; applied by block below
    f"More than {MAX_COMMENT_BLOCK} consecutive comment lines. Cut it to the "
    "rule being stated; if it is a real design note, say which invariant it guards.",
)


def _long_comment_blocks(lines):
    """Yield (path, first_lineno, text) for each run of added comment lines
    longer than MAX_COMMENT_BLOCK (contiguous by line number)."""
    run: list[tuple[str, int, str]] = []
    def flush():
        if len(run) > MAX_COMMENT_BLOCK:
            yield run[0]
        run.clear()
    for path, lineno, text in lines:
        if run and (path != run[-1][0] or lineno != run[-1][1] + 1 or not _COMMENT_LINE.match(text)):
            yield from flush()
        if _COMMENT_LINE.match(text):
            run.append((path, lineno, text))
    yield from flush()


def main() -> int:
    findings = []
    added = [t for t in staged_added_lines() if is_checked(t[0])]
    for path, lineno, text in added:
        for rule in RULES:
            if rule.pattern.search(text):
                findings.append((path, lineno, rule, text))
    for path, lineno, text in _long_comment_blocks(added):
        findings.append((path, lineno, LONG_BLOCK_RULE, text))
    if not findings:
        return 0
    print(f"code-hygiene: {len(findings)} problem(s) in staged changes\n", file=sys.stderr)
    for path, lineno, rule, text in findings:
        print(f"  {path}:{lineno}  [{rule.name}]", file=sys.stderr)
        print(f"      {text.strip()[:120]}", file=sys.stderr)
        print(f"      -> {rule.hint}\n", file=sys.stderr)
    print("Fix the lines above, or bypass once with: git commit --no-verify", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
