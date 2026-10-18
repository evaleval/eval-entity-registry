"""Seed-time org attribution for slash-less draft ids whose leading token is
a known developer org (`cohere-march-2024`). The id stays verbatim; only
org_id is set. An entry that duplicates a real repo of the same developer
(`deepseek-v2-lite-chat` vs `deepseek-ai/DeepSeek-V2-Lite-Chat`) is instead
returned for a fold: org-tagging it alone would collide on the humanized
display name."""
import re
from collections import defaultdict

_LEAD_RE = re.compile(r"([A-Za-z0-9]+)([-.])(.+)")
_PLACEHOLDER_ORGS = frozenset({"unknown", "none", "na", ""})
_SEP = re.compile(r"[-_.\s]+")


def _dev_of(e: dict, hf_to_dev: dict) -> str | None:
    org = e.get("org_id")
    if org and str(org).lower() not in _PLACEHOLDER_ORGS:
        return str(org).lower()
    cid = str(e.get("id", ""))
    if "/" in cid:
        return hf_to_dev.get(cid.split("/", 1)[0].lower())
    m = _LEAD_RE.match(cid)
    return hf_to_dev.get(m.group(1).lower()) if m else None


def _name_key(cid: str, dev: str | None) -> str:
    """Model name normalized after stripping a leading developer token, so
    `deepseek-ai/DeepSeek-V2-Lite-Chat` and `deepseek-v2-lite-chat` share a key."""
    name = cid.split("/", 1)[1] if "/" in cid else cid
    nl = name.lower()
    if dev:
        for tok in (dev, dev.replace("-", ""), dev.split("-")[0]):
            if tok and nl.startswith(tok):
                nl = nl[len(tok):]
                break
    return _SEP.sub("", nl).strip("-_. ")


def attribute_orgs(entries: list[dict], hf_to_dev: dict[str, str]):
    """Attribute org_id in place for standalone malformed-org drafts; return
    duplicates of a real repo (same developer + same org-token-stripped name)
    in the merge map {draft -> real repo} for the fold.

    Returns (entries, merge_map)."""
    seen: dict[tuple, list[str]] = defaultdict(list)
    for e in entries:
        dev = _dev_of(e, hf_to_dev)
        if dev:
            seen[(dev, _name_key(str(e.get("id", "")), dev))].append(str(e.get("id", "")))

    merges: dict[str, str] = {}
    for e in entries:
        cid = str(e.get("id", ""))
        if "/" in cid:
            continue
        org = e.get("org_id")
        if org and str(org).lower() not in _PLACEHOLDER_ORGS:
            continue
        m = _LEAD_RE.match(cid)
        if not m:
            continue
        dev = hf_to_dev.get(m.group(1).lower())
        if not dev:
            continue
        twins = [i for i in seen.get((dev, _name_key(cid, dev)), []) if i != cid and "/" in i]
        if twins:
            merges[cid] = sorted(twins)[0]
        else:
            e["org_id"] = dev
    return entries, merges
