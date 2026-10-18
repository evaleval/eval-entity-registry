#!/usr/bin/env python3
"""
Generate seed/models/sources/models_dev.generated.yaml from https://models.dev/api.json.

Pure models.dev output, no curated overlays: the seed loader applies
`seed/models/core.yaml` and `seed/models/enrichments/aliases.yaml` on top, and
curated entries win on id collision. models.dev covers hosted-API catalogs well
and HF-released open-weight families poorly; core.yaml covers the latter.

Usage:
    python scripts/refresh_from_modelsdev.py              # fetch + write
    python scripts/refresh_from_modelsdev.py --no-fetch   # use /tmp cache
    python scripts/refresh_from_modelsdev.py --dry-run    # diff vs current

Source: https://models.dev (MIT, (c) 2025 models.dev)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import yaml


class UnionFind:
    """Disjoint-set with path compression, shared by both grouping passes so
    their merge semantics stay identical."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self._parent[self.find(a)] = self.find(b)

from eval_entity_registry.lib.collision_fold import _bsizes, collision_key
from eval_entity_registry.lib.seed_io import (
    WEAK_SCALAR_FIELDS,
    resolve_oracle_path,
    safe_load_yaml,
)

from eval_entity_resolver.display import humanize_model_slug
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

# Suffixes stripped to reach the family slug; mirrors the resolver's fuzzy stem
# so dated snapshots fold into one canonical.
_FAMILY_DATE_RES = [
    re.compile(r"-\d{8}$"),                  # YYYYMMDD: -20251101
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),      # YYYY-MM-DD: -2024-05-13
    re.compile(r"-preview-\d{2}-\d{2}$"),    # -preview-05-06 (Google Gemini preview snapshots)
    re.compile(r"-preview-\d{4}-\d{2}-\d{2}$"),  # -preview-2024-05-13 (rare)
    re.compile(r"-preview$"),                # bare -preview
    re.compile(r"-latest$"),                 # -latest hosting tag
    re.compile(r"-v\d+(\.\d+)*$", re.IGNORECASE),  # version suffix: -v0.3, -v1, -v1.0.0
]
# Kept for external callers (tests).
_FAMILY_LATEST_RE = _FAMILY_DATE_RES[-1]
_FAMILY_PREVIEW_RE = _FAMILY_DATE_RES[-2]

# Must match the resolver's _STRIP_SUFFIXES.
_FAMILY_STAGE_SUFFIXES = ("-instruct", "-chat", "-it", "-base")

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"
ORGS_SEED_PATH = REPO_ROOT / "seed" / "orgs.yaml"
CACHE_PATH = Path("/tmp/modelsdev_api.json")
SOURCE_URL = "https://models.dev/api.json"

# models.dev provider slug -> canonical_orgs.id; unlisted providers (re-hosts,
# gateways, regional duplicates) are skipped.
PROVIDER_TO_ORG: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "xai": "xai",
    "cohere": "cohere",
    "mistral": "mistralai",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "moonshotai": "moonshotai",
    "stepfun": "stepfun",
    "minimax": "minimax",
    "zai": "zai",
    "inception": "inception",
    "upstage": "upstage",
    "perplexity": "perplexity",
    "nvidia": "nvidia",
    # Every entry must have a matching seed/orgs.yaml row.
}

# --- inference_platforms single-source --------------------------------------
# The curated platform catalog also seeds seed/inference_platforms.yaml; the
# provider->platform map must stay identical to the runtime map in fuzzy.py,
# so never hand-maintain a parallel dict.
INFERENCE_PLATFORMS_JSON = (
    REPO_ROOT  / "curation" / "inference_platforms.proposed.json"
)


def _load_provider_to_inference_platform(
    path: Path = INFERENCE_PLATFORMS_JSON,
) -> dict[str, str]:
    """{models.dev provider slug -> inference_platforms.id} from the curated
    catalog (every provider-backed platform declares `models_dev_provider`)."""
    data = json.loads(path.read_text())
    mapping: dict[str, str] = {}
    for plat in data.get("platforms", []):
        prov = plat.get("models_dev_provider")
        pid = plat.get("id")
        if not prov or not pid:
            continue
        mapping[prov] = pid
    return mapping


try:
    PROVIDER_TO_INFERENCE_PLATFORM: dict[str, str] = _load_provider_to_inference_platform()
except FileNotFoundError:  # pragma: no cover - catalog file absent
    PROVIDER_TO_INFERENCE_PLATFORM = {}


# --- Author-lab classification + org inference -----------------------------

# Providers whose spelling may anchor a group's authorship, and only when their
# org matches the family-implied org (a lab can re-host others' models too).
def _load_strict_author(path: Path = INFERENCE_PLATFORMS_JSON) -> set[str]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:  # pragma: no cover
        return set()
    return {
        p["models_dev_provider"]
        for p in data.get("platforms", [])
        if p.get("kind") == "author_lab" and p.get("models_dev_provider")
    }


STRICT_AUTHOR: set[str] = _load_strict_author()

# Author-lab provider id -> HF-style org slug.
AUTHOR_PROV_ORG: dict[str, str | None] = {
    "anthropic": "anthropic", "openai": "openai", "google": "google",
    "mistral": "mistralai", "cohere": "cohere", "zai": "zai-org",
    "zhipuai": "zai-org", "alibaba": "qwen", "deepseek": "deepseek-ai",
    "llama": "meta-llama", "minimax": "minimaxai", "moonshotai": "moonshotai",
    "nvidia": "nvidia", "xai": "xai", "xiaomi": "xiaomi",
    "stepfun": "stepfun-ai", "stepfun-ai": "stepfun-ai", "upstage": "upstage",
    "venice": None, "perplexity": "perplexity", "perplexity-agent": "perplexity",
    "nova": "amazon", "sarvam": "sarvam-ai", "inception": "inceptionai",
    "poolside": "poolside", "morph": "morph", "v0": "vercel",
    "lucidquery": None, "inceptron": None,
}

# org inference from family / name tokens (HF-style slugs).
ORG_BY_FAMILY_PREFIX: dict[str, str] = {
    "claude": "anthropic", "gpt": "openai", "o-": "openai", "o": "openai", "gpt-": "openai",
    "gemini": "google", "gemma": "google", "imagen": "google", "learnlm": "google",
    "qwen": "qwen", "qwen3": "qwen", "qwen3.": "qwen",
    "llama": "meta-llama",
    "glm": "zai-org", "glm-": "zai-org",
    "deepseek": "deepseek-ai",
    "minimax": "minimaxai", "mimo": "xiaomi",
    "kimi": "moonshotai",
    "grok": "xai",
    "mistral": "mistralai", "ministral": "mistralai", "devstral": "mistralai",
    "codestral": "mistralai", "mistral-": "mistralai", "mixtral": "mistralai",
    "phi": "microsoft",
    "nemotron": "nvidia", "nova": "amazon", "command": "cohere",
    "command-r": "cohere", "command-a": "cohere",
    "ernie": "baidu", "hunyuan": "tencent", "seed": "bytedance", "doubao": "bytedance",
    "flux": "black-forest-labs", "voyage": "voyageai", "ling": "inclusionai",
    "gpt-oss": "openai",
}


def org_from_family(fam: str | None) -> str | None:
    """Infer an HF-style org slug from a models.dev `family` token."""
    if not fam:
        return None
    f = fam.lower()
    base = f.split("-")[0]
    for key in (f, base):
        if key in ORG_BY_FAMILY_PREFIX:
            return ORG_BY_FAMILY_PREFIX[key]
    for pref, org in ORG_BY_FAMILY_PREFIX.items():
        if f.startswith(pref):
            return org
    return None


# HF-style org slugs -> curated developer ids not already covered by the
# orgs.yaml alias index.
_ORG_SLUG_OVERRIDES: dict[str, str] = {
    "meta-llama": "meta",
    "qwen": "alibaba",
    "deepseek-ai": "deepseek",
    "zai-org": "zai",
}


def _build_org_alias_index() -> dict[str, str]:
    """{lowercased org id / hf_org / alias -> curated org id}, built by the
    resolver's `build_curated_org_map` so it stays identical to the resolver's."""
    from eval_entity_resolver.fold import build_curated_org_map
    data = safe_load_yaml(ORGS_SEED_PATH.read_text()) if ORGS_SEED_PATH.exists() else []
    return build_curated_org_map(data or [])


_DEV_ALIAS_INDEX: dict[str, str] | None = None


def _dev_alias_index() -> dict[str, str]:
    """Module-cached org alias index."""
    global _DEV_ALIAS_INDEX
    if _DEV_ALIAS_INDEX is None:
        _DEV_ALIAS_INDEX = _build_org_alias_index()
    return _DEV_ALIAS_INDEX


def normalize_org_slug(hf_org: str | None, alias_index: dict[str, str]) -> str | None:
    """Map an HF-style org slug to the curated org id when one exists, else
    return it unchanged (an HF-cased org row is auto-created at seed)."""
    if not hf_org:
        return None
    if hf_org in _ORG_SLUG_OVERRIDES:
        return _ORG_SLUG_OVERRIDES[hf_org]
    mapped = alias_index.get(hf_org.lower())
    return mapped if mapped else hf_org


# --- Developer (org) derivation: PREFIX-authoritative ----------------------
# The id's org prefix is the developer. The model NAME only says what a model
# derives from, so name-matching applies only to bare / serving-hosted ids and
# never overrides a prefix (`nvidia/llama-nemotron` is nvidia's, not meta's); a
# curated prefix whose name disagrees is flagged for curation, not flipped.
# Serving hosts: every non-author_lab models.dev provider plus host scaffolding
# tokens; an id under one of these takes its developer from the name.
_SERVING_HOSTS = {
    "accounts", "clarifai", "route", "orcarouter", "workers-ai", "openrouter",
    "stealth", "hf", "cf", "@cf",
    # serving-brand namespaces that appear only as id prefixes
    "volcengine", "fal-ai", "fal", "kilo", "kilo-auto",
    # nano-gpt's `TEE/` namespace is a serving mode, not an uploader org
    "tee",
} | ({p.lower() for p in PROVIDER_TO_INFERENCE_PLATFORM} - {a.lower() for a in STRICT_AUTHOR})
# Leading model-name token -> HF-style developer slug, longest-prefix-first.
_NAME_VENDOR_MAP: list[tuple[str, str]] = sorted([
    ("meta-llama", "meta-llama"), ("codellama", "meta-llama"), ("llama", "meta-llama"),
    ("ministral", "mistralai"), ("mixtral", "mistralai"), ("pixtral", "mistralai"),
    ("codestral", "mistralai"), ("devstral", "mistralai"), ("magistral", "mistralai"),
    ("mistral", "mistralai"),
    ("qwen", "qwen"), ("qwq", "qwen"), ("qvq", "qwen"),
    ("gpt-oss", "openai"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
    ("o4", "openai"), ("whisper", "openai"), ("chatgpt", "openai"), ("codex", "openai"),
    ("dall-e", "openai"), ("text-embedding", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"), ("paligemma", "google"), ("gemma", "google"),
    ("imagen", "google"), ("learnlm", "google"),
    ("deepseek", "deepseek-ai"), ("grok", "xai"), ("chatglm", "zai-org"), ("glm", "zai-org"),
    ("kimi", "moonshotai"), ("moonshot", "moonshotai"), ("minimax", "minimax"),
    ("phi", "microsoft"), ("nemotron", "nvidia"), ("nvlm", "nvidia"), ("mimo", "xiaomi"),
    ("hunyuan", "tencent"), ("ernie", "baidu"), ("doubao", "bytedance"), ("seed", "bytedance"),
    ("command", "cohere"), ("aya", "cohere"), ("nova", "amazon"), ("titan", "amazon"),
    ("solar", "upstage"), ("jamba", "ai21"), ("jurassic", "ai21"), ("sonar", "perplexity"),
    ("hermes", "nousresearch"), ("granite", "ibm"), ("flux", "black-forest-labs"),
    ("voyage", "voyageai"), ("cogito", "deepcogito"), ("falcon", "tiiuae"),
    ("olmo", "allenai"), ("tulu", "allenai"), ("bge", "baai"), ("inflection", "inflection"),
], key=lambda kv: -len(kv[0]))


def developer_from_name(name: str | None) -> str | None:
    """HF-style developer slug from the LEADING vendor token of a model name
    (a derivative's base token mid-name must not hijack it); for bare /
    serving-hosted ids only."""
    if not name:
        return None
    s = re.sub(r"^[a-z]+\.", "", str(name).strip().lower())
    s = s.split("/")[-1]
    s = re.sub(r"[_\s]+", "-", s)
    for tok, dev in _NAME_VENDOR_MAP:
        if re.match(re.escape(tok) + r"([0-9._:\-]|$)", s):
            return dev
    return None


def _derive_group_org(
    recs: list[dict], alias_index: dict[str, str]
) -> tuple[str | None, str | None]:
    """Developer org for a models.dev underlying group (prefix-authoritative).

    Returns (hf_org_slug | None, rehost_review | None); `rehost_review` is the
    disagreeing name-vendor when a curated-org prefix's model name points at a
    different vendor (a possible re-host, e.g. nvidia/whisper)."""
    prefix_orgs: list[str] = []
    name_orgs: list[str] = []
    for r in recs:
        # `hf:` must come off before the org is taken (as normalize_modelsdev_id
        # does), or `hf:nvidia/...` yields a malformed org and a dangling FK.
        raw = re.sub(r"^hf:", "", (r.get("raw") or "").lstrip("~"))
        if "/" in raw and raw.split("/")[0].lower() not in _SERVING_HOSTS:
            prefix_orgs.append(raw.split("/")[0])          # uploader OR curated (raw spelling)
        else:                                              # bare or serving-hosted
            no = developer_from_name(r.get("name"))
            if no:
                name_orgs.append(no)
    if prefix_orgs:
        # Most common spelling, verbatim: equating uploader spelling variants is
        # a curated orgs.yaml claim, not something to infer here.
        low = Counter(p.lower() for p in prefix_orgs).most_common(1)[0][0]
        org = next(p for p in prefix_orgs if p.lower() == low)
        rehost = None
        if alias_index.get(org.lower()) and name_orgs:    # curated prefix + name signal
            nd = Counter(name_orgs).most_common(1)[0][0]
            if normalize_org_slug(nd, alias_index) != normalize_org_slug(org, alias_index):
                rehost = nd
        return org, rehost
    if name_orgs:
        return Counter(name_orgs).most_common(1)[0][0], None
    return None, None


# --- Dedup key + head-pick --------------------------------------------------
_DATE8_RE = re.compile(r"^\d{8}$")
_DATE6_RE = re.compile(r"^\d{6}$")
_NUM_TOKEN_RE = re.compile(r"^\d+[a-z]?$")


def normalize_modelsdev_id(raw: str, strip_variants: bool = True) -> str:
    """Normalize a models.dev model id to an underlying-model spelling
    (host/region/account scaffolding stripped, separators unified).

    `strip_variants=True` also collapses identity variants (-turbo/-thinking/
    -fp8/...) for routing/dedup grouping. Pass False for alias derivation: a
    variant must never emit its base's id as an alias, which double-claims it
    and aborts the seed. Serving tags (`:free`, `-fast`, ...) are stripped
    regardless."""
    s = raw.strip()
    s = s.lstrip("~")  # openrouter '~' latest marker
    s = re.sub(r"^accounts/[^/]+/models/", "", s)
    s = re.sub(r"^hf:", "", s)
    s = re.sub(r"^@cf/", "", s)
    s = re.sub(r"^clarifai/[^/]+/models/", "", s)
    s = re.sub(r"^route/", "", s)
    s = re.sub(r"^orcarouter/", "", s)
    s = s.replace("--", "/")  # sap style
    s = re.sub(r"^databricks-", "", s)
    s = re.sub(r"^azure-", "", s)
    s = re.sub(r"^aws-", "", s)
    s = re.sub(r"^openai-", "", s)
    s = re.sub(r"^anthropic-", "", s)
    s = re.sub(r"^ai21-", "ai21/", s)
    s = re.sub(r"^stealth/", "", s)
    s = _strip_host_region_prefixes(s)
    if "/" in s:
        s = s.split("/")[-1]
    s = _strip_host_region_prefixes(s)
    s = s.lower()
    s = s.replace("@default", "")
    s = re.sub(r"@(\d{8})$", r"-\1", s)
    s = re.sub(r"@.*$", "", s)
    s = _BEDROCK_VER_RE.sub("", s)
    # A trailing `:NNN<unit>` is a size spec (`gpt-oss:120b`), not a serving tag;
    # the unit is required so `:1024` / `:1.0` still strip as tags.
    s = re.sub(r":(\d+(?:\.\d+)?[bmkt])$", r"-\1", s)
    prev = None
    while prev != s:
        prev = s
        s = _TAG_SUFFIX_RE.sub("", s)           # scaffolding tags: always
        if strip_variants:
            s = _IDENTITY_VARIANT_RE.sub("", s)  # identity variants: routing/dedup only
    s = re.sub(r"(\d)\.(\d)", r"\1-\2", s)
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"\(.*$", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


_HOST_PREFIXES = ["amazon.", "anthropic.", "qwen.", "meta.", "google.", "cohere.",
                  "ai21.", "deepseek.", "mistral."]
_REGION_PREFIXES = ["us.", "eu.", "jp.", "au.", "apac.", "global."]
# Serving/routing tags: pure scaffolding, stripped even for alias derivation
# (`gpt-4o:free` is `gpt-4o`); `-tee` / `-maas` are serving modes. A `:NNNb`
# size spec is converted before this fires.
_TAG_SUFFIX_RE = re.compile(r"(:.*$)|(-fast$)|(-precision$)|(-free$)|(-maas$)|(-tee$)")
# Prefix spelling of the `-tee` serving marker; the raw form stays an alias on
# the stripped target.
_TEE_PREFIX_RE = re.compile(r"^TEE/", re.IGNORECASE)
# Identity variants (a different canonical): stripped only for routing/dedup,
# never for alias derivation (see normalize_modelsdev_id).
_IDENTITY_VARIANT_RE = re.compile(
    r"(-thinking$)|(-think$)|(-reasoner$)|(-reasoning$)"
    r"|(-turbo$)"
    r"|(-fp8$)|(-bf16$)|(-int8$)|(-awq$)|(-gptq$)"
)
# Back-compat union for external callers/tests.
_VARIANT_SUFFIX_RE = re.compile(
    _TAG_SUFFIX_RE.pattern + "|" + _IDENTITY_VARIANT_RE.pattern
)
_BEDROCK_VER_RE = re.compile(r"-v\d+:\d+$")


def _strip_host_region_prefixes(s: str) -> str:
    changed = True
    while changed:
        changed = False
        for rp in _REGION_PREFIXES:
            if s.startswith(rp):
                s = s[len(rp):]
                changed = True
        for hp in _HOST_PREFIXES:
            if s.startswith(hp):
                s = s[len(hp):]
                changed = True
    return s


def canon_key_ordered(norm: str) -> str:
    """Underlying-model key (order-preserving)."""
    s = re.sub(r"-(latest|old|new)$", "", norm)
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    s = re.sub(r"-\d{2}-\d{2}$", "", s)
    toks = [t for t in s.split("-") if not _DATE8_RE.match(t) and not _DATE6_RE.match(t)]
    if toks and re.match(r"^v\d+$", toks[-1]):
        toks.pop()
    return "-".join(toks)


def safe_sig(key: str) -> str:
    """Token-multiset signature for the merge pass. Words sorted, version
    numbers order-preserved (so `gpt-4-5` != `gpt-5-4`)."""
    toks = key.split("-")
    words = sorted(t for t in toks if not _NUM_TOKEN_RE.match(t))
    nums = [t for t in toks if _NUM_TOKEN_RE.match(t)]
    return "|".join(words) + "#" + "-".join(nums)


def build_underlying_groups(api_json: dict) -> dict[str, list[dict]]:
    """Group every (provider, model) record into underlying groups via a
    union-find merge pass. Returns {root_key -> [records]}, each record
    {provider, raw, norm, key, family, release, name, open_weights, record}."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for prov, pdata in api_json.items():
        for mid, mr in (pdata.get("models") or {}).items():
            n = normalize_modelsdev_id(mid)
            k = canon_key_ordered(n)
            if not k:
                continue
            groups[k].append(dict(
                provider=prov, raw=mid, norm=n, key=k,
                family=mr.get("family"), release=mr.get("release_date"),
                name=mr.get("name"), open_weights=mr.get("open_weights"),
                record=mr,
            ))

    # Union-find: merge ordered keys sharing the same token-multiset signature,
    # but only when their families don't CONFLICT (guards anagram merges).
    uf = UnionFind()

    for k in groups:
        uf.find(k)
    mset: dict[str, list[str]] = defaultdict(list)
    for k in groups:
        mset[safe_sig(k)].append(k)
    for _ms, ks in sorted(mset.items()):
        if len(ks) > 1:
            # Sorted so grouping cannot flip on upstream key order; fam_a
            # accumulates so the merge is transitive (k2 bridging k1 and k3).
            ks = sorted(ks)
            base = ks[0]
            fam_a = {r["family"] for r in groups[base] if r["family"]}
            for o in ks[1:]:
                fam_b = {r["family"] for r in groups[o] if r["family"]}
                if fam_a & fam_b or not fam_a or not fam_b:
                    uf.union(o, base)
                    fam_a |= fam_b

    merged: dict[str, list[dict]] = defaultdict(list)
    for k, recs in groups.items():
        merged[uf.find(k)].extend(recs)
    return merged


def pick_underlying(root: str, recs: list[dict]) -> dict:
    """Choose the authoritative (org, display_name, release, open_weights) and
    `head_spelling` (the cleanest spelling to mint from) for an underlying
    group. The org is HF-style; callers map it via normalize_org_slug()."""
    fams = [r["family"] for r in recs if r["family"]]
    fam_org = org_from_family(Counter(fams).most_common(1)[0][0]) if fams else None
    author_recs = [r for r in recs if r["provider"] in STRICT_AUTHOR]
    # Deterministic pick (shortest normalized form, then raw), not models.dev's
    # provider order, so the minted id cannot flip on upstream reorders.
    true_author_recs = sorted(
        (r for r in author_recs
         if fam_org is None or AUTHOR_PROV_ORG.get(r["provider"]) == fam_org),
        key=lambda r: (len(r["norm"]), r["raw"]),
    )
    org, rehost_review = _derive_group_org(recs, _dev_alias_index())
    has_author = bool(true_author_recs)  # gates family-tree vs single mint downstream

    if true_author_recs:
        disp = true_author_recs[0]["name"] or root
        head_spelling = true_author_recs[0]["raw"]
    else:
        names = [r["name"] for r in recs if r["name"]]
        disp = Counter(names).most_common(1)[0][0] if names else root
        head_spelling = _cleanest_spelling(root, recs)

    rels = [r["release"] for r in recs if r["release"]]
    release = min(rels) if rels else None
    ow = any(r["open_weights"] for r in recs)
    return {
        "author_org": org,
        "display_name": disp,
        "release_date": release,
        "open_weights": ow,
        "has_author_lab_entry": has_author,
        "head_spelling": head_spelling,
        "rehost_review": rehost_review,
    }


def _cleanest_spelling(root: str, recs: list[dict]) -> str:
    """Cleanest raw spelling to mint from when no author lab anchors the group:
    a record whose normalized form equals the root, else the shortest; raw
    tie-break for determinism."""
    exact = [r for r in recs if r["norm"] == root]
    pool = exact or recs
    return sorted(pool, key=lambda r: (len(r["norm"]), r["raw"]))[0]["raw"]


def _fetch(use_cache: bool) -> dict:
    if use_cache and CACHE_PATH.exists():
        print(f"[refresh] using cache: {CACHE_PATH}", file=sys.stderr)
        return json.loads(CACHE_PATH.read_text())
    print(f"[refresh] fetching {SOURCE_URL}", file=sys.stderr)
    # models.dev's CDN rejects the default Python-urllib User-Agent with a 403.
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "evalcard-registry-refresh/1.0 (+https://github.com/evaleval/evalcard-registry)"},
    )
    with urllib.request.urlopen(req) as r:
        raw = r.read().decode()
    CACHE_PATH.write_text(raw)
    return json.loads(raw)


def _slugify(value: str) -> str:
    """Lowercase, collapse whitespace/underscores to hyphens, drop display-only
    punctuation; dots, hyphens and slashes are preserved."""
    s = value.strip().lower()
    s = re.sub(r"[()\[\]{}]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _family_for(model: dict) -> str:
    """Group key for a model record (the lineage canonical slug).

    Built from `name` rather than `id` because models.dev ids mangle separators
    (`qwen2-5-14b-instruct` for `Qwen2.5-14B-Instruct`). Date/preview/latest
    and training-stage suffixes are stripped, mirroring the resolver's fuzzy
    stem. models.dev's own `family` field is too coarse (every Claude Opus
    major under one slug).
    """
    raw = _slugify(model.get("name") or model.get("id", ""))
    # Suffixes stack (`-preview-05-06-instruct`); each strip shrinks, so the
    # fixpoint terminates.
    while True:
        before = raw
        for pat in _FAMILY_DATE_RES:
            raw = pat.sub("", raw)
        for suffix in sorted(_FAMILY_STAGE_SUFFIXES, key=len, reverse=True):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                break
        if raw == before:
            break
    return raw


# --- Suffix → (relationship, axis) classification --------------------------
# Suffix token -> typed `parents` edge, same enum as core.yaml. Axis semantics:
#   version        dated snapshot / vN marker: same API identity, new release
#   training_stage base / instruct / chat / it
#   tier           branded sibling (haiku/sonnet, mini/nano, flash/pro): makes
#                  no scale claim, so it is never a `size` edge
#   size           a disclosed scale token (7b, 8x7b, a16b); see _is_size_token
#   mode           runtime/decoding mode that is not a training stage
_TOKEN_CLASSIFICATIONS: dict[str, tuple[str, str | None]] = {
    "instruct": ("variant", "training_stage"),
    "it": ("variant", "training_stage"),
    "chat": ("variant", "training_stage"),
    "base": ("variant", "training_stage"),
    "pt": ("variant", "training_stage"),
    "sft": ("variant", "training_stage"),
    "haiku": ("variant", "tier"),
    "sonnet": ("variant", "tier"),
    "opus": ("variant", "tier"),
    "mini": ("variant", "tier"),
    "nano": ("variant", "tier"),
    "micro": ("variant", "tier"),
    "flash": ("variant", "tier"),
    "pro": ("variant", "tier"),
    "small": ("variant", "tier"),
    "medium": ("variant", "tier"),
    "large": ("variant", "tier"),
    "lite": ("variant", "tier"),
    "thinking": ("variant", "mode"),
    "reasoning": ("variant", "mode"),
    "nothink": ("variant", "mode"),
    "guard": ("variant", "mode"),
    "safeguard": ("variant", "mode"),
    "moderation": ("variant", "mode"),
    "vision": ("variant", "modality"),
    "vl": ("variant", "modality"),
    "coder": ("variant", "domain"),
    "code": ("variant", "domain"),
    "math": ("variant", "domain"),
    "turbo": ("quantized", None),
    "fp8": ("quantized", None),
    "fp16": ("quantized", None),
    "bf16": ("quantized", None),
    "int4": ("quantized", None),
    "int8": ("quantized", None),
    "awq": ("quantized", None),
    "gptq": ("quantized", None),
    "gguf": ("quantized", None),
}
_TIER_TOKENS = frozenset(
    k for k, (_rel, axis) in _TOKEN_CLASSIFICATIONS.items() if axis == "tier"
)
_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$", re.IGNORECASE)
_DATE_8_RE = re.compile(r"^\d{8}$")
_DATE_4_RE = re.compile(r"^\d{4}$")
_DATE_3_RE = re.compile(r"^\d{3}$")
_MOE_ACTIVE_RE = re.compile(r"^a\d+b$", re.IGNORECASE)
_SIZE_TOKEN_RE = re.compile(r"^\d+(\.\d+)?b$", re.IGNORECASE)
_MOE_EXPERT_RE = re.compile(r"^\d+x\d+(\.\d+)?b$", re.IGNORECASE)


def _is_size_token(token: str) -> bool:
    """True iff `token` is a disclosed scale token (`7b`, `1.5b`, `8x7b`,
    `a16b`): the only route to a `size` edge, since models.dev has no params
    field."""
    t = token.lower()
    return bool(_SIZE_TOKEN_RE.match(t) or _MOE_EXPERT_RE.match(t) or _MOE_ACTIVE_RE.match(t))


def _classify_token(token: str) -> tuple[str, str | None] | None:
    t = token.lower()
    # Named tokens first, so a tier is never read as a scale claim.
    if t in _TOKEN_CLASSIFICATIONS:
        return _TOKEN_CLASSIFICATIONS[t]
    if _VERSION_RE.match(t) or _DATE_8_RE.match(t) or _DATE_4_RE.match(t) or _DATE_3_RE.match(t):
        return ("variant", "version")
    if _is_size_token(t):
        return ("variant", "size")
    return None


def _classify_suffix_segments(suffix: str) -> list[tuple[str, str | None, str]]:
    """Greedy left-to-right parse of a hyphen/dot-separated suffix into
    (relationship, axis, token) segments; an unclassifiable tail becomes one
    (variant, version, tail) segment so at least one parent edge is emitted.
    """
    if not suffix:
        return []
    norm = re.sub(r"[._]+", "-", suffix.lower()).strip("-")
    if not norm:
        return []
    tokens = norm.split("-")
    segments: list[tuple[str, str | None, str]] = []
    i = 0
    while i < len(tokens):
        # YYYY-MM-DD across 3 tokens
        if i + 3 <= len(tokens):
            window = "-".join(tokens[i:i + 3])
            if re.match(r"^\d{4}-\d{2}-\d{2}$", window):
                segments.append(("variant", "version", window))
                i += 3
                continue
        # vN-N across 2 tokens (slugified v0.3 → v0-3)
        if i + 2 <= len(tokens):
            window = "-".join(tokens[i:i + 2])
            if re.match(r"^v\d+-\d+$", window) or re.match(r"^\d{4}-\d{2}$", window):
                segments.append(("variant", "version", window))
                i += 2
                continue
        cls = _classify_token(tokens[i])
        if cls is None:
            tail = "-".join(tokens[i:])
            segments.append(("variant", "version", tail))
            return segments
        relationship, axis = cls
        segments.append((relationship, axis, tokens[i]))
        i += 1
    return segments


def _build_family_entries(
    org_id: str,
    family_slug: str,
    models: list[dict],
    group_recs: list[dict] | None = None,
    alias_index: dict[str, str] | None = None,
) -> list[dict]:
    """Emit a family's entries: the root (parents=[]) then one child per
    snapshot/variant with a typed `parents` edge; `group_recs` enables
    OpenRouter id adoption. Compound suffixes materialize their intermediate
    canonicals because the loader's parents-merge is union-by-id, so a parent
    id disagreeing with core.yaml would add a spurious edge.
    """
    family_canonical_id = f"{org_id}/{family_slug}"

    # ---- Family root display_name + aggregated metadata ----
    display_name = ""
    for m in models:
        if _slugify(m.get("id", "")) == family_slug:
            display_name = m.get("name") or humanize_model_slug(family_slug)
            break
    if not display_name:
        display_name = humanize_model_slug(family_slug)

    open_weights = any(m.get("open_weights") for m in models)
    release_dates = sorted({m["release_date"] for m in models if m.get("release_date")})
    release_date = release_dates[0] if release_dates else None
    snapshot_ids = sorted({_slugify(m["id"]) for m in models if m.get("id")})

    # The root carries the union of snapshot modalities; leaves keep their own.
    family_input_modalities: set[str] = set()
    family_output_modalities: set[str] = set()
    for m in models:
        mods = m.get("modalities") or {}
        for v in (mods.get("input") or []):
            if isinstance(v, str) and v.strip():
                family_input_modalities.add(v.strip())
        for v in (mods.get("output") or []):
            if isinstance(v, str) and v.strip():
                family_output_modalities.add(v.strip())
    family_input_modalities_list = sorted(family_input_modalities) or None
    family_output_modalities_list = sorted(family_output_modalities) or None

    metadata: dict = {"snapshots": snapshot_ids}
    if release_dates:
        metadata["release_dates"] = release_dates
    if any(m.get("knowledge") for m in models):
        metadata["knowledge_cutoffs"] = sorted({
            m["knowledge"] for m in models if m.get("knowledge")
        })

    root_aliases =[f"{org_id}/{family_slug}"] if f"{org_id}/{family_slug}" != family_canonical_id else []

    family_root_entry = {
        "id": family_canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": family_slug,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": open_weights,
        "release_date": release_date,
        "input_modalities": family_input_modalities_list,
        "output_modalities": family_output_modalities_list,
        "tags": ["open-weight"] if open_weights else [],
        "aliases": root_aliases,
        "metadata": json.dumps(metadata, sort_keys=True),
        "review_status": "reviewed",
    }

    # ---- Child entries: one per snapshot/variant whose id != family_slug ----
    out_entries: list[dict] = [family_root_entry]
    seen_ids: dict[str, dict] = {family_canonical_id: family_root_entry}

    # `family_slug` may carry dots (`qwen2.5-7b`) while ids are dashed: compare
    # dashed, but build chain ids from the dotted form (the lab's spelling).
    family_slug_dashed = re.sub(r"\.", "-", family_slug)

    for m in models:
        snap_dashed = _slugify(m.get("id", ""))
        if not snap_dashed:
            continue
        if snap_dashed == family_slug_dashed or snap_dashed == family_slug:
            continue
        # A non-prefix snapshot id (mirror entry) would emit a malformed child.
        if not snap_dashed.startswith(family_slug_dashed + "-"):
            continue
        suffix = snap_dashed[len(family_slug_dashed) + 1:]
        segments = _classify_suffix_segments(suffix)
        if not segments:
            continue

        # Walk the chain, materializing intermediates as anchor entries.
        current_id = family_canonical_id
        for idx, (relationship, axis, token) in enumerate(segments):
            new_id = f"{current_id}-{token}"
            is_leaf = (idx == len(segments) - 1)
            parent_edge = {"id": current_id, "relationship": relationship}
            if axis:
                parent_edge["axis"] = axis

            if new_id in seen_ids:
                current_id = new_id
                continue

            # Leaf: source metadata plus the dashed models.dev id as an alias
            # (resolves even when the canonical is dotted). Intermediates are
            # anchor-only.
            child_aliases: list[str] = []
            child_release: str | None = None
            child_open_weights = open_weights
            child_input_modalities: list[str] | None = None
            child_output_modalities: list[str] | None = None
            if is_leaf:
                child_aliases = sorted({snap_dashed, f"{org_id}/{snap_dashed}"})
                if m.get("release_date"):
                    child_release = m["release_date"]
                child_open_weights = bool(m.get("open_weights")) or open_weights
                mods = m.get("modalities") or {}
                _ci = sorted({v.strip() for v in (mods.get("input") or []) if isinstance(v, str) and v.strip()})
                _co = sorted({v.strip() for v in (mods.get("output") or []) if isinstance(v, str) and v.strip()})
                child_input_modalities = _ci or None
                child_output_modalities = _co or None

            entry = {
                "id": new_id,
                "display_name": (m.get("name") or humanize_model_slug(new_id)) if is_leaf else humanize_model_slug(new_id),
                "org_id": org_id,
                "family": family_slug,
                "architecture": None,
                "params_billions": None,
                "parents": [parent_edge],
                "open_weights": child_open_weights,
                "release_date": child_release,
                "input_modalities": child_input_modalities,
                "output_modalities": child_output_modalities,
                "tags": ["open-weight"] if child_open_weights else [],
                "aliases": child_aliases,
                "metadata": "{}",
                "review_status": "reviewed",
            }
            seen_ids[new_id] = entry
            out_entries.append(entry)
            current_id = new_id

    # A variant/dated OpenRouter key renames only the entity it names, never
    # the root; children of a renamed root keep their invented ids.
    if group_recs:
        _adopt_openrouter_ids(
            out_entries, group_recs, org_id, alias_index or _dev_alias_index()
        )
    return out_entries


def _provider_alias_forms(
    raw: str, org_id: str | None, provider: str | None = None
) -> list[str]:
    """Surface forms a provider's raw spelling should resolve through: the raw
    spelling when it carries no host/account scaffolding; an OpenRouter
    `org/slug` key verbatim plus its tag-stripped form (router pseudo-endpoints
    excluded); the normalized form and its org-prefixed variant. Host-scaffolded
    raws (`@cf/qwen/...`) are emitted only in normalized form."""
    forms: set[str] = set()
    if not raw:
        return []
    if "/" not in raw and not raw.startswith("@") and not raw.startswith("~"):
        forms.add(raw)
        slug_raw = _slugify(raw)
        if slug_raw:
            forms.add(slug_raw)
    elif _TEE_PREFIX_RE.match(raw):
        forms.add(raw)
    if (
        provider == _OPENROUTER_PROVIDER
        and "/" in raw
        and not _is_router_pseudo_endpoint(raw)
    ):
        forms.add(raw)
        cleaned = _clean_openrouter_key(raw)
        if cleaned:
            forms.add(cleaned)
    # strip_variants=False: stripping would emit the base's id as a variant's
    # alias (`gpt-4-turbo` -> `gpt-4`) and abort the seed.
    norm = normalize_modelsdev_id(raw, strip_variants=False)
    slug = _slugify(norm)
    if slug:
        leaf = slug.rsplit("/", 1)[-1]
        forms.add(leaf)
        if org_id:
            forms.add(f"{org_id}/{leaf}")
    return sorted(f for f in forms if f and f.count("/") <= 1)


def _variant_identity(s: str) -> str:
    """Variant-preserving leaf identity of a spelling (serving tags stripped,
    size/mode/quant/stage/version tokens kept): same identity == same
    canonical."""
    return _slugify(normalize_modelsdev_id(s, strip_variants=False)).rsplit("/", 1)[-1]


_IDENTITY_VARIANT_TOKENS = frozenset(re.findall(r"\(-([a-z0-9]+)\$\)", _IDENTITY_VARIANT_RE.pattern)) | frozenset({
    # Weight formats an HF repo name carries that a served key never does: a
    # hit adding one is a different upload, not the model.
    "gguf", "exl2", "mlx", "nf4", "4bit", "8bit", "bnb", "hqq", "ggml", "onnx",
    "q4", "q5", "q6", "q8", "fp4", "fp16", "fp32", "w4a16", "w8a8", "w8a16",
})


def _variant_tokens(s: str) -> set[str]:
    """Identity-variant tokens anywhere in a spelling's leaf (the suffix regex
    misses a mid-name one such as `…-Reasoning-2512`)."""
    leaf = s.rsplit("/", 1)[-1].lower()
    return set(re.split(r"[-_.:\s]+", leaf)) & _IDENTITY_VARIANT_TOKENS


# ---------------------------------------------------------------------------
# OpenRouter id adoption. Id ladder: a real HF id (rung 1), then an OpenRouter
# catalog key (rung 2), then an invented `{org}/{slug}` (rung 3); `org_id` is
# the curated developer on every rung.
# ---------------------------------------------------------------------------
_OPENROUTER_PROVIDER = "openrouter"

# Curated `junk` ids are spellings that must never be canonical; an OpenRouter
# key equal to one is not adopted (reconcile still folds it onto its target).
_REHOST_REPOINT_JSON = REPO_ROOT / "curation" / "rehost_repoint.json"
_REHOST_JUNK_IDS: frozenset[str] | None = None


def _rehost_junk_ids() -> frozenset[str]:
    global _REHOST_JUNK_IDS
    if _REHOST_JUNK_IDS is None:
        junk: set[str] = set()
        if _REHOST_REPOINT_JSON.exists():
            for e in json.loads(_REHOST_REPOINT_JSON.read_text()) or []:
                j = e.get("junk") if isinstance(e, dict) else None
                if isinstance(j, str) and j:
                    junk.add(j)
        _REHOST_JUNK_IDS = frozenset(junk)
    return _REHOST_JUNK_IDS


def _is_router_pseudo_endpoint(raw: str) -> bool:
    """`openrouter/*` keys (`auto`, `free`, ...) are routing products, not
    models: never adopted as a canonical id and never emitted as an alias
    form. The bare raws behind them stay minted (other providers list them
    directly)."""
    return raw.lstrip("~").lower().startswith("openrouter/")


def _clean_openrouter_key(raw: str) -> str | None:
    """Adoptable `org/leaf` form of an OpenRouter key (`~` marker and serving
    tags stripped). None for a router pseudo-endpoint, a tag-only leaf, or a
    moving `…-latest` pointer (names no fixed release)."""
    key = raw.lstrip("~")
    if "/" not in key or _is_router_pseudo_endpoint(raw):
        return None
    org, _, leaf = key.partition("/")
    prev = None
    while prev != leaf:
        prev = leaf
        leaf = _TAG_SUFFIX_RE.sub("", leaf)
    if not org or not leaf or leaf.endswith("-latest"):
        return None
    return f"{org}/{leaf}"


def _identity_sig(s: str) -> str:
    """Order-insensitive, variant-preserving identity signature: `claude-haiku-3`
    matches `claude-3-haiku`, but any added/removed token is a different
    identity."""
    return safe_sig(_variant_identity(s))


def _entity_identity_sigs(
    cid: str, org_id: str | None, alias_index: dict[str, str]
) -> set[str]:
    """Identity sigs an entry answers to: its own, plus the brand-stripped one
    when the leaf's leading token is a curated alias of its dev org
    (`perplexity/perplexity-sonar-…` == `perplexity/sonar-…`), the same rule
    `_candidate_name_norms` applies."""
    ident = _variant_identity(cid)
    sigs = {safe_sig(ident)}
    if org_id:
        dev = alias_index.get(org_id.lower(), org_id)
        toks = ident.split("-")
        if len(toks) > 1 and alias_index.get(toks[0]) == dev:
            sigs.add(safe_sig("-".join(toks[1:])))
    return sigs


def _openrouter_org_agrees(
    key_org: str, org_id: str | None, alias_index: dict[str, str]
) -> bool:
    """True iff the OpenRouter key's org folds to the entry's developer, so a
    `google/…` key can never re-attribute an `unsloth` re-upload."""
    if not org_id:
        return False
    from eval_entity_resolver.fold import _norm_org_key

    fold = lambda o: alias_index.get(o.lower(), o)  # noqa: E731
    a, b = fold(key_org), fold(org_id)
    return a == b or _norm_org_key(a) == _norm_org_key(b)


def _adopt_openrouter_ids(
    entries: list[dict],
    group_recs: list[dict],
    org_id: str | None,
    alias_index: dict[str, str],
) -> None:
    """Rename an invented `{org}/{slug}` entry to the cleaned OpenRouter key
    naming the same variant-preserving identity (the invented id becomes an
    alias). HF-deferred entries never rename; org must agree; ties go to the
    shortest key, then alphabetical."""
    cands = sorted(
        {
            k
            for r in group_recs
            if r.get("provider") == _OPENROUTER_PROVIDER
            for k in (_clean_openrouter_key(r.get("raw") or ""),)
            if k and k not in _rehost_junk_ids()
        },
        key=lambda k: (len(k), k),
    )
    if not cands:
        return
    taken = {e["id"] for e in entries if e.get("id")}
    renames: dict[str, str] = {}
    for e in entries:
        cid = e.get("id")
        if not cid or "/" not in cid:
            continue  # org-less mints keep their invented slug
        if cid in cands:
            # Tagged by `_tag_openrouter_key_ids` after reconcile, not here:
            # `_pick_winner` ranks the flag, and tagging an equal-spelling
            # coincidence now would flip a reviewed author entry into its
            # re-host twin.
            continue
        if _entry_meta(e).get("hf_deferred"):
            continue  # canonical is the real HF id — never demoted
        sigs = _entity_identity_sigs(cid, e.get("org_id"), alias_index)
        for key in cands:
            if key in taken:
                continue
            if _identity_sig(key) not in sigs:
                continue
            if not _openrouter_org_agrees(
                key.split("/", 1)[0], e.get("org_id"), alias_index
            ):
                continue
            renames[cid] = key
            taken.add(key)
            e["id"] = key
            e["aliases"] = sorted(set(e.get("aliases") or []) | {cid})
            meta = _entry_meta(e)
            meta["openrouter_adopted"] = True
            e["metadata"] = json.dumps(meta, sort_keys=True)
            break
    if renames:
        for e in entries:
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and edge.get("id") in renames:
                    edge["id"] = renames[edge["id"]]


def _tag_openrouter_key_ids(entries: list[dict], api_json: dict) -> list[dict]:
    """Tag `metadata.openrouter_adopted` on every full entry whose id is a
    cleaned OpenRouter key (in place; returns `entries`). Must run after
    `_generate_models` so `_pick_winner`'s ranking of the flag is not
    perturbed; HF-deferred entries stay untagged."""
    or_keys = {
        k
        for m in ((api_json.get(_OPENROUTER_PROVIDER) or {}).get("models") or {})
        for k in (_clean_openrouter_key(m),)
        if k
    }
    for e in entries:
        cid = e.get("id")
        if not cid or "display_name" not in e or cid not in or_keys:
            continue
        meta = _entry_meta(e)
        if meta.get("hf_deferred") or meta.get("openrouter_adopted") is True:
            continue
        meta["openrouter_adopted"] = True
        e["metadata"] = json.dumps(meta, sort_keys=True)
    return entries


def _attach_provider_aliases(
    entries: list[dict],
    group_recs: list[dict],
    org_id: str | None,
) -> None:
    """Attach every provider spelling in the group to the entry whose id/alias
    set contains it, tagged with its inference_platform under the working
    `alias_platforms` map (flattened into aliases + metadata by
    `_finalize_entries`)."""
    by_form: dict[str, dict] = {}
    for e in entries:
        by_form[e["id"]] = e
        for a in e.get("aliases", []):
            by_form.setdefault(a, e)
    root = next((e for e in entries if not e.get("parents")), entries[0] if entries else None)

    _identity = _variant_identity

    def _target_sigs(target: dict) -> set[str]:
        # An HF-deferred id carries tokens the models.dev spellings lack; the
        # authority equated them, so its id-shaped aliases count as identities
        # too. Display names are excluded: `_identity` drops a parenthetical,
        # so "(Thinking)" would read as the base.
        sigs = {safe_sig(_identity(target["id"]))}
        if _entry_meta(target).get("hf_deferred") is True:
            for a in [*(target.get("aliases") or []), *(target.get("alias_platforms") or {})]:
                if not re.search(r"\s", a):
                    sigs.add(safe_sig(_identity(a)))
        return sigs

    for r in group_recs:
        platform = PROVIDER_TO_INFERENCE_PLATFORM.get(r["provider"])
        raw = r["raw"]
        # Route via the normalized leaf so a host-prefixed mirror still lands
        # on the right entry.
        norm_slug = _slugify(normalize_modelsdev_id(raw)).rsplit("/", 1)[-1]
        target = (
            by_form.get(raw)
            or by_form.get(norm_slug)
            or (by_form.get(f"{org_id}/{norm_slug}") if org_id else None)
            or root
        )
        if target is None:
            continue
        # Only attach to a target of the same variant-preserving identity: a
        # `-fp8`/`-instruct` record landing on its base would alias the
        # variant's id onto it and abort the seed. `safe_sig` is
        # order-insensitive, so `claude-haiku-3` still matches an adopted
        # `claude-3-haiku`.
        if safe_sig(_identity(raw)) not in _target_sigs(target):
            continue
        ap = target.setdefault("alias_platforms", {})
        for form in _provider_alias_forms(raw, org_id, r.get("provider")):
            if form == target["id"]:
                continue
            # A form already owned by another entry in this group must not be
            # claimed twice.
            owner = by_form.get(form)
            if owner is not None and owner is not target:
                continue
            # First non-null platform wins per form.
            if form not in ap or (ap.get(form) is None and platform):
                ap[form] = platform


# ---------------------------------------------------------------------------
# Mint decision. Before minting an off-HF `{org}/{slug}` canonical, defer to a
# real HF repo (the canonical IS the HF id) only on a normalized-identity match
# with org agreement after the curated dev-org remap, never on a name-only
# match across developers. Default to mint. Authority: the frozen HF oracle
# plus the published hub_stats index.
# ---------------------------------------------------------------------------

HF_ORACLE_JSON = resolve_oracle_path()

# The published HF id index (local fixture, else the dataset copy) is the
# generator's only HF knowledge: no per-model HF API calls.
HUB_STATS_INDEX_PATH = REPO_ROOT / "fixtures" / "hub_stats_index.parquet"
HF_DATASET_REPO = "evaleval/entity-registry-data"
HUB_STATS_INDEX_TABLE = "hub_stats_index"

_HF_AUTHORITY: dict[str, dict[str, str]] | None = None
_INDEX_AUTHORITY: dict[str, dict[str, str]] | None = None

# Curated umbrella ids (`deepseek/deepseek`) are never one specific HF repo,
# so they are never deferred, whatever a rolling first-party key claims.
COLLISION_OVERRIDES_PATH = REPO_ROOT / "seed" / "models" / "collision_overrides.yaml"
PARENTS_ENRICH_PATH = REPO_ROOT / "seed" / "models" / "enrichments" / "parents.yaml"
_NON_LINEAGE_BASES: frozenset[str] | None = None


def _non_lineage_bases() -> frozenset[str]:
    global _NON_LINEAGE_BASES
    if _NON_LINEAGE_BASES is None:
        data = (
            safe_load_yaml(COLLISION_OVERRIDES_PATH.read_text())
            if COLLISION_OVERRIDES_PATH.exists() else {}
        ) or {}
        _NON_LINEAGE_BASES = frozenset(data.get("non_lineage_bases") or [])
    return _NON_LINEAGE_BASES


_PROTECTED_ROOTS: frozenset[str] | None = None


def _multi_child_roots() -> frozenset[str]:
    """Ids two or more committed entries name as a `variant` parent: such a
    family root is not one repo, whatever a first-party key equates its bare
    name with (a repo with many finetunes or quants is still one repo)."""
    children: dict[str, set[str]] = defaultdict(set)
    for path in (CORE_PATH, PARENTS_ENRICH_PATH, HUB_STATS_PATH, TIER3_PATH, SEED_PATH):
        if not path.exists():
            continue
        data = safe_load_yaml(path.read_text()) or []
        items = data.get("entries") if isinstance(data, dict) else data
        for e in items or []:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            raw = e.get("parents")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except ValueError:
                    raw = []
            for p in raw or []:
                if not isinstance(p, dict) or p.get("relationship") != "variant":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid != e["id"]:
                    children[pid].add(e["id"])
    return frozenset(pid for pid, kids in children.items() if len(kids) >= 2)


def _protected_roots() -> frozenset[str]:
    """Ids never deferred, twin-absorbed or folded onto an HF repo: the curated
    umbrellas plus every committed multi-child family root."""
    global _PROTECTED_ROOTS
    if _PROTECTED_ROOTS is None:
        _PROTECTED_ROOTS = _non_lineage_bases() | _multi_child_roots()
    return _PROTECTED_ROOTS

# --- shared org-aware fold inputs (used by reconcile_generated_against_existing) ---
_HF_TO_DEV: dict[str, str] | None = None


def _hf_to_dev() -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug, via the resolver's shared
    `build_curated_org_map` so the org-aware fold agrees with the resolver."""
    global _HF_TO_DEV
    if _HF_TO_DEV is None:
        from eval_entity_resolver.fold import build_curated_org_map
        data = safe_load_yaml(ORGS_SEED_PATH.read_text()) if ORGS_SEED_PATH.exists() else []
        _HF_TO_DEV = build_curated_org_map(data or [])
    return _HF_TO_DEV


_ORACLE_FIXED_IDS_CACHE: frozenset[str] | None = None


def _oracle_fixed_ids() -> frozenset[str]:
    """Real HF repo ids from the frozen oracle (fixed_exact/near_miss), so the
    fold index recognises HF repos not yet written to any source."""
    global _ORACLE_FIXED_IDS_CACHE
    if _ORACLE_FIXED_IDS_CACHE is None:
        ids: set[str] = set()
        if HF_ORACLE_JSON.exists():
            oracle = json.loads(HF_ORACLE_JSON.read_text()).get("resolutions", {})
            for v in oracle.values():
                if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss"):
                    fx = v.get("fixed_hf_model_id")
                    if isinstance(fx, str) and "/" in fx:
                        ids.add(fx)
        _ORACLE_FIXED_IDS_CACHE = frozenset(ids)
    return _ORACLE_FIXED_IDS_CACHE


def _build_hf_authority(
    oracle_path: Path = HF_ORACLE_JSON,
    alias_index: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """{dev_org: {normalized_name: real_hf_model_id}} over the frozen oracle's
    fixed_exact / fixed_near_miss entries; the HF org is remapped through the
    curated dev-org map so the bucket key is the developer (`Qwen`->`alibaba`),
    the name via the resolver's `normalize`."""
    from eval_entity_resolver.normalization import normalize as _norm

    ai = alias_index if alias_index is not None else _dev_alias_index()
    out: dict[str, dict[str, str]] = defaultdict(dict)
    if not oracle_path.exists():
        return out
    oracle = json.loads(oracle_path.read_text()).get("resolutions", {})
    for _raw, meta in oracle.items():
        if meta.get("resolution_status") not in ("fixed_exact", "fixed_near_miss"):
            continue
        fixed = meta.get("fixed_hf_model_id")
        if not isinstance(fixed, str) or "/" not in fixed:
            continue
        hf_org, hf_name = fixed.split("/", 1)
        dev_org = ai.get(hf_org.lower(), hf_org.lower())
        out[dev_org].setdefault(_norm(hf_name), fixed)
    return out


def _load_hub_stats_index(path: Path = HUB_STATS_INDEX_PATH):
    """The published HF id index as a DataFrame (`id`, `downloads`): local
    fixture, else the dataset copy; neither fails loudly so the cron never
    silently mints what is on HF."""
    import pandas as pd

    if path.exists():
        return pd.read_parquet(path, columns=["id", "downloads"])
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"{HUB_STATS_INDEX_TABLE}/part-0.parquet",
            repo_type="dataset",
        )
    except Exception as exc:
        raise RuntimeError(
            f"hub_stats_index unavailable: no {path} and the dataset copy "
            f"({HF_DATASET_REPO}) could not be fetched: {exc}"
        ) from exc
    return pd.read_parquet(local, columns=["id", "downloads"])


def _build_index_authority(
    index, alias_index: dict[str, str] | None = None
) -> dict[str, dict[str, str]]:
    """{dev_org: {normalized_name: hf_id}} over the hub_stats index, bucketed
    like `_build_hf_authority`; on a shared name-norm the higher download
    count wins."""
    from eval_entity_resolver.normalization import normalize as _norm

    ai = alias_index if alias_index is not None else _dev_alias_index()
    out: dict[str, dict[str, str]] = defaultdict(dict)
    ranked = index.sort_values(["downloads", "id"], ascending=[False, True])
    for hf_id in ranked["id"].tolist():
        if "/" not in hf_id:
            continue
        hf_org, hf_name = hf_id.split("/", 1)
        dev_org = ai.get(hf_org.lower(), hf_org.lower())
        out[dev_org].setdefault(_norm(hf_name), hf_id)
    return out


_INDEX_IDS: dict[str, str] | None = None


def _index_authority() -> dict[str, dict[str, str]]:
    global _INDEX_AUTHORITY, _INDEX_IDS
    if _INDEX_AUTHORITY is None:
        index = _load_hub_stats_index()
        _INDEX_AUTHORITY = _build_index_authority(index)
        # A key that IS an HF repo id (case-folded) names its repo, whatever
        # other namespace shares the name.
        _INDEX_IDS = {}
        for hf_id in index.sort_values(["downloads", "id"], ascending=[False, True])["id"].tolist():
            _INDEX_IDS.setdefault(hf_id.lower(), hf_id)
    return _INDEX_AUTHORITY


def _index_ids() -> dict[str, str]:
    if _INDEX_IDS is None:
        _index_authority()
    return _INDEX_IDS or {}


def _merge_authorities(
    oracle: dict[str, dict[str, str]], index: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Oracle wins on a conflicting (dev_org, name-norm)."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for dev_org, bucket in index.items():
        out[dev_org].update(bucket)
    for dev_org, bucket in oracle.items():
        out[dev_org].update(bucket)
    return out


_ORACLE_AUTHORITY: dict[str, dict[str, str]] | None = None


def _oracle_authority() -> dict[str, dict[str, str]]:
    global _ORACLE_AUTHORITY
    if _ORACLE_AUTHORITY is None:
        _ORACLE_AUTHORITY = _build_hf_authority()
    return _ORACLE_AUTHORITY


def _hf_authority() -> dict[str, dict[str, str]]:
    global _HF_AUTHORITY
    if _HF_AUTHORITY is None:
        _HF_AUTHORITY = _merge_authorities(_oracle_authority(), _index_authority())
    return _HF_AUTHORITY


def _candidate_name_norms(
    spellings: list[str], dev_org: str | None, alias_index: dict[str, str]
) -> set[str]:
    """Normalized name forms a group's spellings should match HF on: each
    leaf, plus the leaf with a leading brand token stripped when that token is
    a curated alias of the dev org (`qwen-qwq-32b` -> `qwq-32b`)."""
    from eval_entity_resolver.normalization import normalize as _norm

    norms: set[str] = set()
    for sp in spellings:
        if not sp:
            continue
        leaf = sp.rsplit("/", 1)[-1]
        n = _norm(leaf)
        if not n:
            continue
        norms.add(n)
        toks = n.split(" ")
        if dev_org and len(toks) > 1:
            stripped = " ".join(toks[1:])
            if alias_index.get(toks[0]) == dev_org and stripped:
                norms.add(stripped)
    return norms


def _hf_defer_target(
    candidate_id: str,
    org_id: str | None,
    spellings: list[str],
    alias_index: dict[str, str],
    authority: dict[str, dict[str, str]] | None = None,
    accept: Callable[[str], bool] | None = None,
) -> str | None:
    """The real HF id to defer to when the group's normalized names match one
    within the SAME dev-org bucket, else None (mint). `accept` vetoes a hit and
    the search continues. No org, or a match only under another developer,
    always mints."""
    if org_id is None:
        return None
    auth = authority if authority is not None else _hf_authority()
    # Uncurated orgs are bucketed by lowercased HF slug; the group's org_id
    # keeps the raw prefix spelling.
    bucket = auth.get(org_id) or auth.get(org_id.lower())
    if not bucket:
        return None
    forms = list(spellings)
    if candidate_id:
        forms.append(candidate_id)
    # Deterministic order (a set would churn with the hash seed), shortest norm
    # first: that is the base identity, so a decorated spelling never pulls a
    # base group onto a variant repo.
    for name_norm in sorted(
        _candidate_name_norms(forms, org_id, alias_index),
        key=lambda n: (n.count(" "), len(n), n),
    ):
        hit = bucket.get(name_norm)
        if hit is not None and (accept is None or accept(hit)):
            return hit
    return None


def _group_hf_target(
    mint_id: str,
    org_id: str | None,
    group_recs: list[dict],
    spellings: list[str],
    alias_index: dict[str, str],
    authority: dict[str, dict[str, str]] | None = None,
) -> str | None:
    """`_hf_defer_target` over a group's keys one tier at a time (mint id,
    `canonical_model_id` keys, `org/name` provider ids, other spellings); the
    first tier that answers wins, so a dated key is never outranked by a
    shorter undated name. A hit must keep the whole mint name and add no
    identity-variant token the mint lacks (a bare `qwen3-vl-30b-a3b` never
    promotes to the `…-Thinking` repo)."""
    if org_id is None or mint_id in _non_lineage_bases():
        return None
    from eval_entity_resolver.normalization import normalize as _norm

    # A multi-child family root folds only onto a repo carrying its own bare
    # name, never onto one child a first-party key equates it with.
    root_bound = mint_id in _protected_roots()

    def _own_name(hf_id: str) -> bool:
        return _norm(hf_id.rsplit("/", 1)[-1]) == _norm(mint_id.rsplit("/", 1)[-1])

    # The oracle is consulted first without tiers or veto so committed
    # deferrals stay put; an injected `authority` (tests) gets no oracle.
    oracle = _oracle_authority() if authority is None else {}
    if oracle:
        hit = _hf_defer_target(
            mint_id, org_id, spellings, alias_index, oracle,
            accept=_own_name if root_bound else None,
        )
        if hit is not None:
            return hit
    def _deglued(s: str) -> str | None:
        # Normalization would weld a `.`/`:`-glued brand (Bedrock's
        # `openai.gpt-oss-safeguard-120b`) into the name.
        m = re.match(r"^(?:[^/]+/)?([A-Za-z0-9_-]+)[.:](.+)$", s)
        if m and (m.group(1).lower() == org_id or alias_index.get(m.group(1).lower()) == org_id):
            return m.group(2)
        return None

    mint_forms = [mint_id] + [d for d in (_deglued(mint_id),) if d]
    mint_keys = [
        re.sub(r"[^a-z0-9]", "", n)
        for n in _candidate_name_norms(mint_forms, org_id, alias_index)
    ]
    mint_variants = _variant_tokens(mint_id)

    from eval_entity_registry.lib.collision_fold import _bsizes

    mint_sizes = _bsizes(mint_id.rsplit("/", 1)[-1])

    def _keeps_mint_name(hf_id: str) -> bool:
        if _variant_tokens(hf_id) - mint_variants:
            return False
        # A stated size must match (`model-7b` is a subsequence of
        # `Model-70B`); a sizeless key may land on a sized repo.
        if mint_sizes and _bsizes(hf_id.rsplit("/", 1)[-1]) != mint_sizes:
            return False
        hf_key = re.sub(r"[^a-z0-9]", "", hf_id.rsplit("/", 1)[-1].lower())
        for key in mint_keys:
            it = iter(hf_key)
            if all(ch in it for ch in key):
                return True
        return False

    cmid_counts = Counter(
        c for r in group_recs
        for c in ((r.get("record") or {}).get("canonical_model_id"),)
        if isinstance(c, str) and c
    )
    # A verbatim HF repo id is tried before any name match, so a same-named
    # repo in a sibling namespace cannot outrank it on downloads.
    if authority is None:
        verbatim = [c for c, _n in sorted(cmid_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        verbatim += [r["raw"] for r in group_recs if r.get("raw") and "/" in r["raw"]]
        for cand in verbatim:
            hf_id = _index_ids().get(cand.lower())
            if hf_id is None:
                continue
            hf_org = hf_id.split("/", 1)[0].lower()
            if alias_index.get(hf_org, hf_org) != org_id and hf_org != org_id.lower():
                continue
            if _keeps_mint_name(hf_id) and (not root_bound or _own_name(hf_id)):
                return hf_id

    tiers: list[list[str]] = [[mint_id]]
    tiers += [
        [c] for c, _n in sorted(cmid_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    tiers.append([r["raw"] for r in group_recs if r.get("raw") and "/" in r["raw"]])
    tiers.append([d for s in [mint_id, *(r["raw"] for r in group_recs if r.get("raw"))]
                  for d in (_deglued(s),) if d])
    tiers.append(spellings + [r["name"] for r in group_recs if r.get("name")])
    for tier in tiers:
        if not tier:
            continue
        hit = _hf_defer_target(
            "", org_id, tier, alias_index, authority,
            accept=lambda h: _keeps_mint_name(h) and (not root_bound or _own_name(h)),
        )
        if hit is not None:
            return hit
    return None


def _hf_deferred_entry(
    hf_id: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
    mint_id: str,
    display_name: str,
) -> dict:
    """HF-deferred record: the canonical id is the real HF repo id, with
    models.dev metadata merged on and the models.dev spellings as aliases."""
    from eval_entity_resolver.normalization import normalize as _norm

    # A group can carry a minority canonical key naming another model; only
    # keys naming THIS repo become aliases.
    hf_norm = _norm(hf_id.rsplit("/", 1)[-1])
    cmids = sorted({
        c for r in group_recs
        for c in ((r.get("record") or {}).get("canonical_model_id"),)
        if isinstance(c, str) and c
        and hf_norm in _candidate_name_norms([c], org_id, _dev_alias_index())
    })
    aliases = []
    for a in (mint_id, display_name, *cmids):
        if a and a != hf_id and a not in aliases:
            aliases.append(a)
    return {
        "id": hf_id,
        "display_name": head["display_name"] or humanize_model_slug(hf_id.split("/", 1)[-1]),
        "org_id": org_id,
        "family": None,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": head["open_weights"],
        "release_date": head["release_date"],
        "input_modalities": None,
        "output_modalities": None,
        "tags": ["open-weight"] if head["open_weights"] else [],
        "aliases": aliases,
        "metadata": json.dumps(
            {
                "underlying_key": head.get("root_key", hf_id),
                "providers": sorted({r["provider"] for r in group_recs}),
                "hf_deferred": True,
            },
            sort_keys=True,
        ),
        "review_status": "reviewed",
        "resolution_source": "models_dev",
    }


def _mint_rehost_entry(
    root_key: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
) -> list[dict]:
    """Mint one canonical for a re-host-only / closed-API group with no
    author-lab tree: `{org}/{slug}`, or a bare slug tagged org-unknown. Returns
    [entry]; provider aliases are attached by the caller."""
    head_raw = head["head_spelling"] or root_key
    # Normalize first so host scaffolding (`@cf/`, `accounts/.../`) never
    # leaks slashes into the id.
    slug = _slugify(normalize_modelsdev_id(head_raw)) or _slugify(root_key)
    slug = slug.rsplit("/", 1)[-1]
    canonical_id = f"{org_id}/{slug}" if org_id else slug
    display_name = head["display_name"] or humanize_model_slug(slug)
    open_weights = head["open_weights"]
    release_date = head["release_date"]
    tags = ["open-weight"] if open_weights else []
    if org_id is None:
        tags = tags + ["org-unknown"]
    entry = {
        "id": canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": slug,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": open_weights,
        "release_date": release_date,
        "input_modalities": None,
        "output_modalities": None,
        "tags": tags,
        "aliases": [],
        "metadata": json.dumps(
            {"underlying_key": root_key, "providers": sorted({r["provider"] for r in group_recs})},
            sort_keys=True,
        ),
        "review_status": "draft" if not head["has_author_lab_entry"] else "reviewed",
        "resolution_source": "models_dev",
    }
    return [entry]


def _mint_or_defer_rehost(
    root_key: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
    alias_index: dict[str, str],
) -> list[dict]:
    """Re-host path mint decision: an HF-deferred record when the group
    resolves to a real HF repo, else the minted `{org}/{slug}` canonical."""
    # Must mirror _mint_rehost_entry's slug derivation.
    head_raw = head["head_spelling"] or root_key
    slug = (_slugify(normalize_modelsdev_id(head_raw)) or _slugify(root_key)).rsplit("/", 1)[-1]
    mint_id = f"{org_id}/{slug}" if org_id else slug
    display_name = head["display_name"] or humanize_model_slug(slug)

    spellings = [mint_id, head_raw, display_name, slug]
    spellings += [r["raw"] for r in group_recs if r.get("raw")]

    hf_id = _group_hf_target(mint_id, org_id, group_recs, spellings, alias_index)
    if hf_id is not None:
        head = {**head, "root_key": root_key}
        return [_hf_deferred_entry(hf_id, org_id, head, group_recs, mint_id, display_name)]
    entries = _mint_rehost_entry(root_key, org_id, head, group_recs)
    _adopt_openrouter_ids(entries, group_recs, org_id, alias_index)
    return entries


def _defer_author_records(
    root_key: str,
    org_id: str,
    author_recs: list[dict],
    group_recs: list[dict],
    alias_index: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Author-lab path mint decision, per record: a record naming an HF repo
    becomes an HF-deferred entry and leaves the family build. Returns
    (deferred_entries, remaining_records)."""
    deferred: list[dict] = []
    remaining: list[dict] = []
    for r in author_recs:
        mint_id = f"{org_id}/{_slugify(r['raw'])}"
        display_name = r.get("name") or humanize_model_slug(_slugify(r["raw"]))
        hf_id = _group_hf_target(
            mint_id, org_id, group_recs, [r["raw"], display_name], alias_index
        )
        if hf_id is None:
            remaining.append(r)
            continue
        head = {
            "display_name": r.get("name"),
            "open_weights": bool(r.get("open_weights")),
            "release_date": r.get("release"),
            "root_key": root_key,
        }
        deferred.append(_hf_deferred_entry(hf_id, org_id, head, [r], mint_id, display_name))
    return deferred, remaining


def _generate_models(api_json: dict, known_org_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Group -> mint -> alias over the full catalog: each underlying group
    yields an author-lab family tree or a minted re-host canonical, with every
    provider spelling aliased in. Returns (entries, skipped_no_org); a
    non-empty skipped_no_org is a hard error (org missing from seed/orgs.yaml).
    """
    out: list[dict] = []
    skipped_providers: list[str] = []
    skipped_no_org: list[str] = []
    alias_index = _build_org_alias_index()

    groups = build_underlying_groups(api_json)

    for root_key, recs in sorted(groups.items()):
        recs = [r for r in recs if r["provider"] in PROVIDER_TO_INFERENCE_PLATFORM]
        if not recs:
            continue

        head = pick_underlying(root_key, recs)
        hf_org = head["author_org"]
        org_id = normalize_org_slug(hf_org, alias_index)

        # A curated PROVIDER_TO_ORG author's org id wins over the HF-style slug;
        # the two can diverge (`inceptionai` vs `inception`).
        curated_author_recs = [
            r for r in recs
            if r["provider"] in PROVIDER_TO_ORG
            and "/" not in r["raw"]
            and head["has_author_lab_entry"]
            # only treat as author when its curated org agrees with the family-org
            and (
                org_id is None
                or PROVIDER_TO_ORG[r["provider"]] == org_id
                or normalize_org_slug(AUTHOR_PROV_ORG.get(r["provider"]), alias_index) == org_id
            )
        ]
        if curated_author_recs:
            org_id = PROVIDER_TO_ORG[curated_author_recs[0]["provider"]]

        author_recs = curated_author_recs

        if head["has_author_lab_entry"] and author_recs and org_id:
            if org_id not in known_org_ids:
                skipped_no_org.append(
                    f"{author_recs[0]['provider']} -> {org_id} (group {root_key})"
                )
                continue
            deferred, author_recs = _defer_author_records(
                root_key, org_id, author_recs, recs, alias_index
            )
            by_family: dict[str, list[dict]] = defaultdict(list)
            for r in author_recs:
                by_family[_family_for(r["record"])].append(r["record"])
            group_entries: list[dict] = []
            for family_slug, models in sorted(by_family.items()):
                if not family_slug:
                    continue
                group_entries.extend(_build_family_entries(
                    org_id, family_slug, models,
                    group_recs=recs, alias_index=alias_index,
                ))
            # Family entries first: a deferred entry must not become the
            # provider-alias fallback root while a family root exists.
            group_entries.extend(deferred)
            if not group_entries:
                group_entries = _mint_or_defer_rehost(root_key, org_id, head, recs, alias_index)
        else:
            group_entries = _mint_or_defer_rehost(root_key, org_id, head, recs, alias_index)

        _attach_provider_aliases(group_entries, recs, org_id)
        out.extend(group_entries)

    if skipped_providers:
        print(
            f"[refresh] skipped {len(skipped_providers)} provider records not in "
            f"PROVIDER_TO_INFERENCE_PLATFORM",
            file=sys.stderr,
        )
    # An org-less mint cannot be bucketed against the HF authority; when its
    # name is that of an HF-deferred entry of this run, it folds onto it.
    from eval_entity_resolver.normalization import normalize as _norm

    hf_by_name: dict[str, list[dict]] = defaultdict(list)
    for e in out:
        if _entry_meta(e).get("hf_deferred") is True and e.get("display_name"):
            hf_by_name[_norm(e["display_name"])].append(e)
    kept: list[dict] = []
    for e in out:
        # An ambiguous name (2+ HF-deferred entries) is no identity evidence.
        cands = (
            hf_by_name.get(_norm(e["display_name"]), [])
            if "/" not in e["id"] and e.get("display_name") and e.get("open_weights") is True
            and _entry_meta(e).get("hf_deferred") is not True else []
        )
        tgt = cands[0] if len(cands) == 1 else None
        if tgt is None:
            kept.append(e)
            continue
        aliases = list(tgt.get("aliases") or [])
        for a in (e["id"], *(e.get("aliases") or [])):
            if a and a != tgt["id"] and a not in aliases:
                aliases.append(a)
        tgt["aliases"] = aliases
        if isinstance(e.get("alias_platforms"), dict):
            tgt.setdefault("alias_platforms", {}).update(e["alias_platforms"])
    out = kept

    # A model reached from two dedup groups would otherwise emit twice.
    return _reconcile_intra_output(_dedup_entries(out)), skipped_no_org


def _pick_winner(group: list[dict]) -> dict:
    """Authoritative entry among same-model dups, by id-ladder rung: HF-deferred,
    OpenRouter-adopted, org-qualified, HF-true-cased, reviewed, shortest id
    (drops doubled-brand mints like `cohere/cohere-command-a`), alphabetical."""
    def _meta_flag(e: dict, flag: str) -> bool:
        try:
            return json.loads(e.get("metadata") or "{}").get(flag) is True
        except (ValueError, TypeError):
            return False

    def key(e: dict):
        cid = e.get("id") or ""
        return (
            0 if _meta_flag(e, "hf_deferred") else 1,
            0 if _meta_flag(e, "openrouter_adopted") else 1,
            0 if "/" in cid else 1,            # org-qualified beats bare org-less
            0 if any(c.isupper() for c in cid) else 1,  # HF-true casing
            0 if e.get("review_status") == "reviewed" else 1,
            len(cid),
            cid,
        )

    return sorted(group, key=key)[0]


def _reconcile_intra_output(entries: list[dict]) -> list[dict]:
    """Reconcile one generation's output so no emitted alias equals (exact or
    normalized) another canonical's id: merge same-model dups (same org_id and
    intersecting brand-aware name norms; `_pick_winner` keeps id+casing,
    losers become aliases, parent edges repointed), then strip alias-steals (a
    variant carrying a different model's id as an alias; not a merge)."""
    from eval_entity_resolver.normalization import normalize as _norm

    alias_index = _dev_alias_index()

    # ---- pass 1: org-aware same-model union-find ----
    uf = UnionFind()

    for e in entries:
        uf.find(e["id"])

    # (a) Same normalized id collides at seed time regardless of org, so they
    # must merge (bare org-less mint vs its org-qualified twin, dot/case twins).
    norm_id_bucket: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        norm_id_bucket[_norm(e["id"])].append(e["id"])
    for ids in norm_id_bucket.values():
        for other in ids[1:]:
            uf.union(other, ids[0])

    # (b) Same org_id and intersecting brand-aware name norms.
    bucket: dict[tuple, list[str]] = defaultdict(list)
    for e in entries:
        org = e.get("org_id")
        if not org:
            continue  # org-less entries merge only via (a)
        for nm in _candidate_name_norms([e["id"]], org, alias_index):
            bucket[(org, nm)].append(e["id"])
    for ids in bucket.values():
        for other in ids[1:]:
            uf.union(other, ids[0])

    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[uf.find(e["id"])].append(e)

    merged: list[dict] = []
    loser_to_winner: dict[str, str] = {}
    for grp in groups.values():
        if len(grp) == 1:
            merged.append(grp[0])
            continue
        winner = _pick_winner(grp)
        walias = set(winner.get("aliases") or [])
        wap = winner.setdefault("alias_platforms", {})
        # An hf_deferred winner often has these empty; fill from losers.
        _PREFER_NONEMPTY = (
            "params_billions", "family", "architecture",
            "input_modalities", "output_modalities",
        )
        wparents = {edge.get("id"): edge for edge in (winner.get("parents") or []) if isinstance(edge, dict)}
        for e in grp:
            if e is winner:
                continue
            loser_to_winner[e["id"]] = winner["id"]
            walias.update(e.get("aliases") or [])
            if e["id"] != winner["id"]:
                walias.add(e["id"])
            for k, v in (e.get("alias_platforms") or {}).items():
                if k not in wap or (wap.get(k) is None and v):
                    wap[k] = v
            winner["open_weights"] = bool(winner.get("open_weights")) or bool(e.get("open_weights"))
            if e.get("review_status") == "reviewed":
                winner["review_status"] = "reviewed"
            for f in _PREFER_NONEMPTY:
                if not winner.get(f) and e.get(f):
                    winner[f] = e[f]
            rds = [d for d in (winner.get("release_date"), e.get("release_date")) if d]
            if rds:
                winner["release_date"] = min(rds)
            for edge in (e.get("parents") or []):
                if isinstance(edge, dict) and edge.get("id") not in wparents:
                    wparents[edge["id"]] = edge
        if wparents:
            winner["parents"] = list(wparents.values())
        winner["aliases"] = sorted(walias)
        merged.append(winner)

    for e in merged:
        for edge in (e.get("parents") or []):
            if isinstance(edge, dict) and edge.get("id") in loser_to_winner:
                edge["id"] = loser_to_winner[edge["id"]]

    # ---- pass 2: strip alias-steals (alias == a different canonical's id) ----
    # `alias_platforms` too: _finalize_entries flattens it back into aliases.
    ids = {e["id"] for e in merged}
    id_norm: dict[str, str] = {}
    for cid in ids:
        id_norm.setdefault(_norm(cid), cid)

    def _steals_id(form: str, cid: str) -> bool:
        if not form or form == cid:
            return False
        if form in ids:
            return True
        owner = id_norm.get(_norm(form))
        return owner is not None and owner != cid

    for e in merged:
        cid = e["id"]
        e["aliases"] = sorted({a for a in (e.get("aliases") or []) if not _steals_id(a, cid)})
        ap = e.get("alias_platforms")
        if isinstance(ap, dict):
            e["alias_platforms"] = {k: v for k, v in ap.items() if not _steals_id(k, cid)}

    # ---- pass 3: a form claimed by 2+ canonicals aborts the seed (the loader
    # promotes display_name to a global alias, so it counts too). Keep it on the
    # natural owner (id-name normalizes to it), else the first by id.
    claimers: dict[str, set[str]] = defaultdict(set)
    for e in merged:
        cid = e["id"]
        for a in e.get("aliases") or []:
            claimers[a].add(cid)
        for k in (e.get("alias_platforms") or {}):
            claimers[k].add(cid)
        dn = e.get("display_name")
        if isinstance(dn, str) and dn:
            claimers[dn].add(cid)

    keeper: dict[str, str] = {}
    for form, owners in claimers.items():
        if len(owners) < 2:
            continue
        natural = sorted(c for c in owners if _norm(c.split("/", 1)[-1]) == _norm(form))
        keeper[form] = (natural or sorted(owners))[0]

    if keeper:
        for e in merged:
            cid = e["id"]
            e["aliases"] = sorted(
                a for a in (e.get("aliases") or []) if a not in keeper or keeper[a] == cid
            )
            ap = e.get("alias_platforms")
            if isinstance(ap, dict):
                e["alias_platforms"] = {k: v for k, v in ap.items() if k not in keeper or keeper[k] == cid}
            dn = e.get("display_name")
            if isinstance(dn, str) and dn in keeper and keeper[dn] != cid:
                e["display_name"] = cid.split("/", 1)[-1]

    return sorted(merged, key=lambda e: e["id"])


def _dedup_entries(entries: list[dict]) -> list[dict]:
    """Collapse entries sharing a canonical id: union aliases / tags /
    alias_platforms, reviewed beats draft, earliest release_date wins."""
    by_id: dict[str, dict] = {}
    for e in entries:
        cur = by_id.get(e["id"])
        if cur is None:
            by_id[e["id"]] = e
            continue
        cur["aliases"] = sorted(set(cur.get("aliases", [])) | set(e.get("aliases", [])))
        cur["tags"] = sorted(set(cur.get("tags", [])) | set(e.get("tags", [])))
        ap = cur.setdefault("alias_platforms", {})
        for k, v in (e.get("alias_platforms") or {}).items():
            if k not in ap or (ap.get(k) is None and v):
                ap[k] = v
        cur["open_weights"] = bool(cur.get("open_weights")) or bool(e.get("open_weights"))
        if e.get("review_status") == "reviewed":
            cur["review_status"] = "reviewed"
        rd =[d for d in (cur.get("release_date"), e.get("release_date")) if d]
        cur["release_date"] = min(rd) if rd else None
    return sorted(by_id.values(), key=lambda e: e["id"])


def _load_known_org_ids() -> set[str]:
    if not ORGS_SEED_PATH.exists():
        return set()
    data = safe_load_yaml(ORGS_SEED_PATH.read_text()) or []
    return {e["id"] for e in data if "id" in e}


_HEADER = """# Generated from models.dev (https://models.dev) — DO NOT EDIT BY HAND.
# To update: edit seed/models/core.yaml (curated canonicals win at load
# time), then run `python scripts/refresh_from_modelsdev.py` to regenerate
# this file.
#
# Source: https://models.dev/api.json (MIT, (c) 2025 models.dev)
# Last refresh date is in git history — see
# `git log -1 -- seed/models/sources/models_dev.generated.yaml`.
#
# This file is one data source among potentially several under
# `seed/models/sources/`. It contains pure models.dev output — no curated
# overlays. The seed CLI loader merges sources → core → enrichments at
# load time (field-level merge: aliases / tags UNION).
#
# Each entry collapses all snapshots / dated variants of a model family
# into one canonical id (`<org>/<family-slug>`). The resolver's fuzzy stem
# step (in eval_entity_resolver/strategies/fuzzy.py) strips date suffixes,
# thinking budgets, hosting provider tags, etc., so per-snapshot raw IDs
# resolve to this family canonical without needing per-snapshot entries.
#
# `aliases` lists snapshot IDs we observed in models.dev for this family.
# Enrich records ({id, aliases} onto an existing canonical) may carry a
# `weak:` scalar map donated from the suppressed/folded mint; the seed
# loader applies those only to fields every full entry left empty and
# core.yaml does not explicitly carry.
"""


def _finalize_entries(entries: list[dict]) -> list[dict]:
    """Flatten the working `alias_platforms` map into the persisted shape: keys
    unioned into `aliases`, non-null platforms kept as
    metadata['alias_platforms'] so the loader can wire the platform FK."""
    finalized: list[dict] = []
    for e in entries:
        ap = e.pop("alias_platforms", None) or {}
        aliases = set(e.get("aliases", []))
        aliases.update(ap.keys())
        aliases.discard(e["id"])
        e["aliases"] = sorted(aliases)
        if ap:
            meta = json.loads(e.get("metadata") or "{}")
            meta["alias_platforms"] = {k: v for k, v in sorted(ap.items()) if v}
            e["metadata"] = json.dumps(meta, sort_keys=True)
        finalized.append(e)
    return finalized


def _write_yaml(entries: list[dict], path: Path) -> str:
    body = yaml.safe_dump(_finalize_entries(entries), sort_keys=False, allow_unicode=True, width=200)
    return _HEADER + "\n" + body


# ---------------------------------------------------------------------------
# Full-catalog split + org de-orphan. HF source-of-truth wins every collision:
# an HF-present model becomes an alias-only enrichment onto the HF-cased
# canonical; only models.dev-only models are minted.
# ---------------------------------------------------------------------------

CATALOG_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml"
HF_ORACLE_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hf_oracle.generated.yaml"
HUB_STATS_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
TIER3_PATH = REPO_ROOT / "seed" / "models" / "sources" / "tier3_inferred.generated.yaml"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
ENRICH_ALIASES_PATH = REPO_ROOT / "seed" / "models" / "enrichments" / "aliases.yaml"
ORGS_GENERATED_PATH = REPO_ROOT / "seed" / "orgs.generated.yaml"

# Existing sources the catalog's id+alias forms must not clash with; SEED_PATH
# is included so the catalog stays purely additive.
_CATALOG_EXISTING_SOURCES = (
    HF_ORACLE_PATH,
    SEED_PATH,
    HUB_STATS_PATH,
    TIER3_PATH,
    CORE_PATH,
    ENRICH_ALIASES_PATH,
)

# Same set minus SEED_PATH, which this path rewrites and so must not dedup
# against; CORE_PATH stays so a mint colliding with a curated id is suppressed
# instead of clobbering core.
_NONCATALOG_EXISTING_SOURCES = (
    HF_ORACLE_PATH,
    HUB_STATS_PATH,
    TIER3_PATH,
    CORE_PATH,
    ENRICH_ALIASES_PATH,
)


def _load_core_skip_ids() -> set[str]:
    """Ids core.yaml suppresses (`skip_ids` + `skip_source_ids`); a committed
    entry with one of these is not carried forward."""
    if not CORE_PATH.exists():
        return set()
    doc = safe_load_yaml(CORE_PATH.read_text())
    if not isinstance(doc, dict):
        return set()
    return set(doc.get("skip_ids") or []) | set(doc.get("skip_source_ids") or [])


def _entry_meta(e: dict) -> dict:
    """An entry's metadata JSON as a dict; malformed metadata degrades to {}
    with a warning rather than crashing the cron."""
    try:
        meta = json.loads(e.get("metadata") or "{}")
    except (ValueError, TypeError):
        print(f"[refresh] WARN: malformed metadata on {e.get('id')!r}; treating as empty", file=sys.stderr)
        return {}
    return meta if isinstance(meta, dict) else {}


def _entry_alias_platforms(e: dict) -> dict:
    """The {alias -> inference_platform} provenance map from a persisted
    entry's metadata JSON ({} when absent/malformed)."""
    ap = _entry_meta(e).get("alias_platforms")
    return ap if isinstance(ap, dict) else {}


def _donor_scalars(e: dict) -> dict:
    """Non-empty weak-scalar fields a suppressed mint donates onto its owner's
    enrich record (from top-level fields or an existing `weak:` map).
    `parents` edges are never donated: they could dangle."""
    weak = e.get("weak")
    weak = weak if isinstance(weak, dict) else {}
    out: dict = {}
    for f in WEAK_SCALAR_FIELDS:
        v = e.get(f)
        if v in (None, "", [], {}):
            v = weak.get(f)
        if v not in (None, "", [], {}):
            out[f] = v
    return out


def _merge_weak(rec: dict, scalars: dict) -> None:
    """Merge donated scalars into rec['weak'] (existing keys win), in
    WEAK_SCALAR_FIELDS order so output stays byte-deterministic."""
    if not scalars:
        return
    weak = rec.get("weak")
    weak = dict(weak) if isinstance(weak, dict) else {}
    for k, v in scalars.items():
        weak.setdefault(k, v)
    rec["weak"] = {k: weak[k] for k in WEAK_SCALAR_FIELDS if k in weak}


def _make_form_owner_lookup(sources: tuple[Path, ...]) -> Callable[[str], str | None]:
    """form -> owning canonical id over `sources` (exact, then normalized)."""
    from eval_entity_resolver.normalization import normalize as _norm

    exact, normed = _build_existing_index(sources)

    def _owner(form: str) -> str | None:
        return exact.get(form) or normed.get(_norm(form))

    return _owner


def _union_alias_platforms(entry: dict, donor_ap: dict) -> None:
    """Union `donor_ap` into entry's metadata.alias_platforms for forms the
    entry carries as aliases (existing keys win)."""
    aliases = set(entry.get("aliases") or [])
    add = {k: v for k, v in donor_ap.items() if v and k in aliases}
    if not add:
        return
    meta = _entry_meta(entry)
    ap = meta.get("alias_platforms") or {}
    meta["alias_platforms"] = dict(sorted({**add, **ap}.items()))
    entry["metadata"] = json.dumps(meta, sort_keys=True)


def _twin_key(cid: str) -> str:
    """Org-aware collision key: org folded through the curated dev-org map,
    name keyed like the seed loader's `collision_key`, so a committed id and
    its respelling (`veo-3-1` vs `veo3-1`) share a key."""
    if "/" in cid:
        org, name = cid.split("/", 1)
        dev = _hf_to_dev().get(org.lower())
        if dev:
            return f"{dev}/{collision_key(name)}"
    return collision_key(cid)


def _version_marker_sig(name: str) -> str:
    """Sorted `vN` tokens of a raw leaf, split before any tag/Bedrock strip:
    `nova-premier-v1:0` carries `v1`, bare `nova-premier` none, so the two
    never twin."""
    toks = re.split(r"[-_.:/\s]+", name.lower())
    return "+".join(sorted(t for t in toks if re.fullmatch(r"v\d+", t)))


def _extended_twin_keys(cid: str) -> set[str]:
    """Order/brand-insensitive fallback twin keys (`dev-org/identity-sig#vN`)
    for respellings the ordered `_twin_key` misses (`claude-haiku-3` vs
    adopted `claude-3-haiku`); a variant/size/version difference still never
    twins. Used only after a `_twin_key` miss."""
    if "/" not in cid:
        return {f"{_identity_sig(cid)}#{_version_marker_sig(cid)}"}
    org, name = cid.split("/", 1)
    dev = _hf_to_dev().get(org.lower()) or org.lower()
    vsig = _version_marker_sig(name)
    keys = {f"{dev}/{_identity_sig(name)}#{vsig}"}
    toks = _variant_identity(name).split("-")
    if len(toks) > 1 and _hf_to_dev().get(toks[0]) == dev:
        keys.add(f"{dev}/{safe_sig('-'.join(toks[1:]))}#{vsig}")
    return keys


def _pick_twin_candidate(cid: str, candidates: list[dict]) -> dict | None:
    """Carry-forward twin for `cid`: an exact identity match among
    size-compatible candidates, else the separator-only match only when it is
    unambiguous (`qwen3.8` and `qwen-3-8` share a `_twin_key` but are not the
    same identity).
    """
    from eval_entity_registry.lib.collision_fold import _bsizes

    size_matches = [e for e in candidates if _bsizes(cid) == _bsizes(e["id"])]
    if not size_matches:
        return None

    def exact_identity(e: dict) -> bool:
        return (
            _identity_sig(cid) == _identity_sig(e["id"])
            and _version_marker_sig(cid) == _version_marker_sig(e["id"])
        )

    identity_matches = [e for e in size_matches if exact_identity(e)]
    if identity_matches:
        return sorted(identity_matches, key=lambda e: e["id"])[0]
    if len(size_matches) == 1:
        return size_matches[0]
    return None


def _carry_forward_committed(
    fresh: list[dict], committed: list[dict], adopt_migration: bool = False
) -> tuple[list[dict], dict[str, str]]:
    """Merge the committed generated file into a fresh rewrite so an upstream
    removal never deletes a resolvable form: a surviving id gets its committed
    aliases back; a respelled twin keeps the committed id among equal id-ladder
    rungs (stability rule; `adopt_migration=True` inverts it for the one-shot
    OpenRouter run); a removed id is retained tagged `upstream_status: removed`
    unless core.yaml skips it. Returns (batch, committed_claims), the latter
    mapping every committed id/alias to its surviving owner.
    """
    by_id = {e["id"]: e for e in fresh if e.get("id")}
    by_twin: dict[str, list[dict]] = defaultdict(list)
    by_twin_ext: dict[str, list[dict]] = defaultdict(list)
    by_form: dict[str, dict] = {}
    for e in sorted(fresh, key=lambda x: x.get("id") or ""):
        if e.get("id"):
            by_twin[_twin_key(e["id"])].append(e)
            for k in _extended_twin_keys(e["id"]):
                by_twin_ext[k].append(e)
            for a in (e.get("aliases") or []):
                if a:
                    by_form.setdefault(a, e)
    skip = _load_core_skip_ids()
    claims: dict[str, str] = {}
    out = list(fresh)
    retained = 0
    absorbed = 0
    stabilized: dict[str, str] = {}  # fresh-spelling id -> committed id it took
    promoted: dict[str, str] = {}    # committed id absorbed -> fresh id that outranked it
    pending_rung2 = 0                # OpenRouter-key twins deferred by the stability rule
    skip_owner: Callable[[str], str | None] | None = None
    for c in committed:
        cid = c.get("id")
        if not cid:
            continue
        cur = by_id.get(cid)
        dn = c.get("display_name")
        # Twins: full committed entries only (an enrich record's id is another
        # canonical, never donatable as an alias); a curated umbrella is never
        # one repo, so nothing is its twin.
        twin = None
        twin_via_form = False
        if (
            cur is None and cid not in skip and "display_name" in c
            and cid not in _non_lineage_bases()
        ):
            t = _pick_twin_candidate(cid, by_twin.get(_twin_key(cid), []))
            if t is None:
                t = _pick_twin_candidate(
                    cid,
                    [
                        e
                        for k in sorted(_extended_twin_keys(cid))
                        for e in by_twin_ext.get(k, [])
                    ],
                )
            if t is None:
                # Form-level twin: the fresh entry carries the committed id as
                # an alias. Only a serving-host-prefixed committed id (malformed,
                # never wins the stability rule) or an invented id the HF
                # authority has since promoted (token sets differ, so the twin
                # keys cannot link them) absorbs this way.
                t = by_form.get(cid)
                if t is not None and not (
                    ("/" in cid and cid.split("/", 1)[0].lower() in _SERVING_HOSTS)
                    or (
                        _entry_meta(t).get("hf_deferred") is True
                        and _entry_meta(c).get("hf_deferred") is not True
                        # A multi-child family root only onto the repo that
                        # carries its own bare name.
                        and (
                            cid not in _protected_roots()
                            or _variant_identity(cid) == _variant_identity(t["id"])
                        )
                    )
                ):
                    t = None
                twin_via_form = t is not None
            twin = t
        # Stability rule, rung-monotone: the committed id wins among equal
        # rungs; a fresh hf_deferred twin (rung 1) is never renamed back to a
        # non-HF id (that would demote the HF id to an alias) and absorbs the
        # committed id instead; rung 2 over 3 stays gated behind the one-shot
        # migration flag, with the deferred debt counted. A twin that already
        # took a committed id this run is never renamed twice.
        promote_twin = (
            twin is not None
            and not twin_via_form
            and "display_name" in twin
            and _entry_meta(twin).get("hf_deferred") is True
            and _entry_meta(c).get("hf_deferred") is not True
        )
        rename_twin = (
            twin is not None
            and not adopt_migration
            and not promote_twin
            and not twin_via_form
            and twin["id"] not in stabilized
            and "display_name" in twin
        )
        if (
            twin is not None
            and not adopt_migration
            and not promote_twin
            and _entry_meta(twin).get("openrouter_adopted") is True
            and _entry_meta(c).get("openrouter_adopted") is not True
            and _entry_meta(c).get("hf_deferred") is not True
        ):
            pending_rung2 += 1
        owner = (
            cur["id"] if cur is not None
            else (cid if rename_twin else (twin["id"] if twin is not None else cid))
        )
        claims.setdefault(cid, owner)
        for a in (c.get("aliases") or []):
            if a:
                claims.setdefault(a, owner)
        if isinstance(dn, str) and dn:
            claims.setdefault(dn, owner)
        if cur is not None:
            back = set(c.get("aliases") or []) - {cid}
            if isinstance(dn, str) and dn and dn not in (cid, cur.get("display_name")):
                back.add(dn)
            if back:
                cur["aliases"] = sorted(set(cur.get("aliases") or []) | back)
                _union_alias_platforms(cur, _entry_alias_platforms(c))
            continue
        if cid in skip:
            scalars = _donor_scalars(c)
            if scalars:
                if skip_owner is None:
                    skip_owner = _make_form_owner_lookup(_NONCATALOG_EXISTING_SOURCES)
                fold_owner = skip_owner(cid)
                if fold_owner is not None and fold_owner != cid:
                    out.append({"id": fold_owner, "weak": scalars})
            continue
        if twin is not None:
            if rename_twin:
                fresh_spelling = twin["id"]
                twin["id"] = cid
                stabilized[cid] = fresh_spelling
                by_id[cid] = twin
                back = (set(c.get("aliases") or []) | {fresh_spelling}) - {cid}
            else:
                promoted[cid] = twin["id"]
                back = (set(c.get("aliases") or []) | {cid}) - {twin["id"]}
            if isinstance(dn, str) and dn and dn not in (twin["id"], twin.get("display_name")):
                back.add(dn)
            twin["aliases"] = sorted(set(twin.get("aliases") or []) | back)
            _union_alias_platforms(twin, _entry_alias_platforms(c))
            absorbed += 1
            continue
        rc = dict(c)
        if "display_name" in rc:  # full entry, not an alias-only enrich record
            meta = _entry_meta(rc)
            meta["upstream_status"] = "removed"
            rc["metadata"] = json.dumps(meta, sort_keys=True)
            retained += 1  # enrich records are carried too but not counted here
        out.append(rc)
    if stabilized or promoted:
        renamed = {old: new for new, old in stabilized.items()}
        renamed.update(promoted)
        # A promote-then-rename chain must compose to the final survivor, or
        # edges and claims land on a renamed-away id.
        for old in list(renamed):
            seen = {old}
            target = renamed[old]
            while target in renamed and target not in seen:
                seen.add(target)
                target = renamed[target]
            renamed[old] = target
        for e in out:
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and edge.get("id") in renamed:
                    edge["id"] = renamed[edge["id"]]
        # Claim values must name surviving ids (reconcile keys on them).
        for k, v in list(claims.items()):
            if v in renamed:
                claims[k] = renamed[v]
        if promoted:
            # A non-regenerated source still referencing a demoted id fails
            # the dedup gate; this names the ids for the cron log.
            print(
                "[refresh] carry-forward: promoted higher-rung fresh id(s): "
                + ", ".join(
                    f"{old} -> {renamed[old]}" for old in sorted(promoted)
                ),
                file=sys.stderr,
            )
    if stabilized:
        print(
            f"[refresh] carry-forward: kept {len(stabilized)} committed id(s) "
            f"over a respelled fresh twin (stability rule)",
            file=sys.stderr,
        )
    if not adopt_migration:
        print(
            f"[refresh] stability rule: {pending_rung2} pending rung-2 id "
            f"promotion(s) (OpenRouter key over invented id) awaiting the next "
            f"--adopt-openrouter-ids-migration run",
            file=sys.stderr,
        )
    if retained:
        print(
            f"[refresh] carry-forward: retained {retained} committed entr(ies) "
            f"absent from today's upstream",
            file=sys.stderr,
        )
    if absorbed:
        print(
            f"[refresh] carry-forward: absorbed {absorbed} respelled committed "
            f"id(s) onto today's {'fresh spelling' if adopt_migration else 'surviving entry'}",
            file=sys.stderr,
        )
    return out, claims


def _iter_unskipped_source_entries(
    sources: tuple[Path, ...],
) -> Iterable[tuple[Path, dict]]:
    """Yield active source records, excluding generated ids core.yaml skips:
    a skipped record treated as an owner would key an enrichment by an entity
    the loader discards. core.yaml's own definitions always count.
    """
    skipped = _load_core_skip_ids()
    for path in sources:
        for entry in _catalog_load_list(path):
            cid = entry.get("id") if isinstance(entry, dict) else None
            if path != CORE_PATH and cid in skipped:
                continue
            yield path, entry


def _build_existing_index(
    sources: tuple[Path, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """(existing_exact, existing_norm) maps {form -> owning canonical id} over
    every id / display_name / alias in `sources`; the normalized map uses the
    resolver's `normalize`, like the seed validator's steal-guard. First
    writer wins on a duplicate form."""
    from eval_entity_resolver.normalization import normalize as _norm

    existing_exact: dict[str, str] = {}
    existing_norm: dict[str, str] = {}

    def _add_exact(form: str, cid: str) -> None:
        if form:
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)

    for _path, e in _iter_unskipped_source_entries(sources):
        cid = e.get("id")
        if not cid:
            continue
        _add_exact(cid, cid)
        dn = e.get("display_name")
        if dn:
            _add_exact(dn, cid)
        for a in (e.get("aliases") or []):
            _add_exact(a, cid)
    return existing_exact, existing_norm


def _make_steal_guard(
    existing_exact: dict[str, str], existing_norm: dict[str, str]
) -> Callable[[str, str], bool]:
    """`_steals(form, cid)`: True iff `form` (exact or normalized) is owned by
    a different canonical id. Must match regenerate_catalog's inner `_steals`."""
    from eval_entity_resolver.normalization import normalize as _norm

    def _steals(form: str, cid: str) -> bool:
        owner = existing_exact.get(form)
        if owner is not None and owner != cid:
            return True
        nowner = existing_norm.get(_norm(form))
        return nowner is not None and nowner != cid

    return _steals


def reconcile_generated_against_existing(
    entries: list[dict],
    sources: tuple[Path, ...] = _NONCATALOG_EXISTING_SOURCES,
    committed_claims: dict[str, str] | None = None,
) -> list[dict]:
    """Core-aware reconciliation for the non-catalog write path: a mint
    colliding (exact or normalized) with an existing canonical under another
    id is dropped, its forms move onto the owner as an enrich record, and
    parent edges are repointed. `committed_claims` pre-seeds form hygiene so
    a new mint never steals a carried-forward alias. Returns survivors +
    enrich records, id-sorted.
    """
    from eval_entity_resolver.fold import build_hf_index, decide_fold
    from eval_entity_resolver.normalization import normalize as _norm

    existing_exact, existing_norm = _build_existing_index(sources)
    _steals = _make_steal_guard(existing_exact, existing_norm)

    # Only a display_name-bearing record defines a canonical; an alias-only
    # enrich record must not suppress the batch's own definition of that id,
    # which would leave it defined nowhere.
    defining_ids: set[str] = set()
    for _path, e in _iter_unskipped_source_entries(sources):
        if e.get("id") and "display_name" in e:
            defining_ids.add(e["id"])
    # Suppression owners must be defined somewhere (derived sources and this
    # batch included); suppressing onto an enrich-only id makes a bare row.
    owner_defining: set[str] = set(defining_ids)
    for _path, e in _iter_unskipped_source_entries(
        (CATALOG_OUT_PATH, TIER3_PATH)
    ):
        if e.get("id") and "display_name" in e:
            owner_defining.add(e["id"])
    for e in entries:
        if isinstance(e, dict) and e.get("id") and "display_name" in e:
            owner_defining.add(e["id"])

    def _owner_of(form: str) -> str | None:
        return existing_exact.get(form) or existing_norm.get(_norm(form))

    # Broader index for donation + hygiene only: the derived catalog / tier3
    # sources are excluded from the partition steal-guard (it must not dedup
    # against derived output) but still say which forms already belong to
    # whom.
    _hy_exact, _hy_norm = _build_existing_index(sources + (CATALOG_OUT_PATH, TIER3_PATH))

    def _owner_broad(form: str) -> str | None:
        return _hy_exact.get(form) or _hy_norm.get(_norm(form))

    # Claims by defined ids only: a dead enrich record's self-claim must not
    # keep it alive.
    _def_exact: dict[str, str] = {}
    _def_norm: dict[str, str] = {}
    for _path, e2 in _iter_unskipped_source_entries(
        sources + (CATALOG_OUT_PATH, TIER3_PATH)
    ):
        cid2 = e2.get("id")
        if not cid2 or cid2 not in owner_defining:
            continue
        for form2 in [cid2, e2.get("display_name"), *(e2.get("aliases") or [])]:
            if form2:
                _def_exact.setdefault(form2, cid2)
                _def_norm.setdefault(_norm(form2), cid2)
    for e2 in entries:
        cid2 = e2.get("id") if isinstance(e2, dict) else None
        if not cid2 or "display_name" not in e2:
            continue
        for form2 in [cid2, e2.get("display_name"), *(e2.get("aliases") or [])]:
            if form2:
                _def_exact.setdefault(form2, cid2)
                _def_norm.setdefault(_norm(form2), cid2)

    def _defined_owner_of(form: str) -> str | None:
        return _def_exact.get(form) or _def_norm.get(_norm(form))

    # Org-aware fold (the resolver's decide_fold, so both agree on sameness):
    # a mint naming a real HF repo under a different, org-decoupled id
    # (`alibaba/qwen-2-5-14b-instruct` -> `Qwen/Qwen2.5-14B-Instruct`) defers.
    existing_entries = [
        entry for _path, entry in _iter_unskipped_source_entries(sources)
    ]
    hf_to_dev = _hf_to_dev()
    _hf_ids, _alias_to_hf, _by_org_name, _ = build_hf_index(
        existing_entries, hf_to_dev, _oracle_fixed_ids()
    )

    def _fold_target(e: dict) -> str | None:
        if e.get("id") in _non_lineage_bases():
            return None
        f = decide_fold(e, _hf_ids, _alias_to_hf, _by_org_name, hf_to_dev)
        if f is None:
            return None
        tgt = f["hf_target"]
        return tgt if tgt and tgt != e.get("id") else None

    # Partition: suppressed when the id steals an existing canonical or
    # org-aware-folds to a real HF repo.
    survivors: list[dict] = []
    suppressed: list[dict] = []
    suppressed_owner: dict[str, str] = {}
    for e in entries:
        cid = e.get("id")
        owner: str | None = None
        if cid and _steals(cid, cid):
            o = _owner_of(cid)
            if o and o != cid and o in owner_defining:
                owner = o
        # An exact re-emit of an existing canonical enriches it rather than
        # surviving as a duplicate, and never goes through decide_fold, whose
        # fuzzy tier strips variant markers and would drag a distinct
        # `-Instruct` canonical onto its base.
        if owner is None and cid and cid in defining_ids:
            suppressed.append(e)
            suppressed_owner[cid] = cid
            continue
        if owner is None and cid:
            owner = _fold_target(e)
        if owner and owner != cid:
            suppressed.append(e)
            suppressed_owner[cid] = owner
            continue
        # An enrich record keyed by an id no source defines would survive as a
        # bare canonical: repoint it onto the defined entity carrying the id as
        # an alias, else drop loudly so the oracle gates catch a real loss.
        if cid and "display_name" not in e and cid not in owner_defining:
            ob = _defined_owner_of(cid)
            if ob is not None and ob != cid:
                suppressed.append(e)
                suppressed_owner[cid] = ob
                continue
            print(
                f"[refresh] WARNING: dropping orphaned enrich record {cid!r} "
                f"(owner vanished from every source); forms dropped: "
                f"{[cid, *(e.get('aliases') or [])]}",
                file=sys.stderr,
            )
            continue
        survivors.append(e)

    # No early return on empty `suppressed`: the survivor hygiene below must
    # run regardless.

    # Donors iterate in id order so the first donor wins per scalar field.
    sibling_ids = {e["id"] for e in survivors}
    _committed = committed_claims or {}
    enrich_aliases: dict[str, set[str]] = defaultdict(set)
    enrich_scalars: dict[str, dict] = defaultdict(dict)
    for e in sorted(suppressed, key=lambda x: x["id"]):
        cid = e["id"]
        owner = suppressed_owner[cid]
        # A cid owned by a different canonical is a distinct model wrongly
        # folded here; donating its id or scalars would merge two models.
        # Ownership evidence is tiered: an independent-source claim
        # (`_owner_of`) vetoes, except a stale claim by a non-defined id the
        # carry-forward absorbed onto `owner`. A derived-layer claim
        # (`_owner_broad`) by cid itself, or by an id committed_claims assigns
        # to this owner, is the catalog's echo of the mint being suppressed
        # and must not block the donation; any other derived claim still does.
        ind_cid = _owner_of(cid)
        cid_owner = _owner_broad(cid)
        foreign_cid = (
            ind_cid is not None
            and ind_cid != owner
            and (ind_cid in owner_defining or _committed.get(ind_cid) != owner)
        ) or (
            cid_owner is not None
            and cid_owner not in (owner, cid)
            and _committed.get(cid_owner) != owner
        )
        if not _steals(cid, owner) or cid != owner:
            # A mint id that normalizes to the owner's is already supplied by
            # the owner; donating it would double-claim the form.
            if _norm(cid) != _norm(owner) and not foreign_cid:
                enrich_aliases[owner].add(cid)
        if not foreign_cid:
            for k, v in _donor_scalars(e).items():
                enrich_scalars[owner].setdefault(k, v)
        # display_name is a resolvable form (a raw value can normalized-match
        # it), so it is donated too. Same tiered evidence as `foreign_cid`; a
        # sibling survivor's id or a form committed to a third entity also
        # vetoes.
        for a in [e.get("display_name"), *(e.get("aliases") or [])]:
            if not a or a == owner:
                continue
            other_ind = _owner_of(a)
            if other_ind is not None and other_ind != owner and (
                other_ind in owner_defining or _committed.get(other_ind) != owner
            ):
                continue
            other = _owner_broad(a)
            if other is not None and other not in (owner, cid) and (
                _committed.get(other) != owner
            ):
                continue
            if a in sibling_ids:
                continue
            co = _committed.get(a)
            if co is not None and co not in (owner, cid):
                continue
            enrich_aliases[owner].add(a)

    for e in survivors:
        parents = e.get("parents")
        if not isinstance(parents, list):
            continue
        for edge in parents:
            if isinstance(edge, dict) and edge.get("id") in suppressed_owner:
                edge["id"] = suppressed_owner[edge["id"]]

    # Survivor form hygiene: a survivor may still carry an alias/display_name
    # owned by an existing canonical or by a sibling survivor (this path
    # rewrites the file wholesale, so a base aliasing its own-batch `-Instruct`
    # sibling double-claims it, which the seed rejects). An entry's own id
    # wins; a sibling's id beats any alias; a shared non-id form goes to the
    # first claimant by id, pre-seeded with the committed claims so a new mint
    # cannot steal a carried-forward form.
    claimed: dict[str, str] = dict(committed_claims or {})
    for e in sorted(survivors, key=lambda x: x["id"]):
        cid = e["id"]

        def _foreign(form: str) -> bool:
            # Independent-source claim: a veto, unless by a non-defined id the
            # carry-forward absorbed onto cid.
            o_ind = _owner_of(form)
            if o_ind is not None and o_ind != cid and (
                o_ind in owner_defining or claimed.get(o_ind) != cid
            ):
                return True
            # Derived-layer claim: when the carry-forward assigned the form or
            # the claiming id to this survivor, it is the previous cron's echo
            # of the absorbed id; stripping on it would delete the forms from
            # every source (`veo3-1`-class forms exist only as explicit
            # aliases, since normalize does not equate letter-digit twins).
            o = _owner_broad(form)
            if o is not None and o != cid and (
                claimed.get(o) != cid and claimed.get(form) != cid
            ):
                return True
            if form in sibling_ids and form != cid:
                return True
            c = claimed.get(form)
            return c is not None and c != cid

        kept: list[str] = []
        for a in (e.get("aliases") or []):
            if not a or a == cid or _foreign(a):
                continue
            claimed.setdefault(a, cid)
            kept.append(a)
        e["aliases"] = sorted(set(kept))
        ap = e.get("alias_platforms")
        if isinstance(ap, dict):
            e["alias_platforms"] = {k: v for k, v in ap.items() if k != cid and not _foreign(k)}
        # Persisted `metadata.alias_platforms` keys must stay a subset of
        # `aliases`, or the next run's carry-forward drops them and the output
        # is not idempotent.
        meta_ap = _entry_alias_platforms(e)
        if meta_ap:
            alias_set = set(e["aliases"])
            kept_ap = {k: v for k, v in meta_ap.items() if k in alias_set}
            if kept_ap != meta_ap:
                meta = _entry_meta(e)
                if kept_ap:
                    meta["alias_platforms"] = dict(sorted(kept_ap.items()))
                else:
                    meta.pop("alias_platforms", None)
                e["metadata"] = json.dumps(meta, sort_keys=True)
        dn = e.get("display_name")
        if isinstance(dn, str) and _foreign(dn):
            cand = cid.split("/", 1)[-1]
            e["display_name"] = cid if _foreign(cand) else cand
            # Declared now so one pass reaches the fixpoint the carry-forward
            # would otherwise take two runs to reach.
            if e["display_name"] != cid and e["display_name"] not in (e.get("aliases") or []):
                e["aliases"] = sorted({*(e.get("aliases") or []), e["display_name"]})

    enrich_records: list[dict] = []
    for owner in sorted(set(enrich_aliases) | set(enrich_scalars)):
        rec: dict = {"id": owner}
        if enrich_aliases.get(owner):
            rec["aliases"] = sorted(enrich_aliases[owner])
        _merge_weak(rec, enrich_scalars.get(owner) or {})
        if len(rec) > 1:
            enrich_records.append(rec)
    out = survivors + enrich_records

    # Consolidate records sharing an id, or each regen carries one more
    # duplicate enrich record forever.
    by_id: dict[str, dict] = {}
    consolidated: list[dict] = []
    for e in out:
        cid = e.get("id")
        if not cid:
            consolidated.append(e)
            continue
        cur = by_id.get(cid)
        if cur is None:
            by_id[cid] = e
            consolidated.append(e)
            continue
        if "display_name" in e and "display_name" not in cur:
            e, cur = cur, e
            idx = consolidated.index(e)
            consolidated[idx] = cur
            by_id[cid] = cur
        if e.get("aliases"):
            cur["aliases"] = sorted(set(cur.get("aliases") or []) | set(e["aliases"]))
        if "display_name" not in cur:
            # A full entry never absorbs weak scalars.
            _merge_weak(cur, e.get("weak") or {})
    return sorted(consolidated, key=lambda e: e.get("id") or "")

_CATALOG_HEADER = """# AUTO-GENERATED by scripts/refresh_from_modelsdev.py (catalog split) — DO NOT HAND-EDIT.
# models.dev full-catalog seed. Two record kinds:
#   * fresh canonical mints: models.dev-only (not-on-HF) models — closed-API
#     families (Claude/GPT/Gemini/Grok) + the re-host/community tail. No HF
#     collision, so HF source-of-truth is not violated.
#   * alias-only enrichments {id, aliases}: a models.dev model that IS HF-present
#     (already a canonical). The existing HF-cased canonical wins; only the
#     provider-spelling aliases (carrying inference_platform in
#     metadata.alias_platforms) union onto it, plus the donor's scalars under
#     `weak:` (the seed loader fills only still-empty fields with those). No
#     duplicate canonical is minted.
# The re-cased seed/models/sources/models_dev.generated.yaml is left intact;
# this file is purely additive. Regenerated by the daily refresh-models cron.
"""


def _catalog_load_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    d = safe_load_yaml(path.read_text())
    if isinstance(d, dict):
        d = d.get("entries", [])
    return d or []


def regenerate_catalog(full: list[dict], adopt_migration: bool = False) -> None:
    """Split the finalized full catalog (`full`) against the existing canonical
    universe, write models_dev_catalog.generated.yaml, and reconcile HF-derived
    community orgs into orgs.generated.yaml. HF wins; only models.dev-only
    models are minted."""
    from eval_entity_resolver.normalization import normalize as _norm

    existing_form_to_cid: dict[str, str] = {}
    existing_exact: dict[str, str] = {}
    existing_norm: dict[str, str] = {}

    def _add_form(form: str, cid: str) -> None:
        if form:
            existing_form_to_cid.setdefault(form.lower(), cid)

    def _add_exact(form: str, cid: str) -> None:
        if form:
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)

    # Only defined canonicals may own forms: a catalog record keyed by a
    # renamed-away enrich id would materialize a bare canonical.
    defined_anywhere: set[str] = set()
    active_existing = list(
        _iter_unskipped_source_entries(_CATALOG_EXISTING_SOURCES)
    )
    for _path, e in active_existing:
        if e.get("id") and "display_name" in e:
            defined_anywhere.add(e["id"])

    for _path, e in active_existing:
        cid = e.get("id")
        if not cid or cid not in defined_anywhere:
            continue
        _add_form(cid, cid)
        _add_exact(cid, cid)
        dn = e.get("display_name")
        if dn:
            _add_exact(dn, cid)
        for a in (e.get("aliases") or []):
            _add_form(a, cid)
            _add_exact(a, cid)

    def _steals(form: str, cid: str) -> bool:
        owner = existing_exact.get(form)
        if owner is not None and owner != cid:
            return True
        nowner = existing_norm.get(_norm(form))
        return nowner is not None and nowner != cid

    # Same org-aware fold as reconcile_generated_against_existing, so both
    # paths agree: a mint naming an HF repo under an org-decoupled id
    # (`zai/GLM-5` -> `zai-org/GLM-5`) enriches it instead of shadowing it.
    from eval_entity_resolver.fold import build_hf_index as _bhi, decide_fold as _df

    _existing_entries = [e for _path, e in active_existing]
    _hf_to_dev_map = _hf_to_dev()
    _chf_ids, _calias, _cby_org, _ = _bhi(_existing_entries, _hf_to_dev_map, _oracle_fixed_ids())

    def _catalog_fold_target(e: dict) -> str | None:
        f = _df(e, _chf_ids, _calias, _cby_org, _hf_to_dev_map)
        return f["hf_target"] if f and f["hf_target"] != e.get("id") else None

    fresh: list[dict] = []
    enrich: list[dict] = []
    # One enrich record per owner; every donor consolidates onto it so the
    # written catalog never carries duplicates.
    enrich_by_id: dict[str, dict] = {}
    fresh_seen_lc: dict[str, dict] = {}
    fresh_seen_twin: dict[str, dict] = {}
    fresh_form_owner: dict[str, str] = {}

    # The committed catalog's claims (display_names included: the loader
    # promotes them to global aliases) pre-seed fresh_form_owner so a fresh
    # split cannot steal a committed form.
    committed = _catalog_load_list(CATALOG_OUT_PATH)
    core_skip = _load_core_skip_ids()
    for c in committed:
        ccid = c.get("id")
        if not ccid or ccid in core_skip:
            continue
        # A committed enrich record keyed by an id no source defines is dead;
        # its claims must not block the forms' real owners.
        if "display_name" not in c and ccid not in defined_anywhere:
            continue
        fresh_form_owner.setdefault(ccid, ccid)
        for a in (c.get("aliases") or []):
            if a:
                fresh_form_owner.setdefault(a, ccid)
        cdn = c.get("display_name")
        if isinstance(cdn, str) and cdn:
            fresh_form_owner.setdefault(cdn, ccid)

    def _enrich_target(
        cid: str,
        aliases: list[str],
        ap: dict | None,
        donor_id: str | None = None,
        scalars: dict | None = None,
    ) -> None:
        # Forms pre-claimed by `donor_id` (a committed record being donated
        # onto `cid`) are donatable.
        ok = (None, cid) if donor_id is None else (None, cid, donor_id)
        keep = sorted({
            a for a in aliases
            if a and a != cid and not _steals(a, cid)
            and fresh_form_owner.get(a) in ok
        })
        for a in keep:
            fresh_form_owner[a] = cid
        ap2: dict = {}
        if ap:
            ap2 = {
                k: v for k, v in ap.items()
                if k != cid and not _steals(k, cid)
                and fresh_form_owner.get(k) in ok
            }
        rec = enrich_by_id.get(cid)
        if rec is None:
            rec = {"id": cid}
            if keep:
                rec["aliases"] = keep
            if ap2:
                rec["metadata"] = json.dumps({"alias_platforms": ap2}, sort_keys=True)
            _merge_weak(rec, scalars or {})
            if len(rec) > 1:
                enrich.append(rec)
                enrich_by_id[cid] = rec
            return
        if keep:
            rec["aliases"] = sorted(set(rec.get("aliases") or []) | set(keep))
        if ap2:
            meta = _entry_meta(rec)
            cur_ap = meta.get("alias_platforms")
            cur_ap = cur_ap if isinstance(cur_ap, dict) else {}
            meta["alias_platforms"] = dict(sorted({**ap2, **cur_ap}.items()))
            rec["metadata"] = json.dumps(meta, sort_keys=True)
        _merge_weak(rec, scalars or {})

    def _forms_of(e: dict) -> list[str]:
        forms = [e["id"]]
        if e.get("display_name"):
            forms.append(e["display_name"])
        forms.extend(a for a in (e.get("aliases") or []) if a)
        return forms

    for e in full:
        cid = e["id"]
        cid_low = cid.lower()
        meta = json.loads(e.get("metadata") or "{}")
        ap = meta.get("alias_platforms") or {}

        cased = existing_form_to_cid.get(cid_low)
        if cased is not None:
            _enrich_target(cased, e.get("aliases", []), ap, scalars=_donor_scalars(e))
            continue
        owner_exact = (
            existing_exact.get(cid)
            or existing_norm.get(_norm(cid))
            or (existing_exact.get(e.get("display_name")) if e.get("display_name") else None)
        )
        if owner_exact is not None:
            _enrich_target(owner_exact, [cid] + list(e.get("aliases", [])), ap, scalars=_donor_scalars(e))
            continue
        fold_tgt = _catalog_fold_target(e)
        if fold_tgt is not None:
            _enrich_target(fold_tgt, [cid] + list(e.get("aliases", [])), ap, scalars=_donor_scalars(e))
            continue
        if cid_low in fresh_seen_lc:
            prior = fresh_seen_lc[cid_low]
            cur = set(prior.get("aliases", []))
            for a in (a for a in e.get("aliases", []) if a):
                if a == prior["id"] or existing_exact.get(a) is not None:
                    continue
                owner = fresh_form_owner.get(a)
                if owner is not None and owner != prior["id"]:
                    continue
                cur.add(a)
                fresh_form_owner[a] = prior["id"]
            prior["aliases"] = sorted(cur)
            continue
        peer = fresh_form_owner.get(cid)
        if peer is not None and peer != cid:
            continue

        clean: list[str] = []
        for a in e.get("aliases", []):
            if not a or a == cid or _steals(a, cid):
                continue
            owner = fresh_form_owner.get(a)
            if owner is not None and owner != cid:
                continue
            clean.append(a)
        e["aliases"] = sorted(set(clean))
        dn = e.get("display_name")

        def _claimed(form: str) -> bool:
            return existing_exact.get(form) is not None or (
                fresh_form_owner.get(form) not in (None, cid))

        if dn and _claimed(dn):
            cand = cid.split("/", 1)[-1]
            e["display_name"] = cand if not _claimed(cand) else cid
            # Declared now so one pass reaches the carry-forward's fixpoint.
            if e["display_name"] != cid and e["display_name"] not in e["aliases"]:
                e["aliases"] = sorted({*e["aliases"], e["display_name"]})
        for form in _forms_of(e):
            fresh_form_owner.setdefault(form, cid)
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)
        fresh_seen_lc[cid_low] = e
        fresh_seen_twin.setdefault(_twin_key(cid), e)
        for k in _extended_twin_keys(cid):
            fresh_seen_twin.setdefault(k, e)
        if ap:
            ap2 = {k: v for k, v in ap.items() if k in e["aliases"]}
            if ap2:
                meta["alias_platforms"] = ap2
            else:
                meta.pop("alias_platforms", None)
            e["metadata"] = json.dumps(meta, sort_keys=True)
        fresh.append(e)

    # Carry the committed catalog forward so an upstream removal never deletes
    # a resolvable form: aliases unioned back, vanished ids re-emitted against
    # their canonical or donated onto the one that absorbed them, else
    # retained tagged `metadata.upstream_status: removed`.
    def _committed_forms(c: dict) -> list[str]:
        """Donatable forms: aliases plus display_name (a loader-promoted global
        alias, so dropping it on a fold deletes a resolvable form)."""
        forms = list(c.get("aliases") or [])
        cdn = c.get("display_name")
        if isinstance(cdn, str) and cdn and cdn not in forms:
            forms.append(cdn)
        return forms

    def _union_back(rec: dict, c: dict, donor_id: str | None = None) -> None:
        # `donor_id`: c's committed id when it differs from rec's; it and the
        # forms it pre-claimed become aliases.
        rcid = rec["id"]
        ok = (None, rcid) if donor_id is None else (None, rcid, donor_id)
        cur = set(rec.get("aliases") or [])
        forms = _committed_forms(c) + ([donor_id] if donor_id else [])
        for a in forms:
            if not a or a in (rcid, rec.get("display_name")) or a in cur or _steals(a, rcid):
                continue
            if fresh_form_owner.get(a) not in ok:
                continue
            cur.add(a)
            fresh_form_owner[a] = rcid
        # No `aliases: []` on a weak-only record: the fresh emission omits the
        # key and the carry-forward must round-trip byte-identically.
        if cur or "aliases" in rec:
            rec["aliases"] = sorted(cur)
        _union_alias_platforms(rec, _entry_alias_platforms(c))
        # A full mint never absorbs weak scalars.
        if "display_name" not in rec:
            _merge_weak(rec, _donor_scalars(c))

    retained = 0
    stabilized_cat: dict[str, str] = {}  # committed id kept -> fresh spelling renamed away
    promoted_cat: dict[str, str] = {}    # committed id absorbed -> higher-rung fresh id
    pending_rung2_cat = 0                # OpenRouter-key twins deferred by the stability rule
    for c in committed:
        ccid = c.get("id")
        if not ccid:
            continue
        if ccid in core_skip:
            # Same rule as _carry_forward_committed: scalars still donate
            # weakly onto a claiming owner; aliases stay suppressed.
            scalars = _donor_scalars(c)
            if scalars:
                sk_owner = existing_exact.get(ccid) or existing_norm.get(_norm(ccid))
                if sk_owner is not None and sk_owner != ccid:
                    _enrich_target(sk_owner, [], None, scalars=scalars)
            continue
        prior = fresh_seen_lc.get(ccid.lower())
        if prior is None and "display_name" in c:
            # Respelled twin, full committed mints only.
            t = fresh_seen_twin.get(_twin_key(ccid)) or next(
                (fresh_seen_twin[k] for k in sorted(_extended_twin_keys(ccid))
                 if k in fresh_seen_twin),
                None,
            )
            if t is not None and _bsizes(ccid) == _bsizes(t["id"]):
                prior = t
        if prior is not None:
            # Stability rule, rung-monotone, as in _carry_forward_committed
            # (a case-only respell counts: committed casing wins).
            promote = (
                prior["id"] != ccid
                and _entry_meta(prior).get("hf_deferred") is True
                and _entry_meta(c).get("hf_deferred") is not True
            )
            if (
                prior["id"] != ccid
                and not adopt_migration
                and not promote
                and _entry_meta(prior).get("openrouter_adopted") is True
                and _entry_meta(c).get("openrouter_adopted") is not True
                and _entry_meta(c).get("hf_deferred") is not True
            ):
                pending_rung2_cat += 1
            if (
                prior["id"] != ccid
                and not adopt_migration
                and not promote
                and "display_name" in prior
                and prior["id"] not in stabilized_cat
            ):
                old = prior["id"]
                prior["id"] = ccid
                stabilized_cat[ccid] = old
                fresh_seen_lc[ccid.lower()] = prior
                fresh_seen_twin.setdefault(_twin_key(ccid), prior)
                for _m in (fresh_form_owner, existing_exact):
                    for k, v in list(_m.items()):
                        if v == old:
                            _m[k] = ccid
                for k, v in list(existing_norm.items()):
                    if v == old:
                        existing_norm[k] = ccid
                prior["aliases"] = sorted(set(prior.get("aliases") or []) | {old})
                fresh_form_owner[old] = ccid
                _union_back(prior, c, donor_id=old)
                continue
            if prior["id"] != ccid:
                promoted_cat[ccid] = prior["id"]
            _union_back(prior, c, donor_id=ccid if prior["id"] != ccid else None)
            continue
        target = enrich_by_id.get(ccid)
        if target is not None:
            _union_back(target, c)
            continue
        owner = existing_exact.get(ccid) or existing_norm.get(_norm(ccid))
        if owner == ccid:
            _enrich_target(ccid, _committed_forms(c), _entry_alias_platforms(c), scalars=_donor_scalars(c))
            continue
        if owner is not None:
            _enrich_target(owner, [ccid, *_committed_forms(c)], _entry_alias_platforms(c), donor_id=ccid, scalars=_donor_scalars(c))
            continue
        if "display_name" in c:
            rc = dict(c)
            meta = _entry_meta(rc)
            meta["upstream_status"] = "removed"
            rc["metadata"] = json.dumps(meta, sort_keys=True)
            fresh.append(rc)
            fresh_seen_lc[ccid.lower()] = rc
            fresh_seen_twin.setdefault(_twin_key(ccid), rc)
            for k in _extended_twin_keys(ccid):
                fresh_seen_twin.setdefault(k, rc)
            for form in _forms_of(rc):
                fresh_form_owner.setdefault(form, ccid)
            retained += 1
    if stabilized_cat or promoted_cat:
        renamed = {old: new for new, old in stabilized_cat.items()}
        renamed.update(promoted_cat)
        # A promote-then-rename chain must compose to the final survivor.
        for old in list(renamed):
            seen = {old}
            target = renamed[old]
            while target in renamed and target not in seen:
                seen.add(target)
                target = renamed[target]
            renamed[old] = target
        for e in fresh:
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and edge.get("id") in renamed:
                    edge["id"] = renamed[edge["id"]]
        if promoted_cat:
            print(
                "[refresh] catalog carry-forward: promoted higher-rung fresh "
                "id(s): "
                + ", ".join(
                    f"{old} -> {renamed[old]}" for old in sorted(promoted_cat)
                ),
                file=sys.stderr,
            )
    if stabilized_cat:
        print(
            f"[refresh] catalog carry-forward: kept {len(stabilized_cat)} committed "
            f"id(s) over a respelled fresh twin (stability rule)",
            file=sys.stderr,
        )
    if not adopt_migration:
        print(
            f"[refresh] catalog stability rule: {pending_rung2_cat} pending "
            f"rung-2 id promotion(s) (OpenRouter key over invented id) awaiting "
            f"the next --adopt-openrouter-ids-migration run",
            file=sys.stderr,
        )
    if retained:
        print(
            f"[refresh] catalog carry-forward: retained {retained} committed "
            f"mint(s) absent from today's upstream",
            file=sys.stderr,
        )

    # Snap each fresh mint's org_id to the canonical casing before `missing`
    # is computed, or a lowercased arrival (`sao10k`) mints a twin of the
    # existing `Sao10K` org via the case-sensitive set-difference below.
    _canon_org = _build_org_canonicalizer()
    for e in fresh:
        for field in ("org_id", "lineage_origin_model_org_id"):
            if e.get(field):
                e[field] = _canon_org(e[field])

    # Stable key order regardless of which donor reached the owner first.
    out_entries = fresh + [
        {k: r[k] for k in ("id", "aliases", "metadata", "weak") if k in r}
        for r in enrich
    ]
    body = yaml.safe_dump(out_entries, sort_keys=False, allow_unicode=True, width=200)
    CATALOG_OUT_PATH.write_text(_CATALOG_HEADER + "\n" + body)

    # --- Org reconciliation (two-tier rule) --------------------------------
    curated_org_ids = {e["id"] for e in _catalog_load_list(ORGS_SEED_PATH) if "id" in e}
    gen_orgs = _catalog_load_list(ORGS_GENERATED_PATH)
    gen_org_ids = {e["id"] for e in gen_orgs if "id" in e}
    referenced = {e.get("org_id") for e in fresh if e.get("org_id")}
    missing = sorted(referenced - curated_org_ids - gen_org_ids)
    if missing:
        for oid in missing:
            gen_orgs.append({
                "id": oid, "display_name": oid, "hf_org": oid,
                "kind": "community", "tags": "[]", "metadata": "{}",
                "review_status": "reviewed",
            })
        gen_header = (
            ORGS_GENERATED_PATH.read_text().split("\n- ", 1)[0].rstrip()
            if ORGS_GENERATED_PATH.exists() else ""
        )
        if not gen_header.startswith("#"):
            gen_header = "# AUTO-GENERATED — HF-derived community orgs."
        ORGS_GENERATED_PATH.write_text(
            gen_header + "\n"
            + yaml.safe_dump(gen_orgs, sort_keys=False, allow_unicode=True, width=200)
        )
        print(f"[refresh] catalog: reconciled {len(missing)} missing community org(s): {missing}", file=sys.stderr)
    print(
        f"[refresh] catalog: {len(fresh)} fresh mint(s) (not-on-HF), "
        f"{len(enrich)} alias-only enrichment(s) (HF-present) -> {CATALOG_OUT_PATH}",
        file=sys.stderr,
    )


# Model sources whose org_ids must have a canonical_orgs row.
_ALL_MODEL_SOURCES = (
    HF_ORACLE_PATH, SEED_PATH, HUB_STATS_PATH, CATALOG_OUT_PATH, TIER3_PATH, CORE_PATH,
)
# Sources whose org_ids carry HF-true casing.
_HF_TRUE_CASING_SOURCES = (HF_ORACLE_PATH, HUB_STATS_PATH)
ORGS_DISTINCT_ALLOWLIST_PATH = REPO_ROOT / "seed" / "orgs_distinct_allowlist.yaml"


def _load_distinct_org_allowlist() -> set[str]:
    if not ORGS_DISTINCT_ALLOWLIST_PATH.exists():
        return set()
    return {
        x
        for x in (safe_load_yaml(ORGS_DISTINCT_ALLOWLIST_PATH.read_text()) or [])
        if isinstance(x, str)
    }


def _write_source_entries(path: Path, entries: list[dict]) -> None:
    """Rewrite a generated/core source file's entries, preserving its `# header`
    and the `{skip_ids,...,entries}` dict shape where present."""
    text = path.read_text() if path.exists() else ""
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    doc = safe_load_yaml(text) if text else None
    if isinstance(doc, dict) and "entries" in doc:
        out = {**doc, "entries": entries}
    else:
        out = entries
    body = yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200)
    path.write_text((header + "\n" if header else "") + body)


def _build_org_canonicalizer() -> Callable[[str | None], str | None]:
    """`_canon(org_spelling) -> canonical_org_id`: curated developer id, else
    HF-true community casing, else verbatim, honoring the distinct-org
    allowlist. Shared by `canonicalize_model_org_ids` and `regenerate_catalog`
    so casing never oscillates between runs; real-HF sources are the primary
    casing authority, all sources plus existing org rows the fallback."""
    from eval_entity_resolver.fold import build_curated_org_map, build_community_casing, canonicalize_org

    curated_map = build_curated_org_map(_catalog_load_list(ORGS_SEED_PATH))
    distinct = _load_distinct_org_allowlist()

    primary = build_community_casing([
        e.get("org_id") for p in _HF_TRUE_CASING_SOURCES for e in _catalog_load_list(p)
        if isinstance(e.get("org_id"), str)
    ])
    fallback = build_community_casing(
        [e.get("org_id") for p in _ALL_MODEL_SOURCES for e in _catalog_load_list(p)
         if isinstance(e.get("org_id"), str)]
        + [e["id"] for p in (ORGS_SEED_PATH, ORGS_GENERATED_PATH)
           for e in _catalog_load_list(p) if isinstance(e.get("id"), str)]
    )
    community = {**fallback, **primary}

    def _canon(org):
        if not isinstance(org, str) or not org:
            return org
        return canonicalize_org(org, curated_map, community, distinct)

    return _canon


def canonicalize_model_org_ids(write_core: bool = False) -> int:
    """Canonicalize every model's org_id + lineage_origin_model_org_id across
    all sources to one spelling per developer; returns the fields rewritten.
    `write_core=False` never rewrites core.yaml (the YAML re-dump would hoist
    its inline comments), reporting a needed change for manual fix instead."""
    _canon = _build_org_canonicalizer()

    rewritten = 0
    for path in _ALL_MODEL_SOURCES:
        entries = _catalog_load_list(path)
        if not entries:
            continue
        core_readonly = path == CORE_PATH and not write_core
        changed = False
        for e in entries:
            if not isinstance(e, dict):
                continue
            for field in ("org_id", "lineage_origin_model_org_id"):
                old = e.get(field)
                new = _canon(old)
                if new != old:
                    if core_readonly:
                        print(
                            f"::warning::core.yaml {e.get('id')}: {field} {old!r} "
                            f"should be {new!r} — fix core.yaml by hand (the cron "
                            f"never rewrites the curated file)"
                        )
                        continue
                    e[field] = new
                    rewritten += 1
                    changed = True
        if changed:
            _write_source_entries(path, entries)
    print(f"[refresh] org-canonicalize: rewrote {rewritten} org field(s) to the "
          f"canonical developer spelling", file=sys.stderr)
    return rewritten


def reconcile_all_orgs(write_core: bool = False) -> None:
    """Mint a community canonical_orgs row for every org_id referenced by any
    model source (per-generator reconciliation covers only each generator's
    own mints) that lacks one. Runs last in the regen. Never deletes or
    renames a curated row; an org a curated lab already claims (id / hf_org /
    alias) is never minted as a community twin."""
    # Canonicalize first so the rows minted below are over canonical org_ids.
    canonicalize_model_org_ids(write_core=write_core)

    curated_claims: set[str] = set()
    for e in _catalog_load_list(ORGS_SEED_PATH):
        for form in (e.get("id"), e.get("hf_org"), *(e.get("aliases") or [])):
            if isinstance(form, str) and form:
                curated_claims.add(form.lower())
    gen_orgs = _catalog_load_list(ORGS_GENERATED_PATH)
    gen_org_ids = {e["id"] for e in gen_orgs if isinstance(e.get("id"), str)}

    referenced: set[str] = set()
    for path in _ALL_MODEL_SOURCES:
        for e in _catalog_load_list(path):
            oid = e.get("org_id")
            if isinstance(oid, str) and oid:
                referenced.add(oid)

    # Prune generated org rows no model references whose case/separator key
    # matches a different, referenced org: leftovers of a prior casing (a
    # distinct uploader would still be referenced by its own models).
    from eval_entity_resolver.fold import _norm_org_key
    distinct = _load_distinct_org_allowlist()
    ref_keys = {_norm_org_key(o): o for o in referenced}
    pruned: list[str] = []
    kept_orgs = []
    for e in gen_orgs:
        oid = e.get("id")
        if (isinstance(oid, str) and oid and oid not in referenced
                and oid not in distinct
                and _norm_org_key(oid) in ref_keys
                and ref_keys[_norm_org_key(oid)] != oid):
            pruned.append(oid)
            continue
        kept_orgs.append(e)
    gen_orgs = kept_orgs
    gen_org_ids = {e["id"] for e in gen_orgs if isinstance(e.get("id"), str)}

    missing = sorted(
        oid for oid in referenced
        if oid not in gen_org_ids and oid.lower() not in curated_claims
    )
    for oid in missing:
        gen_orgs.append({
            "id": oid, "display_name": oid, "hf_org": oid,
            "kind": "community", "tags": "[]", "metadata": "{}",
            "review_status": "reviewed",
        })
    if not missing and not pruned:
        print("[refresh] org-reconcile: no dangling org_ids", file=sys.stderr)
        return
    gen_header = (
        ORGS_GENERATED_PATH.read_text().split("\n- ", 1)[0].rstrip()
        if ORGS_GENERATED_PATH.exists() else ""
    )
    if not gen_header.startswith("#"):
        gen_header = "# AUTO-GENERATED — HF-derived community orgs."
    ORGS_GENERATED_PATH.write_text(
        gen_header + "\n" + yaml.safe_dump(gen_orgs, sort_keys=False, allow_unicode=True, width=200)
    )
    print(f"[refresh] org-reconcile: minted {len(missing)} community org(s), pruned "
          f"{len(pruned)} stale twin(s) (e.g. mint {missing[:4]}, prune {pruned[:4]})", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true", help="use cached /tmp/modelsdev_api.json")
    p.add_argument("--dry-run", action="store_true", help=f"print diff vs current {SEED_PATH}; don't write")
    p.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="write to this PREVIEW path instead of the committed generated YAML "
        "(for inspection; leaves the committed file untouched)",
    )
    p.add_argument(
        "--catalog",
        action="store_true",
        help="ONLY (re)generate the models_dev_catalog.generated.yaml "
        "split (de-orphan) + reconcile orgs.generated.yaml; does NOT rewrite "
        "models_dev.generated.yaml. The cron runs this as a second step after "
        "the source write so the catalog splits against the settled re-cased "
        "models_dev.",
    )
    p.add_argument(
        "--reconcile-orgs",
        action="store_true",
        help="ONLY run the universal org reconcile: mint a community canonical_orgs "
        "row for every org_id referenced by ANY model source (incl. tier3) that "
        "lacks one. Run as the LAST regen step (after tier3) — does not fetch or "
        "rewrite model sources. Never WRITES core.yaml (reads it for authority; "
        "warns when a core org_id spelling needs a manual fix).",
    )
    p.add_argument(
        "--adopt-openrouter-ids-migration",
        action="store_true",
        help="ONE-SHOT migration switch for the OpenRouter id adoption "
        "(specs/model-id-resolution PLAN.md G3): on a committed/fresh twin "
        "match the FRESH externally-adopted id wins and the committed id "
        "becomes an alias. Without it the stability rule holds — the "
        "committed id wins — so the daily cron can never thrash ids on "
        "OpenRouter listing changes.",
    )
    p.add_argument(
        "--reconcile-orgs-write-core",
        action="store_true",
        help="like --reconcile-orgs, but ALSO rewrites core.yaml org_id "
        "spellings (a full YAML re-dump that hoists core's inline comments). "
        "Manual/one-shot use only (regenerate_sources.sh) — the cron must "
        "never write the curated file.",
    )
    args = p.parse_args()

    if args.reconcile_orgs or args.reconcile_orgs_write_core:
        reconcile_all_orgs(write_core=args.reconcile_orgs_write_core)
        return 0

    api = _fetch(use_cache=args.no_fetch)
    known_orgs = _load_known_org_ids()
    if not known_orgs:
        print(f"[refresh] ERROR: {ORGS_SEED_PATH} not found or empty. Seed orgs first.", file=sys.stderr)
        return 1

    if args.catalog:
        if args.preview_out is not None:
            p.error(
                "--preview-out is not supported with --catalog: the catalog "
                "split always writes the committed catalog + orgs files"
            )
        generated, skipped_no_org = _generate_models(api, known_orgs)
        if skipped_no_org:
            print(f"[refresh] ERROR: {len(skipped_no_org)} provider(s) -> unknown org_id", file=sys.stderr)
            return 1
        regenerate_catalog(
            _tag_openrouter_key_ids(_finalize_entries(generated), api),
            adopt_migration=args.adopt_openrouter_ids_migration,
        )
        return 0

    generated, skipped_no_org = _generate_models(api, known_orgs)
    if skipped_no_org:
        print(
            f"[refresh] ERROR: {len(skipped_no_org)} provider(s) mapped to unknown org_id "
            f"(must exist in {ORGS_SEED_PATH}):",
            file=sys.stderr,
        )
        for entry in skipped_no_org:
            print(f"  - {entry}", file=sys.stderr)
        print(
            "[refresh] Add the missing orgs to seed/orgs.yaml or fix the "
            "PROVIDER_TO_ORG mapping in this script, then re-run.",
            file=sys.stderr,
        )
        return 1
    # Finalize before reconciliation: reconcile reads only `aliases`, so
    # provider spellings still buried in `alias_platforms` would leak past it
    # and surface on the wrong canonical when _write_yaml flattens them.
    generated = _tag_openrouter_key_ids(_finalize_entries(generated), api)
    generated, committed_claims = _carry_forward_committed(
        generated, _catalog_load_list(SEED_PATH),
        adopt_migration=args.adopt_openrouter_ids_migration,
    )
    before = len(generated)
    generated = reconcile_generated_against_existing(
        generated, committed_claims=committed_claims
    )
    if len(generated) != before:
        print(
            f"[refresh] reconciliation: suppressed {before - len(generated)} "
            f"mint(s) colliding (normalized) with an existing canonical",
            file=sys.stderr,
        )
    # Without this a lowercased arrival (`sao10k`) would split identity
    # against the existing `Sao10K` org row.
    _canon_org = _build_org_canonicalizer()
    for e in generated:
        for field in ("org_id", "lineage_origin_model_org_id"):
            if e.get(field):
                e[field] = _canon_org(e[field])
    out_path = args.preview_out or SEED_PATH
    new_text = _write_yaml(generated, out_path)

    if args.preview_out is not None and not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_text)
        print(
            f"[refresh] PREVIEW: wrote {len(generated)} model entries to {out_path} "
            f"(committed {SEED_PATH} untouched)",
            file=sys.stderr,
        )
        return 0

    if args.dry_run:
        if SEED_PATH.exists():
            old = SEED_PATH.read_text()
            if old == new_text:
                print("[refresh] no changes")
            else:
                import difflib
                diff = difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=str(SEED_PATH),
                    tofile=f"{SEED_PATH} (generated)",
                )
                sys.stdout.writelines(diff)
        else:
            print(new_text)
        return 0

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_PATH.write_text(new_text)
    deferred_ids = {
        e["id"] for e in generated
        if "display_name" in e and _entry_meta(e).get("hf_deferred") is True
    }
    index_promoted = deferred_ids - _oracle_fixed_ids()
    print(
        f"[refresh] wrote {len(generated)} model entries to {SEED_PATH} "
        f"({len(deferred_ids)} hf-deferred, {len(index_promoted)} HF-promoted "
        f"via hub_stats_index)",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
