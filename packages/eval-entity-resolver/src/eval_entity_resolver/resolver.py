"""Matches a raw value to a canonical id via the strategy chain (exact →
HF id check → normalized → fuzzy → no_match) and, given a `CanonicalStore`,
enriches the result with the canonical's metadata and hierarchy. The
enriched `ResolutionResult` is the rich in-process shape the HTTP API
projects down from."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from eval_entity_resolver.alias_store import AliasStore
from eval_entity_resolver.canonical_store import CanonicalStore, _hf_repo_id_of
from eval_entity_resolver.eee import _keyword_extract, prepare_eval_name_segments
from eval_entity_resolver.normalization import normalize
from eval_entity_resolver.models import (
    HfIdHit,
    ResolutionResult,
    ResolverConfig,
    looks_like_hf_id,
)
from eval_entity_resolver.strategies.exact import exact_match
from eval_entity_resolver.strategies.normalized import normalized_match
from eval_entity_resolver.strategies.fuzzy import fuzzy_match

# Strictly between exact (1.0) and fuzzy (`_STEM_CONFIDENCE`, 0.90) so the
# tier is recoverable from the confidence alone.
_NORMALIZED_CONFIDENCE = 0.95

# Two-character segments are language codes or version fragments, and the
# registry holds two-letter canonicals (`if`, `mc`, `nq`) they would collide with.
_MIN_BENCHMARK_SEGMENT_LEN = 3
_MIN_METRIC_SEGMENT_LEN = 3

# Trailing version token on a harness string (`lm_eval 0.4.12`,
# `inspect_ai inspect_ai:0.3.80.dev25+g91a50728.d20250331`); the optional
# `<lib>:` prefix is validated against the name in `_strip_harness_version`.
_HARNESS_VERSION_TAIL_RE = re.compile(
    r"^(?P<name>.*\S)\s+"
    r"(?:(?P<lib>[A-Za-z][\w.\-]*):)?"
    r"v?(?P<version>\d+(?:\.\d+)+"
    r"(?:[.\-]?(?:dev|rc|a|b|alpha|beta|post|final)\d*)?"
    r"(?:\+[\w.]+)?)$",
    re.IGNORECASE,
)


def _strip_harness_version(raw_value: str) -> Optional[str]:
    """Bare harness name when `raw_value` ends in a whitespace-separated version
    token, else None. A `<lib>:` prefix on the token is accepted only when `<lib>`
    normalizes equal to the name, so `foo bar:1.0` is not stripped."""
    if not raw_value:
        return None
    m = _HARNESS_VERSION_TAIL_RE.match(raw_value.strip())
    if not m:
        return None
    if m.group("lib") and normalize(m.group("lib")) != normalize(m.group("name")):
        return None
    return m.group("name")


@dataclass
class StructuredBenchmark:
    """One dotted `evaluation_name` resolved against the benchmark vocabulary.
    `benchmark_raw` is the surface form to record downstream (winning segment
    plus any subset segments).

    `subset_is_identity` says whether `subset` names a PART of the benchmark
    (`mmlu.mmlu.anatomy`: `anatomy` is one subject, False) or merely spells the
    benchmark at greater length (`polyglotoxicityprompts.polyglotoxicityprompts.
    small_overall`: the joined spelling is an alias of the benchmark, True).
    `subset` is populated in both cases."""

    canonical_id: str
    benchmark_raw: str
    subset: Optional[str]
    subset_is_identity: bool = False

    @property
    def observation_role(self) -> str:
        """`"whole"` or `"part"`: what this match verified the row observed.

        The third answer, `"unknown"`, is the ABSENCE of a `StructuredBenchmark`
        (a name `prepare_eval_name_segments` declines, e.g. HELM's
        `MMLU All Subjects`). Callers must map `None` to unknown, not to whole:
        an unparsed name is not a verified observation of anything."""
        return "part" if (self.subset and not self.subset_is_identity) else "whole"


@dataclass
class StructuredMetric:
    """One namespaced `metric_config.metric_id` resolved against the metric
    vocabulary. `matched_segment` is the surface form that carried the identity
    (the whole id when a whole-id alias won); `qualifier` is the tail after it,
    which narrows the metric to a scoring variant (`gpqa.accuracy.strict`), or
    None when the tail carries no information."""

    canonical_id: str
    matched_segment: str
    qualifier: Optional[str]


def _is_aggregate_marker(segment: str) -> bool:
    """`overall` / `<group>_overall` mark an aggregation level, not a metric
    variant, so they never become a qualifier."""
    lowered = segment.lower()
    return lowered == "overall" or lowered.endswith("_overall")


class Resolver:
    def __init__(
        self,
        store: AliasStore,
        config: Optional[ResolverConfig] = None,
        canonical_store: Optional[CanonicalStore] = None,
        hf_id_checker: Optional[Callable[[str], Optional[HfIdHit]]] = None,
    ) -> None:
        """Without `canonical_store` only the basic match fields (canonical_id,
        strategy, confidence) are populated. With `hf_id_checker`, HF-shaped
        model raw values are checked against it: a verbatim (case-insensitive)
        confirmation outranks any alias match; a separator-collapse confirmation
        ranks between the normalized-alias and fuzzy steps."""
        self.store = store
        self.config = config or ResolverConfig()
        self.canonical_store = canonical_store
        self.hf_id_checker = hf_id_checker

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        config: Optional[ResolverConfig] = None,
    ) -> "Resolver":
        """Fully-enriching resolver over `<path>/*.parquet` (alias + canonical
        tables)."""
        return cls(
            AliasStore.from_parquet(path),
            config=config,
            canonical_store=CanonicalStore.from_parquet(path),
        )

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        config: Optional[ResolverConfig] = None,
    ) -> "Resolver":
        """Fully-enriching resolver over a HF Dataset repo."""
        return cls(
            AliasStore.from_hf(repo_id),
            config=config,
            canonical_store=CanonicalStore.from_hf(repo_id),
        )

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str] = None,
        mode: str = "resolve",
        check_hf: bool = True,
    ) -> ResolutionResult:
        """`check_hf=False` disables the injected HF id check for this call, so
        internal candidate probing never draws live-lookup budget."""
        canonical_id = exact_match(raw_value, entity_type, source_config, self.store)

        hf_hit = (
            self._maybe_check_hf_id(raw_value, entity_type, canonical_id)
            if check_hf else None
        )
        if hf_hit is not None and hf_hit.verbatim:
            if canonical_id is not None and canonical_id == hf_hit.hf_id:
                result = self._enrich(
                    raw_value, entity_type, source_config, canonical_id, "exact", 1.0
                )
                return self._stamp_hf_attestation(result, hf_hit)
            # A string that IS a real HF repo id resolves to itself, over any
            # alias mapping.
            return self._hf_checker_result(
                raw_value, entity_type, source_config, hf_hit, "exact", 1.0
            )

        if canonical_id is not None:
            result = self._enrich(
                raw_value, entity_type, source_config, canonical_id, "exact", 1.0
            )
            if hf_hit is not None and canonical_id == hf_hit.hf_id:
                result = self._stamp_hf_attestation(result, hf_hit)
            return result

        # A curated normalized alias outranks a separator-collapsed HF hit.
        if _NORMALIZED_CONFIDENCE >= self.config.threshold:
            canonical_id = normalized_match(raw_value, entity_type, self.store, source_config)
            if canonical_id is not None:
                result = self._enrich(
                    raw_value, entity_type, source_config,
                    canonical_id, "normalized", _NORMALIZED_CONFIDENCE,
                )
                if hf_hit is not None and canonical_id == hf_hit.hf_id:
                    result = self._stamp_hf_attestation(result, hf_hit)
                return result

        if hf_hit is not None and not hf_hit.verbatim:
            return self._hf_checker_result(
                raw_value, entity_type, source_config,
                hf_hit, "normalized", _NORMALIZED_CONFIDENCE,
            )

        if mode == "exact":
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=None,
                strategy="no_match",
                confidence=0.0,
            )

        canonical_id, confidence, inferred_platform = fuzzy_match(
            raw_value, entity_type, self.config.threshold, self.store, source_config,
            org_dev_map=self._org_fold_map(), known_orgs=self._known_orgs(),
        )
        if canonical_id is not None:
            result = self._enrich(
                raw_value, entity_type, source_config,
                canonical_id, "fuzzy", confidence,
            )
            # An explicit host token in the raw id (`together/`, `-bedrock`) is
            # the strongest per-run platform fact; absent one, leave the field.
            if inferred_platform is not None:
                result.inference_platform = inferred_platform
            return result

        return self._no_match_or_harness_retry(
            raw_value, entity_type, source_config, mode
        )

    def _no_match_or_harness_retry(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        mode: str,
    ) -> ResolutionResult:
        """The `no_match` exit, with one additive step for harnesses: a versioned
        harness string (`lm_eval 0.4.12`) whose bare name IS a registered harness
        resolves to it.

        Runs only after every strategy missed (so a per-version alias always wins
        first), only for `entity_type == "harness"`, and never in exact mode (the
        strip is an inference step). Reported as `normalized` at 0.95 with
        `resolution_detail = {harness_version_stripped, bare_name, bare_tier}`;
        a harness hit that stripped nothing keeps an empty `resolution_detail`.
        """
        if (
            entity_type == "harness"
            and mode != "exact"
            and _NORMALIZED_CONFIDENCE >= self.config.threshold
        ):
            bare = _strip_harness_version(raw_value)
            if bare is not None:
                tier = "exact"
                canonical = exact_match(bare, "harness", source_config, self.store)
                if canonical is None:
                    tier = "normalized"
                    canonical = normalized_match(
                        bare, "harness", self.store, source_config
                    )
                if canonical is not None:
                    result = self._enrich(
                        raw_value, "harness", source_config,
                        canonical, "normalized", _NORMALIZED_CONFIDENCE,
                    )
                    result.resolution_detail = {
                        "harness_version_stripped": raw_value.strip()[len(bare):].strip(),
                        "bare_name": bare,
                        "bare_tier": tier,
                    }
                    return result
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=None,
            strategy="no_match",
            confidence=0.0,
        )

    def resolve_structured_metric_id(
        self,
        raw_id: Optional[str],
        source_config: Optional[str] = None,
        catch_all_ids: frozenset = frozenset(),
    ) -> Optional[str]:
        """Positionless structural resolution for a namespaced
        `metric_config.metric_id` (`lmarena.elo.overall`, `vals_ai.mgsm.mgsm_de.accuracy`).

        Segment roles vary by adapter (the metric may sit last, in the middle, or
        be absent), so every segment after the adapter namespace is resolved
        against the metric vocabulary and membership decides. Exact and
        normalized tiers only: a fuzzy near-miss on a short token is a different
        metric, not a spelling variant.

        `catch_all_ids` marks no-information buckets (e.g. the generic `score`);
        a catch-all hit counts as no disclosure, so a field name never outranks
        prose naming the real metric.

        Returns the canonical id when the whole id is a registered non-catch-all
        alias, else when exactly one distinct non-catch-all metric is disclosed;
        None otherwise (no hits, only catch-all hits, or conflicting hits).
        """
        match = self.resolve_structured_metric(raw_id, source_config, catch_all_ids)
        return match.canonical_id if match is not None else None

    def resolve_structured_metric(
        self,
        raw_id: Optional[str],
        source_config: Optional[str] = None,
        catch_all_ids: frozenset = frozenset(),
    ) -> Optional["StructuredMetric"]:
        """`resolve_structured_metric_id` with the match span reported.

        `qualifier` is the tail after the matched segment, which narrows one
        metric to a variant the source reports separately (`gpqa.accuracy.strict`
        vs `gpqa.accuracy.diamond`); pooling on canonical id alone would median
        unrelated numbers. Only aggregate markers (`overall`, `<group>_overall`)
        are excluded from the tail; a numeric segment IS a variant
        (`bench.accuracy.1` vs `.5` are k-shot readings). Segments before the
        match are never part of it. A whole-id alias match reports the whole id
        as `matched_segment` with no qualifier.

        Returns None exactly when `resolve_structured_metric_id` returns None.
        """
        if not raw_id or not isinstance(raw_id, str):
            return None
        raw_id = raw_id.strip()
        if ("." not in raw_id and "/" not in raw_id) or any(c.isspace() for c in raw_id):
            return None
        segments = [s for s in re.split(r"[./]", raw_id) if s]
        if len(segments) < 2:
            return None
        # A whole-id alias outranks any single-segment reading. Exact tier only:
        # `normalize()` deletes dots, so a normalized whole-id lookup could
        # equate distinct ids.
        whole = exact_match(raw_id, "metric", source_config, self.store)
        if whole is not None and whole not in catch_all_ids:
            return StructuredMetric(whole, raw_id, None)
        hits: list[tuple[int, str]] = []
        for index, segment in enumerate(segments[1:], start=1):
            # segments[0] is the adapter namespace
            canonical = exact_match(
                segment, "metric", source_config, self.store
            ) or normalized_match(segment, "metric", self.store, source_config)
            if canonical is not None:
                hits.append((index, canonical))
        specific = {c for _, c in hits if c not in catch_all_ids}
        if len(specific) != 1:
            return None
        canonical_id = next(iter(specific))
        # The LAST segment naming the metric ends the identity: a restatement of
        # the metric is not a qualifier of itself.
        matched_index = max(i for i, c in hits if c == canonical_id)
        tail = [
            s for s in segments[matched_index + 1:] if not _is_aggregate_marker(s)
        ]
        return StructuredMetric(
            canonical_id, segments[matched_index], ".".join(tail) or None
        )

    def resolve_structured_benchmark(
        self,
        raw_name: Optional[str],
        source_config: Optional[str] = None,
    ) -> Optional["StructuredBenchmark"]:
        """Positionless structural resolution for a dotted `evaluation_name`
        (`bbq.bbq.overall`, `MMLU.MMLU-Pro.overall`, `vals_ai.mmlu_pro.biology`).

        Segment roles vary by adapter (the benchmark may sit first, second or
        third), so every segment is probed against the benchmark vocabulary and
        membership decides. Before probing: identical adjacent segments collapse,
        aggregate markers and numeric segments are dropped, a leading segment
        naming the row's `source_config` is dropped, and a trailing metric
        segment is dropped (`bfcl.live.live_accuracy`).

        Each hit is re-tried with its trailing segments attached, shallowest
        first, and the first surface form the registry knows wins. That keeps
        the MOST SPECIFIC reading: a leading namespace loses to the benchmark it
        qualifies (`MMLU.MMLU-Pro` is MMLU-Pro), while a subset that only means
        something under its parent stays with it (`vals_ai.mmlu_pro.math` is
        MMLU-Pro's math subject, not the MATH dataset). Segments after the
        winner are the subset.

        If nothing resolves after dropping the source segment, it is probed like
        any other, so a datastore folder that is itself a registered benchmark
        is the fallback identity for the names under it, subset kept:
        `vals_ai.programbench.strict` under `vals-ai` is
        `("vals-ai", "vals ai programbench strict", "programbench strict")`.
        Registering a folder name as a benchmark therefore claims every
        otherwise-unknown dotted name beneath it as one slice each.

        `subset_is_identity` distinguishes a real subset (the segment hit won)
        from trailing segments that belong to the identity (the joined form
        won). Returns None when no segment resolves.
        """
        segments = prepare_eval_name_segments(raw_name)
        if segments is None:
            return None
        probe_segments = segments
        dropped_source = False
        if (
            len(segments) > 1
            and source_config
            and normalize(segments[0]) == normalize(source_config)
        ):
            probe_segments = segments[1:]
            dropped_source = True
        match = self._probe_benchmark_segments(probe_segments, source_config)
        if match is None and dropped_source:
            match = self._probe_benchmark_segments(
                segments, source_config, alias_tiers_only_at=0
            )
        return match

    def _probe_benchmark_segments(
        self,
        segments: list[str],
        source_config: Optional[str],
        *,
        alias_tiers_only_at: int = -1,
    ) -> Optional["StructuredBenchmark"]:
        """Probe every segment against the benchmark vocabulary and spell the
        winner (see `resolve_structured_benchmark`). The segment at
        `alias_tiers_only_at` (the folder-named segment of the fallback retry)
        takes the exact/normalized tiers only: fuzzy run-configuration stripping
        on a source folder name would be a new inference surface. `-1` gives
        every segment the full chain."""
        if len(segments) > 1 and self._segment_is_metric(segments[-1], source_config):
            segments = segments[:-1]

        hits: list[tuple[int, str]] = []
        for i, segment in enumerate(segments):
            canonical_id: Optional[str] = None
            if len(segment) >= _MIN_BENCHMARK_SEGMENT_LEN:
                if i == alias_tiers_only_at:
                    canonical_id = exact_match(
                        segment, "benchmark", source_config, self.store
                    ) or normalized_match(
                        segment, "benchmark", self.store, source_config
                    )
                else:
                    canonical_id = self.resolve(
                        segment, "benchmark", source_config, check_hf=False
                    ).canonical_id
            if canonical_id is not None:
                hits.append((i, canonical_id))
                continue
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            if not self._is_namespace_segment(segment, nxt, source_config):
                # An unrecognized non-namespace segment ends the identity path:
                # anything deeper is a subset of an unknown benchmark
                # (`vals_ai.programbench.strict`) and must not match on its own.
                break
        if not hits:
            return None

        def _spell(parts: list[str]) -> str:
            return " ".join(p.replace("_", " ") for p in parts)

        for index, canonical in hits:
            benchmark_raw = _spell(segments[index:])
            if index == len(segments) - 1:
                return StructuredBenchmark(canonical, benchmark_raw, None)
            joined = self.resolve(
                benchmark_raw, "benchmark", source_config, check_hf=False
            )
            if joined.canonical_id is not None:
                # The joined spelling wins whenever it resolves (fuzzy included),
                # but the identity flag requires an EXACT alias: only that says
                # the registry spells this surface form, as written, as the
                # benchmark. The normalized tier absorbs every subject name
                # (`mmlu anatomy` → `mmlu`), which would turn each subject row
                # into a whole-benchmark observation.
                claimed = exact_match(
                    benchmark_raw, "benchmark", source_config, self.store
                )
                joined_subset = _spell(segments[index + 1:]) or None
                return StructuredBenchmark(
                    joined.canonical_id, benchmark_raw, joined_subset,
                    subset_is_identity=(
                        claimed is not None
                        or self._subset_names_the_benchmark(
                            joined_subset, joined.canonical_id
                        )
                    ),
                )

        index, canonical = hits[-1]
        subset = _spell(segments[index + 1:]) or None
        return StructuredBenchmark(
            canonical, _spell(segments[index:]), subset,
            subset_is_identity=self._subset_names_the_benchmark(subset, canonical),
        )

    @staticmethod
    def _subset_names_the_benchmark(
        subset: Optional[str],
        canonical: str,
    ) -> bool:
        """True when the reported subset is the benchmark's own name spelled
        differently (`vals_ai.swebench.overall` lands on the `vals-ai` folder and
        reports `swebench` as the subset), so it narrows nothing.

        The test is SELF-SPELLING, not resolution: a benchmark's aliases include
        the names of its PARTS (`conclusion_tasks` is an alias of `legalbench`),
        so "resolves to the same canonical" would call a part aggregate a
        whole-benchmark reading. Exact identity of the joined spelling is
        checked in `_probe_benchmark_segments`; this adds only self-spelling.
        """
        if not subset:
            return False

        def _squash(value: str) -> str:
            # `normalize` keeps a separator as a space (`swe-bench` → `swe
            # bench`) while sources spell `swebench`.
            return "".join(ch for ch in normalize(value) if ch.isalnum())

        return _squash(subset) == _squash(canonical)

    def _is_namespace_segment(
        self, segment: str, next_segment: Optional[str], source_config: Optional[str]
    ) -> bool:
        """A segment that qualifies the benchmark rather than being one, so the
        probe may look past it: a known composite or family, or a stem the next
        segment extends (`alpaca_eval` before `alpaca_eval_v2`)."""
        for entity_type in ("composite", "family"):
            if exact_match(
                segment, entity_type, source_config, self.store
            ) or normalized_match(segment, entity_type, self.store, source_config):
                return True
        return next_segment is not None and normalize(next_segment).startswith(
            normalize(segment) + " "
        )

    def _segment_is_metric(self, segment: str, source_config: Optional[str]) -> bool:
        """True when a trailing segment resolves as a metric or reads as one to
        the metric extractor (`bfcl.live.live_accuracy`), so a metric tail is
        never mistaken for a subset."""
        spelled = segment.replace("_", " ")
        if exact_match(segment, "metric", source_config, self.store):
            return True
        if len(segment) < _MIN_METRIC_SEGMENT_LEN:
            # A two-character tail is ambiguous (`mr` is a Marathi slice as much
            # as the mean-recall alias `mR`): only a byte-exact alias may claim it.
            return False
        if normalized_match(segment, "metric", self.store, source_config):
            return True
        return _keyword_extract(spelled.lower()) is not None

    # ------------------------------------------------------------------
    # HF id check (injected)
    # ------------------------------------------------------------------

    def _maybe_check_hf_id(
        self,
        raw_value: str,
        entity_type: str,
        exact_canonical_id: Optional[str],
    ) -> Optional[HfIdHit]:
        """Run the injected HF id checker only when it can change the outcome.

        Skipped for a byte-equal exact hit whose row is already HF-attested (the
        checker could only agree) or carries an off-HF provenance claim
        (`models_dev`/`NA`/`curated`: a closed-API or OpenRouter id, where a live
        check would 404 on every resolve). A case-only or name-inferred agreement
        still runs it, because the checker is what recovers HF-true casing."""
        if self.hf_id_checker is None or entity_type != "model":
            return None
        if not looks_like_hf_id(raw_value):
            return None
        if exact_canonical_id is not None and exact_canonical_id == raw_value:
            if self._hf_check_skippable(exact_canonical_id):
                return None
        return self.hf_id_checker(raw_value)

    def _hf_check_skippable(self, canonical_id: str) -> bool:
        if self.canonical_store is None:
            return False
        ent = self.canonical_store.lookup("model", canonical_id)
        if ent is None:
            return False
        if _hf_repo_id_of(ent, canonical_id) is not None:
            return True
        src = ent.get("resolution_source")
        return isinstance(src, str) and src in ("models_dev", "NA", "curated")

    def _hf_checker_result(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        hit: HfIdHit,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Result for a winning HF id check hit: full enrichment when the id is
        registered, else a bare result flagged `hf_attested_unregistered` for the
        service layer to auto-create or serve without minting."""
        known = (
            self.canonical_store is not None
            and self.canonical_store.lookup("model", hit.hf_id) is not None
        )
        if known:
            result = self._enrich(
                raw_value, entity_type, source_config, hit.hf_id, strategy, confidence
            )
            return self._stamp_hf_attestation(result, hit)
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=hit.hf_id,
            strategy=strategy,
            confidence=confidence,
            resolution_source=hit.source,
            ancestry=[],
            resolution_detail={"granularity": None, "hf_repo_id": hit.hf_id},
            hf_attestation=hit.hf_id,
            hf_attested_unregistered=True,
        )

    @staticmethod
    def _stamp_hf_attestation(result: ResolutionResult, hit: HfIdHit) -> ResolutionResult:
        """Surface a runtime attestation in `resolution_detail.hf_repo_id` even
        when the row carries no HF provenance; only when the response canonical
        IS the attested id."""
        if result.canonical_id == hit.hf_id:
            result.hf_attestation = hit.hf_id
            if isinstance(result.resolution_detail, dict):
                result.resolution_detail["hf_repo_id"] = hit.hf_id
        return result

    def _org_fold_map(self) -> dict:
        """Org-fold map for the fuzzy org-agreement guard: `org_dev_map` plus the
        org alias rows (the curated alias tier, `AI2` -> `allenai`, is not a
        canonical_orgs column). Alias rows win on collision. Keyed lowercase;
        cached."""
        cached = getattr(self, "_org_fold_map_cache", None)
        if cached is not None:
            return cached
        from eval_entity_resolver.fold import _ORG_ALIASES

        m: dict = dict(self.canonical_store.org_dev_map) if self.canonical_store else dict(_ORG_ALIASES)
        if self.store is not None:
            for raw, cid in self.store.iter_alias_pairs("org"):
                m[raw.lower()] = cid
        self._org_fold_map_cache = m
        return m

    def _known_orgs(self) -> frozenset:
        """canonical_orgs ids. The fuzzy guard never separator-merges two
        prefixes that resolve to DIFFERENT ids in this set, so uploaders kept
        distinct in canonical_orgs stay distinct at resolve time. Cached; empty
        without a canonical_store."""
        cached = getattr(self, "_known_orgs_cache", None)
        if cached is not None:
            return cached
        ids: set[str] = set()
        if self.canonical_store is not None:
            df = self.canonical_store._tables.get("org")
            if df is not None and not df.empty and "id" in df.columns:
                ids = {str(i) for i in df["id"] if isinstance(i, str)}
        self._known_orgs_cache = frozenset(ids)
        return self._known_orgs_cache

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def build_result(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        canonical_id: str,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Enriched `ResolutionResult` for a canonical_id the caller already
        knows (bypassing the strategy chain); same enrichment as `resolve()`."""
        return self._enrich(raw_value, entity_type, source_config, canonical_id, strategy, confidence)

    def _enrich(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        matched_canonical_id: str,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Populate the rich response fields from the matched row; without a
        canonical_store they stay None."""
        if self.canonical_store is None:
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=matched_canonical_id,
                strategy=strategy,
                confidence=confidence,
            )

        cs = self.canonical_store
        matched_entity = cs.lookup(entity_type, matched_canonical_id)
        review_status = (matched_entity or {}).get("review_status") if matched_entity else None

        if entity_type == "model":
            fields = cs.model_metadata_fields(matched_canonical_id, matched_entity)
            # review_status must describe the canonical the response names.
            if fields["canonical_id"] != matched_canonical_id:
                root_entity = cs.lookup("model", fields["canonical_id"])
                if root_entity:
                    review_status = root_entity.get("review_status") or review_status
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=fields["canonical_id"],
                strategy=strategy,
                confidence=confidence,
                review_status=review_status,
                parent_canonical_id=cs.parent_canonical_id("model", matched_entity),
                resolved_leaf_id=fields["resolved_leaf_id"],
                root_model_id=fields["root_model_id"],
                lineage_origin_org_id=fields["lineage_origin_org_id"],
                # `.get`: a store predating these keys must still load.
                model_group_id=fields.get("model_group_id"),
                model_family_id=fields.get("model_family_id"),
                lineage_origin_model_id=fields.get("lineage_origin_model_id"),
                lineage_origin_model_org_id=fields.get("lineage_origin_model_org_id"),
                inference_platform=fields.get("inference_platform"),
                resolution_source=fields.get("resolution_source"),
                resolution_granularity=fields.get("resolution_granularity"),
                parents=fields["parents"],
                open_weights=fields["open_weights"],
                release_date=fields["release_date"],
                params_billions=fields["params_billions"],
                ancestry=cs.compute_ancestry("model", fields["canonical_id"], matched_entity),
                resolution_detail=cs.resolution_detail(
                    "model", fields["canonical_id"], matched_entity=matched_entity
                ),
            )

        if entity_type == "benchmark":
            fam = cs.benchmark_family_enrichment(matched_canonical_id)
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=matched_canonical_id,
                strategy=strategy,
                confidence=confidence,
                review_status=review_status,
                parent_canonical_id=cs.parent_canonical_id(entity_type, matched_entity),
                family_key=fam["family_key"],
                category=fam["category"],
                composite_keys=fam["composite_keys"],
                ancestry=cs.compute_ancestry("benchmark", matched_canonical_id, matched_entity),
                resolution_detail=cs.resolution_detail(
                    "benchmark", matched_canonical_id,
                    raw_value=raw_value, matched_entity=matched_entity,
                ),
            )

        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=matched_canonical_id,
            strategy=strategy,
            confidence=confidence,
            review_status=review_status,
            parent_canonical_id=cs.parent_canonical_id(entity_type, matched_entity),
            ancestry=cs.compute_ancestry(entity_type, matched_canonical_id, matched_entity),
            resolution_detail=cs.resolution_detail(
                entity_type, matched_canonical_id,
                raw_value=raw_value, matched_entity=matched_entity,
            ),
        )
