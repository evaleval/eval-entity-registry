"""Wraps the eval-entity-resolver package: runs the resolver, auto-creates
draft canonicals on no_match, writes an alias per resolution, and appends
to the resolution log."""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import replace as _dc_replace
from datetime import datetime, timezone
from typing import Optional

from eval_entity_resolver import AliasStore, CanonicalStore, Resolver, ResolverConfig, ResolutionResult
from eval_entity_resolver.display import humanize_model_slug

from eval_entity_registry.config import settings
from eval_entity_registry.store.hf_store import RegistryStore
from eval_entity_registry.store import queries


# Base-family tokens (mirrors scripts/generate_tier3_inferred_seed.py). Only
# detects a candidate base; an edge is emitted only when it alias-confirms.
_TIER3_BASE_TOKENS = (
    "tinyllama", "openhermes", "openchat", "mixtral",
    "llama", "mistral", "qwen", "gemma", "yi", "phi", "gpt", "claude",
    "gemini", "falcon", "bloom", "deepseek", "baichuan", "cohere", "command",
    "neural", "solar", "nous", "zephyr", "orca", "dolphin",
)
# An id like `unknown/foo` has no extractable org.
_PLACEHOLDER_ORG_PREFIXES = {"unknown", "none", "null", "model", "models", "local"}


_ENTITY_TABLE = {
    "model": "canonical_models",
    "benchmark": "canonical_benchmarks",
    "metric": "canonical_metrics",
    "harness": "eval_harnesses",
    "org": "canonical_orgs",
}

def _slugify(value: str) -> str:
    """Lowercase slug for auto-created entity ids; UUID-derived when the input
    reduces to nothing (e.g. all punctuation)."""
    slug = value.lower().strip()
    slug = re.sub(r"[^\w\s\-/]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        slug = f"auto-{str(uuid.uuid4())[:8]}"
    return slug


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_with_pending(registry_store: RegistryStore, name: str) -> "pd.DataFrame":
    """Table DataFrame with pending-buffer rows appended: auto-created drafts
    are buffered until `flush_pending`, and without the overlay the
    resolver's `CanonicalStore` snapshot cannot see them. Concat cannot
    duplicate keys because `upsert_entity` routes existing ids to in-place
    update and only new ids to pending."""
    import pandas as pd
    base_df = registry_store.table(name) if registry_store.has_table(name) else pd.DataFrame()
    pending = getattr(registry_store, "_pending", {}).get(name, [])
    if not pending:
        return base_df
    pending_df = pd.DataFrame(pending)
    if base_df.empty:
        return pending_df
    return pd.concat([base_df, pending_df], ignore_index=True)


def _build_alias_store(registry_store: RegistryStore) -> AliasStore:
    """Build an AliasStore from the registry's in-memory aliases table."""
    aliases_df = registry_store.table("aliases")
    return AliasStore(aliases_df, read_only=True)


def _build_canonical_store(registry_store: RegistryStore) -> CanonicalStore:
    """CanonicalStore over the in-memory canonical tables, pending rows
    overlaid (see `_table_with_pending`). families_df/composites_df are
    needed for benchmark `family_key` / `category`."""
    return CanonicalStore(
        models_df=_table_with_pending(registry_store, "canonical_models"),
        benchmarks_df=_table_with_pending(registry_store, "canonical_benchmarks"),
        metrics_df=_table_with_pending(registry_store, "canonical_metrics"),
        harnesses_df=_table_with_pending(registry_store, "eval_harnesses"),
        orgs_df=_table_with_pending(registry_store, "canonical_orgs") if registry_store.has_table("canonical_orgs") else None,
        families_df=registry_store.table("canonical_families") if registry_store.has_table("canonical_families") else None,
        composites_df=registry_store.table("canonical_composites") if registry_store.has_table("canonical_composites") else None,
    )


_RESPONSE_FIELDS = (
    "canonical_id", "strategy", "confidence", "review_status",
    "parent_canonical_id", "resolved_leaf_id", "root_model_id",
    "lineage_origin_org_id",
    # `root_model_id` / `lineage_origin_org_id` are deprecated aliases of
    # `model_group_id` / `lineage_origin_model_org_id`, kept for older consumers.
    "model_group_id", "model_family_id", "lineage_origin_model_id",
    "lineage_origin_model_org_id", "inference_platform",
    "resolution_source", "resolution_granularity",
    "parents", "open_weights",
    "release_date", "params_billions",
    "family_key", "composite_keys", "category",
    # The HTTP route projects its lean shape from these.
    "ancestry", "resolution_detail",
)


def _result_to_dict(result: ResolutionResult, *, created_new: bool) -> dict:
    """Service-contract dict from a `ResolutionResult`, plus `created_new`
    (service state the resolver does not track)."""
    return {field: getattr(result, field) for field in _RESPONSE_FIELDS} | {
        "created_new": created_new,
    }


def _no_match_result() -> dict:
    """No-match dict with the full response key set."""
    return {field: None for field in _RESPONSE_FIELDS} | {
        "strategy": "no_match",
        "confidence": 0.0,
        "created_new": False,
    }


class ResolutionService:
    def __init__(self, registry_store: RegistryStore) -> None:
        import threading
        self.store = registry_store
        self._resolver: Optional[Resolver] = None
        # Duplicate strings ("Accuracy" appears in every record) skip the chain.
        self._resolve_cache: dict[tuple[str, str, Optional[str]], dict] = {}
        # Hub-stats indices snapshot the aliases / orgs tables; cleared by
        # `invalidate_resolver()` so later lookups see just-added canonicals.
        # The lock guards the lazy build under FastAPI's threadpool.
        self._hub_stats_client = None
        self._hub_stats_indices: Optional[tuple[dict[str, str], dict[str, str]]] = None
        self._hub_stats_indices_lock = threading.Lock()
        # normalized HF id -> HF-true id from `hub_stats_index`; feeds the
        # HfIdVerifier. Invalidated alongside the hub-stats indices.
        self._hf_id_index: Optional[dict[str, str]] = None
        # Outlives resolver invalidations so its result cache and breaker
        # state survive entity churn.
        from eval_entity_registry.services.hf_id_verifier import HfIdVerifier
        self._hf_id_verifier = HfIdVerifier(index_provider=self._build_hf_id_index)

    def _get_resolver(self) -> Resolver:
        if self._resolver is None:
            alias_store = _build_alias_store(self.store)
            canonical_store = _build_canonical_store(self.store)
            config = ResolverConfig(threshold=settings.resolver_auto_merge_threshold)
            self._resolver = Resolver(
                alias_store, config, canonical_store=canonical_store,
                hf_id_checker=self._hf_id_verifier.check,
            )
        return self._resolver

    def invalidate_resolver(self) -> None:
        """Force a resolver rebuild after alias or entity changes. Also clears
        the hub-stats indices so a child seen later in the same run can
        resolve its `baseModels` parent to a just-created draft."""
        self._resolver = None
        with self._hub_stats_indices_lock:
            self._hub_stats_indices = None
            self._hf_id_index = None

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        source_field: Optional[str],
        sync_run_id: Optional[str] = None,
        rerun: bool = False,
        mode: str = "resolve",
    ) -> dict:
        """Resolve a raw value to a canonical entity.

        Returns `_RESPONSE_FIELDS` plus `created_new`; every outcome
        (match, auto-draft, no_match) emits the same key set. `mode="exact"`
        disables fuzzy inference and every side effect on entity data (no
        drafts, no alias writes). Field semantics: `api/schemas.py`."""
        if not raw_value or not raw_value.strip():
            return _no_match_result()

        cache_key = (raw_value, entity_type, source_config, mode)
        if not rerun and cache_key in self._resolve_cache:
            return self._resolve_cache[cache_key]

        # Read-only: no side effects. The HF id check runs inside the chain,
        # so a real-but-unminted HF id resolves to itself without minting.
        if settings.read_only:
            resolver = self._get_resolver()
            result: ResolutionResult = resolver.resolve(
                raw_value, entity_type, source_config, mode=mode
            )
            result_dict = (
                _result_to_dict(result, created_new=False)
                if result.canonical_id is not None else _no_match_result()
            )
            self._memoize(cache_key, result_dict)
            return result_dict

        # An existing alias is not served from the alias row: it carries only
        # the canonical_id, and `build_result` from an id alone cannot recover
        # leaf identity. Re-run the chain (an O(1) exact hit) and overlay the
        # stored strategy/confidence so callers still see the original audit.
        prefetched: Optional[ResolutionResult] = None
        if not rerun:
            existing = queries.get_alias(self.store, raw_value, entity_type, source_config)
            if existing:
                resolver = self._get_resolver()
                fresh = resolver.resolve(raw_value, entity_type, source_config, mode=mode)
                checker_backed = (
                    fresh.hf_attestation is not None
                    and fresh.canonical_id == fresh.hf_attestation
                )
                if fresh.canonical_id == existing["canonical_id"]:
                    # A checker-backed result keeps its runtime attestation;
                    # stale stored audit fields must not overwrite it.
                    enriched = fresh if checker_backed else _dc_replace(
                        fresh,
                        strategy=existing["strategy"],
                        confidence=existing["confidence"],
                    )
                elif (
                    checker_backed
                    and fresh.canonical_id.lower() == existing["canonical_id"].lower()
                ):
                    # Same repo (HF ids are case-insensitively unique): keep
                    # the registered row, no flag, no mint.
                    enriched = resolver.build_result(
                        raw_value, entity_type, source_config,
                        existing["canonical_id"], fresh.strategy, fresh.confidence,
                    )
                elif checker_backed:
                    if mode == "exact":
                        # Exact mode is side-effect-free: no flag, no mint.
                        enriched = fresh
                    else:
                        # The attested HF id wins; the alias is flagged for
                        # review, never silently repointed. Fall through so an
                        # unregistered attested id still gets minted.
                        self._flag_alias_hf_disagreement(existing, fresh.hf_attestation)
                        prefetched = fresh
                else:
                    # The canonical moved since the alias was written; the
                    # alias row is the source of truth, accept the leaf clobber.
                    enriched = resolver.build_result(
                        raw_value, entity_type, source_config,
                        existing["canonical_id"], existing["strategy"], existing["confidence"],
                    )
                if prefetched is None:
                    result_dict = _result_to_dict(enriched, created_new=False)
                    self._memoize(cache_key, result_dict)
                    return result_dict

        resolver = self._get_resolver()
        result = (
            prefetched if prefetched is not None
            else resolver.resolve(raw_value, entity_type, source_config, mode=mode)
        )

        # Exact mode: no drafts, org mints, alias writes or log rows, on a
        # match or a miss.
        if mode == "exact":
            result_dict = (
                _result_to_dict(result, created_new=False)
                if result.canonical_id is not None else _no_match_result()
            )
            self._memoize(cache_key, result_dict)
            return result_dict

        created_new = False
        alias_status = "auto"

        if result.canonical_id is not None and result.hf_attested_unregistered:
            # HF-attested but unregistered. A case-variant canonical (e.g. a
            # lowercased draft minted before the index knew the repo) is the
            # same repo: reuse it rather than mint a shadow duplicate.
            case_variant = self._find_case_variant_model(result.canonical_id)
            if case_variant is not None:
                canonical_id = case_variant
                result = resolver.build_result(
                    raw_value, entity_type, source_config,
                    case_variant, result.strategy, result.confidence,
                )
            else:
                # Mint under the checker's HF-true id, never a slugified
                # re-derivation, so no alias or eval_results row dangles.
                canonical_id = self._auto_create_entity(
                    entity_type, raw_value, preset_hf_id=result.canonical_id
                )
                created_new = True
        elif result.canonical_id is not None:
            canonical_id = result.canonical_id
            alias_status = "auto"
        else:
            canonical_id = self._auto_create_entity(entity_type, raw_value)
            alias_status = "uncertain"
            created_new = True

        strategy_used = result.strategy if result.canonical_id else "auto_draft"

        # A version-stripped harness match records its provenance on the
        # alias row so per-version rows stay auditable.
        detail = result.resolution_detail or {}
        strip_note = None
        if isinstance(detail, dict) and detail.get("harness_version_stripped"):
            strip_note = (
                f"harness version stripped: {detail['harness_version_stripped']} "
                f"(bare {detail.get('bare_name')}, {detail.get('bare_tier')} tier)"
            )

        alias_data = {
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": source_field,
            "status": alias_status,
            "strategy": strategy_used,
            "confidence": result.confidence,
            "notes": strip_note,
        }
        if rerun:
            existing_row = queries.get_alias(self.store, raw_value, entity_type, source_config)
            existing_alias_id = existing_row.get("id") if existing_row else None
            if existing_alias_id and (
                existing_row.get("status") == "confirmed"
                and existing_row.get("canonical_id") != canonical_id
            ):
                # A confirmed alias is never auto-repointed or demoted; the
                # disagreement is recorded for review.
                queries.update_alias(
                    self.store,
                    alias_id=existing_alias_id,
                    updates={
                        "notes": (
                            f"rerun disagreement: resolver returned "
                            f"{canonical_id} ({strategy_used})"
                        ),
                    },
                )
            elif existing_alias_id:
                updates = {
                    "canonical_id": canonical_id,
                    "status": alias_status,
                    "strategy": strategy_used,
                    "confidence": result.confidence,
                }
                if strip_note is not None:
                    # Without this the audit trail is lost on the first rerun.
                    updates["notes"] = strip_note
                queries.update_alias(
                    self.store,
                    alias_id=existing_alias_id,
                    updates=updates,
                )
            else:
                try:
                    queries.add_alias(self.store, alias_data, buffered=True)
                except ValueError:
                    pass
        else:
            try:
                queries.add_alias(self.store, alias_data, buffered=True)
            except ValueError:
                pass  # already added earlier in this run

        if sync_run_id:
            queries.append_resolution_log(
                self.store,
                {
                    "sync_run_id": sync_run_id,
                    "raw_value": raw_value,
                    "entity_type": entity_type,
                    "source_config": source_config,
                    "strategy": strategy_used,
                    "confidence": result.confidence,
                    "canonical_id": canonical_id,
                    "created_new": created_new,
                },
            )

        # A new entity changes the alias index and the canonical-store snapshot.
        if created_new:
            self.invalidate_resolver()

        # On a match `result` already carries the leaf and its metadata, and
        # `build_result` from the id alone would lose the leaf. On auto-create
        # the new canonical is the leaf; it may still sit in the pending
        # buffer, so a missed review_status is forced to 'draft'.
        if created_new:
            resolver = self._get_resolver()
            enriched = resolver.build_result(
                raw_value, entity_type, source_config,
                canonical_id, strategy_used, result.confidence,
            )
        else:
            enriched = result
        result_dict = _result_to_dict(enriched, created_new=created_new)
        if created_new and result_dict.get("review_status") is None:
            result_dict["review_status"] = "draft"
        self._memoize(cache_key, result_dict)
        return result_dict

    def _memoize(self, cache_key: tuple, result_dict: dict) -> None:
        """Memoize for duplicate raw values. A no_match for an HF-shaped model
        value is not memoized: the HF id checker may confirm it later
        (negative-TTL expiry, index refresh, breaker close)."""
        raw_value, entity_type = cache_key[0], cache_key[1]
        if (
            result_dict.get("canonical_id") is None
            and entity_type == "model"
            and self._looks_like_hf_id(raw_value)
        ):
            return
        self._resolve_cache[cache_key] = result_dict

    def _find_case_variant_model(self, hf_id: str) -> Optional[str]:
        """Registered canonical model id equal to `hf_id` up to casing, or
        None. Linear scan; only reached on the rare attested-but-unregistered path."""
        df = self.store.table("canonical_models")
        if df is None or df.empty:
            return None
        lower = hf_id.lower()
        for existing_id in df["id"]:
            if isinstance(existing_id, str) and existing_id.lower() == lower:
                return existing_id
        return None

    def _flag_alias_hf_disagreement(self, existing: dict, hf_id: str) -> None:
        """Record an HF-attestation disagreement on the alias without
        repointing it: a non-confirmed row drops to `uncertain` for review,
        a confirmed row keeps its status."""
        alias_id = existing.get("id")
        if not alias_id:
            return
        updates: dict = {"notes": f"hf-id-check disagreement: attested {hf_id}"}
        if existing.get("status") != "confirmed":
            updates["status"] = "uncertain"
        queries.update_alias(self.store, alias_id=alias_id, updates=updates)

    def _auto_create_entity(
        self, entity_type: str, raw_value: str, preset_hf_id: Optional[str] = None,
    ) -> str:
        table = _ENTITY_TABLE[entity_type]

        # Hub-stats runs before id finalisation because it may return the
        # HF-true `hf_id`: an HF-confirmed model is minted in HF casing with
        # resolution_source="hf" / review_status="reviewed", never slugified.
        # Best-effort: `enrichment` is `{}` on miss or error.
        enrichment: dict = {}
        if entity_type == "model" and (preset_hf_id or self._looks_like_hf_id(raw_value)):
            # Provisional id for self-edge suppression only; the real id is
            # finalised from `hf_id` below.
            enrichment = self._lookup_hub_stats(
                preset_hf_id or raw_value, target_canonical=_slugify(raw_value)
            ) or {}
        if preset_hf_id and not (
            isinstance(enrichment.get("hf_id"), str) and enrichment["hf_id"].strip()
        ):
            # The checker already attested existence; adopt the id even when
            # hub-stats misses it.
            enrichment["hf_id"] = preset_hf_id

        hf_confirmed = entity_type == "model" and isinstance(
            enrichment.get("hf_id"), str
        ) and enrichment["hf_id"].strip()

        now = _now()
        if hf_confirmed:
            from eval_entity_registry.services.hub_stats import (
                hf_id_to_canonical_cased,
            )
            hf_id = enrichment["hf_id"]
            candidate_id, _hf_org_id = hf_id_to_canonical_cased(
                hf_id, self._build_hf_to_dev()
            )
            display = hf_id.split("/", 1)[1] if "/" in hf_id else hf_id
        else:
            candidate_id = _slugify(raw_value)
            # Other entity types' raw names are usually already display-ready.
            if entity_type == "model":
                display = humanize_model_slug(raw_value) or raw_value
            else:
                display = raw_value

        df = self.store.table(table)
        if (df["id"] == candidate_id).any():
            candidate_id = f"{candidate_id}-{str(uuid.uuid4())[:8]}"

        base = {
            "id": candidate_id,
            "display_name": display,
            "metadata": "{}",
            "review_status": "reviewed" if hf_confirmed else "draft",
            "created_at": now,
            "updated_at": now,
        }
        if entity_type == "model":
            # HF-confirmed: the org comes from the HF-true repo so it matches
            # the seeded canonical's org; otherwise the raw prefix is resolved.
            if hf_confirmed:
                org_id = _hf_org_id
                self._ensure_hf_org(org_id)
            else:
                org_id = self._resolve_model_org_id(raw_value)
            # A slug-fallback draft is a name-based inference. With no
            # extractable org the org is not guessed: org_id stays None and
            # the draft is tagged `org-unknown` for review.
            tier3 = (entity_type == "model") and not hf_confirmed
            org_unknown = tier3 and not self._has_extractable_org(raw_value)
            if org_unknown:
                org_id = None
            base.update({
                "developer": None,
                "org_id": org_id,
                "family": None,
                "architecture": None,
                "params_billions": None,
                "parents": "[]",
                "model_group_id": None,
                "lineage_origin_model_org_id": None,
                "open_weights": None,
                "tags": '["org-unknown"]' if org_unknown else "[]",
                "resolution_source": "hf" if hf_confirmed else (
                    "inferred" if tier3 else None
                ),
                "resolution_granularity": "variant" if tier3 else None,
            })
            # Enrichment overrides the defaults above; `hf_id` is provenance,
            # not a column.
            for k, v in enrichment.items():
                if k == "hf_id":
                    continue
                if v is not None:
                    base[k] = v
            # Alias-only fallback for when hub-stats misses (stale, disabled,
            # rate-limited); no-op when a version-axis edge already exists.
            if self._looks_like_hf_id(raw_value):
                self._maybe_infer_family_parent(base, raw_value, candidate_id)
            # Lineage is independent of the missing org, so org-less drafts
            # may still link to a base.
            if tier3:
                self._maybe_infer_tier3_base(base, raw_value, candidate_id)
        elif entity_type == "benchmark":
            base.update({"description": None, "dataset_repo": None, "parent_benchmark_id": None, "tags": "[]"})
        elif entity_type == "metric":
            base.update({"score_type": None, "lower_is_better": False, "min_score": None, "max_score": None})
        elif entity_type == "harness":
            base.update({"version": None, "fork_url": None})
        elif entity_type == "org":
            base.update({
                "parent_org_id": None,
                "website": None,
                "logo_url": None,
                "hf_org": None,
                "kind": "unknown",
                "tags": "[]",
            })

        queries.upsert_entity(self.store, table, base, buffered=True)
        return candidate_id

    def _maybe_infer_family_parent(
        self, base: dict, raw_value: str, candidate_id: str,
    ) -> None:
        """Add a `{variant, axis: version}` edge to `base['parents']` when the
        raw value's snapshot shape resolves via the alias index. Independent of
        hub-stats so releases not yet in the parquet still get linked."""
        try:
            existing = json.loads(base.get("parents") or "[]")
        except (ValueError, TypeError):
            existing = []
        if any(
            p.get("relationship") == "variant" and p.get("axis") == "version"
            for p in existing
            if isinstance(p, dict)
        ):
            return
        from eval_entity_registry.services.hub_stats import infer_family_parent_edge
        try:
            aliases_to_canonical, _ = self._build_hub_stats_indices()
        except Exception:
            return
        edge = infer_family_parent_edge(
            raw_value, aliases_to_canonical, target_canonical=candidate_id,
        )
        if edge is None:
            return
        existing.append(edge)
        base["parents"] = json.dumps(existing)

    @staticmethod
    def _has_extractable_org(raw_value: str) -> bool:
        """True iff the raw value has a single `/` with a non-placeholder left side."""
        if not raw_value or raw_value.count("/") != 1:
            return False
        org, name = (p.strip() for p in raw_value.split("/", 1))
        if not org or not name:
            return False
        return org.lower() not in _PLACEHOLDER_ORG_PREFIXES

    def _maybe_infer_tier3_base(
        self, base: dict, raw_value: str, candidate_id: str,
    ) -> None:
        """Add one `{finetune}` edge to `base['parents']` when the raw value
        carries a base-family token and a shorter stem alias-confirms
        (exact/normalized, never fuzzy) to a canonical that is not the draft
        itself. No-op when a parent already exists or nothing confirms; an
        edge is never invented. Mirrors scripts/generate_tier3_inferred_seed.py."""
        try:
            existing = json.loads(base.get("parents") or "[]")
        except (ValueError, TypeError):
            existing = []
        if existing:  # version-axis or hub-stats edge already present
            return

        name = raw_value.split("/", 1)[1] if "/" in raw_value else raw_value
        toks = [t for t in re.split(r"[\s\-_/.:]+", name.lower()) if t]
        tokset = set(toks)
        base_tok = None
        for fam in _TIER3_BASE_TOKENS:
            if fam in tokset or any(
                t.startswith(fam) and t[len(fam):][:1].isdigit() for t in toks
            ):
                base_tok = fam
                break
        if base_tok is None:
            return

        from eval_entity_registry.services.hub_stats import normalize as _nz
        raw_name_nz = _nz(name)
        org_prefix = (
            raw_value.split("/", 1)[0] if self._has_extractable_org(raw_value) else None
        )
        try:
            start = next(
                i for i, t in enumerate(toks)
                if t == base_tok or (t.startswith(base_tok) and t[len(base_tok):][:1].isdigit())
            )
        except StopIteration:
            start = 0
        tail = toks[start:]
        resolver = self._get_resolver()
        seen: set[str] = set()
        for end in range(len(tail), 0, -1):
            stem = "-".join(tail[:end])
            candidates = []
            if org_prefix:
                hf_to_dev = self._build_hf_to_dev()
                dev = hf_to_dev.get(org_prefix.lower(), org_prefix)
                candidates.append(f"{dev}/{stem}")
            candidates.append(stem)
            for cand in candidates:
                if cand in seen or _nz(cand) == _nz(raw_value):
                    continue
                seen.add(cand)
                # Stem probes bypass the HF checker: each would spend live
                # budget and could attest an unregistered repo.
                res = resolver.resolve(cand, "model", check_hf=False)
                hit = res.canonical_id
                if not hit or res.strategy not in ("exact", "normalized"):
                    continue
                # An attested-but-unregistered id would be a dangling parent FK.
                if res.hf_attested_unregistered:
                    continue
                hit_name = hit.split("/", 1)[1] if "/" in hit else hit
                # A "base" that is the draft's own identity is not a parent.
                if _nz(hit) == _nz(raw_value) or _nz(hit_name) == raw_name_nz:
                    continue
                if _nz(hit) == _nz(candidate_id):
                    continue
                existing.append({"id": hit, "relationship": "finetune"})
                base["parents"] = json.dumps(existing)
                return

    @staticmethod
    def _looks_like_hf_id(raw_value: str) -> bool:
        """Single `/` with non-empty parts on both sides; bare names and
        multi-slash paths never trigger a hub-stats lookup."""
        if not raw_value or raw_value.count("/") != 1:
            return False
        org, name = raw_value.split("/", 1)
        return bool(org.strip()) and bool(name.strip())

    def _lookup_hub_stats(
        self, hf_id: str, target_canonical: Optional[str] = None,
    ) -> Optional[dict]:
        """Partial draft dict from a live hub-stats lookup of `hf_id`; None on
        miss or any error. `target_canonical` lets the family-version
        inference suppress a self-edge."""
        if not settings.hub_stats_lookup_enabled:
            return None
        try:
            client = self._get_hub_stats_client()
            row = client.lookup(hf_id)
        except Exception:
            return None
        if row is None:
            return None
        from eval_entity_registry.services import hub_stats as _hs
        try:
            aliases_to_canonical, org_alias_map = self._build_hub_stats_indices()
            return _hs.enrich_draft_from_row(
                row, aliases_to_canonical, org_alias_map,
                target_canonical=target_canonical,
            )
        except Exception:
            return None

    def _get_hub_stats_client(self):
        """Lazily built, reused across lookups."""
        if self._hub_stats_client is None:
            from eval_entity_registry.services.hub_stats import HubStatsClient
            self._hub_stats_client = HubStatsClient()
        return self._hub_stats_client

    def _build_hub_stats_indices(self) -> tuple[dict[str, str], dict[str, str]]:
        """Indices for `enrich_draft_from_row`: normalized alias -> canonical_id
        and normalized HF org -> org_id. Cached until `invalidate_resolver()`;
        lock-guarded against concurrent lazy builds."""
        cached = self._hub_stats_indices
        if cached is not None:
            return cached
        with self._hub_stats_indices_lock:
            if self._hub_stats_indices is not None:
                return self._hub_stats_indices
            from eval_entity_registry.services.hub_stats import normalize as _hsnorm

            aliases_df = self.store.table("aliases")
            models_df = self.store.table("canonical_models")

            a2c: dict[str, str] = {}
            # Canonical ids first: an id outranks another canonical's alias on
            # collision. Must match refresh_from_hub_stats.load_existing_canonical_aliases
            # or the live and offline lineage graphs silently diverge.
            for _, row in models_df.iterrows():
                cid = row.get("id")
                if isinstance(cid, str):
                    a2c.setdefault(_hsnorm(cid), cid)
            for _, row in aliases_df.iterrows():
                if row.get("entity_type") != "model":
                    continue
                raw = row.get("raw_value")
                cid = row.get("canonical_id")
                if isinstance(raw, str) and "/" in raw and isinstance(cid, str):
                    a2c.setdefault(_hsnorm(raw), cid)

            # The full store-backed org map; a reduced id/hf_org-only map drops
            # curated aliases (CohereLabs -> cohere) and diverges from the bulk refresh.
            org_map = self._build_hf_to_dev()

            self._hub_stats_indices = (a2c, org_map)
            return self._hub_stats_indices

    def _build_hf_id_index(self) -> dict[str, str]:
        """`normalized HF id -> HF-true id` from `hub_stats_index` (`{}` when
        absent or empty); lets the resolve path confirm an exact HF id never
        minted into the registry. Cached until `invalidate_resolver`."""
        cached = self._hf_id_index
        if cached is not None:
            return cached
        with self._hub_stats_indices_lock:
            if self._hf_id_index is not None:
                return self._hf_id_index
            from eval_entity_registry.services.hub_stats import normalize as _hsnorm

            index: dict[str, str] = {}
            if not self.store.has_table("hub_stats_index"):
                self._hf_id_index = index
                return index
            df = self.store.table("hub_stats_index")
            if df is None or df.empty:
                self._hf_id_index = index
                return index
            has_norm = "id_norm" in df.columns
            for _, row in df.iterrows():
                hf_id = row.get("id")
                if not isinstance(hf_id, str) or not hf_id.strip():
                    continue
                key = row.get("id_norm") if has_norm else None
                if not isinstance(key, str) or not key:
                    key = _hsnorm(hf_id)
                # Two distinct repos on one id_norm degrade the entry to a
                # verbatim-only dict {lower_id: true_id}.
                existing = index.get(key)
                if existing is None:
                    index[key] = hf_id
                elif isinstance(existing, str):
                    if existing.lower() != hf_id.lower():
                        index[key] = {
                            existing.lower(): existing,
                            hf_id.lower(): hf_id,
                        }
                else:
                    existing.setdefault(hf_id.lower(), hf_id)
            self._hf_id_index = index
            return index

    def _build_hf_to_dev(self) -> dict[str, str]:
        """HF-org-lowercase -> developer slug via the shared store-backed map.
        Org alias rows are included because canonical_orgs has no aliases
        column, so the curated alias tier reaches the live path too."""
        from eval_entity_resolver.fold import build_org_dev_map_from_store

        adf = self.store.table("aliases")
        org_alias_pairs = (
            zip(adf[adf["entity_type"] == "org"]["raw_value"],
                adf[adf["entity_type"] == "org"]["canonical_id"])
            if adf is not None and not adf.empty else []
        )
        return build_org_dev_map_from_store(
            self.store.table("canonical_orgs").to_dict("records"), org_alias_pairs
        )

    def _ensure_hf_org(self, org_id: Optional[str]) -> None:
        """Create a community `canonical_orgs` row for an HF-derived org id so
        the new model's `org_id` FK resolves; no-op in read-only mode or when
        the row exists. Mirrors the `orgs.generated.yaml` shape."""
        if not org_id or settings.read_only:
            return
        orgs_df = self.store.table("canonical_orgs")
        if (orgs_df["id"] == org_id).any():
            return
        now = _now()
        queries.upsert_entity(
            self.store,
            "canonical_orgs",
            {
                "id": org_id,
                "display_name": org_id,
                "parent_org_id": None,
                "website": None,
                "logo_url": None,
                "hf_org": org_id,
                "kind": "community",
                "tags": "[]",
                "metadata": "{}",
                "review_status": "reviewed",
                "created_at": now,
                "updated_at": now,
            },
            buffered=True,
        )

    def _resolve_model_org_id(self, raw_value: str) -> Optional[str]:
        """Canonical org id for an HF-shaped `org/name` raw value. Goes through
        `self.resolve()` so an unknown org gets a draft + alias and the model's
        `org_id` FK holds; read-only mode resolves without creating."""
        if "/" not in raw_value:
            return None
        raw_org = raw_value.split("/", 1)[0].strip()
        if not raw_org:
            return None
        if settings.read_only:
            return self._get_resolver().resolve(raw_org, "org", None).canonical_id
        # source_field=None: an implicit resolve, not a caller request.
        result_dict = self.resolve(raw_org, "org", None, None)
        return result_dict.get("canonical_id")

    def _find_alias_id(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
    ) -> Optional[str]:
        df = self.store.table("aliases")
        mask = (df["raw_value"] == raw_value) & (df["entity_type"] == entity_type)
        if source_config:
            mask = mask & (df["source_config"] == source_config)
        else:
            mask = mask & df["source_config"].isna()
        rows = df[mask]
        return rows.iloc[0]["id"] if not rows.empty else None
