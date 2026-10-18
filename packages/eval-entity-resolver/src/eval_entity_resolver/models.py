from dataclasses import dataclass
from typing import Literal, Optional


ResolutionStrategy = Literal["exact", "normalized", "fuzzy", "no_match"]
# "exact" stops after the alias and HF-id-check steps and returns no_match
# instead of inferring; callers must not auto-create on it.
ResolveMode = Literal["resolve", "exact"]
# A slice/subset is deliberately NOT a type: it is an alias-fold onto its parent
# benchmark, surfaced through `resolution_detail`.
EntityType = Literal[
    "model", "benchmark", "metric", "harness", "org", "composite", "family"
]


def looks_like_hf_id(raw_value: str) -> bool:
    """A single `/` with non-empty parts on both sides; multi-slash paths are
    deliberately excluded as likely malformed."""
    if not raw_value or raw_value.count("/") != 1:
        return False
    org, name = raw_value.split("/", 1)
    return bool(org.strip()) and bool(name.strip())


@dataclass
class HfIdHit:
    """A confirmed HF repo id from an injected `hf_id_checker`.

    - `hf_id`: the repo id in HF-true casing.
    - `verbatim`: True when the raw value equals `hf_id` case-insensitively
      (HF repo ids are case-insensitively unique, so this identifies the repo
      exactly); False for a separator-collapse match.
    - `source`: `hub_stats_index` or `hf_live`; becomes `resolution_source`
      when the hit wins for an unregistered id.
    """
    hf_id: str
    verbatim: bool
    source: str


@dataclass
class ResolutionResult:
    """Outcome of one `Resolver.resolve` call.

    Core fields (always populated):
      - `raw_value`, `entity_type`, `source_config`: echo of the inputs.
      - `canonical_id`: the matched LEAF canonical (None on no_match); for
        models the group root is in `model_group_id`.
      - `strategy`, `confidence` (0.0–1.0).

    Enrichment fields (None unless the `Resolver` has a `CanonicalStore`):
      - `review_status`: review state of the matched canonical.
      - `parent_canonical_id`: the `variant` edge for models; the `parent_*_id`
        column for benchmarks/orgs.
      - `resolved_leaf_id`: equals `canonical_id`. Models only.
      - `root_model_id`: DEPRECATED alias of `model_group_id`, null when the
        leaf is its own group root; remove once no consumer reads it.
      - `model_group_id`: identity-group root (folds version, quantized and
        mode edges); self for a singleton. Models only.
      - `model_family_id`: family-release root. Models only.
      - `lineage_origin_model_id`: deepest non-variant ancestor (what the
        model was built from). Models only.
      - `lineage_origin_org_id`: DEPRECATED alias of
        `lineage_origin_model_org_id`; remove once no consumer reads it.
      - `lineage_origin_model_org_id`: that ancestor's org_id. Models only.
      - `inference_platform`: FK to inference_platforms.id. Models only.
      - `resolution_source`: {hf|models_dev|curated|inferred|none}.
      - `resolution_granularity`: {variant|group|family}.
      - `parents`: typed-edge list of the matched leaf. Models only.
      - `open_weights`, `release_date` (YYYY-MM or YYYY-MM-DD),
        `params_billions`. Models only.
      - `family_key`: canonical_families.id; the benchmark's own id for a
        singleton family. Benchmarks only.
      - `composite_keys`: canonical_composites.id values where the benchmark
        appears; [] from the resolver (the producer fills it). Benchmarks only.
      - `category`: the family's curated category, or None. Benchmarks only.
    """
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str]
    canonical_id: Optional[str]
    strategy: ResolutionStrategy
    confidence: float
    review_status: Optional[str] = None
    parent_canonical_id: Optional[str] = None
    resolved_leaf_id: Optional[str] = None
    root_model_id: Optional[str] = None
    lineage_origin_org_id: Optional[str] = None
    model_group_id: Optional[str] = None
    model_family_id: Optional[str] = None
    lineage_origin_model_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    inference_platform: Optional[str] = None
    resolution_source: Optional[str] = None
    resolution_granularity: Optional[str] = None
    parents: Optional[list[dict]] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    params_billions: Optional[float] = None
    family_key: Optional[str] = None
    composite_keys: Optional[list[str]] = None
    category: Optional[str] = None
    # Ordered `{canonical_id, level}` from the immediate parent up to the root
    # (`[]` at a root): model -> [group, family]; benchmark -> [family,
    # composite]; family -> [composite]. See `CanonicalStore.compute_ancestry`.
    ancestry: Optional[list[dict]] = None
    # Typed sub-object keyed by entity_type:
    #   model     -> {"granularity": variant|group|family,
    #                 "hf_repo_id": str when runtime-attested, else None
    #                 (null is "not attested", not "not on HF")}
    #   benchmark -> {"level": composite|family|benchmark|slice,
    #                 "matched_subset": str|None}
    #   harness   -> {"harness_version_stripped", "bare_name", "bare_tier"}
    #                 only when a trailing version token was stripped to reach
    #                 the match; {} otherwise
    #   composite|family|metric|org -> {}
    resolution_detail: Optional[dict] = None
    # Repo id the injected HF checker confirmed (HF-true casing), on checker-won
    # results and on alias-won results it agreed with; lets the service layer
    # rank checker-backed results above a stored alias. Never serialized.
    hf_attestation: Optional[str] = None
    # The checker confirmed a repo id that is NOT a registered canonical: the
    # service layer must auto-create it (write mode) and must not emit FK
    # references to it. Never serialized.
    hf_attested_unregistered: bool = False


@dataclass
class ResolverConfig:
    threshold: float = 0.85
