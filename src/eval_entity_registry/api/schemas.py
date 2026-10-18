from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# JSON has no infinity literal: pydantic parses the string form on input and
# main.RegistryJSONResponse renders it on output.
_BOUND_DESCRIPTION = (
    "Metric bound. null = not stated; the string \"Infinity\" / \"-Infinity\" = "
    "unbounded on that side by definition (JSON wire form of an infinite float)."
)


EntityType = Literal[
    "benchmark", "model", "metric", "harness", "org", "composite", "family"
]
ReviewStatus = Literal["draft", "reviewed"]
AliasStatus = Literal["auto", "uncertain", "confirmed", "rejected"]
ParentRelationship = Literal["variant", "finetune", "quantized", "merge", "adapter"]
ParentAxis = Literal[
    "size", "mode", "modality", "domain", "version", "training_stage", "tier"
]
OrgKind = Literal["lab", "community", "individual", "unknown"]


class ParentEdge(BaseModel):
    id: str
    relationship: ParentRelationship
    axis: Optional[ParentAxis] = None


# --- Resolve ---

ResolveMode = Literal["resolve", "exact"]


class ResolveRequest(BaseModel):
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str] = None
    source_field: Optional[str] = None
    # "resolve" runs the full chain (fuzzy inference; draft auto-create in write
    # mode). "exact" stops after exact/normalized alias and HF-id-check: no
    # fuzzy guessing, no draft creation, no alias writes.
    mode: ResolveMode = "resolve"


# --- Lean resolve contract ---
# One shape for every entity type: core match fields + ordered `ancestry` +
# typed `resolution_detail`. Type-specific entity fields (model group/family/
# lineage/params; benchmark family_key/composite_keys/category; family and
# composite members) live only on the entity GET endpoints. The in-process
# `ResolutionResult` stays the rich union.

AncestryLevel = Literal["group", "family", "composite", "benchmark", "slice"]


class AncestryEntry(BaseModel):
    """One parent canonical and its level; the chain runs from the matched entity's immediate parent up to the root."""
    canonical_id: str
    level: AncestryLevel


class ModelResolutionDetail(BaseModel):
    """Type-specific resolution detail for a model match."""
    granularity: Optional[str] = None  # variant | group | family
    # Hugging Face repo id only when runtime-attestable: HF oracle
    # (resolution_source == "hf") or hub-stats-confirmed (metadata.hf_id ==
    # canonical_id). null means "not attested", not "not on HF": models.dev-only
    # and name-inferred entries are null even when the repo exists.
    hf_repo_id: Optional[str] = None


class BenchmarkResolutionDetail(BaseModel):
    """Type-specific resolution detail for a benchmark match.

    `level`: where the matched canonical sits in the benchmark tree;
    `slice` when it carries a `parent_benchmark_id` (e.g. `rewardbench-chat`
    under `rewardbench`). `matched_subset` is independent of `level`: the
    raw surface form whenever it differs from the canonical id by more than
    case (e.g. `"MATH Level 5"` -> `math-level-5`)."""
    level: Optional[str] = None  # composite | family | benchmark | slice
    matched_subset: Optional[str] = None


class ResolveResponse(BaseModel):
    # Echo of the raw input.
    raw_value: Optional[str] = None
    # Echo of the requested entity type; selects the entity GET endpoint for detail.
    entity_type: Optional[str] = None
    # The matched canonical id (None on no_match). For models this is the
    # exact matched leaf; group/family membership is carried in `ancestry`.
    canonical_id: Optional[str] = None
    strategy: str
    confidence: float
    created_new: bool
    resolution_source: Optional[str] = None
    review_status: Optional[str] = None
    # Ordered from the matched entity's immediate parent up to the root; `[]`
    # when self is a root.
    #   model     -> e.g. [{group}, {family}]
    #   benchmark -> e.g. [{family}, {composite}]
    ancestry: list[AncestryEntry] = []
    # Schema selected by `entity_type`; `{}` for composite / family / metric /
    # org. `harness` carries `{harness_version_stripped, bare_name, bare_tier}`
    # only when the match came from stripping a trailing version token
    # (`lm_eval 0.4.12`).
    resolution_detail: dict[str, Any] = {}


# --- Entities ---

class ModelCreate(BaseModel):
    id: str
    display_name: str
    developer: Optional[str] = None
    org_id: Optional[str] = None
    family: Optional[str] = None
    architecture: Optional[str] = None
    params_billions: Optional[float] = None
    parents: list[ParentEdge] = []
    model_group_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class ModelPatch(BaseModel):
    display_name: Optional[str] = None
    developer: Optional[str] = None
    org_id: Optional[str] = None
    family: Optional[str] = None
    architecture: Optional[str] = None
    params_billions: Optional[float] = None
    parents: Optional[list[ParentEdge]] = None
    model_group_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


_PREFERRED_METRIC_DESCRIPTION = (
    "Registry-declared default metric for the merged benchmark view "
    "(canonical_metrics.id). Null = no default declared."
)
_PREFERRED_METRIC_JUDGED_DESCRIPTION = (
    "Is preferred_metric_id produced by an LLM judge? true = yes, "
    "false = no, null = unstated. Says nothing about the benchmark's "
    "other metrics."
)


class BenchmarkCreate(BaseModel):
    id: str
    display_name: str
    description: Optional[str] = None
    dataset_repo: Optional[str] = None
    parent_benchmark_id: Optional[str] = None
    preferred_metric_id: Optional[str] = Field(
        default=None, description=_PREFERRED_METRIC_DESCRIPTION
    )
    preferred_metric_llm_judged: Optional[bool] = Field(
        default=None, description=_PREFERRED_METRIC_JUDGED_DESCRIPTION
    )
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class BenchmarkPatch(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    dataset_repo: Optional[str] = None
    parent_benchmark_id: Optional[str] = None
    preferred_metric_id: Optional[str] = Field(
        default=None, description=_PREFERRED_METRIC_DESCRIPTION
    )
    preferred_metric_llm_judged: Optional[bool] = Field(
        default=None, description=_PREFERRED_METRIC_JUDGED_DESCRIPTION
    )
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


class MetricCreate(BaseModel):
    id: str
    display_name: str
    score_type: Optional[str] = None
    lower_is_better: bool = False
    min_score: Optional[float] = Field(default=None, description=_BOUND_DESCRIPTION)
    max_score: Optional[float] = Field(default=None, description=_BOUND_DESCRIPTION)
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class MetricPatch(BaseModel):
    display_name: Optional[str] = None
    score_type: Optional[str] = None
    lower_is_better: Optional[bool] = None
    min_score: Optional[float] = Field(default=None, description=_BOUND_DESCRIPTION)
    max_score: Optional[float] = Field(default=None, description=_BOUND_DESCRIPTION)
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


class HarnessCreate(BaseModel):
    id: str
    display_name: str
    version: Optional[str] = None
    fork_url: Optional[str] = None
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class HarnessPatch(BaseModel):
    display_name: Optional[str] = None
    version: Optional[str] = None
    fork_url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


# --- Orgs ---

class OrgCreate(BaseModel):
    id: str
    display_name: str
    parent_org_id: Optional[str] = None
    website: Optional[str] = None
    hf_org: Optional[str] = None
    logo_url: Optional[str] = None
    kind: OrgKind = "unknown"
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class OrgPatch(BaseModel):
    display_name: Optional[str] = None
    parent_org_id: Optional[str] = None
    website: Optional[str] = None
    hf_org: Optional[str] = None
    logo_url: Optional[str] = None
    kind: Optional[OrgKind] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


# --- Aliases ---

class AliasPatch(BaseModel):
    canonical_id: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
