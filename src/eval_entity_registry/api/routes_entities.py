import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from eval_entity_registry.api.deps import writable as _writable
from eval_entity_registry.api.schemas import (
    BenchmarkCreate, BenchmarkPatch,
    HarnessCreate, HarnessPatch,
    MetricCreate, MetricPatch,
    ModelCreate, ModelPatch,
    ReviewStatus,
)
from eval_entity_registry.store.hf_store import get_store, RegistryStore
from eval_entity_registry.store import queries

router = APIRouter()


def _get_or_404(store, table, entity_id):
    entity = queries.get_entity(store, table, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"{table} '{entity_id}' not found")
    return entity


_JSON_FIELDS = {
    "tags", "metadata", "parents",
    "benchmark_ids", "composite_keys", "source_configs", "folder_aliases",
}


def _encode(data: dict) -> dict:
    """JSON-encode list/dict fields for parquet storage."""
    out = {}
    for k, v in data.items():
        if k in _JSON_FIELDS and isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _decode(entity: dict) -> dict:
    """JSON-decode string fields that should be list/dict in API responses."""
    out = {}
    for k, v in entity.items():
        if k in _JSON_FIELDS and isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = v
        else:
            out[k] = v
    return out


def _decode_model(store: RegistryStore, entity: dict) -> dict:
    out = _decode(entity)
    org_id = out.get("org_id")
    if org_id and store.has_table("canonical_orgs"):
        org = queries.get_entity(store, "canonical_orgs", org_id)
        out["developer"] = org.get("display_name") if org else None
    return out


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

@router.get("/models")
def list_models(
    search: Optional[str] = None,
    developer: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [
        _decode_model(store, e)
        for e in queries.list_entities(
            store,
            "canonical_models",
            search=search,
            review_status=review_status,
            developer=developer,
        )
    ]


@router.get("/models/{model_id:path}")
def get_model(model_id: str, store: RegistryStore = Depends(get_store)):
    return _decode_model(store, _get_or_404(store, "canonical_models", model_id))


@router.post("/models", status_code=201, dependencies=_writable)
def create_model(body: ModelCreate, store: RegistryStore = Depends(get_store)):
    return _decode_model(
        store,
        queries.upsert_entity(store, "canonical_models", _encode(body.model_dump())),
    )


@router.patch("/models/{model_id:path}", dependencies=_writable)
def patch_model(model_id: str, body: ModelPatch, store: RegistryStore = Depends(get_store)):
    _get_or_404(store, "canonical_models", model_id)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    data["id"] = model_id
    return _decode_model(
        store,
        queries.upsert_entity(store, "canonical_models", _encode(data)),
    )


# ------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------

@router.get("/benchmarks")
def list_benchmarks(
    search: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [_decode(e) for e in queries.list_entities(store, "canonical_benchmarks", search=search, review_status=review_status)]


@router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, store: RegistryStore = Depends(get_store)):
    return _decode(_get_or_404(store, "canonical_benchmarks", benchmark_id))


@router.post("/benchmarks", status_code=201, dependencies=_writable)
def create_benchmark(body: BenchmarkCreate, store: RegistryStore = Depends(get_store)):
    return _decode(queries.upsert_entity(store, "canonical_benchmarks", _encode(body.model_dump())))


@router.patch("/benchmarks/{benchmark_id}", dependencies=_writable)
def patch_benchmark(benchmark_id: str, body: BenchmarkPatch, store: RegistryStore = Depends(get_store)):
    _get_or_404(store, "canonical_benchmarks", benchmark_id)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    data["id"] = benchmark_id
    return _decode(queries.upsert_entity(store, "canonical_benchmarks", _encode(data)))


# ------------------------------------------------------------------
# Families (read-only; no CRUD)
# ------------------------------------------------------------------

@router.get("/families")
def list_families(
    search: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [
        _decode(e)
        for e in queries.list_entities(
            store, "canonical_families", search=search, review_status=review_status
        )
    ]


@router.get("/families/{family_id}")
def get_family(family_id: str, store: RegistryStore = Depends(get_store)):
    """A benchmark family: member `benchmark_ids`, curated `category`, and the `composite_keys` it rolls up into."""
    return _decode(_get_or_404(store, "canonical_families", family_id))


# ------------------------------------------------------------------
# Composites (read-only; no CRUD)
# ------------------------------------------------------------------

@router.get("/composites")
def list_composites(
    search: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [
        _decode(e)
        for e in queries.list_entities(
            store, "canonical_composites", search=search, review_status=review_status
        )
    ]


@router.get("/composites/{composite_id}")
def get_composite(composite_id: str, store: RegistryStore = Depends(get_store)):
    """A composite (leaderboard / suite): its `source_configs` members and the `family_id` it belongs to."""
    return _decode(_get_or_404(store, "canonical_composites", composite_id))


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@router.get("/metrics")
def list_metrics(
    search: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [_decode(e) for e in queries.list_entities(store, "canonical_metrics", search=search, review_status=review_status)]


@router.get("/metrics/{metric_id}")
def get_metric(metric_id: str, store: RegistryStore = Depends(get_store)):
    return _decode(_get_or_404(store, "canonical_metrics", metric_id))


@router.post("/metrics", status_code=201, dependencies=_writable)
def create_metric(body: MetricCreate, store: RegistryStore = Depends(get_store)):
    return _decode(queries.upsert_entity(store, "canonical_metrics", _encode(body.model_dump())))


@router.patch("/metrics/{metric_id}", dependencies=_writable)
def patch_metric(metric_id: str, body: MetricPatch, store: RegistryStore = Depends(get_store)):
    _get_or_404(store, "canonical_metrics", metric_id)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    data["id"] = metric_id
    return _decode(queries.upsert_entity(store, "canonical_metrics", _encode(data)))


# ------------------------------------------------------------------
# Harnesses
# ------------------------------------------------------------------

@router.get("/harnesses")
def list_harnesses(
    search: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [_decode(e) for e in queries.list_entities(store, "eval_harnesses", search=search, review_status=review_status)]


@router.get("/harnesses/{harness_id}")
def get_harness(harness_id: str, store: RegistryStore = Depends(get_store)):
    return _decode(_get_or_404(store, "eval_harnesses", harness_id))


@router.post("/harnesses", status_code=201, dependencies=_writable)
def create_harness(body: HarnessCreate, store: RegistryStore = Depends(get_store)):
    return _decode(queries.upsert_entity(store, "eval_harnesses", _encode(body.model_dump())))


@router.patch("/harnesses/{harness_id}", dependencies=_writable)
def patch_harness(harness_id: str, body: HarnessPatch, store: RegistryStore = Depends(get_store)):
    _get_or_404(store, "eval_harnesses", harness_id)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    data["id"] = harness_id
    return _decode(queries.upsert_entity(store, "eval_harnesses", _encode(data)))
