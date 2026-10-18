import uuid

from fastapi import APIRouter, Depends, Request
from datetime import datetime, timezone

from eval_entity_registry.api.schemas import ResolveRequest, ResolveResponse
from eval_entity_registry.services.resolution_service import ResolutionService
from eval_entity_registry.services.log_writer import ResolveLogWriter

router = APIRouter()


def _svc(request: Request) -> ResolutionService:
    return request.app.state.resolution_service


def _log_writer(request: Request) -> ResolveLogWriter:
    return request.app.state.log_writer


# Type-agnostic fields copied verbatim from the service dict; type-specific
# entity fields live only on the entity GET endpoints (see schemas.ResolveResponse).
_CORE_FIELDS = (
    "canonical_id", "strategy", "confidence", "created_new",
    "resolution_source", "review_status",
)


def _project_response(req: ResolveRequest, result: dict) -> ResolveResponse:
    """Project the rich in-process resolve dict down to the lean HTTP ResolveResponse."""
    detail = result.get("resolution_detail")
    if not isinstance(detail, dict):
        detail = {}
    ancestry = result.get("ancestry") or []
    return ResolveResponse(
        raw_value=req.raw_value,
        entity_type=req.entity_type,
        ancestry=ancestry,
        resolution_detail=detail,
        **{k: result.get(k) for k in _CORE_FIELDS},
    )


def _log_resolve(
    log_writer: ResolveLogWriter,
    request_id: str,
    req: ResolveRequest,
    result: dict,
) -> None:
    ancestry = result.get("ancestry") or []
    log_writer.append({
        "request_id": request_id,
        "raw_value": req.raw_value,
        "entity_type": req.entity_type,
        "source_config": req.source_config,
        "canonical_id": result.get("canonical_id"),
        "strategy": result.get("strategy"),
        "confidence": result.get("confidence"),
        "ancestry_levels": [a.get("level") for a in ancestry if isinstance(a, dict)],
        "resolution_detail": result.get("resolution_detail"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@router.post("/resolve", response_model=ResolveResponse)
def resolve(
    req: ResolveRequest,
    svc: ResolutionService = Depends(_svc),
    log_writer: ResolveLogWriter = Depends(_log_writer),
):
    result = svc.resolve(
        raw_value=req.raw_value,
        entity_type=req.entity_type,
        source_config=req.source_config,
        source_field=req.source_field,
        mode=req.mode,
    )
    _log_resolve(log_writer, str(uuid.uuid4()), req, result)
    return _project_response(req, result)


@router.post("/resolve/batch", response_model=list[ResolveResponse])
def resolve_batch(
    reqs: list[ResolveRequest],
    svc: ResolutionService = Depends(_svc),
    log_writer: ResolveLogWriter = Depends(_log_writer),
):
    request_id = str(uuid.uuid4())
    responses = []
    for r in reqs:
        result = svc.resolve(
            raw_value=r.raw_value,
            entity_type=r.entity_type,
            source_config=r.source_config,
            source_field=r.source_field,
            mode=r.mode,
        )
        _log_resolve(log_writer, request_id, r, result)
        responses.append(_project_response(r, result))
    return responses
