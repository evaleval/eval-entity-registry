"""Alias management routes (list + patch)."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from eval_entity_registry.api.deps import writable as _writable
from eval_entity_registry.api.schemas import AliasPatch, AliasStatus, EntityType
from eval_entity_registry.store.hf_store import get_store, RegistryStore
from eval_entity_registry.store import queries


router = APIRouter()


@router.get("/aliases")
def list_aliases(
    status: Optional[AliasStatus] = None,
    entity_type: Optional[EntityType] = None,
    source_config: Optional[str] = None,
    store: RegistryStore = Depends(get_store),
):
    return queries.list_entities(
        store,
        "aliases",
        review_status=None,
        **{k: v for k, v in {"status": status, "entity_type": entity_type, "source_config": source_config}.items() if v is not None},
    )


@router.patch("/aliases/{alias_id}", dependencies=_writable)
def patch_alias(alias_id: str, body: AliasPatch, store: RegistryStore = Depends(get_store)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    result = queries.update_alias(store, alias_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Alias '{alias_id}' not found")
    return result
