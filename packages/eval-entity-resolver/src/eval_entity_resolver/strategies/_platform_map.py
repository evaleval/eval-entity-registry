"""Host-token → inference_platform map for the fuzzy strategy.

The resolver must not hard-depend on ``eval_entity_registry``, yet the mapping
has one authority, ``seed/inference_platforms.yaml``. So it is loaded through
two channels, in order: the registry's ``lib.inference_platforms_map`` when
importable, else the co-located seed YAML found by walking up to the repo
root. With neither (a bare wheel install) the map is empty and every host token
captures ``None``; strip-for-matching still works. Must not import ``fuzzy``
(cycle).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Missing-developer sentinel; never a platform.
_UNKNOWN_SENTINEL = "unknown"

_HOST_TOKEN_TO_PLATFORM: dict[str, Optional[str]] = {}
_LOADED = False


def _coerce_aliases(raw: Any) -> list[str]:
    """Accept a native list or a JSON-encoded list string."""
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw else []
        return list(decoded) if isinstance(decoded, list) else []
    return list(raw or [])


def _load_via_registry_lib() -> Optional[dict[str, Optional[str]]]:
    """The registry's accessor, if importable."""
    try:
        from eval_entity_registry.lib.inference_platforms_map import (  # type: ignore
            all_host_tokens,
            get_host_token_platform,
        )
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None
    mapping: dict[str, Optional[str]] = {}
    for token in all_host_tokens():
        mapping[token] = get_host_token_platform(token)
    return mapping or None


def _find_seed_yaml() -> Optional[Path]:
    """The co-located seed/inference_platforms.yaml, walking up from this module."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "seed" / "inference_platforms.yaml"
        if candidate.exists():
            return candidate
    return None


def _load_via_seed_yaml() -> Optional[dict[str, Optional[str]]]:
    seed_path = _find_seed_yaml()
    if seed_path is None:
        return None
    try:
        import yaml  # optional dep
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        with open(seed_path) as f:
            platforms = yaml.safe_load(f) or []
    except (OSError, ValueError):
        return None
    mapping: dict[str, Optional[str]] = {}
    for plat in platforms:
        if not isinstance(plat, dict):
            continue
        platform_id = plat.get("id")
        for alias in _coerce_aliases(plat.get("aliases")):
            if alias:
                mapping[alias.lower()] = platform_id
    mapping[_UNKNOWN_SENTINEL] = None
    return mapping or None


def _load() -> None:
    global _LOADED
    _HOST_TOKEN_TO_PLATFORM.clear()
    mapping = _load_via_registry_lib()
    if mapping is None:
        mapping = _load_via_seed_yaml()
    if mapping:
        _HOST_TOKEN_TO_PLATFORM.update(mapping)
    _HOST_TOKEN_TO_PLATFORM[_UNKNOWN_SENTINEL] = None
    _LOADED = True


def _ensure_loaded() -> None:
    if not _LOADED:
        _load()


def get_host_token_platform(token: str) -> Optional[str]:
    """inference_platforms.id for a host token spelling (``'fireworks/'``,
    ``'-bedrock'``), or None when unknown. Case-insensitive."""
    if not token:
        return None
    _ensure_loaded()
    return _HOST_TOKEN_TO_PLATFORM.get(token.lower())


_load()
