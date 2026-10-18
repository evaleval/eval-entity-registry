"""Host-token → inference_platform mapping, inverted from the single
authority ``seed/inference_platforms.yaml``; consumers import it rather than
copying the map.

Imports only stdlib + yaml (no fuzzy, no schemas) to avoid an import cycle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

# Host token meaning "missing developer field"; maps to no platform.
_UNKNOWN_SENTINEL = "unknown"

_SEED_PATH = (
    Path(__file__).resolve().parents[3] / "seed" / "inference_platforms.yaml"
)

_HOST_TOKEN_TO_PLATFORM: dict[str, Optional[str]] = {}
_LOADED = False


def _coerce_aliases(raw) -> list[str]:
    """The seed CLI JSON-encodes the aliases column; the YAML on disk holds native lists."""
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw else []
        return list(decoded) if isinstance(decoded, list) else []
    return list(raw or [])


def _load() -> None:
    global _LOADED
    _HOST_TOKEN_TO_PLATFORM.clear()
    if _SEED_PATH.exists():
        with open(_SEED_PATH) as f:
            platforms = yaml.safe_load(f) or []
        for plat in platforms:
            platform_id = plat.get("id")
            for alias in _coerce_aliases(plat.get("aliases")):
                if alias:
                    _HOST_TOKEN_TO_PLATFORM[alias.lower()] = platform_id
    _HOST_TOKEN_TO_PLATFORM[_UNKNOWN_SENTINEL] = None
    _LOADED = True


def _ensure_loaded() -> None:
    if not _LOADED:
        _load()


def get_host_token_platform(token: str) -> Optional[str]:
    """inference_platforms.id for a host token (e.g. 'fireworks/', '-bedrock'), case-insensitive; None when unknown."""
    if not token:
        return None
    _ensure_loaded()
    return _HOST_TOKEN_TO_PLATFORM.get(token.lower())


def all_host_tokens() -> set[str]:
    """Known host tokens (lowercased), including the `unknown` sentinel."""
    _ensure_loaded()
    return set(_HOST_TOKEN_TO_PLATFORM.keys())


# Eager load; no-op when the seed file is absent.
_load()
