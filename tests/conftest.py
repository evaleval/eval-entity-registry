"""Shared test fixtures — reset module-level caches between tests."""
import importlib.util
from pathlib import Path

import pytest
from eval_entity_registry.config import settings
from eval_entity_registry.store import queries, hf_store

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script_module(script_stem, module_name=None):
    """Import scripts/<script_stem>.py as a standalone module without making
    scripts/ a package, so tests can call its functions without running its
    destructive main().

    module_name overrides the registered module name (defaults to script_stem);
    a few tests load the same script under a distinct synthetic name.
    """
    path = _REPO_ROOT / "scripts" / f"{script_stem}.py"
    spec = importlib.util.spec_from_file_location(module_name or script_stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_query_caches():
    """Clear module-level caches and pending buffers around each test."""
    queries._alias_index.clear()
    queries._pending_result_ids.clear()
    store = getattr(hf_store, "_store", None)
    if store is not None and hasattr(store, "_pending"):
        store._pending = {}
    yield
    queries._alias_index.clear()
    queries._pending_result_ids.clear()
    store = getattr(hf_store, "_store", None)
    if store is not None and hasattr(store, "_pending"):
        store._pending = {}


@pytest.fixture(autouse=True)
def _disable_hub_stats_lookup():
    """The hub-stats lookup hits HF; tests that need it re-enable and mock."""
    original = settings.hub_stats_lookup_enabled
    settings.hub_stats_lookup_enabled = False
    yield
    settings.hub_stats_lookup_enabled = original


@pytest.fixture(autouse=True)
def _disable_live_hf_id_check():
    """The verifier's live tier hits the Hub API; tests that need it
    re-enable and mock (test_hf_id_verifier.py)."""
    original = settings.hf_live_id_check_enabled
    settings.hf_live_id_check_enabled = False
    yield
    settings.hf_live_id_check_enabled = original
