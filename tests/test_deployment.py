"""Space deployment behaviour: selective table loading, read-only gating of
writes and resolve side effects, and the resolve log writer."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from eval_entity_registry.main import app
from eval_entity_registry.store import schemas as s
from eval_entity_registry.store.hf_store import RegistryStore, QUERY_TABLE_NAMES, TABLE_NAMES
from eval_entity_registry.services.resolution_service import ResolutionService
from eval_entity_registry.services.log_writer import ResolveLogWriter, _MAX_BUFFER_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(table_names: list[str]) -> RegistryStore:
    store = RegistryStore()
    store._tables = {name: s.empty(name) for name in table_names}
    store._loaded = True
    return store


def _seed_benchmark(store: RegistryStore):
    """Add a known benchmark + alias so resolve can find it."""
    import pandas as pd
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    bm = pd.DataFrame([{
        "id": "math",
        "display_name": "MATH",
        "description": None,
        "dataset_repo": None,
        "parent_benchmark_id": None,
        "tags": "[]",
        "metadata": "{}",
        "review_status": "confirmed",
        "created_at": now,
        "updated_at": now,
    }])
    store.set_table("canonical_benchmarks", pd.concat([store.table("canonical_benchmarks"), bm], ignore_index=True))

    alias = pd.DataFrame([{
        "id": "alias-1",
        "raw_value": "MATH",
        "entity_type": "benchmark",
        "canonical_id": "math",
        "source_config": None,
        "source_field": None,
        "status": "confirmed",
        "strategy": "exact",
        "confidence": 1.0,
        "notes": None,
        "created_at": now,
        "updated_at": now,
    }])
    store.set_table("aliases", pd.concat([store.table("aliases"), alias], ignore_index=True))


@pytest.fixture
def full_store(monkeypatch):
    """All tables (normal mode)."""
    from eval_entity_registry.store import hf_store
    store = _make_store(TABLE_NAMES)
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


@pytest.fixture
def query_store(monkeypatch):
    """Query tables only (read-only mode loading)."""
    from eval_entity_registry.store import hf_store
    store = _make_store(QUERY_TABLE_NAMES)
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Selective table loading
# ---------------------------------------------------------------------------

class TestSelectiveTableLoading:
    def test_query_tables_loaded(self, query_store):
        for t in QUERY_TABLE_NAMES:
            assert query_store.has_table(t)

    def test_pipeline_tables_not_loaded(self, query_store):
        for t in ["eval_results", "resolution_log", "sync_runs"]:
            assert not query_store.has_table(t)

    def test_full_store_has_all_tables(self, full_store):
        for t in TABLE_NAMES:
            assert full_store.has_table(t)


# ---------------------------------------------------------------------------
# Health/stats with missing tables
# ---------------------------------------------------------------------------

class TestHealthMissingTables:
    def test_health_with_query_only_tables(self, query_store, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["entities"]["models"] == 0

    def test_stats_with_query_only_tables(self, query_store, client):
        """Unloaded pipeline tables count as 0, not an error."""
        r = client.get("/api/v1/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["resolution_log"]["total"] == 0
        assert data["sync_runs"]["total"] == 0
        assert data["models"]["total"] == 0
        assert data["aliases"]["total"] == 0


# ---------------------------------------------------------------------------
# Write endpoint gating (read-only mode)
# ---------------------------------------------------------------------------

class TestReadOnlyGating:
    @pytest.fixture(autouse=True)
    def _set_read_only(self, monkeypatch, full_store):
        from eval_entity_registry import config
        monkeypatch.setattr(config.settings, "read_only", True)
        yield
        monkeypatch.setattr(config.settings, "read_only", False)

    def test_create_model_blocked(self, client):
        r = client.post("/api/v1/models", json={
            "id": "test", "display_name": "Test"
        })
        assert r.status_code == 405

    def test_patch_model_blocked(self, client):
        r = client.patch("/api/v1/models/test", json={"display_name": "X"})
        assert r.status_code == 405

    def test_create_benchmark_blocked(self, client):
        r = client.post("/api/v1/benchmarks", json={
            "id": "test", "display_name": "Test"
        })
        assert r.status_code == 405

    def test_create_metric_blocked(self, client):
        r = client.post("/api/v1/metrics", json={
            "id": "test", "display_name": "Test"
        })
        assert r.status_code == 405

    def test_create_harness_blocked(self, client):
        r = client.post("/api/v1/harnesses", json={
            "id": "test", "display_name": "Test"
        })
        assert r.status_code == 405

    def test_patch_alias_blocked(self, client):
        r = client.patch("/api/v1/aliases/some-id", json={"status": "confirmed"})
        assert r.status_code == 405

    def test_get_models_allowed(self, client):
        """GETs stay open in read-only mode."""
        r = client.get("/api/v1/models")
        assert r.status_code == 200

    def test_get_aliases_allowed(self, client):
        r = client.get("/api/v1/aliases")
        assert r.status_code == 200

    def test_malformed_post_model_returns_405_not_422(self, client):
        """The read-only gate runs before body validation."""
        r = client.post("/api/v1/models", json={"garbage": True})
        assert r.status_code == 405

    def test_malformed_post_benchmark_returns_405_not_422(self, client):
        r = client.post("/api/v1/benchmarks", json={"garbage": True})
        assert r.status_code == 405

    def test_malformed_post_metric_returns_405_not_422(self, client):
        r = client.post("/api/v1/metrics", json={"garbage": True})
        assert r.status_code == 405

    def test_malformed_post_harness_returns_405_not_422(self, client):
        r = client.post("/api/v1/harnesses", json={"garbage": True})
        assert r.status_code == 405

    def test_malformed_patch_alias_returns_405_not_422(self, client):
        r = client.patch("/api/v1/aliases/x", json={"status": 123})
        assert r.status_code == 405


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_resolve_invalid_entity_type_returns_422(self, full_store, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "x", "entity_type": "not-a-real-type"
        })
        assert r.status_code == 422

    def test_list_benchmarks_invalid_review_status_returns_422(self, full_store, client):
        r = client.get("/api/v1/benchmarks?review_status=nonsense")
        assert r.status_code == 422

    def test_list_metrics_invalid_review_status_returns_422(self, full_store, client):
        r = client.get("/api/v1/metrics?review_status=nonsense")
        assert r.status_code == 422

    def test_list_harnesses_invalid_review_status_returns_422(self, full_store, client):
        r = client.get("/api/v1/harnesses?review_status=nonsense")
        assert r.status_code == 422

    def test_list_aliases_invalid_status_returns_422(self, full_store, client):
        r = client.get("/api/v1/aliases?status=nonsense")
        assert r.status_code == 422

    def test_list_aliases_invalid_entity_type_returns_422(self, full_store, client):
        r = client.get("/api/v1/aliases?entity_type=nonsense")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Read-only resolve behavior
# ---------------------------------------------------------------------------

class TestReadOnlyResolve:
    @pytest.fixture(autouse=True)
    def _set_read_only(self, monkeypatch, full_store):
        from eval_entity_registry import config
        monkeypatch.setattr(config.settings, "read_only", True)
        _seed_benchmark(full_store)
        app.state.resolution_service = ResolutionService(full_store)

    def test_resolve_match_returns_result(self, client, full_store):
        initial_aliases = len(full_store.table("aliases"))
        initial_benchmarks = len(full_store.table("canonical_benchmarks"))
        r = client.post("/api/v1/resolve", json={
            "raw_value": "MATH",
            "entity_type": "benchmark",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] == "math"
        assert data["created_new"] is False
        assert len(full_store.table("aliases")) == initial_aliases
        assert len(full_store.table("canonical_benchmarks")) == initial_benchmarks

    def test_resolve_no_match_returns_null(self, client, full_store):
        """Read-only: an unmatched string returns null and creates no draft."""
        r = client.post("/api/v1/resolve", json={
            "raw_value": "CompletelyUnknownBenchmark",
            "entity_type": "benchmark",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] is None
        assert data["strategy"] == "no_match"
        assert data["created_new"] is False
        assert len(full_store.table("canonical_benchmarks")) == 1

    def test_resolve_no_match_no_alias_written(self, client, full_store):
        """Read-only resolve writes no aliases."""
        initial_aliases = len(full_store.table("aliases"))
        client.post("/api/v1/resolve", json={
            "raw_value": "CompletelyUnknownBenchmark",
            "entity_type": "benchmark",
        })
        assert len(full_store.table("aliases")) == initial_aliases

    def test_resolve_batch_read_only(self, client, full_store):
        r = client.post("/api/v1/resolve/batch", json=[
            {"raw_value": "MATH", "entity_type": "benchmark"},
            {"raw_value": "Unknown", "entity_type": "benchmark"},
        ])
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["canonical_id"] == "math"
        assert data[1]["canonical_id"] is None


# ---------------------------------------------------------------------------
# Resolve log writer
# ---------------------------------------------------------------------------

class TestResolveLogWriter:
    def test_disabled_writer_discards_entries(self):
        writer = ResolveLogWriter("")
        assert not writer.enabled
        writer.append({"raw_value": "test"})
        assert len(writer._buffer) == 0

    def test_enabled_writer_buffers_entries(self):
        writer = ResolveLogWriter("some-bucket")
        assert writer.enabled
        writer.append({"raw_value": "test"})
        assert len(writer._buffer) == 1

    def test_resolve_endpoint_appends_to_log(self, full_store, client):
        _seed_benchmark(full_store)
        app.state.resolution_service = ResolutionService(full_store)
        log_writer = ResolveLogWriter("test-bucket")
        app.state.log_writer = log_writer

        client.post("/api/v1/resolve", json={
            "raw_value": "MATH", "entity_type": "benchmark"
        })
        assert len(log_writer._buffer) == 1
        entry = log_writer._buffer[0]
        assert entry["raw_value"] == "MATH"
        assert entry["canonical_id"] == "math"
        assert "request_id" in entry
        assert "timestamp" in entry

    def test_batch_resolve_shares_request_id(self, full_store, client):
        _seed_benchmark(full_store)
        app.state.resolution_service = ResolutionService(full_store)
        log_writer = ResolveLogWriter("test-bucket")
        app.state.log_writer = log_writer

        client.post("/api/v1/resolve/batch", json=[
            {"raw_value": "MATH", "entity_type": "benchmark"},
            {"raw_value": "Other", "entity_type": "benchmark"},
        ])
        assert len(log_writer._buffer) == 2
        assert log_writer._buffer[0]["request_id"] == log_writer._buffer[1]["request_id"]

    def test_buffer_capped_on_overflow(self):
        """The buffer caps at _MAX_BUFFER_SIZE, dropping the oldest entries."""
        writer = ResolveLogWriter("test-bucket")
        for i in range(_MAX_BUFFER_SIZE + 100):
            writer.append({"raw_value": f"entry-{i}"})
        assert len(writer._buffer) == _MAX_BUFFER_SIZE
        assert writer._buffer[-1]["raw_value"] == f"entry-{_MAX_BUFFER_SIZE + 99}"

    async def test_flush_clears_buffer(self):
        """A successful flush clears the buffer."""
        writer = ResolveLogWriter("test-bucket")
        writer.append({"request_id": "r1", "raw_value": "test", "entity_type": "benchmark",
                        "source_config": None, "canonical_id": "math", "strategy": "exact",
                        "confidence": 1.0, "timestamp": "2026-01-01T00:00:00"})

        async def mock_to_thread(func, *args, **kwargs):
            pass

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            await writer._flush()

        assert len(writer._buffer) == 0

    async def test_flush_re_adds_entries_on_failure(self):
        """A failed flush returns its entries to the buffer."""
        writer = ResolveLogWriter("test-bucket")
        writer.append({"request_id": "r1", "raw_value": "test", "entity_type": "benchmark",
                        "source_config": None, "canonical_id": "math", "strategy": "exact",
                        "confidence": 1.0, "timestamp": "2026-01-01T00:00:00"})

        async def mock_to_thread(func, *args, **kwargs):
            raise RuntimeError("upload failed")

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            await writer._flush()

        assert len(writer._buffer) == 1
        assert writer._buffer[0]["raw_value"] == "test"

    async def test_flush_empty_buffer_is_noop(self):
        """An empty flush is a no-op."""
        writer = ResolveLogWriter("test-bucket")
        await writer._flush()
        assert len(writer._buffer) == 0


# ---------------------------------------------------------------------------
# Read-only resolve edge cases
# ---------------------------------------------------------------------------

class TestReadOnlyResolveEdgeCases:
    @pytest.fixture(autouse=True)
    def _set_read_only(self, monkeypatch, full_store):
        from eval_entity_registry import config
        monkeypatch.setattr(config.settings, "read_only", True)
        app.state.resolution_service = ResolutionService(full_store)

    def test_resolve_empty_string(self, client, full_store):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "",
            "entity_type": "benchmark",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] is None
        assert data["strategy"] == "no_match"
        assert data["created_new"] is False

    def test_resolve_whitespace_only(self, client, full_store):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "   ",
            "entity_type": "benchmark",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] is None
        assert data["created_new"] is False
