"""Tests for `scripts/ci_dispatch_seed.py` and the refresh workflow shape gate."""

from __future__ import annotations

import copy
import io
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import load_script_module

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHA = "a" * 40


@pytest.fixture(scope="module")
def mod():
    return load_script_module("ci_dispatch_seed")


def _run_url(run_id: int) -> str:
    return f"https://github.com/evaleval/eval-entity-registry/actions/{'runs'}/{run_id}"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        self.now += dt


# ---------- retry classification ----------


def test_retryable_distinguishes_transient_and_permanent_errors(mod):
    err = mod.GitHubAPIError
    # Transport / truncated JSON (status_code=None) is always retryable.
    assert mod._is_retryable(err(None, "timeout"), allow_404=False) is True
    assert mod._is_retryable(err(429, "rate limit"), allow_404=False) is True
    assert mod._is_retryable(err(502, "bad gateway"), allow_404=False) is True
    assert mod._is_retryable(err(503, "unavailable"), allow_404=False) is True

    # 404 is retryable ONLY while polling a newly dispatched run ID, never on dispatch.
    assert mod._is_retryable(err(404, "not found"), allow_404=True) is True
    assert mod._is_retryable(err(404, "not found"), allow_404=False) is False

    # Auth and validation failures are permanent; rate-limit 403s are retryable.
    assert mod._is_retryable(err(401, "unauthorized"), allow_404=True) is False
    assert mod._is_retryable(err(422, "invalid"), allow_404=True) is False
    assert mod._is_retryable(err(403, "forbidden"), allow_404=True) is False
    assert (
        mod._is_retryable(
            err(403, "secondary rate limit", headers={"Retry-After": "10"}),
            allow_404=False,
        )
        is True
    )
    assert (
        mod._is_retryable(
            err(403, "quota exhausted", headers={"X-RateLimit-Remaining": "0"}),
            allow_404=False,
        )
        is True
    )


# ---------- dispatch + run_id resolution ----------


def test_dispatch_uses_returned_workflow_run_id_without_listing(mod):
    calls: list[tuple[str, str, Any]] = []

    def fake_req(method: str, url: str, *, token: str, payload: Any = None):
        calls.append((method, url, payload))
        assert token == "tok"
        return {"workflow_run_id": 77, "html_url": _run_url(77)}

    run_id = mod.dispatch_and_resolve_run_id(
        "evaleval/eval-entity-registry",
        source_run_id="99-1",
        expected_sha=_SHA,
        token="tok",
        request_fn=fake_req,
        sleep=lambda _: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )
    assert run_id == 77
    assert len(calls) == 1
    method, url, payload = calls[0]
    assert method == "POST"
    assert url.endswith("/repos/evaleval/eval-entity-registry/actions/workflows/seed.yml/dispatches")
    assert payload == {
        "ref": "main",
        "inputs": {"source_run_id": "99-1", "expected_sha": _SHA},
        "return_run_details": True,
    }


def test_dispatch_falls_back_to_title_correlation_on_204(mod):
    clock = _Clock()
    list_calls = {"n": 0}

    def fake_req(method: str, url: str, *, token: str, payload: Any = None):
        if method == "POST":
            return None  # 204 No Content
        list_calls["n"] += 1
        if list_calls["n"] == 1:
            raise mod.GitHubAPIError(503, "transient")
        if list_calls["n"] == 2:
            return {"workflow_runs": [{"id": 11, "display_title": "Seed + publish (other-1)"}]}
        return {
            "workflow_runs": [
                {"id": 11, "display_title": "Seed + publish (other-1)"},
                {"id": 42, "display_title": "Seed + publish (99-1)"},
            ]
        }

    run_id = mod.dispatch_and_resolve_run_id(
        "evaleval/eval-entity-registry",
        source_run_id="99-1",
        expected_sha=_SHA,
        token="tok",
        request_fn=fake_req,
        sleep=clock.sleep,
    )
    assert run_id == 42
    assert list_calls["n"] == 3
    assert clock.sleeps == [5.0, 5.0]


def test_dispatch_adopts_created_run_instead_of_double_dispatching(mod):
    """If POST drops the connection after server-side creation, pre-retry lookup
    adopts the existing correlated run without sending a second POST."""
    clock = _Clock()
    posts = {"n": 0}

    def fake_req(method: str, url: str, *, token: str, payload: Any = None):
        if method == "POST":
            posts["n"] += 1
            raise mod.GitHubAPIError(None, "connection reset after write")
        return {"workflow_runs": [{"id": 88, "display_title": "Seed + publish (99-1)"}]}

    run_id = mod.dispatch_and_resolve_run_id(
        "evaleval/eval-entity-registry",
        source_run_id="99-1",
        expected_sha=_SHA,
        token="tok",
        request_fn=fake_req,
        sleep=clock.sleep,
    )
    assert run_id == 88
    assert posts["n"] == 1
    assert clock.sleeps == [5.0]


def test_dispatch_retries_post_when_not_yet_created(mod):
    clock = _Clock()
    posts = {"n": 0}

    def fake_req(method: str, url: str, *, token: str, payload: Any = None):
        if method == "POST":
            posts["n"] += 1
            if posts["n"] == 1:
                raise mod.GitHubAPIError(502, "gateway hiccup")
            return {"workflow_run_id": 91}
        return {"workflow_runs": []}

    run_id = mod.dispatch_and_resolve_run_id(
        "evaleval/eval-entity-registry",
        source_run_id="99-1",
        expected_sha=_SHA,
        token="tok",
        request_fn=fake_req,
        sleep=clock.sleep,
    )
    assert run_id == 91
    assert posts["n"] == 2
    assert clock.sleeps == [5.0]


@pytest.mark.parametrize("code", [401, 403, 404, 422])
def test_dispatch_fails_fast_on_permanent_http_errors(mod, code):
    calls = {"n": 0}

    def fake_req(method: str, url: str, *, token: str, payload: Any = None):
        calls["n"] += 1
        raise mod.GitHubAPIError(code, "permanent")

    with pytest.raises(mod.GitHubAPIError):
        mod.dispatch_and_resolve_run_id(
            "evaleval/eval-entity-registry",
            source_run_id="99-1",
            expected_sha=_SHA,
            token="tok",
            request_fn=fake_req,
            sleep=lambda _: (_ for _ in ()).throw(AssertionError("must not sleep")),
        )
    assert calls["n"] == 1


# ---------- run completion polling ----------


def test_wait_survives_startup_404_replica_flap_and_midrun_502(mod, capsys):
    """Startup read-replica 404s, non-sticky 200->404 flapping, and mid-run 502s
    all recover and reset consecutive_errors once valid responses arrive."""
    clock = _Clock()
    responses = iter(
        [
            mod.GitHubAPIError(404, "not indexed yet"),
            mod.GitHubAPIError(404, "still lagging"),
            {"status": "queued", "conclusion": None, "html_url": _run_url(42)},
            mod.GitHubAPIError(404, "lagging replica"),
            {"status": "in_progress", "conclusion": None, "html_url": _run_url(42)},
            {"status": "in_progress", "conclusion": None, "html_url": _run_url(42)},
            mod.GitHubAPIError(502, "bad gateway"),
            mod.GitHubAPIError(None, "socket reset"),
            {"status": "completed", "conclusion": "success", "html_url": _run_url(42)},
        ]
    )

    def fake_req(method: str, url: str, *, token: str):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    out = mod.wait_for_run_completion(
        "evaleval/eval-entity-registry",
        42,
        token="tok",
        poll_seconds=15.0,
        max_consecutive_errors=3,  # 4 total errors occur, never >=3 consecutively
        request_fn=fake_req,
        sleep=clock.sleep,
        clock=clock,
    )
    assert out == _run_url(42)
    err = capsys.readouterr().err
    # Duplicate in_progress poll does not spam duplicate status lines.
    assert err.count("status=in_progress") == 1
    assert err.count("status=queued") == 1
    assert err.count("status=completed conclusion=success") == 1


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", "startup_failure", None])
def test_wait_refuses_every_non_success_conclusion(mod, conclusion):
    clock = _Clock()

    def fake_req(method: str, url: str, *, token: str):
        return {"status": "completed", "conclusion": conclusion, "html_url": _run_url(42)}

    with pytest.raises(mod.ChildRunFailedError, match=f"conclusion={conclusion!r}"):
        mod.wait_for_run_completion(
            "evaleval/eval-entity-registry",
            42,
            token="tok",
            request_fn=fake_req,
            sleep=clock.sleep,
            clock=clock,
        )
    assert clock.sleeps == []


def test_wait_aborts_on_consecutive_error_cap_and_fast_fails_auth(mod):
    clock = _Clock()

    def always_404(method: str, url: str, *, token: str):
        raise mod.GitHubAPIError(404, "missing")

    with pytest.raises(RuntimeError, match="4 consecutive API errors"):
        mod.wait_for_run_completion(
            "evaleval/eval-entity-registry",
            42,
            token="tok",
            poll_seconds=10.0,
            max_consecutive_errors=4,
            request_fn=always_404,
            sleep=clock.sleep,
            clock=clock,
        )
    assert clock.sleeps == [10.0, 10.0, 10.0]

    def bad_auth(method: str, url: str, *, token: str):
        raise mod.GitHubAPIError(403, "Resource not accessible by integration")

    clock2 = _Clock()
    with pytest.raises(mod.GitHubAPIError):
        mod.wait_for_run_completion(
            "evaleval/eval-entity-registry",
            42,
            token="tok",
            request_fn=bad_auth,
            sleep=clock2.sleep,
            clock=clock2,
        )
    assert clock2.sleeps == []


def test_wait_enforces_overall_deadline(mod):
    clock = _Clock()

    def stuck_in_progress(method: str, url: str, *, token: str):
        return {"status": "in_progress", "conclusion": None}

    with pytest.raises(TimeoutError, match="timed out after 35s"):
        mod.wait_for_run_completion(
            "evaleval/eval-entity-registry",
            42,
            token="tok",
            poll_seconds=15.0,
            timeout_seconds=35.0,
            request_fn=stuck_in_progress,
            sleep=clock.sleep,
            clock=clock,
        )
    assert clock.sleeps == [15.0, 15.0, 5.0]


# ---------- HTTP client + CLI entry point ----------


def test_request_json_maps_http_and_transport_errors(mod, monkeypatch):
    def raise_http(req, timeout=30.0):
        fp = io.BytesIO(b'{"message":"Not Found"}')
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {"Retry-After": "3"}, fp)

    monkeypatch.setattr(mod.urllib.request, "urlopen", raise_http)
    with pytest.raises(mod.GitHubAPIError) as exc_info:
        mod._request_json("GET", "https://api.github.test/x", token="tok")
    assert exc_info.value.status_code == 404
    assert exc_info.value.headers["retry-after"] == "3"
    assert "Not Found" in str(exc_info.value)

    def raise_url(req, timeout=30.0):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(mod.urllib.request, "urlopen", raise_url)
    with pytest.raises(mod.GitHubAPIError) as exc_info2:
        mod._request_json("GET", "https://api.github.test/x", token="tok")
    assert exc_info2.value.status_code is None


def test_cli_beacon_retry_resolves_origin_main_and_rejects_bad_inputs(mod, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REPOSITORY", "evaleval/eval-entity-registry")
    monkeypatch.setenv("GH_TOKEN", "tok")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "b" * 40)

    seen: dict[str, Any] = {}

    def fake_dispatch(repo, **kw):
        seen["repo"] = repo
        seen.update(kw)
        return 55

    monkeypatch.setattr(mod, "dispatch_and_resolve_run_id", fake_dispatch)
    monkeypatch.setattr(mod, "wait_for_run_completion", lambda repo, rid, **kw: _run_url(rid))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ci_dispatch_seed.py", "--source-run-id", "77-1", "--expected-sha", ""],
    )
    assert mod.main() == 0
    assert seen["expected_sha"] == "b" * 40
    assert seen["source_run_id"] == "77-1"

    # Unreadable origin/main fails closed with exit 1.
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: None)
    assert mod.main() == 1
    assert "cannot resolve 40-hex origin/main SHA" in capsys.readouterr().err

    # Malformed explicit --expected-sha exits 2 via argparse.
    monkeypatch.setattr(
        sys,
        "argv",
        ["ci_dispatch_seed.py", "--source-run-id", "77-1", "--expected-sha", "not-a-sha"],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2

    # Missing token fails closed with exit 1.
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ci_dispatch_seed.py", "--source-run-id", "77-1", "--expected-sha", _SHA],
    )
    assert mod.main() == 1


# ---------- workflow-shape gate (real workflows + bad-input refusal) ----------


@pytest.mark.parametrize("filename", ["refresh-models.yml", "refresh-hub-stats.yml"])
def test_refresh_workflows_satisfy_dispatch_and_beacon_invariants(mod, filename):
    path = _REPO_ROOT / ".github" / "workflows" / filename
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    mod.validate_refresh_workflow(doc, filename)


def test_workflow_validator_refuses_bad_workflow_shapes(mod):
    base_path = _REPO_ROOT / ".github" / "workflows" / "refresh-models.yml"
    good = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    def _step(doc, name):
        for s in doc["jobs"]["refresh"]["steps"]:
            if s.get("name") == name:
                return s
        raise KeyError(name)

    # 1. Reintroducing `gh run watch` is refused.
    bad = copy.deepcopy(good)
    _step(bad, "Publish and wait for verified completion")["run"] += "\ngh run watch 1 --exit-status"
    with pytest.raises(ValueError, match="forbidden `gh run watch`"):
        mod.validate_refresh_workflow(bad, "refresh-models.yml")

    # 2. Reintroducing inline `gh workflow run` is refused.
    bad = copy.deepcopy(good)
    _step(bad, "Publish and wait for verified completion")["run"] = "gh workflow run seed.yml"
    with pytest.raises(ValueError, match="forbidden inline `gh workflow run`"):
        mod.validate_refresh_workflow(bad, "refresh-models.yml")

    # 3. Dropping EXPECTED_SHA or pushed_sha wiring is refused.
    bad = copy.deepcopy(good)
    _step(bad, "Publish and wait for verified completion")["env"]["EXPECTED_SHA"] = "main"
    with pytest.raises(ValueError, match="pushed_sha"):
        mod.validate_refresh_workflow(bad, "refresh-models.yml")

    # 4. Clearing the stall beacon without `steps.deploy.outputs.needed == 'true'` is refused.
    bad = copy.deepcopy(good)
    _step(bad, "Clear stall beacon")["if"] = "success()"
    with pytest.raises(ValueError, match="Clear stall beacon"):
        mod.validate_refresh_workflow(bad, "refresh-models.yml")

    # 5. Raising the stall beacon without `cancelled()` is refused.
    bad = copy.deepcopy(good)
    _step(bad, "Raise/refresh stall beacon")["if"] = "failure()"
    with pytest.raises(ValueError, match="Raise/refresh stall beacon"):
        mod.validate_refresh_workflow(bad, "refresh-models.yml")
