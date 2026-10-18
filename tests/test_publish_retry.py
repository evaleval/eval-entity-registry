"""Unit tests for the publish script's transient-error backoff.

The HF publish job intermittently hits 429 (rate limit) when several publish
runs hit the Hub at once; huggingface_hub's own retry caps at 8s and gives up.
_with_backoff adds a longer outer retry that distinguishes transient errors
(429 / 5xx — worth waiting out) from permanent ones (auth / not-found).
"""
from __future__ import annotations

import sys

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from conftest import load_script_module


@pytest.fixture(scope="module")
def mod():
    return load_script_module("publish_registry_data")


def _http_error(status_code: int) -> HfHubHTTPError:
    resp = httpx.Response(status_code, request=httpx.Request("GET", "http://hf.test"))
    return HfHubHTTPError("boom", response=resp)


def test_is_retryable_only_for_rate_limit_and_5xx(mod):
    assert mod._is_retryable_http(_http_error(429)) is True
    assert mod._is_retryable_http(_http_error(503)) is True
    assert mod._is_retryable_http(_http_error(500)) is True
    assert mod._is_retryable_http(_http_error(403)) is False  # auth — won't heal
    assert mod._is_retryable_http(_http_error(404)) is False  # not found
    assert mod._is_retryable_http(ValueError("nope")) is False  # no response attr


def test_retries_on_429_then_succeeds(mod):
    waits: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(429)
        return "ok"

    out = mod._with_backoff(fn, what="test", base_seconds=15.0, sleep=waits.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert waits == [15.0, 30.0]  # exponential backoff before attempts 2 and 3


def test_does_not_retry_on_auth_error(mod):
    waits: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _http_error(403)

    with pytest.raises(HfHubHTTPError):
        mod._with_backoff(fn, what="test", sleep=waits.append)
    assert calls["n"] == 1  # raised immediately, no retry
    assert waits == []


def test_exhausts_attempts_then_raises(mod):
    waits: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _http_error(429)

    with pytest.raises(HfHubHTTPError):
        mod._with_backoff(fn, what="test", attempts=4, base_seconds=15.0, sleep=waits.append)
    assert calls["n"] == 4  # all attempts used
    assert waits == [15.0, 30.0, 60.0]  # slept before attempts 2, 3, 4; not after the last


def test_production_freshness_guard_accepts_current_main(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "abc123")

    mod._assert_checkout_is_current_main()


def test_production_freshness_guard_rejects_stale_or_unverifiable(
    mod, monkeypatch
):
    monkeypatch.setattr(mod, "_git_sha", lambda: "old123")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "new456")
    with pytest.raises(RuntimeError, match="stale checkout"):
        mod._assert_checkout_is_current_main()

    monkeypatch.setattr(mod, "_origin_main_sha", lambda: None)
    with pytest.raises(RuntimeError, match="cannot verify"):
        mod._assert_checkout_is_current_main()


def test_expected_head_guard_accepts_matching_checkout(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "a" * 40)

    mod._assert_checkout_is_expected("a" * 40)


def test_expected_head_guard_rejects_mismatch_or_unknown_head(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "b" * 40)
    with pytest.raises(RuntimeError, match="is not the expected head"):
        mod._assert_checkout_is_expected("a" * 40)

    monkeypatch.setattr(mod, "_git_sha", lambda: None)
    with pytest.raises(RuntimeError, match="cannot read checkout HEAD"):
        mod._assert_checkout_is_expected("a" * 40)


def _main_with_argv(mod, monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["publish_registry_data.py", *argv])
    return mod.main()


def test_cli_rejects_malformed_expected_head(mod, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main_with_argv(
            mod,
            monkeypatch,
            "--skip-seed",
            "--require-origin-main-head",
            "--expected-head",
            "not-a-sha",
        )
    assert exc.value.code == 2


def test_cli_rejects_expected_head_without_production_guard(mod, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main_with_argv(
            mod, monkeypatch, "--skip-seed", "--expected-head", "a" * 40
        )
    assert exc.value.code == 2


# ---------- superseded-vs-diverged freshness guard ----------

def test_freshness_guard_returns_current_when_head_matches(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "a" * 40)
    assert mod._assert_checkout_is_current_main() == "current"
    assert mod._assert_checkout_is_current_main(skip_if_superseded=True) == "current"


def test_freshness_guard_skips_only_when_main_moved_forward(mod, monkeypatch):
    """Superseded (our commit is an ancestor of the new head) is a no-op ONLY
    when the caller opted in; a diverged head always raises."""
    monkeypatch.setattr(mod, "_git_sha", lambda: "old" + "0" * 37)
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "new" + "0" * 37)
    seen = []
    monkeypatch.setattr(mod, "_is_ancestor", lambda a, b: (seen.append((a, b)), True)[1])
    assert mod._assert_checkout_is_current_main(skip_if_superseded=True) == "superseded"
    assert seen == [("old" + "0" * 37, "new" + "0" * 37)]
    # Default (no opt-in): stale is still a hard failure, no ancestry probe.
    seen.clear()
    with pytest.raises(RuntimeError, match="stale checkout"):
        mod._assert_checkout_is_current_main()
    assert seen == []
    # Diverged / force-pushed / unfetchable: fail closed even with opt-in.
    monkeypatch.setattr(mod, "_is_ancestor", lambda a, b: False)
    with pytest.raises(RuntimeError, match="stale checkout"):
        mod._assert_checkout_is_current_main(skip_if_superseded=True)


def test_is_ancestor_fails_closed_when_fetch_fails(mod, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise subprocess.CalledProcessError(128, a[0])

    monkeypatch.setattr(mod.subprocess, "run", boom)
    assert mod._is_ancestor("a" * 40, "b" * 40) is False


def test_freshness_check_only_writes_github_output(mod, monkeypatch, tmp_path):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(mod, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "b" * 40)
    monkeypatch.setattr(mod, "_is_ancestor", lambda a, b: True)
    rc = _main_with_argv(
        mod, monkeypatch, "--freshness-check-only", "--require-origin-main-head",
        "--skip-if-superseded",
    )
    assert rc == 0
    assert out.read_text().strip() == "superseded=true"

    out.write_text("")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "a" * 40)
    rc = _main_with_argv(
        mod, monkeypatch, "--freshness-check-only", "--require-origin-main-head",
        "--skip-if-superseded",
    )
    assert rc == 0
    assert out.read_text().strip() == "superseded=false"

    # Diverged → non-zero, and superseded=false so a gated step never runs.
    out.write_text("")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "c" * 40)
    monkeypatch.setattr(mod, "_is_ancestor", lambda a, b: False)
    rc = _main_with_argv(
        mod, monkeypatch, "--freshness-check-only", "--require-origin-main-head",
        "--skip-if-superseded",
    )
    assert rc == 1
    assert out.read_text().strip() == "superseded=false"

    # --expected-head is honoured first: a pinned child that checked out the
    # wrong commit fails even if main is "current".
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "a" * 40)
    rc = _main_with_argv(
        mod, monkeypatch, "--freshness-check-only", "--require-origin-main-head",
        "--skip-if-superseded", "--expected-head", "b" * 40,
    )
    assert rc == 1


def test_cli_rejects_skip_flags_without_production_guard(mod, monkeypatch):
    with pytest.raises(SystemExit):
        _main_with_argv(mod, monkeypatch, "--skip-if-superseded")
    with pytest.raises(SystemExit):
        _main_with_argv(mod, monkeypatch, "--freshness-check-only")
