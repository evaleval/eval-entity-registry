"""Dispatch `seed.yml` for a pushed commit and wait for verified completion.

The refresh crons hold `registry-seed-writer` while `seed.yml` holds
`registry-data-publish`. Because GitHub Actions allows only one concurrency
group per job (job-level concurrency overrides workflow-level concurrency),
the parent cron dispatches `seed.yml` and waits here until it finishes,
keeping both locks held so a watermark-only cron commit cannot advance `main`
mid-publish and trigger `--skip-if-superseded`.

Polls `GET /repos/{repo}/actions/runs/{run_id}` directly rather than calling
`gh run watch`, which polls `/runs/{id}` and `/runs/{id}/jobs` without retry
and aborts on read-replica `404`s immediately after dispatch or transient
`5xx` responses mid-run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_API_URL = "https://api.github.com"
DEFAULT_WORKFLOW = "seed.yml"
DEFAULT_REF = "main"
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 5400.0
DEFAULT_MAX_CONSECUTIVE_ERRORS = 12
DISPATCH_ATTEMPTS = 3
DISPATCH_BASE_SECONDS = 5.0
FIND_RUN_ATTEMPTS = 30
FIND_RUN_INTERVAL_SECONDS = 5.0

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

RequestFn = Callable[..., Optional[dict[str, Any]]]


class GitHubAPIError(Exception):
    """HTTP or transport failure talking to the GitHub REST API."""

    def __init__(
        self,
        status_code: Optional[int],
        message: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


class ChildRunFailedError(RuntimeError):
    """The correlated `seed.yml` workflow run finished without `success`."""


def _is_retryable(exc: GitHubAPIError, *, allow_404: bool) -> bool:
    """True for transient GitHub API errors; permanent auth/input errors fail fast.

    `allow_404` is enabled only when polling a run ID just returned by dispatch,
    because GitHub's read replicas lag behind run creation and across non-sticky
    backends. On `POST .../dispatches` or workflow-runs listing, `404` means a
    missing workflow/repo or missing permission and fails immediately.
    """
    code = exc.status_code
    if code is None:
        return True
    if allow_404 and code == 404:
        return True
    if code in (408, 429) or 500 <= code < 600:
        return True
    if code == 403:
        # GitHub signals primary/secondary rate limits via 403 with Retry-After
        # or zero remaining quota; permission errors have neither.
        return (
            "retry-after" in exc.headers
            or exc.headers.get("x-ratelimit-remaining") == "0"
        )
    return False


def _request_json(
    method: str,
    url: str,
    *,
    token: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Optional[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "eval-entity-registry-ci",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(500).decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        resp_headers = dict(exc.headers.items()) if exc.headers else {}
        raise GitHubAPIError(
            exc.code,
            f"HTTP {exc.code} on {method} {url}" + (f": {detail}" if detail else ""),
            headers=resp_headers,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GitHubAPIError(None, f"transport error on {method} {url}: {exc}") from exc

    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubAPIError(None, f"invalid JSON on {method} {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GitHubAPIError(None, f"expected JSON object on {method} {url}")
    return parsed


def _origin_main_sha() -> Optional[str]:
    """Current `origin/main` SHA for beacon-retry dispatches when nothing was pushed."""
    try:
        out = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        parts = out.stdout.split()
        return parts[0] if parts else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _resolve_expected_sha(raw_sha: Optional[str]) -> str:
    """Validate an explicit 40-hex SHA or resolve `origin/main` on beacon retry."""
    candidate = (raw_sha or "").strip()
    if not candidate:
        resolved = _origin_main_sha()
        if not resolved or not _SHA_RE.fullmatch(resolved):
            raise RuntimeError(
                "cannot resolve 40-hex origin/main SHA for beacon-retry dispatch"
            )
        print(
            f"[dispatch] no pushed SHA supplied; resolved origin/main -> {resolved}",
            file=sys.stderr,
        )
        return resolved
    if not _SHA_RE.fullmatch(candidate):
        raise ValueError(f"--expected-sha must be a full 40-hex commit sha, got {candidate!r}")
    return candidate


def _find_run_by_title(
    repo: str,
    workflow: str,
    ref: str,
    expected_title: str,
    *,
    token: str,
    api_url: str,
    request_fn: RequestFn,
) -> Optional[int]:
    wf_q = urllib.parse.quote(workflow, safe="")
    ref_q = urllib.parse.quote(ref, safe="")
    url = (
        f"{api_url.rstrip('/')}/repos/{repo}/actions/workflows/{wf_q}/runs"
        f"?event=workflow_dispatch&branch={ref_q}&per_page=50"
    )
    data = request_fn("GET", url, token=token)
    if not data:
        return None
    runs = data.get("workflow_runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if isinstance(run, dict) and run.get("display_title") == expected_title:
            run_id = run.get("id")
            if isinstance(run_id, int) or (isinstance(run_id, str) and run_id.isdigit()):
                return int(run_id)
    return None


def dispatch_and_resolve_run_id(
    repo: str,
    *,
    workflow: str = DEFAULT_WORKFLOW,
    ref: str = DEFAULT_REF,
    source_run_id: str,
    expected_sha: str,
    token: str,
    api_url: str = DEFAULT_API_URL,
    request_fn: RequestFn = _request_json,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Dispatch `workflow` and return the created `workflow_run_id`.

    Uses `return_run_details=True` so GitHub returns `workflow_run_id` directly,
    with title-correlation fallback for `204` responses or ambiguous transport
    drops after server-side run creation (preventing duplicate dispatches).
    """
    expected_title = f"Seed + publish ({source_run_id})"
    wf_q = urllib.parse.quote(workflow, safe="")
    dispatch_url = f"{api_url.rstrip('/')}/repos/{repo}/actions/workflows/{wf_q}/dispatches"
    payload = {
        "ref": ref,
        "inputs": {
            "source_run_id": source_run_id,
            "expected_sha": expected_sha,
        },
        "return_run_details": True,
    }

    for attempt in range(1, DISPATCH_ATTEMPTS + 1):
        try:
            resp = request_fn("POST", dispatch_url, token=token, payload=payload)
            if resp is not None:
                raw_id = resp.get("workflow_run_id")
                if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
                    return int(raw_id)
            break
        except GitHubAPIError as exc:
            if not _is_retryable(exc, allow_404=False):
                raise
            wait = DISPATCH_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"  [dispatch] transient error (attempt {attempt}/{DISPATCH_ATTEMPTS}): "
                f"{exc}; waiting {wait:.0f}s before checking runs…",
                file=sys.stderr,
            )
            sleep(wait)
            try:
                existing = _find_run_by_title(
                    repo,
                    workflow,
                    ref,
                    expected_title,
                    token=token,
                    api_url=api_url,
                    request_fn=request_fn,
                )
            except GitHubAPIError as lookup_exc:
                if not _is_retryable(lookup_exc, allow_404=False):
                    raise
                existing = None
            if existing is not None:
                print(
                    f"  [dispatch] adopted already-created run {existing} ({expected_title})",
                    file=sys.stderr,
                )
                return existing
            if attempt == DISPATCH_ATTEMPTS:
                raise

    for attempt in range(1, FIND_RUN_ATTEMPTS + 1):
        try:
            run_id = _find_run_by_title(
                repo,
                workflow,
                ref,
                expected_title,
                token=token,
                api_url=api_url,
                request_fn=request_fn,
            )
            if run_id is not None:
                return run_id
        except GitHubAPIError as exc:
            if not _is_retryable(exc, allow_404=False) or attempt == FIND_RUN_ATTEMPTS:
                raise
            print(
                f"  [dispatch] transient error listing runs "
                f"(attempt {attempt}/{FIND_RUN_ATTEMPTS}): {exc}",
                file=sys.stderr,
            )
        if attempt < FIND_RUN_ATTEMPTS:
            sleep(FIND_RUN_INTERVAL_SECONDS)

    raise RuntimeError(
        f"could not locate dispatched {workflow} run titled {expected_title!r}"
    )


def wait_for_run_completion(
    repo: str,
    run_id: int,
    *,
    token: str,
    api_url: str = DEFAULT_API_URL,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
    request_fn: RequestFn = _request_json,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Poll `GET /repos/{repo}/actions/runs/{run_id}` until `completed`.

    Returns `html_url` on `conclusion == "success"`; raises `ChildRunFailedError`
    on any other terminal conclusion, `RuntimeError` after `max_consecutive_errors`
    unbroken transient errors, and `TimeoutError` at `timeout_seconds`.
    """
    url = f"{api_url.rstrip('/')}/repos/{repo}/actions/runs/{run_id}"
    html_url = f"https://github.com/{repo}/actions/runs/{run_id}"
    deadline = clock() + timeout_seconds
    consecutive_errors = 0
    last_state: Optional[tuple[Optional[str], Optional[str]]] = None

    while True:
        try:
            data = request_fn("GET", url, token=token)
            if data is None:
                raise GitHubAPIError(None, f"empty response on GET {url}")
        except GitHubAPIError as exc:
            if not _is_retryable(exc, allow_404=True):
                raise
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"giving up on {html_url} after {consecutive_errors} "
                    f"consecutive API errors (last: {exc})"
                ) from exc
            if clock() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout_seconds:.0f}s waiting for "
                    f"{html_url} (last error: {exc})"
                ) from exc
            print(
                f"  [wait] transient error polling run {run_id} "
                f"({consecutive_errors}/{max_consecutive_errors}): {exc}; "
                f"sleeping {poll_seconds:.0f}s…",
                file=sys.stderr,
            )
            sleep(min(poll_seconds, max(0.0, deadline - clock())))
            continue

        consecutive_errors = 0
        raw_html = data.get("html_url")
        if isinstance(raw_html, str) and raw_html:
            html_url = raw_html
        status = data.get("status")
        conclusion = data.get("conclusion")
        if (status, conclusion) != last_state:
            print(
                f"[wait] run {run_id} status={status} "
                f"conclusion={conclusion or '-'} ({html_url})",
                file=sys.stderr,
            )
            last_state = (status, conclusion)

        if status == "completed":
            if conclusion == "success":
                return html_url
            raise ChildRunFailedError(
                f"correlated seed run {run_id} ended with "
                f"conclusion={conclusion!r}: {html_url}"
            )

        if clock() >= deadline:
            raise TimeoutError(
                f"timed out after {timeout_seconds:.0f}s waiting for "
                f"{html_url} (last status={status!r})"
            )
        sleep(min(poll_seconds, max(0.0, deadline - clock())))


def validate_refresh_workflow(doc: Mapping[str, Any], workflow_name: str) -> None:
    """Enforce the seed-dispatch and stall-beacon invariants on a refresh workflow."""
    jobs = doc.get("jobs")
    if not isinstance(jobs, Mapping):
        raise ValueError(f"{workflow_name}: missing jobs mapping")

    for job_id, job in jobs.items():
        if not isinstance(job, Mapping):
            continue
        for step in job.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            run_cmd = step.get("run")
            if isinstance(run_cmd, str):
                if "gh run watch" in run_cmd:
                    raise ValueError(
                        f"{workflow_name} ({job_id}): forbidden `gh run watch` in step "
                        f"{step.get('name')!r}; use scripts/ci_dispatch_seed.py"
                    )
                if "gh workflow run" in run_cmd:
                    raise ValueError(
                        f"{workflow_name} ({job_id}): forbidden inline `gh workflow run` "
                        f"in step {step.get('name')!r}; use scripts/ci_dispatch_seed.py"
                    )

    refresh = jobs.get("refresh")
    if not isinstance(refresh, Mapping):
        raise ValueError(f"{workflow_name}: missing `refresh` job")
    steps_by_name = {
        s.get("name"): s
        for s in (refresh.get("steps") or [])
        if isinstance(s, Mapping) and isinstance(s.get("name"), str)
    }

    pub = steps_by_name.get("Publish and wait for verified completion")
    if pub is None:
        raise ValueError(
            f"{workflow_name}: missing `Publish and wait for verified completion` step"
        )
    if pub.get("if") != "steps.deploy.outputs.needed == 'true'":
        raise ValueError(
            f"{workflow_name}: publish step must guard on steps.deploy.outputs.needed == 'true'"
        )
    env = pub.get("env")
    if not isinstance(env, Mapping) or "GH_TOKEN" not in env or "EXPECTED_SHA" not in env:
        raise ValueError(
            f"{workflow_name}: publish step must pass GH_TOKEN and EXPECTED_SHA in env"
        )
    if "steps.commit.outputs.pushed_sha" not in str(env.get("EXPECTED_SHA", "")):
        raise ValueError(
            f"{workflow_name}: EXPECTED_SHA must read steps.commit.outputs.pushed_sha"
        )
    pub_run = str(pub.get("run") or "")
    for required in ("scripts/ci_dispatch_seed.py", "--source-run-id", "--expected-sha"):
        if required not in pub_run:
            raise ValueError(
                f"{workflow_name}: publish step run command missing {required!r}"
            )

    clear = steps_by_name.get("Clear stall beacon")
    if clear is None:
        raise ValueError(f"{workflow_name}: missing `Clear stall beacon` step")
    clear_if = str(clear.get("if") or "")
    if "success()" not in clear_if or "steps.deploy.outputs.needed == 'true'" not in clear_if:
        raise ValueError(
            f"{workflow_name}: `Clear stall beacon` must require success() and "
            "steps.deploy.outputs.needed == 'true'"
        )

    raise_step = steps_by_name.get("Raise/refresh stall beacon")
    if raise_step is None:
        raise ValueError(f"{workflow_name}: missing `Raise/refresh stall beacon` step")
    raise_if = str(raise_step.get("if") or "")
    if "failure()" not in raise_if or "cancelled()" not in raise_if:
        raise ValueError(
            f"{workflow_name}: `Raise/refresh stall beacon` must run on failure() || cancelled()"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name repository slug (defaults to $GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help=f"Workflow file or ID to dispatch (default: {DEFAULT_WORKFLOW}).",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help=f"Branch ref to dispatch against (default: {DEFAULT_REF}).",
    )
    parser.add_argument(
        "--source-run-id",
        required=True,
        help="Correlation key passed to seed.yml (e.g. '<run_id>-<attempt>').",
    )
    parser.add_argument(
        "--expected-sha",
        default="",
        help="40-hex commit pushed by caller; empty resolves current origin/main.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help=f"Seconds between run status polls (default: {DEFAULT_POLL_SECONDS:g}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Overall deadline waiting for completion (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_ERRORS,
        help="Abort after this many unbroken transient poll failures.",
    )
    args = parser.parse_args()

    repo = args.repo.strip()
    if not _REPO_RE.fullmatch(repo):
        parser.error("--repo (or $GITHUB_REPOSITORY) must be 'owner/name'")
    source_run_id = args.source_run_id.strip()
    if not source_run_id:
        parser.error("--source-run-id must be non-empty")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0 or args.max_consecutive_errors < 1:
        parser.error("poll-seconds, timeout-seconds, and max-consecutive-errors must be positive")

    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        print("::error::GH_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    try:
        expected_sha = _resolve_expected_sha(args.expected_sha)
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    api_url = os.environ.get("GITHUB_API_URL", DEFAULT_API_URL)
    print(
        f"[dispatch] triggering {args.workflow} on {repo}@{args.ref} "
        f"(source_run_id={source_run_id}, expected_sha={expected_sha})",
        file=sys.stderr,
    )
    try:
        run_id = dispatch_and_resolve_run_id(
            repo,
            workflow=args.workflow,
            ref=args.ref,
            source_run_id=source_run_id,
            expected_sha=expected_sha,
            token=token,
            api_url=api_url,
        )
        html_url = wait_for_run_completion(
            repo,
            run_id,
            token=token,
            api_url=api_url,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
            max_consecutive_errors=args.max_consecutive_errors,
        )
    except (GitHubAPIError, ChildRunFailedError, RuntimeError, TimeoutError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    print(f"[done] verified completion of {html_url}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
