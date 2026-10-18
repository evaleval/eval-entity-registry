"""Publish the registry's canonical_* parquets to evaleval/entity-registry-data.

  1. Run the seed CLI against `seed/`, materialising every canonical_* table
     to `fixtures/`.
  2. Compute a content hash over the published bytes, the idempotency key:
     when the previous publish on HF carries the same hash, skip the upload.
  3. Write a `manifest.json` sidecar: schema_version (registry.<MAJOR>.<MINOR>),
     content_hash, seed_git_sha, generated_at, per-table row counts.
  4. Upload `fixtures/` + `manifest.json` to the HF Dataset repo as a single
     revision.

Usage:
    uv run python scripts/publish_registry_data.py [--dry-run]

`--dry-run` writes the manifest and skips the push. ``--skip-seed`` is for CI
after it has already seeded and tested ``fixtures/``, so the bytes validated by
the workflow are the bytes offered to the Hub.

Consumers assert that `manifest.json.schema_version`'s major matches the major
they expect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

# Major: removed/renamed columns in any canonical_* table; minor: added
# columns. The producer asserts the major matches what it expects.
SCHEMA_VERSION = "registry.3.3"

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed"
FIXTURES_DIR = REPO_ROOT / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
HF_DATASET_REPO = "evaleval/entity-registry-data"

# Published by their own cron, never by this flat publish; a local build would
# otherwise leak a stray copy. Mirrors hf_store._CRON_OWNED_TABLES.
_CRON_OWNED_PARQUETS = {"hub_stats_index.parquet"}

# Shipped alongside the parquets so the producer needs no registry checkout.
SHIPPED_YAML = ("slice_overrides.yaml", "display_overrides.yaml")


def _git_sha() -> Optional[str]:
    """HEAD SHA of the registry repo; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _origin_main_sha() -> Optional[str]:
    """Current ``origin/main`` SHA read from the remote; ``None`` when it cannot
    be read (callers fail closed)."""
    try:
        out = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.split()[0] if out.stdout.split() else None
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def _is_ancestor(local_sha: str, remote_sha: str) -> bool:
    """True when ``remote_sha`` descends from ``local_sha``; False when history
    cannot be fetched. CI checkouts are shallow, so the remote commit is fetched
    first with enough depth to reach our checkout."""
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "--depth=200", "origin", remote_sha],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        rc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", local_sha, remote_sha],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).returncode
        return rc == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _assert_checkout_is_current_main(skip_if_superseded: bool = False) -> str:
    """Freshness guard: ``"current"`` when HEAD is origin/main; with
    ``skip_if_superseded``, ``"superseded"`` when origin/main moved forward past
    HEAD (safe to skip: every forward move of main carries its own publish of a
    superset of this tree). Any other mismatch raises, so a delayed older run
    can never roll dataset HEAD back."""
    local_sha = _git_sha()
    remote_sha = _origin_main_sha()
    if not local_sha or not remote_sha:
        raise RuntimeError(
            "cannot verify checkout freshness against origin/main; refusing publish"
        )
    if local_sha == remote_sha:
        return "current"
    if skip_if_superseded and _is_ancestor(local_sha, remote_sha):
        return "superseded"
    raise RuntimeError(
        f"stale checkout {local_sha} is not current origin/main {remote_sha}; "
        "refusing publish"
    )


def _write_github_output(key: str, value: str) -> None:
    """Append ``key=value`` to ``$GITHUB_OUTPUT`` when running inside a step."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")


def _assert_checkout_is_expected(expected: str) -> None:
    """Re-assert the caller's pinned SHA at publish time, in case the dispatched
    run was resolved to a different commit than the one it was asked for."""
    local_sha = _git_sha()
    if not local_sha:
        raise RuntimeError(
            f"cannot read checkout HEAD to compare against expected {expected}; "
            "refusing publish"
        )
    if local_sha != expected:
        raise RuntimeError(
            f"checkout {local_sha} is not the expected head {expected}; "
            "refusing publish"
        )


def _content_hash(fixtures_dir: Path) -> str:
    """SHA-256 over the sorted published files (parquets + YAML overrides).
    manifest.json is excluded because it contains the hash; file names are
    hashed with the bytes so an added/missing file changes the hash."""
    h = hashlib.sha256()
    files = sorted(
        [p for p in fixtures_dir.glob("*.parquet") if p.name not in _CRON_OWNED_PARQUETS]
        + [p for p in fixtures_dir.glob("*.yaml") if p.name != "manifest.json"]
    )
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _row_counts(fixtures_dir: Path) -> dict[str, int]:
    """Per-table row counts for the manifest, from parquet metadata only."""
    import pyarrow.parquet as pq

    out: dict[str, int] = {}
    for p in sorted(fixtures_dir.glob("*.parquet")):
        if p.name in _CRON_OWNED_PARQUETS:
            continue
        try:
            md = pq.read_metadata(p)
            out[p.stem] = md.num_rows
        except Exception as exc:
            # A broken parquet must not block the publish.
            print(f"  [warn] couldn't read row count for {p.name}: {exc}", file=sys.stderr)
    return out


def _run_seed() -> int:
    """Run the seed CLI in LOCAL_MODE so it writes fixtures/ without pushing."""
    env = dict(os.environ)
    env["LOCAL_MODE"] = "true"
    proc = subprocess.run(
        ["uv", "run", "eval-entity-registry", "seed", "--local"],
        cwd=REPO_ROOT,
        env=env,
    )
    return proc.returncode


_T = TypeVar("_T")


def _is_retryable_http(exc: Exception) -> bool:
    """True for transient HF errors (429, 5xx); 401/403/404 do not heal on
    retry and propagate immediately."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code == 429 or (isinstance(code, int) and 500 <= code < 600)


def _with_backoff(
    fn: Callable[[], _T],
    *,
    what: str,
    attempts: int = 5,
    base_seconds: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run fn(), retrying transient HF errors with a long exponential backoff
    (15s, 30s, 60s, …). huggingface_hub's own retry caps at 8s and gives up
    after a few tries, too shallow for a sustained 429 burst."""
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except HfHubHTTPError as exc:
            if attempt == attempts or not _is_retryable_http(exc):
                raise
            wait = base_seconds * (2 ** (attempt - 1))
            print(
                f"  [retry] {what}: {type(exc).__name__} "
                f"(attempt {attempt}/{attempts}); waiting {wait:.0f}s…",
                file=sys.stderr,
            )
            sleep(wait)
    # Unreachable: the final attempt either returns or raises above.
    raise RuntimeError(f"{what}: retry loop exited without a result")


def _read_remote_manifest() -> Optional[dict]:
    """The remote manifest.json, or None when the dataset or manifest does not
    exist yet or any HF call fails."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return None

    try:
        local = _with_backoff(
            lambda: hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename="manifest.json",
                repo_type="dataset",
            ),
            what="read remote manifest",
        )
        with open(local) as f:
            return json.load(f)
    except (
        RepositoryNotFoundError,
        EntryNotFoundError,
        HfHubHTTPError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        # Transient errors were already retried; failing open here means we
        # publish rather than skip.
        print(f"  [info] no remote manifest available ({type(exc).__name__}); will publish",
              file=sys.stderr)
        return None


def _push(fixtures_dir: Path, manifest: dict) -> None:
    """Upload fixtures/ + manifest.json to HF as a single dataset revision; the
    commit message carries the seed git SHA so consumers can pin to it."""
    from huggingface_hub import HfApi

    api = HfApi()
    sha = manifest.get("seed_git_sha") or "unknown"
    commit_msg = f"Publish registry data (seed @ {sha[:8] if sha != 'unknown' else 'unknown'})"
    api.upload_folder(
        folder_path=str(fixtures_dir),
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
        allow_patterns=["*.parquet", "manifest.json", "*.yaml"],
        ignore_patterns=sorted(_CRON_OWNED_PARQUETS),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run seed + write manifest locally; skip HF push. Used in PR CI.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Use already materialized fixtures (CI only; avoids re-seeding after tests).",
    )
    parser.add_argument(
        "--require-origin-main-head",
        action="store_true",
        help="Fail unless checkout HEAD is the current remote main (production CI).",
    )
    parser.add_argument(
        "--expected-head",
        metavar="SHA",
        help="Fail unless checkout HEAD is this exact 40-hex commit (the SHA the "
             "dispatching workflow pushed and asked the child to publish).",
    )
    parser.add_argument(
        "--skip-if-superseded",
        action="store_true",
        help="With --require-origin-main-head: when origin/main has moved FORWARD "
             "past the checkout, exit 0 without publishing (the newer head's own "
             "publish supersedes this one) instead of failing. A diverged or "
             "unverifiable head still fails.",
    )
    parser.add_argument(
        "--freshness-check-only",
        action="store_true",
        help="Run only the head guards (--expected-head / --require-origin-main-head) "
             "and exit; writes superseded=true|false to $GITHUB_OUTPUT. Lets CI skip "
             "the seed + test minutes for a run that is already superseded.",
    )
    args = parser.parse_args()

    if args.expected_head is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_head):
            parser.error("--expected-head must be a full 40-hex commit sha")
        if not args.require_origin_main_head:
            parser.error("--expected-head requires --require-origin-main-head")
    if (args.skip_if_superseded or args.freshness_check_only) and not args.require_origin_main_head:
        parser.error("--skip-if-superseded / --freshness-check-only require --require-origin-main-head")

    if args.freshness_check_only:
        try:
            if args.expected_head:
                _assert_checkout_is_expected(args.expected_head)
            verdict = _assert_checkout_is_current_main(skip_if_superseded=args.skip_if_superseded)
        except RuntimeError as exc:
            print(f"[freshness] [error] {exc}", file=sys.stderr)
            _write_github_output("superseded", "false")
            return 1
        superseded = verdict == "superseded"
        _write_github_output("superseded", "true" if superseded else "false")
        if superseded:
            print(
                f"[freshness] checkout {_git_sha()} is superseded by origin/main "
                f"{_origin_main_sha()}; that head's own publish covers this tree — skipping.",
                file=sys.stderr,
            )
        else:
            print("[freshness] checkout is current origin/main.", file=sys.stderr)
        return 0

    if args.skip_seed:
        print("[1/4] Using pre-seeded fixtures…", file=sys.stderr)
    else:
        print("[1/4] Running seed CLI…", file=sys.stderr)
        rc = _run_seed()
        if rc != 0:
            print(f"seed failed with exit code {rc}", file=sys.stderr)
            return rc

    if not FIXTURES_DIR.is_dir():
        print(f"fixtures/ missing after seed; expected {FIXTURES_DIR}", file=sys.stderr)
        return 1

    print("[2/4] Staging YAML overrides into fixtures/…", file=sys.stderr)
    import shutil as _shutil
    for name in SHIPPED_YAML:
        src = SEED_DIR / name
        if src.exists():
            _shutil.copy2(src, FIXTURES_DIR / name)
            print(f"  staged: {name}", file=sys.stderr)
        else:
            print(f"  [warn] {src} missing; consumers will fall back to defaults",
                  file=sys.stderr)

    print("[2/4] Computing content hash…", file=sys.stderr)
    content_hash = _content_hash(FIXTURES_DIR)
    print(f"  content_hash: {content_hash[:16]}…", file=sys.stderr)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "content_hash": content_hash,
        "seed_git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_counts": _row_counts(FIXTURES_DIR),
    }

    print("[3/4] Writing manifest.json…", file=sys.stderr)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  schema_version: {SCHEMA_VERSION}", file=sys.stderr)
    print(f"  row_counts: {sum(manifest['row_counts'].values())} rows across "
          f"{len(manifest['row_counts'])} tables", file=sys.stderr)

    if args.dry_run:
        print("[4/4] --dry-run: skipping HF upload.", file=sys.stderr)
        return 0

    if args.require_origin_main_head:
        if args.expected_head:
            print("[4/4] Verifying checkout is the expected head…", file=sys.stderr)
            try:
                _assert_checkout_is_expected(args.expected_head)
            except RuntimeError as exc:
                print(f"  [error] {exc}", file=sys.stderr)
                return 1
        print("[4/4] Verifying checkout is current origin/main…", file=sys.stderr)
        try:
            verdict = _assert_checkout_is_current_main(
                skip_if_superseded=args.skip_if_superseded
            )
        except RuntimeError as exc:
            print(f"  [error] {exc}", file=sys.stderr)
            return 1
        if verdict == "superseded":
            print(
                f"  superseded: origin/main moved forward to {_origin_main_sha()} during "
                "this run; that head's own publish covers this tree — skipping upload.",
                file=sys.stderr,
            )
            return 0

    print("[4/4] Checking remote manifest for idempotency…", file=sys.stderr)
    remote = _read_remote_manifest()
    if remote and remote.get("seed_git_sha") == manifest.get("seed_git_sha"):
        print("  remote seed_git_sha matches local; skipping push (no-op).",
              file=sys.stderr)
        return 0
    if remote and remote.get("content_hash") == content_hash:
        print("  remote content_hash matches local; skipping push (no-op).",
              file=sys.stderr)
        return 0

    print(f"  pushing to {HF_DATASET_REPO}…", file=sys.stderr)
    _with_backoff(lambda: _push(FIXTURES_DIR, manifest), what="upload to HF")
    print("  done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
