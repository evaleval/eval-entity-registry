"""Pinned, local snapshot of `cfahlgren1/hub-stats` for the bulk scripts.

Why this exists (see specs/hub-stats-snapshot-fetch.md): the HF parquet-converter
endpoint (`/api/datasets/<repo>/parquet/<config>/<split>/<n>.parquet`, the
`PARQUET_URL` the live `HubStatsClient` still uses) answers 400 for several
minutes after *any* commit to the upstream dataset — the dataset-viewer deletes
and rebuilds the whole `refs/convert/parquet` branch. Upstream commits six files
every day around 13:18–13:36 UTC, and the crons regularly fired inside that
window. The raw `main/models.parquet` is byte-identical to the converted file
and lives on a path the viewer never touches.

The scripts therefore (1) resolve the dataset's current commit sha + the file's
content hash in ONE resolver-bucket request, (2) short-circuit on an unchanged
hash, else (3) `hf_hub_download` the file at that pinned sha (one resumable,
hash-verified stream) and query it locally. Nothing here is used by the
deployed Space.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

HUB_STATS_REPO = "cfahlgren1/hub-stats"
HUB_STATS_FILE = "models.parquet"


@dataclass(frozen=True)
class HubStatsSnapshot:
    repo_id: str
    revision: str  # dataset commit sha (40 hex)
    filename: str
    sha256: str  # LFS content sha256 (== `x-linked-etag`); NOT the xet hash
    size: int
    local_path: Optional[Path] = None  # None until downloaded

    @property
    def short(self) -> str:
        return f"{self.repo_id}@{self.revision[:12]} {self.filename} sha256={self.sha256[:12]} size={self.size}"


def resolve_snapshot(
    repo_id: str = HUB_STATS_REPO,
    filename: str = HUB_STATS_FILE,
    token: Optional[str] = None,
) -> HubStatsSnapshot:
    """One HEAD on `resolve/main/<filename>` → commit sha, LFS sha256, size.
    Raises RuntimeError if the Hub stops reporting either (a layout change we
    want to see loudly, not paper over)."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(repo_id, filename, repo_type="dataset")
    meta = get_hf_file_metadata(url, token=token)
    if not meta.commit_hash or not meta.etag or meta.size is None:
        raise RuntimeError(
            f"hub-stats snapshot: incomplete metadata for {repo_id}/{filename} "
            f"(commit_hash={meta.commit_hash!r}, etag={meta.etag!r}, size={meta.size!r})"
        )
    return HubStatsSnapshot(
        repo_id=repo_id,
        revision=meta.commit_hash,
        filename=filename,
        sha256=meta.etag.strip('"'),
        size=int(meta.size),
    )


def download_snapshot(
    snap: HubStatsSnapshot,
    token: Optional[str] = None,
    cache_dir: Optional[str | os.PathLike] = None,
) -> HubStatsSnapshot:
    """`hf_hub_download` the file at the snapshot's pinned sha. The hub client
    checks free disk space first and xet verifies content by hash; we add a
    size check as a cheap belt-and-braces guard."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            snap.repo_id,
            snap.filename,
            repo_type="dataset",
            revision=snap.revision,
            token=token,
            cache_dir=cache_dir,
        )
    )
    actual = path.stat().st_size
    if actual != snap.size:
        raise RuntimeError(
            f"hub-stats snapshot: downloaded size {actual} != expected {snap.size} for {snap.short}"
        )
    return replace(snap, local_path=path)


def hf_source(snap: HubStatsSnapshot) -> str:
    """DuckDB-native pinned remote source (`hf://datasets/<repo>@<sha>/<file>`)
    for a caller that chooses to range-read instead of downloading. Needs
    `CREATE SECRET (TYPE HUGGINGFACE, TOKEN ...)` for authenticated reads. The
    crons do not use this; they download."""
    return f"hf://datasets/{snap.repo_id}@{snap.revision}/{snap.filename}"


def is_remote_source(source: str) -> bool:
    return source.startswith(("http://", "https://", "hf://"))
