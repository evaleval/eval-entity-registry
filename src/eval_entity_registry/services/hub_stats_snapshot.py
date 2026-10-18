"""Pinned snapshot of `cfahlgren1/hub-stats` `main/models.parquet`.

All hub-stats readers go through here instead of the HF parquet-converter
endpoint (`/api/datasets/<repo>/parquet/...`): the dataset-viewer rebuilds that
endpoint after every upstream commit and it answers 400 meanwhile, which hit
the daily crons. The raw file is byte-identical and never rebuilt.

`resolve_snapshot` pins a commit sha and the file's content hash in one
request; `download_snapshot` fetches that exact revision for local queries;
`hf_source` is the equivalent pinned `hf://` path for a caller that prefers
range reads.
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
    Raises RuntimeError if any of them is missing."""
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
    """`hf_hub_download` the file at the snapshot's pinned sha, then check its
    size against the snapshot."""
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
    """Pinned DuckDB source `hf://datasets/<repo>@<sha>/<file>`. Authenticated
    reads need `CREATE SECRET (TYPE HUGGINGFACE, TOKEN ...)`."""
    return f"hf://datasets/{snap.repo_id}@{snap.revision}/{snap.filename}"


def is_remote_source(source: str) -> bool:
    return source.startswith(("http://", "https://", "hf://"))
