"""Shared seed-YAML I/O helpers for the generator/refresh scripts, kept in
the installed package so candidate loading and org-map construction cannot
drift between them."""
from __future__ import annotations

from pathlib import Path

import yaml

# Scalar fields a generated enrich record may donate under its `weak:` map.
# The loader applies them last, filling only fields every full entry left
# empty and core.yaml does not carry (even as null). Shared so the generator
# and the loader agree on the field set.
WEAK_SCALAR_FIELDS = (
    "release_date",
    "open_weights",
    "params_billions",
    "input_modalities",
    "output_modalities",
    "architecture",
)


def safe_load_yaml(text: str):
    """Parse trusted seed YAML with ``CSafeLoader`` when libyaml is available.

    ``yaml.safe_load`` always picks the Python loader; the refresh suite reads
    these files dozens of times and the C loader cuts minutes from it.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(text, Loader=loader)


def resolve_oracle_path(name: str = "hf_model_id_resolution.json") -> Path:
    """Absolute path to a frozen curation input (the HF oracle JSON by default).

    Prefers the committed copy under ``curation/`` (the only one present in a
    single-repo CI checkout), falling back to the workspace parent for local
    dev. Gate suite and generators both call this so the order cannot drift."""
    repo_root = Path(__file__).resolve().parents[3]
    tracked = repo_root / "curation" / name
    return tracked if tracked.exists() else repo_root.parent / name


def load_entries_from_yaml(path: Path) -> list[dict]:
    """Entry list from a seed-model YAML file: a flat list or the
    ``{skip_ids, skip_source_ids, entries}`` dict shape. ``[]`` for a missing file."""
    if not path.exists():
        return []
    raw = safe_load_yaml(path.read_text()) or []
    return (raw.get("entries") if isinstance(raw, dict) else raw) or []


def build_hf_to_dev_from_orgs_yaml(orgs_path: Path) -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug via the shared
    ``eval_entity_resolver.fold.build_curated_org_map``. The alias tier (not
    just ``hf_org``) is what folds e.g. minimaxai->minimax, ai2->allenai."""
    from eval_entity_resolver.fold import build_curated_org_map

    orgs = safe_load_yaml(orgs_path.read_text()) or [] if orgs_path.exists() else []
    return build_curated_org_map(orgs)
