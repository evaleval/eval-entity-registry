"""scripts/generate_hf_oracle_seed.py: an hf_oracle mint's id is the real HF
repo id, marked `metadata.hf_id == id`, and no pass may rewrite or suppress it
in favour of a colliding curated slug. Offline; main() is never run.
"""
from __future__ import annotations

import json

from conftest import load_script_module


def _load_module():
    return load_script_module("generate_hf_oracle_seed", "gen_hf_oracle_core_guard")


def test_authoritative_metadata_marks_repo_as_its_own_hf_id():
    mod = _load_module()
    meta = json.loads(mod.authoritative_metadata("meta-llama/Llama-2-7b"))
    assert meta["hf_id"] == "meta-llama/Llama-2-7b"


def test_canon_id_returns_the_real_hf_repo_verbatim():
    """The org is folded into org_id only, never into the id."""
    mod = _load_module()
    hf_to_dev = {"meta-llama": "meta", "qwen": "alibaba"}
    cid, org_id = mod.canon_id("meta-llama/Llama-2-7b", hf_to_dev)
    assert cid == "meta-llama/Llama-2-7b"
    assert org_id == "meta"
    cid2, org2 = mod.canon_id("Qwen/Qwen2.5-7B", hf_to_dev)
    assert cid2 == "Qwen/Qwen2.5-7B"
    assert org2 == "alibaba"


def test_mint_id_equals_its_metadata_hf_id_so_it_is_never_rewritten():
    """id == metadata.hf_id is the authoritative marker downstream consumers
    key on; mirrors the mint construction in main()."""
    mod = _load_module()
    tgt, _org = mod.canon_id("meta-llama/Meta-Llama-3-8B-Instruct", {"meta-llama": "meta"})
    entry = {"id": tgt, "metadata": mod.authoritative_metadata(tgt)}
    assert json.loads(entry["metadata"])["hf_id"] == entry["id"]


def test_nearmiss_separator_rename_is_aliased_but_cross_uploader_is_blocked():
    """A separator/case rename of the same uploader is aliased; a cross-uploader
    migration or size change is identity-changing and blocked."""
    mod = _load_module()
    hf_to_dev = {"meta-llama": "meta", "qwen": "alibaba"}
    assert mod.nearmiss_changes_identity(
        "DeepAutoAI/Explore_Llama-3.1-8B-Inst",
        "DeepAuto-AI/Explore_Llama-3.1-8B-Inst", hf_to_dev) is None
    assert mod.nearmiss_changes_identity(
        "AI4free/Dhanishtha", "HelpingAI/Dhanishtha", hf_to_dev) is not None
    assert mod.nearmiss_changes_identity(
        "foo/model-7b", "foo/model-13b", hf_to_dev) is not None


def test_no_mint_suppressor_against_core():
    """No reconcile pass may suppress an authoritative mint over a normalized
    collision with a curated slug; pinned by the absence of the hook."""
    mod = _load_module()
    assert not hasattr(mod, "reconcile_mints_against_core"), (
        "do not add a reconcile pass that suppresses authoritative HF mints "
        "(id == metadata.hf_id) in favor of a colliding curated slug"
    )
