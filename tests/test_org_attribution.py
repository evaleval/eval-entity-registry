"""Org attribution for malformed-org draft ids (no '/', org glued by - or .):
standalone drafts get org_id; drafts duplicating a real repo come back as a
merge map."""
from eval_entity_registry.lib.org_attribution import attribute_orgs

HF_TO_DEV = {"cohere": "cohere", "deepseek": "deepseek", "nvidia": "nvidia",
             "writer": "writer", "anthropic": "anthropic", "qwen": "alibaba"}


def _e(id, org_id=None, aliases=None):
    return {"id": id, "org_id": org_id, "aliases": aliases or []}


def test_standalone_draft_gets_org_no_merge():
    e = _e("cohere-march-2024")
    _, merges = attribute_orgs([e], HF_TO_DEV)
    assert e["org_id"] == "cohere" and e["id"] == "cohere-march-2024"
    assert merges == {}


def test_deepseek_trap_attributes_org_keeps_id():
    e = _e("deepseek-coder")  # no exact twin: attribute only, keep the id
    _, merges = attribute_orgs([e], HF_TO_DEV)
    assert e["org_id"] == "deepseek" and e["id"] == "deepseek-coder"
    assert merges == {}


def test_real_repo_dup_returns_merge():
    draft = _e("deepseek-v2-lite-chat")
    real = _e("deepseek-ai/DeepSeek-V2-Lite-Chat", org_id="deepseek")
    _, merges = attribute_orgs([draft, real], HF_TO_DEV)
    assert merges == {"deepseek-v2-lite-chat": "deepseek-ai/DeepSeek-V2-Lite-Chat"}
    assert draft.get("org_id") is None  # folded, not attributed


def test_dot_namespace_dup_detected():
    draft = _e("nvidia.nemotron-nano-9b-v2")
    real = _e("nvidia/NVIDIA-Nemotron-Nano-9B-v2", org_id="nvidia")
    _, merges = attribute_orgs([draft, real], HF_TO_DEV)
    assert merges == {"nvidia.nemotron-nano-9b-v2": "nvidia/NVIDIA-Nemotron-Nano-9B-v2"}


def test_unknown_leading_token_left_alone():
    e = _e("randomthing-foo-bar")
    _, merges = attribute_orgs([e], HF_TO_DEV)
    assert e["org_id"] is None and merges == {}


def test_existing_org_not_overwritten():
    e = _e("meta-llama-3", org_id="meta")
    _, merges = attribute_orgs([e], {"meta": "meta", **HF_TO_DEV})
    assert e["org_id"] == "meta"
