"""The shared org-aware fold (eval_entity_resolver.fold.decide_fold) used by
the generators and the gate. Offline, pure-function.
"""
from __future__ import annotations

from eval_entity_resolver.fold import build_hf_index, decide_fold, name_norm

# HF org spelling -> developer slug, as the generators build it.
HF_TO_DEV = {
    "coherelabs": "cohere", "cohere": "cohere",
    "qwen": "alibaba", "alibaba": "alibaba",
    "meta-llama": "meta", "meta": "meta",
    "huggingfacetb": "huggingface",
}

# A small real-HF universe (resolution_source='hf' = a real repo canonical).
HF_ENTRIES = [
    {"id": "CohereLabs/c4ai-command-r-v01", "resolution_source": "hf"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "resolution_source": "hf"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "resolution_source": "hf"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B", "resolution_source": "hf"},
]


def _idx():
    return build_hf_index(HF_ENTRIES, HF_TO_DEV)


def test_name_norm_collapses_all_separators_to_one_token():
    # A mangled models.dev spelling equals the HF casing under name_norm.
    assert name_norm("qwen-2-5-14b-instruct") == name_norm("Qwen2.5-14B-Instruct")
    assert name_norm("c4ai-command-r-v01") == name_norm("c4ai-command-r-v01")


def test_dev_org_slug_folds_to_real_hf_repo():
    hf_ids, a2c, by_org, _ = _idx()
    # Org agreement after remap: the dev-org mint folds onto the real repo.
    for cid, org, target in [
        ("cohere/c4ai-command-r-v01", "cohere", "CohereLabs/c4ai-command-r-v01"),
        ("alibaba/qwen-2-5-14b-instruct", "alibaba", "Qwen/Qwen2.5-14B-Instruct"),
        ("huggingface/smollm2-1.7b", "huggingface", "HuggingFaceTB/SmolLM2-1.7B"),
    ]:
        f = decide_fold({"id": cid, "org_id": org, "aliases": []}, hf_ids, a2c, by_org, HF_TO_DEV)
        assert f is not None and f["hf_target"] == target, f"{cid} should fold to {target}, got {f}"


def test_no_fold_without_org_agreement():
    hf_ids, a2c, by_org, _ = _idx()
    # Same name, different developer: no cross-vendor merge.
    f = decide_fold(
        {"id": "openai/c4ai-command-r-v01", "org_id": "openai", "aliases": []},
        hf_ids, a2c, by_org, HF_TO_DEV,
    )
    assert f is None


def test_no_fold_for_a_genuinely_different_size():
    hf_ids, a2c, by_org, _ = _idx()
    # A different size is a different model.
    f = decide_fold(
        {"id": "alibaba/qwen-2-5-72b-instruct", "org_id": "alibaba", "aliases": []},
        hf_ids, a2c, by_org, HF_TO_DEV,
    )
    assert f is None


def test_alias_tier_does_not_merge_across_distinct_developers():
    # A bare alias legitimately shared by two developers (google's gemma and an
    # unsloth re-upload) must not fold one into the other via the alias tier.
    h2d = {**HF_TO_DEV, "google": "google", "unsloth": "unsloth"}
    hf_entries = HF_ENTRIES + [
        {"id": "google/gemma-3-4b-it", "resolution_source": "hf", "aliases": ["gemma-3-4b-it"]},
    ]
    hf_ids, a2c, by_org, _ = build_hf_index(hf_entries, h2d)
    assert a2c.get("gemma-3-4b-it") == "google/gemma-3-4b-it"
    f = decide_fold(
        {"id": "unsloth/gemma-3-4b-it", "org_id": "unsloth", "aliases": ["gemma-3-4b-it"]},
        hf_ids, a2c, by_org, h2d,
    )
    assert f is None, f"unsloth must not fold into google via a shared bare alias, got {f}"
    # A same-developer alias link still folds.
    g = decide_fold(
        {"id": "google/gemma-3-4b-it-bf16", "org_id": "google", "aliases": ["gemma-3-4b-it"]},
        hf_ids, a2c, by_org, h2d,
    )
    assert g is not None and g["hf_target"] == "google/gemma-3-4b-it"
