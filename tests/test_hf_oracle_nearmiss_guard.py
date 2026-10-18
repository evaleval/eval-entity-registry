"""The near-miss identity guard in generate_hf_oracle_seed.py: an HF redirect
that changes model identity (size / cross-developer) must never become a
confirmed alias. The generator itself is not run."""
from conftest import load_script_module


def _mod():
    return load_script_module("generate_hf_oracle_seed", "gen_hf_oracle")


def test_nearmiss_identity_guard():
    m = _mod()
    g = m.nearmiss_changes_identity
    h = {"meta-llama": "meta", "facebook": "meta", "cohereforai": "cohere"}

    # identity-changing: blocked
    assert g("pankajmathur/orca_mini_v9_2_14B", "pankajmathur/orca_mini_phi-4", h)  # denylist
    assert g("foo/bar-3b", "foo/bar-7b", h)            # genuine size change 3B->7B
    assert g("aorg/some-model", "borg/some-model", h)  # cross-developer org

    # same model / benign normalisation: allowed
    assert g("meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-3.1-8B", h) is None   # prefix drop
    assert g("Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-Instruct-2507", h) is None  # MoE active-param, not a 2nd size
    assert g("x/Yi-1.5-34B", "x/Yi-1.5-34B-Chat", h) is None                          # version token, not a size
    assert g("unknown/yi-vl", "01-ai/yi-vl", h) is None                               # org-less placeholder -> real org is fine
    assert g("meta-llama/foo-7b", "facebook/foo-7b", h) is None                       # curated org fold (meta-llama==facebook==meta)

    # The denylist sidecar must actually load.
    assert len(m._AUDIT_BAD_NEARMISS) >= 50
