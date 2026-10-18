"""Seed-time normalize-collision fold: the SAME model minted under different
separator spellings collapses into ONE canonical, with size-conflict and curated
guards against false merges."""
import json

from eval_entity_registry.lib.collision_fold import fold_collisions, collision_key


def _e(id, source=None, aliases=None, parents=None, **kw):
    return {"id": id, "resolution_source": source, "aliases": aliases or [],
            "parents": parents, **kw}


def test_folds_separator_spellings_into_one_winner():
    entries = [_e("google/gemini-1.5-pro", "inferred", ["raw-a"]),
               _e("google/gemini-1-5-pro", "inferred", ["raw-b"])]
    out, remap = fold_collisions(entries)
    assert len(out) == 1
    w = out[0]
    # dotted spelling wins; loser id + both raw aliases land on it
    assert w["id"] == "google/gemini-1.5-pro"
    assert remap == {"google/gemini-1-5-pro": "google/gemini-1.5-pro"}
    assert "google/gemini-1-5-pro" in w["aliases"]
    assert {"raw-a", "raw-b"} <= set(w["aliases"])


def test_fold_keeps_loser_display_name_as_alias():
    """The loser's display_name is a loader-promoted alias, so folding must
    carry it onto the winner rather than delete a resolvable surface form."""
    entries = [
        _e("google/veo3-1", "models_dev", display_name="veo3-1"),
        _e("google/veo-3-1", "models_dev", ["veo-3-1"], display_name="Veo-3.1-Fast"),
    ]
    out, remap = fold_collisions(entries)
    assert len(out) == 1
    w = out[0]
    assert remap == {"google/veo-3-1": "google/veo3-1"}
    assert {"google/veo-3-1", "veo-3-1", "Veo-3.1-Fast"} <= set(w["aliases"])


def test_real_hf_spelling_wins_over_inferred():
    entries = [_e("abacus-ai/llama3-smaug-8b", "inferred"),
               _e("abacusai/Llama-3-Smaug-8B", "hf")]
    out, remap = fold_collisions(entries)
    assert out[0]["id"] == "abacusai/Llama-3-Smaug-8B"  # hf source wins
    assert remap == {"abacus-ai/llama3-smaug-8b": "abacusai/Llama-3-Smaug-8B"}


def test_size_conflict_is_not_folded():
    # 1.3B params != 13B params — different models, must stay separate.
    entries = [_e("facebook/opt-1.3b", "hf"), _e("facebook/opt-13b", "inferred")]
    out, remap = fold_collisions(entries)
    assert len(out) == 2 and remap == {}


def test_never_fold_override_respected():
    entries = [_e("minimax/minimax-m2.7"), _e("minimax/minimax-m27", "models_dev")]
    out, _ = fold_collisions(entries, never_fold=[["minimax/minimax-m2.7", "minimax/minimax-m27"]])
    assert len(out) == 2


def test_prefer_pins_winner():
    entries = [_e("openai/gpt-5.4", None), _e("openai/gpt-54", "models_dev")]
    key = collision_key("openai/gpt-5.4")
    out, remap = fold_collisions(entries, prefer={key: "openai/gpt-54"})
    assert out[0]["id"] == "openai/gpt-54"


def test_malformed_org_and_placeholder_fold_into_real_org():
    # Org-less, placeholder-org, and real-org spellings collapse onto the real org.
    entries = [
        _e("perplexity-sonar-reasoning", "inferred"),               # org-less
        _e("unknown/perplexity-sonar-reasoning", None, ["raw-x"]),  # placeholder org
        _e("perplexity/sonar-reasoning", "inferred"),               # real org -> winner
    ]
    out, remap = fold_collisions(entries)
    assert len(out) == 1
    w = out[0]
    assert w["id"] == "perplexity/sonar-reasoning"
    assert set(remap) == {"perplexity-sonar-reasoning", "unknown/perplexity-sonar-reasoning"}
    assert {"perplexity-sonar-reasoning", "unknown/perplexity-sonar-reasoning", "raw-x"} <= set(w["aliases"])


def test_singleton_malformed_org_is_left_alone():
    # No real-org twin exists, so nothing folds.
    entries = [_e("deepseek-coder", "inferred"), _e("deepseek/deepseek-v2", "inferred")]
    out, remap = fold_collisions(entries)
    assert len(out) == 2 and remap == {}


def test_parent_edges_repointed_to_winner():
    # A third model parented on the LOSER id must repoint to the winner.
    child = _e("org/child-snap-2025-01-01", "inferred",
               parents=json.dumps([{"id": "org/model-1-5", "relationship": "variant", "axis": "version"}]))
    entries = [_e("org/model-1.5", "inferred"), _e("org/model-1-5", "inferred"), child]
    out, remap = fold_collisions(entries)
    assert remap == {"org/model-1-5": "org/model-1.5"}
    child_out = next(e for e in out if e["id"] == "org/child-snap-2025-01-01")
    edges = json.loads(child_out["parents"])
    assert edges[0]["id"] == "org/model-1.5"  # repointed loser -> winner
