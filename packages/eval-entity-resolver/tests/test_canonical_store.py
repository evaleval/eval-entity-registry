"""CanonicalStore enrichment: a Resolver given a CanonicalStore returns the
same rich fields the HTTP API does (leaf id, group root, parents, lineage,
open_weights, release_date, params_billions, ancestry, resolution_detail)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import uuid

import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, CanonicalStore, Resolver, ResolverConfig
from eval_entity_resolver.alias_store import _empty_df as _empty_alias_df


def _alias_store(*rows) -> AliasStore:
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for raw_value, entity_type, canonical_id, source_config, status in rows:
        records.append({
            "id": str(uuid.uuid4()),
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": None,
            "status": status,
            "strategy": "confirmed",
            "confidence": 1.0,
            "notes": None,
            "created_at": now,
            "updated_at": now,
        })
    df = pd.DataFrame(records) if records else _empty_alias_df()
    return AliasStore(df)


def _models_df(*rows) -> pd.DataFrame:
    """canonical_models DataFrame with the columns the resolver reads;
    unspecified fields are None."""
    cols = [
        "id", "display_name", "org_id", "parents", "model_group_id",
        "lineage_origin_model_org_id", "open_weights", "release_date",
        "params_billions", "tags", "metadata", "review_status",
    ]
    out = []
    for r in rows:
        row = {c: r.get(c) for c in cols}
        out.append(row)
    return pd.DataFrame(out)


# ---------- bare resolver (no canonical_store) — fields stay None ----------

def test_bare_resolver_returns_only_basic_fields():
    """Without a CanonicalStore every enrichment field is None."""
    store = _alias_store(("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"))
    resolver = Resolver(store)  # no canonical_store
    r = resolver.resolve("meta/llama-3-8b", "model")
    assert r.canonical_id == "meta/llama-3-8b"
    assert r.strategy == "exact"
    assert r.review_status is None
    assert r.parents is None
    assert r.open_weights is None
    assert r.release_date is None


# ---------- enriched resolver matches API shape ----------

def test_enriched_resolver_populates_model_metadata():
    """With a CanonicalStore the result carries the matched row's metadata."""
    aliases = _alias_store(("Llama-3.1-8B", "model", "meta/llama-3.1-8b", None, "confirmed"))
    canonicals = CanonicalStore(models_df=_models_df({
        "id": "meta/llama-3.1-8b",
        "display_name": "Llama 3.1 8B",
        "org_id": "meta",
        "parents": json.dumps([{"id": "meta/llama-3.1", "relationship": "variant", "axis": "size"}]),
        "lineage_origin_model_org_id": "meta",
        "open_weights": True,
        "release_date": "2024-07-18",
        "params_billions": 8.0,
        "review_status": "reviewed",
    }))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("Llama-3.1-8B", "model")
    assert r.canonical_id == "meta/llama-3.1-8b"
    assert r.review_status == "reviewed"
    assert r.parent_canonical_id == "meta/llama-3.1"  # variant edge
    assert r.parents == [{"id": "meta/llama-3.1", "relationship": "variant", "axis": "size"}]
    assert r.lineage_origin_org_id == "meta"
    assert r.open_weights is True
    assert r.release_date == "2024-07-18"
    assert r.params_billions == 8.0


def test_enriched_resolver_flips_quantized_chain_to_leaf_with_group_root():
    """A quantized leaf resolves to the LEAF as `canonical_id`; the identity
    root lives in `model_group_id` (and `root_model_id`), and metadata reads
    off the leaf row."""
    aliases = _alias_store(
        ("meta/llama-3.1-8b-instruct-turbo", "model", "meta/llama-3.1-8b-instruct-turbo", None, "confirmed"),
    )
    canonicals = CanonicalStore(models_df=_models_df(
        {
            "id": "meta/llama-3.1-8b-instruct",
            "org_id": "meta", "parents": "[]", "open_weights": True,
            "release_date": "2024-07-18", "params_billions": 8.0,
            "lineage_origin_model_org_id": "meta", "review_status": "reviewed",
        },
        # Leaf metadata is set directly: no seed-time derive pass runs here.
        {
            "id": "meta/llama-3.1-8b-instruct-turbo",
            "org_id": "meta",
            "parents": json.dumps([{"id": "meta/llama-3.1-8b-instruct", "relationship": "quantized"}]),
            "model_group_id": "meta/llama-3.1-8b-instruct",
            "lineage_origin_model_org_id": "meta",
            "open_weights": True, "release_date": "2024-07-18",
            "params_billions": 8.0,
            "review_status": "reviewed",
        },
    ))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("meta/llama-3.1-8b-instruct-turbo", "model")
    assert r.canonical_id == "meta/llama-3.1-8b-instruct-turbo"
    assert r.resolved_leaf_id == "meta/llama-3.1-8b-instruct-turbo"
    assert r.model_group_id == "meta/llama-3.1-8b-instruct"
    assert r.root_model_id == "meta/llama-3.1-8b-instruct"
    assert r.open_weights is True
    assert r.release_date == "2024-07-18"
    assert r.params_billions == 8.0
    assert r.parents == [{"id": "meta/llama-3.1-8b-instruct", "relationship": "quantized"}]


def test_enriched_resolver_benchmark_surfaces_parent():
    """Benchmark parent_canonical_id comes from `parent_benchmark_id`;
    model-only fields stay None."""
    aliases = _alias_store(("MATH Level 5", "benchmark", "math-level-5", None, "confirmed"))
    bm_df = pd.DataFrame([{
        "id": "math-level-5", "display_name": "MATH Level 5",
        "parent_benchmark_id": "math", "review_status": "reviewed",
    }])
    canonicals = CanonicalStore(benchmarks_df=bm_df)
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("MATH Level 5", "benchmark")
    assert r.canonical_id == "math-level-5"
    assert r.parent_canonical_id == "math"
    assert r.review_status == "reviewed"
    assert r.resolved_leaf_id is None
    assert r.parents is None
    assert r.open_weights is None


def test_no_match_carries_no_enrichment():
    """A no_match carries no enrichment."""
    canonicals = CanonicalStore(models_df=_models_df())
    resolver = Resolver(_alias_store(), canonical_store=canonicals)
    r = resolver.resolve("totally-unknown-model", "model")
    assert r.canonical_id is None
    assert r.strategy == "no_match"
    assert r.parents is None
    assert r.open_weights is None
    assert r.review_status is None


# ---------- CanonicalStore directly ----------

def test_canonical_store_lookup_returns_dict_with_na_coerced():
    """lookup coerces NaN/NA to None so callers never see pd.NA."""
    df = pd.DataFrame([
        {"id": "x/y", "open_weights": True, "release_date": None, "params_billions": float("nan")},
    ])
    cs = CanonicalStore(models_df=df)
    row = cs.lookup("model", "x/y")
    assert row is not None
    assert row["open_weights"] is True
    assert row["release_date"] is None
    assert row["params_billions"] is None


def test_canonical_store_lookup_missing_id_returns_none():
    cs = CanonicalStore(models_df=_models_df())
    assert cs.lookup("model", "does-not-exist") is None
    assert cs.lookup("benchmark", "does-not-exist") is None


# ---------- backwards compat ----------

def test_resolver_positional_two_arg_constructor():
    """The positional `Resolver(store, config)` form must keep working:
    downstream pipelines construct it that way."""
    store = _alias_store(("MATH", "benchmark", "math", None, "confirmed"))
    resolver = Resolver(store, ResolverConfig(threshold=0.85))
    r = resolver.resolve("MATH", "benchmark")
    assert r.canonical_id == "math"
    assert r.strategy == "exact"
    assert r.confidence == 1.0
    assert r.review_status is None
    assert r.parents is None


def test_flip_handles_dangling_group_root_fk():
    """A `model_group_id` pointing at an id absent from the store must not
    crash enrichment: canonical_id is the present leaf and the dangling
    root still surfaces in model_group_id."""
    aliases = _alias_store(("orphan/leaf", "model", "orphan/leaf", None, "confirmed"))
    canonicals = CanonicalStore(models_df=_models_df({
        "id": "orphan/leaf",
        "org_id": "orphan",
        "parents": json.dumps([{"id": "orphan/missing-root", "relationship": "quantized"}]),
        "model_group_id": "orphan/missing-root",
        "review_status": "draft",
    }))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("orphan/leaf", "model")
    assert r.canonical_id == "orphan/leaf"
    assert r.resolved_leaf_id == "orphan/leaf"
    assert r.model_group_id == "orphan/missing-root"
    assert r.open_weights is None
    assert r.release_date is None


# ---------- ancestry + typed resolution_detail ----------

def _benchmarks_df(*rows) -> pd.DataFrame:
    cols = ["id", "display_name", "parent_benchmark_id", "review_status"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def _families_df(*rows) -> pd.DataFrame:
    cols = ["id", "display_name", "category", "benchmark_ids",
            "composite_keys", "review_status"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def _composites_df(*rows) -> pd.DataFrame:
    cols = ["id", "display_name", "source_configs", "family_id", "review_status"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def test_model_ancestry_and_detail():
    aliases = _alias_store(("acme/w-7b-it", "model", "acme/w-7b-it", None, "confirmed"))
    df = _models_df({"id": "acme/w-7b-it", "org_id": "acme", "parents": "[]"})
    df["model_family_id"] = ["acme/w"]
    df["model_group_id"] = ["acme/w-7b"]
    df["resolution_granularity"] = ["variant"]
    cs = CanonicalStore(models_df=df)
    r = Resolver(aliases, canonical_store=cs).resolve("acme/w-7b-it", "model")
    assert r.ancestry == [
        {"canonical_id": "acme/w-7b", "level": "group"},
        {"canonical_id": "acme/w", "level": "family"},
    ]
    # Off-HF model (no resolution_source / metadata): hf_repo_id is None.
    assert r.resolution_detail == {"granularity": "variant", "hf_repo_id": None}


def test_model_detail_hf_repo_id():
    """hf_repo_id is set only for an HF-backed canonical (resolution_source==hf
    or metadata.hf_id==id); a shadow slug whose hf_id points elsewhere gets None."""
    aliases = _alias_store(
        ("meta/llama", "model", "meta/llama", None, "confirmed"),
        ("acme/mistral", "model", "acme/mistral", None, "confirmed"),
        ("kimi/slug", "model", "kimi/slug", None, "confirmed"),
        ("junk/meta", "model", "junk/meta", None, "confirmed"),
    )
    df = _models_df(
        {"id": "meta/llama", "org_id": "meta", "parents": "[]"},
        {"id": "acme/mistral", "org_id": "acme", "parents": "[]",
         "metadata": '{"hf_id": "acme/mistral"}'},
        {"id": "kimi/slug", "org_id": "moonshotai", "parents": "[]",
         "metadata": '{"hf_id": "moonshotai/Real-Kimi"}'},
        # metadata that parses to a NON-dict JSON value must not crash the resolve.
        {"id": "junk/meta", "org_id": "junk", "parents": "[]",
         "metadata": "[1, 2, 3]"},
    )
    df["resolution_source"] = ["hf", "models_dev", "inferred", "inferred"]
    cs = CanonicalStore(models_df=df)
    res = lambda raw: Resolver(aliases, canonical_store=cs).resolve(raw, "model")
    assert res("meta/llama").resolution_detail["hf_repo_id"] == "meta/llama"       # source==hf
    assert res("acme/mistral").resolution_detail["hf_repo_id"] == "acme/mistral"   # metadata.hf_id==id
    assert res("kimi/slug").resolution_detail["hf_repo_id"] is None                # shadow: hf_id != id
    assert res("junk/meta").resolution_detail["hf_repo_id"] is None                # non-dict metadata: no crash


def test_benchmark_full_chain_ancestry():
    aliases = _alias_store(("bench-pro", "benchmark", "bench-pro", None, "confirmed"))
    cs = CanonicalStore(
        benchmarks_df=_benchmarks_df({"id": "bench-pro", "display_name": "BP"}),
        families_df=_families_df({
            "id": "fam-x", "benchmark_ids": json.dumps(["bench-pro"]),
            "composite_keys": json.dumps(["comp-suite"]), "category": "reasoning"}),
        composites_df=_composites_df({
            "id": "comp-suite", "family_id": "fam-x",
            "source_configs": json.dumps(["cfg-a"])}),
    )
    r = Resolver(aliases, canonical_store=cs).resolve("bench-pro", "benchmark")
    assert r.ancestry == [
        {"canonical_id": "fam-x", "level": "family"},
        {"canonical_id": "comp-suite", "level": "composite"},
    ]
    assert r.resolution_detail == {"level": "benchmark", "matched_subset": None}


def test_benchmark_slice_detail():
    aliases = _alias_store(("bench-sub", "benchmark", "bench-sub", None, "confirmed"))
    cs = CanonicalStore(benchmarks_df=_benchmarks_df(
        {"id": "bench-pro", "display_name": "BP"},
        {"id": "bench-sub", "display_name": "BS", "parent_benchmark_id": "bench-pro"},
    ))
    r = Resolver(aliases, canonical_store=cs).resolve("bench-sub", "benchmark")
    assert r.resolution_detail["level"] == "slice"
    # bench-sub's family defaults to its parent walk terminus (bench-pro).
    assert r.ancestry == [{"canonical_id": "bench-pro", "level": "family"}]


def test_benchmark_subset_fold_matched_subset():
    aliases = _alias_store(("Anatomy", "benchmark", "mmlu", None, "confirmed"))
    cs = CanonicalStore(benchmarks_df=_benchmarks_df(
        {"id": "mmlu", "display_name": "MMLU"}))
    r = Resolver(aliases, canonical_store=cs).resolve("Anatomy", "benchmark")
    assert r.canonical_id == "mmlu"
    assert r.resolution_detail == {"level": "benchmark", "matched_subset": "Anatomy"}


def test_family_resolves_with_composite_ancestry():
    aliases = _alias_store(("fam-x", "family", "fam-x", None, "confirmed"))
    cs = CanonicalStore(
        families_df=_families_df({
            "id": "fam-x", "composite_keys": json.dumps(["comp-suite"])}),
        composites_df=_composites_df({"id": "comp-suite", "family_id": "fam-x"}),
    )
    r = Resolver(aliases, canonical_store=cs).resolve("fam-x", "family")
    assert r.canonical_id == "fam-x"
    assert r.ancestry == [{"canonical_id": "comp-suite", "level": "composite"}]
    assert r.resolution_detail == {}


def test_composite_resolves_as_root():
    aliases = _alias_store(("comp-suite", "composite", "comp-suite", None, "confirmed"))
    cs = CanonicalStore(composites_df=_composites_df(
        {"id": "comp-suite", "family_id": "fam-x"}))
    r = Resolver(aliases, canonical_store=cs).resolve("comp-suite", "composite")
    assert r.canonical_id == "comp-suite"
    assert r.ancestry == []
    assert r.resolution_detail == {}
