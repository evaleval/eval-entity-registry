"""eval-entity-registry CLI: seed, stats, sync."""
import json
import math
import re
import unicodedata
from numbers import Real
from pathlib import Path
from typing import Optional

import typer
import yaml


# A scope key (`source_config` on a scoped alias or metric fold) is an EEE
# dataset config name verbatim (`openeval`, `helm_capabilities`): never a
# registry id, never normalised. A typo'd or path-like key fails the seed
# instead of scoping an alias to a config that never calls.
_SOURCE_SCOPE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def seed_collision_key(s: str) -> str:
    """Key under which two seed surface forms count as one entity: NFKD +
    combining-mark strip (so "racé" collides with "race"), casefold, and
    [a-z0-9] only. Cross-script confusables still evade."""
    decomposed = unicodedata.normalize("NFKD", str(s))
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", base.casefold())


def _json_encode_if_needed(value):
    """Encode lists/dicts as JSON strings; the canonical_* parquet columns are
    all VARCHAR and seed files mix native lists with pre-encoded strings."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _legacy_parent_model_id_to_parents(entry: dict) -> None:
    """Translate a legacy scalar `parent_model_id` into the typed `parents`
    list in place; no-op when `parents` is already present."""
    if "parents" in entry and entry["parents"] is not None:
        entry.pop("parent_model_id", None)
        return
    legacy = entry.pop("parent_model_id", None)
    if legacy:
        entry["parents"] = [{"id": legacy, "relationship": "variant", "axis": "size"}]

from eval_entity_registry.store.hf_store import get_store
from eval_entity_registry.store import queries, schemas
from eval_entity_registry.lib import collision_fold, org_attribution
from eval_entity_registry.lib.seed_io import WEAK_SCALAR_FIELDS, build_hf_to_dev_from_orgs_yaml
from eval_entity_registry.store.queries import _derive_release_date_from_id, _is_na
from eval_entity_resolver.normalization import normalize as _normalize_alias
from eval_entity_resolver.display import humanize_model_slug


def _humanized_display(entry: dict) -> str:
    """Presentation display_name for a model; shapes only the stored column,
    never alias promotion, so resolution is unchanged.

    A display with a space is a label and is kept as-is (humanizing would
    mangle proper-noun hyphens like `SWE-GPT`); a slug is humanized. A
    leading `unknown/` placeholder org is dropped."""
    cid = str(entry.get("id") or "")
    dn = str(entry.get("display_name") or "").strip()
    name = cid.split("/", 1)[1] if "/" in cid else cid  # org-stripped id
    if not dn or dn == cid:
        # An org-stripped name that already reads as a label is used verbatim.
        out = name if " " in name else humanize_model_slug(cid)
    elif " " not in dn:
        out = humanize_model_slug(dn)
    else:
        out = dn
    # The slug branches strip it via the org split; this catches curated labels.
    if out.startswith("unknown/"):
        out = out[len("unknown/"):]
    return out

app = typer.Typer(help="eval-entity-registry CLI")


def _load_store():
    store = get_store()
    if not store.loaded:
        store.load()
    return store


# ------------------------------------------------------------------
# seed
# ------------------------------------------------------------------

@app.command()
def seed(
    local: bool = typer.Option(False, "--local", help="Write to fixtures/ instead of HF Hub"),
    seed_dir: str = typer.Option("./seed", "--seed-dir"),
    prune_stale: bool = typer.Option(
        False,
        "--prune-stale/--no-prune-stale",
        help="Remove reviewed seed entities and seed aliases absent from the current YAML snapshot.",
    ),
):
    """Load known canonical entities from seed YAML files."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()
    seed_path = Path(seed_dir)

    # Models — three-layer load from seed/models/: sources/*.generated.yaml
    # (never hand-edited) → core.yaml (curated; flat list or
    # {skip_ids, skip_source_ids, entries}) → enrichments/*.yaml (overlays).
    # Field-level merge per entry (_merge_into); `weak:` scalars last.
    def _load_models_merged() -> list[dict]:
        models_dir = seed_path / "models"
        sources_dir = models_dir / "sources"
        core_file = models_dir / "core.yaml"
        enrichments_dir = models_dir / "enrichments"

        source_entries: list[dict] = []
        core_entries: list[dict] = []
        enrichment_entries: list[dict] = []
        skip_ids: set[str] = set()

        def _is_empty(v) -> bool:
            if v is None:
                return True
            if isinstance(v, (list, dict)) and len(v) == 0:
                return True
            if isinstance(v, str) and v.strip() in ("", "[]", "{}"):
                return True
            return False

        # `weak:` scalars never enter the strong merge; collected in
        # deterministic order (files by sorted name, records in file order).
        weak_candidates: list[tuple[str, str, object]] = []
        if sources_dir.is_dir():
            for src_path in sorted(sources_dir.glob("*.generated.yaml")):
                with open(src_path) as f:
                    loaded = yaml.safe_load(f) or []
                if not isinstance(loaded, list):
                    raise typer.BadParameter(f"{src_path} must be a flat list")
                for entry in loaded:
                    if not isinstance(entry, dict) or not entry.get("id"):
                        continue
                    weak = entry.pop("weak", None)
                    if isinstance(weak, dict):
                        for field in WEAK_SCALAR_FIELDS:
                            if not _is_empty(weak.get(field)):
                                weak_candidates.append((entry["id"], field, weak[field]))
                source_entries.extend(loaded)

        skip_source_ids: set[str] = set()
        if core_file.exists():
            with open(core_file) as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, list):
                core_entries = loaded
            elif isinstance(loaded, dict):
                core_entries = loaded.get("entries", []) or []
                skip_ids = set(loaded.get("skip_ids", []) or [])
                # Drops ids from sources/enrichments only, so a generated source's
                # bad aliases cannot be UNION-merged back onto a curated entry.
                skip_source_ids = set(loaded.get("skip_source_ids", []) or [])
            else:
                raise typer.BadParameter(f"{core_file} unexpected shape {type(loaded)}")

        # An explicit core key, even null, is a curated claim that blocks weak
        # fill; weak fills only keys core leaves absent.
        core_explicit: dict[str, set[str]] = {}
        for entry in core_entries:
            if isinstance(entry, dict) and entry.get("id"):
                core_explicit.setdefault(entry["id"], set()).update(entry.keys())

        # aliases.yaml carries alias bridges, parents.yaml curated lineage
        # edges the bulk generators cannot reconstruct from their sources.
        if enrichments_dir.is_dir():
            for enr_path in sorted(enrichments_dir.glob("*.yaml")):
                with open(enr_path) as f:
                    loaded = yaml.safe_load(f) or []
                if not isinstance(loaded, list):
                    raise typer.BadParameter(f"{enr_path} must be a flat list")
                enrichment_entries.extend(loaded)

        def _merge_into(target: dict, src: dict) -> dict:
            """Merge two entries with the same canonical_id: `aliases` / `tags`
            union (case-insensitive); `metadata` per-key, later wins (scalar
            rule when a side is not a JSON object); other scalars prefer
            non-empty then last-write-wins, so a sparse later entry never
            blanks a rich one. Empty = None, "", [], {}, '{}' or '[]'."""
            import json as _json

            existing_aliases = list(target.get("aliases") or [])
            existing_lc = {a.lower() for a in existing_aliases if a}
            new_aliases = list(src.get("aliases") or [])
            for a in new_aliases:
                if a and a.lower() not in existing_lc:
                    existing_aliases.append(a)
                    existing_lc.add(a.lower())

            def _decode_list_field(v):
                """(list, was_json) from a YAML list or JSON-encoded string."""
                if v is None:
                    return [], False
                if isinstance(v, list):
                    return list(v), False
                if isinstance(v, str):
                    s = v.strip()
                    if not s or s in ("[]", "null"):
                        return [], True
                    try:
                        d = _json.loads(s)
                        if isinstance(d, list):
                            return list(d), True
                    except (ValueError, TypeError):
                        pass
                return [v], False

            tgt_tags, tgt_was_json = _decode_list_field(target.get("tags"))
            src_tags, src_was_json = _decode_list_field(src.get("tags"))
            seen_tags_lc = {str(t).lower() for t in tgt_tags}
            for t in src_tags:
                if t is not None and str(t).lower() not in seen_tags_lc:
                    tgt_tags.append(t)
                    seen_tags_lc.add(str(t).lower())
            tags_merged = _json.dumps(tgt_tags) if (tgt_was_json or src_was_json) else tgt_tags

            # `parents` union by edge id; a shared edge is field-merged so a later
            # source can fill `axis` without duplicating it. Target order kept.
            tgt_parents, tgt_p_was_json = _decode_list_field(target.get("parents"))
            src_parents, src_p_was_json = _decode_list_field(src.get("parents"))
            parents_by_id: dict[str, dict] = {}
            parents_order: list[str] = []
            for p in tgt_parents:
                if isinstance(p, dict) and p.get("id"):
                    pid = p["id"]
                    if pid not in parents_by_id:
                        parents_order.append(pid)
                        parents_by_id[pid] = dict(p)
            for p in src_parents:
                if not isinstance(p, dict) or not p.get("id"):
                    continue
                pid = p["id"]
                if pid in parents_by_id:
                    merged_edge = dict(parents_by_id[pid])
                    for k, v in p.items():
                        if _is_empty(v):
                            continue
                        merged_edge[k] = v
                    parents_by_id[pid] = merged_edge
                else:
                    parents_order.append(pid)
                    parents_by_id[pid] = dict(p)
            parents_list = [parents_by_id[pid] for pid in parents_order]
            parents_merged = (
                _json.dumps(parents_list)
                if (tgt_p_was_json or src_p_was_json)
                else parents_list
            )

            # A malformed or non-dict side opts the pair out of the per-key merge.
            def _decode_dict_field(v):
                """Return (dict, was_json, is_dict)."""
                if v is None:
                    return {}, False, True
                if isinstance(v, dict):
                    return dict(v), False, True
                if isinstance(v, str):
                    s = v.strip()
                    if not s or s in ("{}", "null"):
                        return {}, True, True
                    try:
                        d = _json.loads(s)
                        if isinstance(d, dict):
                            return d, True, True
                    except (ValueError, TypeError):
                        pass
                return {}, False, False

            tgt_meta, tgt_m_json, tgt_m_ok = _decode_dict_field(target.get("metadata"))
            src_meta, src_m_json, src_m_ok = _decode_dict_field(src.get("metadata"))
            metadata_handled = tgt_m_ok and src_m_ok
            if metadata_handled:
                meta_merged = {**tgt_meta, **src_meta}  # later source wins per key
                metadata_merged = (
                    _json.dumps(meta_merged, sort_keys=True)
                    if (tgt_m_json or src_m_json)
                    else meta_merged
                )

            merged = dict(target)
            for k, v in src.items():
                if k in ("aliases", "tags", "parents"):
                    continue  # handled separately
                if k == "metadata" and metadata_handled:
                    continue  # handled separately
                if _is_empty(v):
                    continue
                merged[k] = v
            if metadata_handled and ("metadata" in target or "metadata" in src):
                merged["metadata"] = metadata_merged
            merged["aliases"] = existing_aliases
            merged["tags"] = tags_merged
            # No `parents` key unless one side had any.
            if tgt_parents or src_parents:
                merged["parents"] = parents_merged
            return merged

        by_id: dict[str, dict] = {}

        def _absorb(entries: list[dict], extra_skip: set[str] = frozenset()) -> None:
            drop = skip_ids | extra_skip
            for e in entries:
                if "id" not in e:
                    raise typer.BadParameter(f"models seed entry missing id: {e!r}")
                if e["id"] in drop:
                    continue
                _legacy_parent_model_id_to_parents(e)
                if e["id"] in by_id:
                    by_id[e["id"]] = _merge_into(by_id[e["id"]], e)
                else:
                    by_id[e["id"]] = e

        # Sources/enrichments respect both skip_ids and skip_source_ids;
        # core entries respect only skip_ids so curated overrides always apply.
        _absorb(source_entries, extra_skip=skip_source_ids)
        _absorb(core_entries)
        _absorb(enrichment_entries, extra_skip=skip_source_ids)
        merged = list(by_id.values())

        # Before the fold, so a normalised id can collapse with its twin.
        merged, _org_merges = org_attribution.attribute_orgs(
            merged, build_hf_to_dev_from_orgs_yaml(seed_path / "orgs.yaml")
        )

        # collision_overrides.yaml carries the do-not-fold list and winner pins
        # for lib/collision_fold.py.
        ov_file = models_dir / "collision_overrides.yaml"
        never_fold, prefer, curated_merge, non_lineage_bases = [], {}, {}, set()
        if ov_file.is_file():
            ov = yaml.safe_load(ov_file.read_text()) or {}
            never_fold = ov.get("never_fold") or []
            prefer = ov.get("prefer") or {}
            # `merge`: {loser_id -> winner_id} for spellings the automatic fold
            # cannot key together (different org prefix, size-guard false block).
            curated_merge = ov.get("merge") or {}
            # Umbrella ids that stay resolvable but may never be a lineage parent.
            non_lineage_bases = set(ov.get("non_lineage_bases") or [])
        # An HF-true canonical carries the invented ids it replaced as aliases.
        # A side source the refresh never rewrites may still key a record by
        # such an id, which would materialise a second canonical colliding on
        # the alias; fold those onto the HF entry (core ids never).
        core_ids = {e["id"] for e in core_entries}
        # A multi-child family root (`alibaba/qwen3-vl-235b-a22b` over Instruct
        # and Thinking) is not one repo and stays a root, like an umbrella.
        kids: dict[str, set[str]] = {}
        for e in merged:
            for p in collision_fold._edges(e.get("parents")):
                if not isinstance(p, dict) or p.get("relationship") != "variant":
                    continue
                pid = p.get("id")
                if isinstance(pid, str) and pid != e["id"]:
                    kids.setdefault(pid, set()).add(e["id"])
        multi_child = {pid for pid, k in kids.items() if len(k) >= 2}

        def _own_name(old: str, hf_id: str) -> bool:
            return seed_collision_key(old.rsplit("/", 1)[-1]) == seed_collision_key(hf_id.rsplit("/", 1)[-1])

        def _hf_true(e: dict) -> bool:
            meta = e.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except ValueError:
                    meta = None
            return e.get("resolution_source") == "hf" or (
                isinstance(meta, dict) and meta.get("hf_deferred") is True
            )

        hf_true_ids = {e["id"] for e in merged if _hf_true(e)}
        core_keys = {collision_fold.collision_key(i) for i in core_ids}
        promoted_merge: dict[str, str] = {}
        for e in merged:
            if e["id"] not in hf_true_ids:
                continue
            for a in e.get("aliases") or []:
                if a in core_ids or a in non_lineage_bases or a in hf_true_ids:
                    continue
                if a in multi_child and not _own_name(a, e["id"]):
                    continue
                if a != e["id"] and a in by_id:
                    promoted_merge.setdefault(a, e["id"])
        # A loser that is itself a declared winner would make a cycle.
        promoted_merge = {l: w for l, w in promoted_merge.items() if w not in promoted_merge}
        for l, w in promoted_merge.items():
            # A casing twin shares the collision key: pin the HF id as winner so
            # the group fold cannot leave the two remaps pointing at each other.
            key = collision_fold.collision_key(w)
            if collision_fold.collision_key(l) == key and key not in core_keys:
                prefer.setdefault(key, w)
        merged, _remap = collision_fold.fold_collisions(
            merged, never_fold, prefer,
            force_merge={**_org_merges, **promoted_merge, **curated_merge},
            non_lineage_bases=non_lineage_bases,
        )

        # Weak fill runs after the collision fold so a folded-in full entry's
        # scalars also beat weak ones. A weak value lands only on a still-empty
        # field the core entry does not explicitly carry; first contribution wins.
        weak_scalars: dict[tuple[str, str], object] = {}
        for eid, field, value in weak_candidates:
            if eid in skip_ids or eid in skip_source_ids:
                continue
            weak_scalars.setdefault((_remap.get(eid, eid), field), value)
        by_merged_id = {e["id"]: e for e in merged}
        for (eid, field), value in weak_scalars.items():
            target = by_merged_id.get(eid)
            if target is None or field in core_explicit.get(eid, ()):
                continue
            # A dated-snapshot id's own date stamp outranks weak data.
            if field == "release_date" and _derive_release_date_from_id(eid) is not None:
                continue
            if _is_empty(target.get(field)):
                target[field] = value
        return merged

    # ------------------------------------------------------------------
    # Benchmarks — seed/benchmarks_generated/*.yaml (bulk, stable ids so
    # refreshes are idempotent) → seed/benchmarks.yaml (curated, wins).
    # ------------------------------------------------------------------
    def _load_benchmarks_merged() -> list[dict]:
        curated_path = seed_path / "benchmarks.yaml"
        generated_dir = seed_path / "benchmarks_generated"

        def _load_benchmarks_yaml(path: Path) -> list[dict]:
            raw_yaml = path.read_text()
            invalid_judge_flag = re.search(
                r"^\s*preferred_metric_llm_judged:\s*(?!true\b|false\b|null\b|~)(?P<value>\S+)",
                raw_yaml,
                re.MULTILINE,
            )
            if invalid_judge_flag:
                raise typer.BadParameter(
                    f"{path}: invalid preferred_metric_llm_judged value "
                    f"{invalid_judge_flag.group('value')!r}; "
                    "use true, false, null, or ~"
                )
            loaded = yaml.safe_load(raw_yaml) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{path} must be a flat list")
            return loaded

        generated_entries: list[dict] = []
        if generated_dir.is_dir():
            for src_path in sorted(generated_dir.glob("*.yaml")):
                generated_entries.extend(_load_benchmarks_yaml(src_path))

        curated_entries: list[dict] = []
        if curated_path.exists():
            curated_entries = _load_benchmarks_yaml(curated_path)

        def _merge_benchmark(generated: dict, curated: dict) -> dict:
            """Curated wins on every field it specifies; aliases are unioned so
            generator-emitted aliases survive."""
            merged = dict(generated)
            for k, v in curated.items():
                if k == "aliases":
                    continue
                merged[k] = v
            existing = list(generated.get("aliases") or [])
            existing_lc = {a.lower() for a in existing if a}
            for a in (curated.get("aliases") or []):
                if a and a.lower() not in existing_lc:
                    existing.append(a)
                    existing_lc.add(a.lower())
            merged["aliases"] = existing
            return merged

        by_id: dict[str, dict] = {}
        for entry in generated_entries:
            if "id" not in entry:
                raise typer.BadParameter(f"benchmarks generated entry missing id: {entry!r}")
            by_id[entry["id"]] = entry
        for entry in curated_entries:
            if "id" not in entry:
                raise typer.BadParameter(f"benchmarks seed entry missing id: {entry!r}")
            if entry["id"] in by_id:
                by_id[entry["id"]] = _merge_benchmark(by_id[entry["id"]], entry)
            else:
                by_id[entry["id"]] = entry
        # `preferred_metric` is validated against metrics.yaml so a typo cannot
        # silently null the merged-view default.
        metrics_path = seed_path / "metrics.yaml"
        metric_ids: set[str] = set()
        if metrics_path.exists():
            with open(metrics_path) as f:
                metric_ids = {m["id"] for m in (yaml.safe_load(f) or [])}
        for entry in by_id.values():
            pm = entry.pop("preferred_metric", None)
            judged = entry.get("preferred_metric_llm_judged")
            if judged is not None and type(judged) is not bool:
                raise typer.BadParameter(
                    f"benchmark {entry['id']!r}: preferred_metric_llm_judged "
                    "must be a boolean or null"
                )
            if judged is not None and pm is None:
                raise typer.BadParameter(
                    f"benchmark {entry['id']!r}: preferred_metric_llm_judged "
                    "requires preferred_metric"
                )
            if pm is not None:
                if pm not in metric_ids:
                    raise typer.BadParameter(
                        f"benchmark {entry['id']!r}: preferred_metric {pm!r} "
                        f"is not a canonical metric id"
                    )
                entry["preferred_metric_id"] = pm

        _check_benchmark_collisions(list(by_id.values()))
        return list(by_id.values())

    def _check_benchmark_collisions(entries: list[dict]) -> None:
        """Flag (never auto-fold) benchmarks whose ids, display names or global
        aliases share a seed_collision_key (stricter than collision_fold's key
        so spaced aliases are covered); verified-distinct groups live in
        seed/benchmarks_distinct_allowlist.yaml. Also rejects '/' in ids: the
        merged-view URL scheme needs single-segment benchmark ids."""
        _norm = seed_collision_key
        allow_path = seed_path / "benchmarks_distinct_allowlist.yaml"
        allowed: list[frozenset[str]] = []
        if allow_path.exists():
            with open(allow_path) as f:
                for group in yaml.safe_load(f) or []:
                    allowed.append(frozenset(group))

        # A stale or misspelled allowlist entry would silently stop guarding.
        all_ids = {str(e["id"]) for e in entries}
        stale = sorted({bid for grp in allowed for bid in grp} - all_ids)
        if stale:
            raise typer.BadParameter(
                f"benchmarks_distinct_allowlist.yaml names unknown benchmark id(s): {stale}"
            )

        key_owners: dict[str, set[str]] = {}
        for e in entries:
            bid = str(e["id"])
            if "/" in bid:
                raise typer.BadParameter(
                    f"benchmark id {bid!r} contains '/' (reserved for merged-view routing)"
                )
            # id, display_name and aliases all become resolver aliases at seed
            # time, so any of them can carry a dupe in.
            keys = {_norm(bid)}
            if e.get("display_name"):
                keys.add(_norm(e["display_name"]))
            keys |= {_norm(a) for a in (e.get("aliases") or []) if a}
            keys.discard("")
            for k in keys:
                key_owners.setdefault(k, set()).add(bid)
        conflicts = [
            sorted(ids)
            for ids in key_owners.values()
            if len(ids) > 1 and not any(ids <= grp for grp in allowed)
        ]
        if conflicts:
            listing = "; ".join(str(g) for g in sorted(conflicts))
            raise typer.BadParameter(
                f"{len(conflicts)} separator/case-collision group(s) among benchmark "
                f"ids+aliases — fold them in the seed or allowlist in "
                f"benchmarks_distinct_allowlist.yaml: {listing}"
            )

    # ------------------------------------------------------------------
    # Families — seed/families.yaml is keyed {slug: {fields}} for
    # readability; flattened to `id` rows for upsert.
    # ------------------------------------------------------------------
    def _load_families_seed() -> list[dict]:
        path = seed_path / "families.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise typer.BadParameter(f"{path} must be a top-level mapping {{slug: {{...}}}}")

        out: list[dict] = []
        seen_benchmarks: dict[str, str] = {}
        for slug, fields in raw.items():
            if not isinstance(fields, dict):
                raise typer.BadParameter(f"family {slug!r} entry must be a mapping, got {type(fields).__name__}")
            benchmark_ids = list(fields.get("benchmarks") or [])
            for bid in benchmark_ids:
                if bid in seen_benchmarks and seen_benchmarks[bid] != slug:
                    raise typer.BadParameter(
                        f"benchmark {bid!r} listed in two families: "
                        f"{seen_benchmarks[bid]!r} and {slug!r}"
                    )
                seen_benchmarks[bid] = slug
            entry = {
                "id": slug,
                "display_name": fields.get("display") or slug,
                "category": fields.get("category"),
                "benchmark_ids": benchmark_ids,
                "primary_benchmark_key": fields.get("primary_benchmark_key"),
                "folder_aliases": list(fields.get("folder_aliases") or []),
                "composite_keys": list(fields.get("composite_keys") or []),
                "tags": fields.get("tags") or [],
                "metadata": fields.get("metadata") or {},
                "review_status": fields.get("review_status") or "reviewed",
            }
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Composites — {slug: {display, configs: [...], category?, family_id?}}
    # ------------------------------------------------------------------
    def _load_composites_seed() -> list[dict]:
        path = seed_path / "composites.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise typer.BadParameter(f"{path} must be a top-level mapping {{slug: {{...}}}}")

        out: list[dict] = []
        for slug, fields in raw.items():
            if not isinstance(fields, dict):
                raise typer.BadParameter(f"composite {slug!r} entry must be a mapping, got {type(fields).__name__}")
            raw_configs = fields.get("configs")
            if raw_configs is None:
                # Implicit single source_config equal to the slug. Upstream EEE
                # folders are kebab (`arc-agi`) or snake (`helm_classic`), so
                # both spellings ship for the producer's JOIN.
                kebab = slug
                snake = slug.replace("-", "_")
                source_configs = [kebab] if kebab == snake else [kebab, snake]
            else:
                # Scoped members {config, org[, source]} pass through as objects
                # into the JSON-encoded list; str() would ship Python-repr garbage.
                source_configs = []
                for c in raw_configs:
                    if isinstance(c, dict):
                        unknown = sorted(set(c) - {"config", "org", "source"})
                        if unknown or "config" not in c:
                            raise typer.BadParameter(
                                f"composite {slug!r}: scoped configs member "
                                f"{c!r} must have keys {{config, org[, source]}}"
                            )
                        if c.get("source") is not None and c.get("org") is None:
                            raise typer.BadParameter(
                                f"composite {slug!r}: scoped member {c!r} "
                                f"sets `source` without `org`"
                            )
                        member = {"config": str(c["config"])}
                        if c.get("org") is not None:
                            member["org"] = str(c["org"])
                        if c.get("source") is not None:
                            member["source"] = str(c["source"])
                        source_configs.append(member)
                    else:
                        source_configs.append(str(c))
            entry = {
                "id": slug,
                "display_name": fields.get("display") or slug,
                "category": fields.get("category"),
                "source_configs": source_configs,
                "family_id": fields.get("family_id"),
                "tags": fields.get("tags") or [],
                "metadata": fields.get("metadata") or {},
                "review_status": fields.get("review_status") or "reviewed",
            }
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Orgs — curated seed/orgs.yaml + seed/orgs.generated.yaml. On id
    # collision the generated entry is dropped outright (not field-merged):
    # generated entries are thin and never improve a curated record.
    # ------------------------------------------------------------------
    def _load_orgs_merged() -> list[dict]:
        curated_path = seed_path / "orgs.yaml"
        generated_path = seed_path / "orgs.generated.yaml"

        curated: list[dict] = []
        if curated_path.exists():
            with open(curated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{curated_path} must be a flat list")
            curated = loaded

        generated: list[dict] = []
        if generated_path.exists():
            with open(generated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{generated_path} must be a flat list")
            generated = loaded

        curated_ids = {e["id"] for e in curated if "id" in e}
        out = list(curated)
        for e in generated:
            if "id" not in e:
                raise typer.BadParameter(f"orgs.generated.yaml entry missing id: {e!r}")
            if e["id"] not in curated_ids:
                out.append(e)
        return out

    # table name, yaml file, label, entity_type (for alias creation)
    seed_specs = [
        ("canonical_orgs", "__merged_orgs__", "orgs", "org"),
        ("canonical_benchmarks", "__merged_benchmarks__", "benchmarks", "benchmark"),
        ("canonical_metrics", seed_path / "metrics.yaml", "metrics", "metric"),
        ("eval_harnesses", seed_path / "harnesses.yaml", "harnesses", "harness"),
        # family/composite aliases are emitted for consistency; the resolver
        # does not consult them.
        ("canonical_families", "__nested_families__", "families", "family"),
        ("canonical_composites", "__nested_composites__", "composites", "composite"),
        ("canonical_models", "__merged_models__", "models", "model"),
    ]

    alias_count = 0
    # Reported in the summary so a typo'd config shows up as a new key.
    source_scope_keys: set[str] = set()
    # For stale removal; alias key = (raw_value, entity_type, canonical_id, source_config).
    seed_snapshot: list[tuple[str, str, set[str], set[tuple[str, str, str, Optional[str]]]]] = []

    # Two canonicals declaring one (raw_value, entity_type, source_config) in
    # a single pass would make the owner seed-order-dependent, so collisions
    # are collected and raised after the loop. Keyed on this run only: a YAML
    # rename against the persisted store is not a collision. Values are
    # (owner, declarer): the post-redirect target and the entity whose alias
    # list claimed the raw.
    run_alias_owners: dict[tuple[str, str, Optional[str]], tuple[str, str]] = {}
    alias_collisions: list[tuple[str, str, Optional[str], str, str, str, str]] = []

    # O(1) add_alias collision checks; with buffered=True this avoids
    # O(N²) concat-per-row.
    queries._rebuild_alias_index(store)

    for table, yaml_file, label, entity_type in seed_specs:
        table_columns = set(schemas.empty(table).columns)
        if yaml_file == "__merged_models__":
            items = _load_models_merged()
            if not items:
                typer.echo(f"  [skip] no model entries found in seed/models.yaml or _overrides/")
                continue
        elif yaml_file == "__merged_orgs__":
            items = _load_orgs_merged()
            if not items:
                typer.echo(f"  [skip] no org entries found in seed/orgs.yaml or seed/orgs.generated.yaml")
                continue
        elif yaml_file == "__merged_benchmarks__":
            items = _load_benchmarks_merged()
            if not items:
                typer.echo(f"  [skip] no benchmark entries found in seed/benchmarks.yaml or seed/benchmarks_generated/")
                continue
        elif yaml_file == "__nested_families__":
            items = _load_families_seed()
            if not items:
                typer.echo(f"  [skip] no family entries found in seed/families.yaml")
                continue
        elif yaml_file == "__nested_composites__":
            items = _load_composites_seed()
            if not items:
                typer.echo(f"  [skip] no composite entries found in seed/composites.yaml")
                continue
        else:
            if not yaml_file.exists():
                typer.echo(f"  [skip] {yaml_file} not found")
                continue
            with open(yaml_file) as f:
                items = yaml.safe_load(f) or []

        yaml_ids: set[str] = set()
        yaml_alias_keys: set[tuple[str, str, str, Optional[str]]] = set()

        # Anti-shadowing: a non-lab fork's bare display_name normalizing to a
        # lab model's name would win it at confidence 1.0, so only colliding
        # display_name aliases are dropped off non-lab models.
        lab_norms: set[str] = set()
        # Prefer-official: bare leaf -> owner by precedence, not load order.
        # 1. lab_leaf_owner: a lab owns its leaf (a fork whose display is the
        #    bare leaf would otherwise win it). 2. strong_leaf_owner: else the
        #    leaf's unique confirmed (hf/reviewed) entry, applied only against
        #    unconfirmed non-lab drafts so two real repos are never ranked. A
        #    leaf with 2+ confirmed non-lab claimants is a duplicate, not a
        #    name call, and stays unowned.
        lab_leaf_owner: dict[str, str] = {}
        lab_owner_dev: dict[str, str] = {}
        strong_leaf_owner: dict[str, str] = {}
        strong_ids: set[str] = set()
        org_kind: dict[str, str] = {}
        hf_to_dev: dict[str, str] = {}
        org_names: dict[str, str] = {}

        def _entry_dev(it: dict) -> str:
            # A set org_id is authoritative: curated remaps put the true
            # developer there, and trusting the id prefix would launder a
            # mis-attribution into lab status. Only an unset org_id falls back
            # to the prefix, resolved through the same maps
            # derive_model_lineage_fields uses, so the two agree.
            org_id = it.get("org_id")
            if org_id:
                return str(org_id)
            cid = str(it.get("id") or "")
            if "/" not in cid:
                return ""
            prefix = cid.split("/", 1)[0]
            return hf_to_dev.get(prefix.lower()) or org_names.get(prefix.lower()) or prefix

        def _entry_is_lab(it: dict) -> bool:
            return org_kind.get(_entry_dev(it)) == "lab"

        if entity_type == "model":
            # From the YAML, not store.table: orgs were upserted buffered=True
            # and are not flushed yet.
            merged_orgs = [
                o for o in (_load_orgs_merged() or []) if isinstance(o, dict) and o.get("id")
            ]
            org_kind = {str(o["id"]): str(o.get("kind") or "") for o in merged_orgs}
            hf_to_dev = build_hf_to_dev_from_orgs_yaml(seed_path / "orgs.yaml")
            for o in merged_orgs:
                oid = str(o["id"])
                for name in [oid, o.get("hf_org"), o.get("display_name"), *(o.get("aliases") or [])]:
                    if isinstance(name, str) and name:
                        org_names.setdefault(name.lower(), oid)

            strong_claimants: dict[str, set[str]] = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = str(it.get("id") or "")
                # Confirmed = hf-sourced or reviewed; never a tier3 name-inferred
                # draft, where org mis-attributions live.
                confirmed = "/" in cid and (
                    it.get("resolution_source") == "hf"
                    or it.get("review_status") == "reviewed"
                )
                if confirmed:
                    strong_ids.add(cid)
                    strong_claimants.setdefault(
                        _normalize_alias(cid.split("/", 1)[1]), set()
                    ).add(cid)
                if not _entry_is_lab(it):
                    continue
                # Only the forms a lab seeds (display_name + explicit aliases)
                # catch a raw; the id/name-part are not bare aliases, so dropping
                # on them would orphan the raw.
                for s in [it.get("display_name"), *(it.get("aliases") or [])]:
                    if isinstance(s, str) and s:
                        lab_norms.add(_normalize_alias(s))
                if confirmed:
                    # First claimant wins; test_lab_leaf_ownership_is_unambiguous
                    # enforces that confirmed lab models never share a leaf.
                    _leaf_norm = _normalize_alias(cid.split("/", 1)[1])
                    if lab_leaf_owner.setdefault(_leaf_norm, cid) == cid:
                        lab_owner_dev[_leaf_norm] = _entry_dev(it)
            strong_leaf_owner = {
                norm: next(iter(cids))
                for norm, cids in strong_claimants.items()
                if len(cids) == 1
            }

        for original_item in items:
            item = dict(original_item)
            extra_aliases = item.pop("aliases", []) or []
            scoped_aliases = item.pop("scoped_aliases", {}) or {}
            # canonical_* parquet columns are VARCHAR.
            for col in (
                "tags", "metadata", "parents",
                "input_modalities", "output_modalities",
                "benchmark_ids", "folder_aliases", "composite_keys",
                "source_configs",
            ):
                if col in item:
                    item[col] = _json_encode_if_needed(item[col])
            entity_item = {k: v for k, v in item.items() if k in table_columns}
            unknown_keys = sorted(set(item.keys()) - table_columns)
            if unknown_keys:
                typer.echo(
                    f"  [warn] {label} entry {item.get('id', '?')!r} has unknown "
                    f"key(s) {unknown_keys} — silently dropped. Check for typos."
                )
            if "id" not in entity_item:
                raise typer.BadParameter(f"{label} seed entry is missing required id: {original_item!r}")
            # Alias promotion below uses the original `display_name`, so
            # humanizing the stored column leaves resolution unchanged.
            display_name = entity_item.get("display_name", "")
            if entity_type == "model":
                humanized = _humanized_display(entity_item)
                if humanized:
                    entity_item["display_name"] = humanized
            queries.upsert_entity(store, table, entity_item, buffered=True)
            canonical_id = entity_item["id"]
            yaml_ids.add(canonical_id)

            # Scoped aliases match only under their source_config, so short
            # tokens ("Overall") can map to different benchmarks per EEE config.
            global_aliases = {canonical_id, display_name} | set(extra_aliases)
            # Suppress only the auto-promoted display_name of a non-lab model that
            # would shadow a lab (lab_norms). global_aliases is a set, so a string
            # that is also the canonical_id or an explicit alias is never dropped.
            item_dev = _entry_dev(entity_item) if entity_type == "model" else ""
            item_is_lab = org_kind.get(item_dev) == "lab"
            if (
                entity_type == "model"
                and display_name
                and display_name != canonical_id
                and display_name not in extra_aliases
                and not item_is_lab
                and _normalize_alias(display_name) in lab_norms
            ):
                global_aliases.discard(display_name)

            alias_specs: list[tuple[str, Optional[str]]] = [
                (raw, None) for raw in global_aliases if raw
            ]
            for source_cfg, raw_values in scoped_aliases.items():
                if not isinstance(source_cfg, str) or not _SOURCE_SCOPE_KEY_RE.match(source_cfg):
                    raise typer.BadParameter(
                        f"{label} entry {canonical_id!r}: scoped_aliases key "
                        f"{source_cfg!r} is not a valid EEE dataset config name"
                    )
                source_scope_keys.add(source_cfg)
                for raw in raw_values or []:
                    if raw:
                        alias_specs.append((raw, source_cfg))

            for raw_value, source_cfg in alias_specs:
                # Prefer-official redirect (global aliases only): a bare leaf
                # alias points at the leaf's owner (lab_leaf_owner /
                # strong_leaf_owner). The lab tier takes names from non-lab
                # entries and same-developer lab entries, never across distinct
                # labs; the confirmed tier only from unconfirmed non-lab drafts.
                target_cid = canonical_id
                if entity_type == "model" and source_cfg is None:
                    _norm = _normalize_alias(raw_value)
                    _lab = lab_leaf_owner.get(_norm)
                    if _lab and _lab != canonical_id and (
                        not item_is_lab or lab_owner_dev.get(_norm) == item_dev
                    ):
                        target_cid = _lab
                    elif not item_is_lab and canonical_id not in strong_ids:
                        _strong = strong_leaf_owner.get(_norm)
                        if _strong and _strong != canonical_id:
                            target_cid = _strong
                yaml_alias_keys.add((raw_value, entity_type, target_cid, source_cfg))
                _claim_key = (raw_value, entity_type, queries._source_config_key(source_cfg))
                _prev = run_alias_owners.get(_claim_key)
                if _prev is None:
                    run_alias_owners[_claim_key] = (target_cid, canonical_id)
                elif _prev[0] != target_cid:
                    alias_collisions.append(
                        (raw_value, entity_type, _claim_key[2],
                         _prev[0], target_cid, _prev[1], canonical_id)
                    )
                try:
                    queries.add_alias(store, {
                        "raw_value": raw_value,
                        "entity_type": entity_type,
                        "canonical_id": target_cid,
                        "source_config": source_cfg,
                        "source_field": "seed",
                        "status": "confirmed",
                        "strategy": "seed",
                        "confidence": 1.0,
                        "notes": None,
                    }, buffered=True)
                    alias_count += 1
                except ValueError:
                    # An existing row pointing at a different canonical is a YAML
                    # rename: repoint it, or stale-removal at the end of seed
                    # deletes it (its old key is not in yaml_alias_keys).
                    aliases_df = store.table("aliases")
                    mask = (
                        (aliases_df["raw_value"] == raw_value)
                        & (aliases_df["entity_type"] == entity_type)
                        & (aliases_df["status"] != "rejected")
                    )
                    if source_cfg is not None:
                        mask = mask & (aliases_df["source_config"] == source_cfg)
                    else:
                        mask = mask & aliases_df["source_config"].isna()
                    existing = aliases_df[mask]
                    if existing.empty:
                        # Collision from this run's pending buffer: mutate the
                        # pending dict in place (the alias index shares it).
                        # Repoint to target_cid, not canonical_id, so a
                        # prefer-official redirect survives a duplicate claim.
                        for p in queries._get_pending(store, "aliases"):
                            if (p.get("entity_type") == entity_type
                                    and p.get("raw_value") == raw_value
                                    and queries._source_config_key(p.get("source_config")) == queries._source_config_key(source_cfg)
                                    and p.get("status") != "rejected"):
                                if p["canonical_id"] != target_cid:
                                    prev = p["canonical_id"]
                                    p["canonical_id"] = target_cid
                                    p["source_field"] = "seed"
                                    p["status"] = "confirmed"
                                    p["strategy"] = "seed"
                                    p["confidence"] = 1.0
                                    typer.echo(
                                        f"  [rename] alias {raw_value!r} ({entity_type}) "
                                        f"moved {prev!r} -> {target_cid!r} (pending)"
                                    )
                                    alias_count += 1
                                break
                        continue
                    row = existing.iloc[0]
                    if row["canonical_id"] != target_cid:
                        queries.update_alias(store, row["id"], {
                            "canonical_id": target_cid,
                            "source_field": "seed",
                            "status": "confirmed",
                            "strategy": "seed",
                            "confidence": 1.0,
                        })
                        typer.echo(
                            f"  [rename] alias {raw_value!r} ({entity_type}) "
                            f"moved {row['canonical_id']!r} -> {target_cid!r}"
                        )
                        alias_count += 1
                    # else: identical re-seed of an existing alias — no-op.

        seed_snapshot.append((table, entity_type, yaml_ids, yaml_alias_keys))
        typer.echo(f"  {label}: {len(items)}")

    # Raised before flush so dirty data never persists.
    if alias_collisions:
        deduped = sorted(set(alias_collisions))
        lines = "\n".join(
            f"    {rv!r} ({et}{', cfg=' + ck if ck else ''}): "
            f"would point at both {a!r} (declared by {da!r}) "
            f"and {b!r} (declared by {db!r})"
            for (rv, et, ck, a, b, da, db) in deduped
        )
        raise typer.BadParameter(
            f"{len(deduped)} alias collision(s) — the same alias is declared by "
            "more than one canonical in this seed pass, so the owner would be "
            "seed-order-dependent (nondeterministic). Each alias must belong to "
            "exactly one canonical:\n" + lines
        )

    # prune_stale reads store.table() directly, so flush first.
    queries.flush_pending(store)

    # ------------------------------------------------------------------
    # Inference platforms — bypasses the seed_specs loop: its `aliases`
    # column is a stored JSON list of host-token spellings, not alias rows.
    # ------------------------------------------------------------------
    def _load_inference_platforms_merged() -> list[dict]:
        path = seed_path / "inference_platforms.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            loaded = yaml.safe_load(f) or []
        if not isinstance(loaded, list):
            raise typer.BadParameter(f"{path} must be a flat list")
        return loaded

    inf_plat_entries = _load_inference_platforms_merged()
    if inf_plat_entries:
        import pandas as pd

        now = queries._now()
        inf_cols = list(schemas._SCHEMAS["canonical_inference_platforms"].keys())
        inf_plat_rows = []
        for e in inf_plat_entries:
            if "id" not in e or "display_name" not in e or "kind" not in e:
                raise typer.BadParameter(
                    f"inference_platforms entry missing required field: {e!r}"
                )
            row = {
                "id": e.get("id"),
                "display_name": e.get("display_name"),
                "kind": e.get("kind"),
                "aliases": _json_encode_if_needed(e.get("aliases")),
                "canonical_org": e.get("canonical_org"),
                "variant_of": e.get("variant_of"),
                "homepage": e.get("homepage"),
                "created_at": now,
                "updated_at": now,
            }
            inf_plat_rows.append(row)
        inf_plat_df = pd.DataFrame(inf_plat_rows)
        for col in inf_cols:
            if col not in inf_plat_df.columns:
                inf_plat_df[col] = None
        inf_plat_df = inf_plat_df[inf_cols]
        store.set_table("canonical_inference_platforms", inf_plat_df)
        typer.echo(f"  inference_platforms: {len(inf_plat_rows)}")

    # ------------------------------------------------------------------
    # Benchmark metric folds — flat dim table with no entity ids, so it
    # skips the seed_specs/upsert path.
    # ------------------------------------------------------------------
    folds_path = seed_path / "metric_folds.yaml"
    if folds_path.exists():
        import pandas as pd

        with open(folds_path) as f:
            fold_entries = yaml.safe_load(f) or []
        if not isinstance(fold_entries, list):
            raise typer.BadParameter(f"{folds_path} must be a flat list")
        bench_ids = {e["id"] for e in _load_benchmarks_merged()}
        fold_metrics_path = seed_path / "metrics.yaml"
        metrics_by_id = {}
        if fold_metrics_path.exists():
            with open(fold_metrics_path) as f:
                metrics_by_id = {m["id"]: m for m in (yaml.safe_load(f) or [])}
        metric_ids = set(metrics_by_id)
        seen_rows: set[tuple[str, str, Optional[str]]] = set()
        unscoped_pairs: dict[tuple[str, str], str] = {}
        fold_rows = []
        for e in fold_entries:
            b, fm, tm = e.get("benchmark"), e.get("from_metric"), e.get("to_metric")
            source_config = e.get("source_config")
            if not (b and fm and tm):
                raise typer.BadParameter(f"metric_folds entry needs benchmark/from_metric/to_metric: {e!r}")
            if b not in bench_ids:
                raise typer.BadParameter(f"metric_folds: unknown benchmark {b!r}")
            missing = [m for m in (fm, tm) if m not in metric_ids]
            if missing:
                raise typer.BadParameter(f"metric_folds ({b}): unknown metric id(s) {missing}")
            if source_config is not None and not isinstance(source_config, str):
                raise typer.BadParameter(
                    f"metric_folds ({b}): source_config must be a string or null"
                )
            if source_config is not None and not _SOURCE_SCOPE_KEY_RE.match(source_config):
                raise typer.BadParameter(
                    f"metric_folds ({b}): source_config {source_config!r} is not a "
                    "valid EEE dataset config name"
                )
            if source_config is not None:
                source_scope_keys.add(source_config)
            # A scoped self-fold is a pure published-scale conversion; an
            # unscoped one carries no information.
            if fm == tm and source_config is None:
                raise typer.BadParameter(f"metric_folds ({b}): self-fold {fm!r}")
            row_key = (b, fm, source_config)
            if row_key in seen_rows:
                raise typer.BadParameter(
                    f"metric_folds: duplicate fold for ({b}, {fm}, {source_config!r})"
                )
            seen_rows.add(row_key)
            if source_config is None:
                unscoped_pairs[(b, fm)] = tm
            factor = e.get("scale_factor")
            offset = e.get("scale_offset")
            if source_config is None and (factor is not None or offset is not None):
                raise typer.BadParameter(
                    f"metric_folds ({b}): scale_factor/scale_offset conversions require source_config"
                )
            for name, value in (("scale_factor", factor), ("scale_offset", offset)):
                if value is not None and (
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise typer.BadParameter(
                        f"metric_folds ({b}): {name} must be a finite real number, got {value!r}"
                    )
            if factor is not None and factor <= 0:
                raise typer.BadParameter(
                    f"metric_folds ({b}): scale_factor must be a positive number, got {factor!r}"
                )
            if factor is not None or offset is not None:
                target = metrics_by_id[tm]
                min_score, max_score = target.get("min_score"), target.get("max_score")
                if (
                    not isinstance(min_score, Real)
                    or isinstance(min_score, bool)
                    or not math.isfinite(min_score)
                    or not isinstance(max_score, Real)
                    or isinstance(max_score, bool)
                    or not math.isfinite(max_score)
                    or max_score <= min_score
                ):
                    raise typer.BadParameter(
                        f"metric_folds ({b}): {tm!r} must have finite ordered numeric bounds "
                        "when scale_factor or scale_offset is set"
                    )
            fold_rows.append({
                "benchmark_id": b, "from_metric_id": fm,
                "to_metric_id": tm, "source_config": source_config,
                "scale_factor": factor, "scale_offset": offset,
                "note": e.get("note"),
            })
        # A scoped row is either the per-source conversion of a benchmark-wide
        # fold (and must name that fold's target, or one source's metric would
        # mean something else) or a standalone conversion that renames nothing.
        for row in fold_rows:
            source_config = row["source_config"]
            if source_config is None:
                continue
            pair = (row["benchmark_id"], row["from_metric_id"])
            unscoped_target = unscoped_pairs.get(pair)
            if unscoped_target is not None:
                if row["to_metric_id"] != unscoped_target:
                    raise typer.BadParameter(
                        f"metric_folds ({pair[0]}): scoped to_metric must match unscoped row"
                    )
            elif not (
                row["to_metric_id"] == row["from_metric_id"]
                and (row["scale_factor"] is not None or row["scale_offset"] is not None)
            ):
                raise typer.BadParameter(
                    f"metric_folds ({pair[0]}): scoped row requires unscoped row for "
                    f"{pair[1]!r}, unless it is a pure published-scale conversion "
                    "(from_metric == to_metric with scale_factor/scale_offset)"
                )
        # No chains: a fold target must not itself be folded on the same benchmark.
        targets = {(r["benchmark_id"], r["to_metric_id"]) for r in fold_rows}
        chained = sorted(targets & set(unscoped_pairs))
        if chained:
            raise typer.BadParameter(f"metric_folds: chained folds {chained}")
        folds_df = pd.DataFrame(
            fold_rows, columns=list(schemas._SCHEMAS["benchmark_metric_folds"].keys())
        )
        store.set_table("benchmark_metric_folds", folds_df)
        typer.echo(f"  metric_folds: {len(fold_rows)}")

    # Needs the full parents graph, so after every model row is present.
    lineage_counts = queries.derive_model_lineage_fields(store)
    typer.echo(
        f"  derived: model_group_id={lineage_counts['group_set']}, "
        f"model_family_id={lineage_counts['family_set']}, "
        f"lineage_origin_model_id={lineage_counts['lineage_model_set']}, "
        f"lineage_origin_model_org_id={lineage_counts['lineage_org_set']}, "
        f"open_weights_inherited={lineage_counts['open_weights_inherited']}, "
        f"release_date_from_id={lineage_counts['release_date_derived_from_id']}"
    )

    removed_entities = 0
    removed_aliases = 0
    if prune_stale:
        # Only seed-originated rows (strategy == "seed", reviewed entities)
        # are pruned; sync-created aliases and auto-drafts are never touched.
        for table, entity_type, yaml_ids, yaml_alias_keys in seed_snapshot:
            aliases_df = store.table("aliases")
            seed_mask = (aliases_df["strategy"] == "seed") & (aliases_df["entity_type"] == entity_type)
            if seed_mask.any():
                seed_aliases = aliases_df[seed_mask]
                stale_alias_mask = seed_mask.copy()
                for idx in seed_aliases.index:
                    row = seed_aliases.loc[idx]
                    sc = row.get("source_config")
                    if _is_na(sc):
                        sc = None
                    key = (row["raw_value"], row["entity_type"], row["canonical_id"], sc)
                    if key in yaml_alias_keys:
                        stale_alias_mask[idx] = False
                n_stale = stale_alias_mask.sum()
                if n_stale > 0:
                    store.set_table("aliases", aliases_df[~stale_alias_mask].reset_index(drop=True))
                    removed_aliases += int(n_stale)

            entity_df = store.table(table)
            if len(entity_df) > 0:
                stale = entity_df["id"].isin(yaml_ids)
                stale_entities = entity_df[~stale & (entity_df["review_status"] == "reviewed")]
                # FK guard: an org auto-created from a model's id prefix is not
                # in the orgs YAML, and pruning it would dangle the model's
                # org_id; parent_org_id references are kept for the same reason.
                if entity_type == "org" and len(stale_entities) > 0:
                    referenced: set[str] = set()
                    models = store.table("canonical_models")
                    if models is not None and len(models) > 0:
                        for col in ("org_id", "lineage_origin_model_org_id"):
                            if col in models.columns:
                                referenced |= {
                                    str(x) for x in models[col].dropna().astype(str)
                                }
                    if "parent_org_id" in entity_df.columns:
                        referenced |= {
                            str(x) for x in entity_df["parent_org_id"].dropna().astype(str)
                        }
                    stale_entities = stale_entities[
                        ~stale_entities["id"].astype(str).isin(referenced)
                    ]
                # An entity referenced by a sync-created alias is kept.
                current_aliases = store.table("aliases")
                for eid in stale_entities["id"]:
                    entity_aliases = current_aliases[
                        (current_aliases["canonical_id"] == eid)
                        & (current_aliases["entity_type"] == entity_type)
                    ]
                    if len(entity_aliases) == 0 or (entity_aliases["strategy"] == "seed").all():
                        entity_df = entity_df[entity_df["id"] != eid]
                        current_aliases = current_aliases[
                            ~((current_aliases["canonical_id"] == eid)
                              & (current_aliases["entity_type"] == entity_type))
                        ]
                        removed_entities += 1
                store.set_table(table, entity_df.reset_index(drop=True))
                store.set_table("aliases", current_aliases.reset_index(drop=True))

    typer.echo(f"  aliases: {alias_count} added, {removed_aliases} removed")
    typer.echo(
        "  source scope keys: "
        + (", ".join(sorted(source_scope_keys)) if source_scope_keys else "(none)")
    )
    if removed_entities:
        typer.echo(f"  stale entities removed: {removed_entities}")

    store.push_to_hub()
    typer.echo("Seed complete.")


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

@app.command()
def stats(
    local: bool = typer.Option(False, "--local", help="Read from fixtures/ instead of HF Hub"),
):
    """Print registry entity counts and pending review summary."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()

    def _row(table):
        df = store.table(table)
        total = len(df)
        draft = int((df["review_status"] == "draft").sum()) if "review_status" in df.columns else 0
        return total, draft

    for label, table in [
        ("models    ", "canonical_models"),
        ("benchmarks", "canonical_benchmarks"),
        ("metrics   ", "canonical_metrics"),
        ("harnesses ", "eval_harnesses"),
    ]:
        total, draft = _row(table)
        typer.echo(f"  {label}  total={total}  draft={draft}")

    aliases_df = store.table("aliases")
    uncertain = int((aliases_df["status"] == "uncertain").sum()) if "status" in aliases_df.columns else 0
    typer.echo(f"\n  aliases        total={len(aliases_df)}  uncertain={uncertain}")
    typer.echo(f"  eval_results   total={len(store.table('eval_results'))}")
    typer.echo(f"  resolution_log total={len(store.table('resolution_log'))}")
    typer.echo(f"  sync_runs      total={len(store.table('sync_runs'))}")


# ------------------------------------------------------------------
# sync
# ------------------------------------------------------------------

@app.command()
def sync(
    config: Optional[str] = typer.Option(None, "--config", help="EEE config name"),
    all_configs: bool = typer.Option(False, "--all", help="Sync all EEE configs"),
    rerun: bool = typer.Option(False, "--rerun", help="Re-resolve all raw strings even if already aliased"),
    local: bool = typer.Option(False, "--local"),
):
    """Batch sync EEE config(s) into eval_results: one row per
    (model x benchmark x metric) with resolved canonical ids."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    if not config and not all_configs:
        typer.echo("Specify --config <name> or --all", err=True)
        raise typer.Exit(1)

    from eval_entity_registry.services.ingestion import run_sync
    import datasets as ds_lib

    store = _load_store()

    configs_to_run: list[str] = []
    if all_configs:
        configs_to_run = ds_lib.get_dataset_config_names("evaleval/EEE_datastore")
    else:
        configs_to_run = [config]

    failed = []
    for cfg in configs_to_run:
        typer.echo(f"Syncing {cfg}...")
        try:
            counts = run_sync(cfg, store, rerun=rerun)
            typer.echo(f"  {cfg}: {counts}")
        except Exception as e:
            typer.echo(f"  {cfg}: FAILED — {e}", err=True)
            failed.append(cfg)

    typer.echo("Persisting tables...")
    store.push_to_hub()

    if failed:
        typer.echo(f"Done with {len(failed)} failed config(s): {', '.join(failed)}")
    else:
        typer.echo("Done.")
