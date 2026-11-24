"""Every metric surface form (display name or alias) lands on its own
canonical under the producer's order: direct non-catch-all hit, else
`extract_metric` over the text. A form landing elsewhere is a silent
mis-merge even when `Resolver.resolve(raw)` is right. The extraction-only
leg is gated by an exact inventory of known mis-routes, so a new compound
name fails and a fix must shrink the list.
"""
import json
from pathlib import Path

import pytest
import yaml
from eval_entity_resolver import Resolver
from eval_entity_resolver.eee import extract_metric

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "fixtures"

# Surface forms the keyword extractor alone routes elsewhere or nowhere. The
# direct alias leg takes them in production; a new entry is a new hazard, and
# shrinking the list is how a fix is recorded.
KNOWN_EXTRACTION_MISROUTES = frozenset({
    ("acc-at-0-25", "Acc@0.25"),
    ("action-space-generalization-mse", "Action Space Generalization MSE"),
    ("action-top-1-accuracy", "Action Top-1 Accuracy"),
    ("air-bench-chat-gpt-4-score", "AIR-Bench Chat GPT-4 Score"),
    ("arena-hard-score", "Arena-Hard Score"),
    ("average-rank", "Average Rank"),
    ("avg-attempts", "Average guesses used per game"),
    ("bd-rate-weighted-psnr-yuv420", "BD-Rate (Weighted PSNR YUV420)"),
    ("bootstrap-score", "Bootstrap Score"),
    ("bootstrap-score", "Bootstrap score"),
    ("codegolf.score", "codegolf.score"),
    # Benchmark-scoped aliases; the extractor reads only the `score` tail.
    ("unsafe-continuation-rate", "polyglotoxicityprompts.score"),
    ("unsafe-continuation-rate", "realtoxicityprompts.score"),
    ("compilation-success-rate-csr", "Compilation Success Rate (CSR)"),
    ("cvebench.mean", "cvebench.mean"),
    ("cyse2-vulnerability-exploit.mean", "cyse2-vulnerability-exploit.mean"),
    ("detection-auroc-only-logical", "Detection AUROC (only logical)"),
    ("detection-auroc-only-structural", "Detection AUROC (only structural)"),
    ("detection-f1-max", "Detection F1-max"),
    ("dpg-bench-score", "DPG-Bench Score"),
    ("dreamgenbench-average-score", "DreamGenBench Average Score"),
    ("drivelm-p1-3-gpt-score-chatgpt-3-5", "DriveLM P1-3 GPT Score (ChatGPT-3.5)"),
    ("driving-score", "Driving Score"),
    ("element-accuracy", "Element Accuracy"),
    ("f1-at-0-25", "F1@0.25"),
    ("f1-at-0-5", "F1@0.5"),
    ("final-acc", "Final Acc"),
    ("final-acc", "Final Accuracy"),
    ("final-score-fs", "Final Score (FS)"),
    ("framework-accuracy-fa", "Framework Accuracy (FA)"),
    ("generated-scene-consistency-mse", "Generated Scene Consistency MSE"),
    ("geneval-score", "GenEval Score"),
    ("healthbench-score", "HealthBench Score"),
    ("hpsv3", "HPSv3 Score"),
    ("human-normalized-score", "Mean Human Normalized Score"),
    ("imgedit-score", "ImgEdit Score"),
    ("inst-level-loose-accuracy", "Instruction-Level Loose Accuracy"),
    ("inst-level-strict-accuracy", "Instruction-Level Strict Accuracy"),
    ("judge-score-1-10", "Score (1-10)"),
    ("l2-bench.mean", "l2-bench.mean"),
    ("laion-aesthetic-score", "LAION Aesthetic Score"),
    ("lenient-accuracy", "Lenient Accuracy"),
    ("long-context-memory-mse", "Long Context Memory MSE"),
    ("longaudiobench-overall-judge-score", "LongAudioBench Overall Judge Score"),
    ("macro-accuracy", "Macro Accuracy"),
    ("median-human-normalized-score", "Median Human Normalized Score"),
    ("median-win-rate", "Median Win Rate"),
    ("mega-bench-macro-score", "Macro score"),
    ("mmau-pro-open-ended-judge-score", "MMAU-Pro Open-Ended Judge Score"),
    ("mmau-pro-overall-weighted-performance", "MMAU-Pro Overall Weighted Performance"),
    ("mmteb-score", "MMTEB Score"),
    ("mteb-score", "MTEB Score"),
    ("non-lin-score", "Non-Lin Score"),
    ("normalized-score", "Normalized Score"),
    ("noun-top-1-accuracy", "Noun Top-1 Accuracy"),
    ("ocrbench-v2-chinese-score", "Chinese Score"),
    ("ocrbench-v2-english-score", "English Score"),
    ("omnicontext-score", "OmniContext score"),
    ("oneig-overall-score", "OneIG Overall Score"),
    ("operation-f1", "Operation F1"),
    ("paired-accuracy", "Paired Accuracy"),
    ("pie-bench-edit-region-clip-similarity", "PIE-Bench Edit Region CLIP Similarity"),
    ("pie-bench-whole-image-clip-similarity", "PIE-Bench Whole Image CLIP Similarity"),
    ("pointmap-accuracy", "Pointmap Accuracy"),
    ("polistemics.adherence", "Polistemics Rubric Score"),
    ("prompt-level-loose-accuracy", "Prompt-Level Loose Accuracy"),
    ("prompt-level-strict-accuracy", "Prompt-Level Strict Accuracy"),
    ("r-at-1-mean-0-3-and-0-5", "R@1 Mean(0.3 and 0.5)"),
    ("r1-at-0-3", "R1@0.3"),
    ("r1-at-0-5", "R1@0.5"),
    ("r1-at-0-7", "R1@0.7"),
    ("radgraph-f1", "RadGraph F1"),
    ("restoration-score-rs", "Restoration Score (RS)"),
    ("risebench-overall-accuracy", "RISEBench Overall Accuracy"),
    ("scicode.main", "SciCode Main-Problem Solve Rate"),
    ("score-0-10", "Score (0-10)"),
    ("segmentation-au-spro-until-fpr-5-pct", "Segmentation AU-sPRO (until FPR 5%)"),
    ("segmentation-f1-max", "Segmentation F1-max"),
    ("segmentation-f1-private", "Segmentation F1 (Private)"),
    ("segmentation-f1-private-mixed", "Segmentation F1 (Private Mixed)"),
    ("segmentation-f1-public", "Segmentation F1 (Public)"),
    ("stft-dist", "STFT-Dist."),
    ("success-suboptimal-rate", "Success + Suboptimal Rate"),
    ("swebench-verified-mini-mariushobbhahn.mean", "swebench-verified-mini-mariushobbhahn.mean"),
    ("top-5-accuracy", "Top-5 Accuracy"),
    ("total-cost", "Total Cost"),
    ("verb-top-1-accuracy", "Verb Top-1 Accuracy"),
    ("vqascore", "VQA Score"),
    ("waics", "WAICS (Average, excl. Incomplete)"),
    ("wb-score", "WildBench Score"),
    ("wise-score", "WISE Score"),
    ("word-accuracy", "Word Accuracy"),
    ("worldmodelbench-total-score", "WorldModelBench Total Score"),
    # OpenEval / tau-bench channel names, each reached by its own alias.
    ("bleu-accuracy", "BLEU Accuracy"),
    ("rouge-1-accuracy", "ROUGE-1 Accuracy"),
    ("rouge-2-accuracy", "ROUGE-2 Accuracy"),
    ("rouge-l-accuracy", "ROUGE-L Accuracy"),
    ("quasi-postfix-exact-match", "Postfix Quasi-Exact Match"),
    ("voice-response-latency", "Voice Response Latency (mean)"),
    ("voice-yield-latency", "Voice Yield Latency (mean)"),
    ("emobench-emotion-quasi-exact-match", "EmoBench Emotion Quasi-Exact Match"),
    ("emobench-cause-quasi-exact-match", "EmoBench Cause Quasi-Exact Match"),
    ("opentom-fine-postfix-exact-match", "OpenToM Fine Postfix Exact Match"),
    ("opentom-coarse-postfix-exact-match", "OpenToM Coarse Postfix Exact Match"),
    ("do-not-answer-action-class", "Do-Not-Answer Action Class (mean)"),
    ("heuristic-accuracy", "Heuristic Accuracy"),
    ("moralbench-human-weighted-score", "MoralBench Human-Weighted Score"),
    ("pass-hat-2", "Pass^2 (all 2 trials)"),
    ("pass-hat-3", "Pass^3 (all 3 trials)"),
    ("pass-hat-4", "Pass^4 (all 4 trials)"),
    # Artificial Analysis, Papers with Code, BenchPress and Vectara names,
    # each reached by its own display name or alias.
    ("ap50", "AP@0.5"),
    ("ap70", "AP@0.7"),
    ("average-test-pass-rate", "Average Test Pass Rate"),
    ("balanced-accuracy", "Balanced Accuracy"),
    ("balanced-accuracy", "Mean Class Accuracy"),
    ("balanced-accuracy", "Unweighted Accuracy (UA/UAR)"),
    ("balanced-accuracy", "Unweighted Average Recall (UAR)"),
    ("blended-price-per-1m-tokens", "Price per 1M Tokens (blended 3:1)"),
    ("blueprint-bench-similarity-score", "Blueprint-Bench Similarity Score"),
    ("ccc", "Concordance correlation coefficient (CCC)"),
    ("chamfer-distance", "Chamfer Distance (normalized cube, 16K points)"),
    ("clip-image-prompt-similarity", "CLIP Image-Prompt Similarity (×100)"),
    ("cumulative-success-rate", "Cumulative Success Rate (CSR)"),
    ("delta-avg", "δ_avg (Average Position Accuracy)"),
    ("f-score-at-0-2", "F-Score@0.2"),
    ("f-score-at-0-2", "F-Score@0.2 (16K points)"),
    ("fd-pointnet", "FD (PointNet++, 4K FPS points)"),
    ("geneval2-score", "GenEval2 Score"),
    ("healthbench-length-adjusted-score", "HealthBench Length-Adjusted Score"),
    ("imagebind-score", "ImageBind Score"),
    ("ivebench-total-score", "IVEBench Total Score"),
    ("judge-rubric-score", "Judge Rubric Score"),
    ("median-output-tokens-per-second", "Median Output Tokens per Second"),
    ("median-time-to-first-answer-token", "Median Time to First Answer Token"),
    ("median-time-to-first-token", "Median Time to First Token"),
    ("mmteb-rerank-score", "MMTEB Rerank Score"),
    ("mmteb-retrieval-score", "MMTEB Retrieval Score"),
    ("ndtw", "Normalized Dynamic Time Warping"),
    ("omniparsebench-layout", "OmniParseBench Layout Pass Rate"),
    ("omniparsebench-tables", "OmniParseBench Tables Pass Rate"),
    ("omniparsebench-text", "OmniParseBench Text Pass Rate"),
    ("page-grounding-f1", "Page Grounding F1"),
    ("parsebench-charts", "ParseBench Charts Score"),
    ("parsebench-content-faithfulness", "ParseBench Content Faithfulness Score"),
    ("parsebench-overall", "ParseBench Overall Score"),
    ("parsebench-semantic-formatting", "ParseBench Semantic Formatting Score"),
    ("parsebench-tables", "ParseBench Tables Score"),
    ("parsebench-visual-grounding", "ParseBench Visual Grounding Score"),
    ("partial-credit-score", "Partial-Credit Score"),
    ("physics-iq-verified-score-i2v", "Physics-IQ Verified Score (I2V)"),
    ("physics-iq-verified-score-v2v", "Physics-IQ Verified Score (V2V)"),
    ("price-per-1m-input-tokens", "Price per 1M Input Tokens"),
    ("price-per-1m-output-tokens", "Price per 1M Output Tokens"),
    ("qwen-image-bench-quality", "Qwen-Image-Bench Quality Score"),
    ("rank-1-identification", "Rank-1 Identification"),
    ("rank-5-identification", "Rank-5 Identification"),
    ("rubric-criteria-pass-rate", "Rubric Criteria Pass Rate"),
    ("skill-score", "Skill Score"),
    ("skill-score", "Skill Score (%)"),
    ("speecheditbench-acoustic-joint-success", "SpeechEditBench Acoustic Joint Success"),
    ("speecheditbench-compositional-joint-success", "SpeechEditBench Compositional Joint Success"),
    ("speecheditbench-content-joint-success", "SpeechEditBench Content Joint Success"),
    ("speecheditbench-emotion-joint-success", "SpeechEditBench Emotion Joint Success"),
    ("speecheditbench-paralinguistic-joint-success", "SpeechEditBench Paralinguistic Joint Success"),
    ("speecheditbench-prosody-joint-success", "SpeechEditBench Prosody Joint Success"),
    ("speecheditbench-style-joint-success", "SpeechEditBench Style Joint Success"),
    ("spl", "Success weighted by Path Length"),
    ("sql-average-win-rate", "SQL Avg. Win Rate"),
    ("sql-average-win-rate", "SQL Avg. Win Rate (%)"),
    ("task-success-rate", "Task Success Rate (TSR)"),
    ("vbench-2-0-commonsense-score", "VBench-2.0 Commonsense Score"),
    ("vbench-2-0-controllability-score", "VBench-2.0 Controllability Score"),
    ("vbench-2-0-creativity-score", "VBench-2.0 Creativity Score"),
    ("vbench-2-0-human-fidelity-score", "VBench-2.0 Human Fidelity Score"),
    ("vbench-2-0-physics-score", "VBench-2.0 Physics Score"),
    ("vbench-2-0-total-score", "VBench-2.0 Total Score"),
    ("vbench-quality-score", "VBench Quality Score"),
    ("vbench-semantic-score", "VBench Semantic Score"),
    ("vbench-total-score", "VBench Total Score"),
    ("word-grounding-f1", "Word Grounding F1"),
    # Terminal-Bench pass@k, reached by a scoped alias.
    ("pass-at-2", "Pass@2 (any of 2 trials)"),
    ("pass-at-3", "Pass@3 (any of 3 trials)"),
    ("pass-at-4", "Pass@4 (any of 4 trials)"),
    ("pass-at-5", "Pass@5 (any of 5 trials)"),
})

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-entity-registry seed --local`",
)


def _entries():
    return yaml.safe_load((_ROOT / "seed" / "metrics.yaml").read_text())


def _catch_all_ids():
    return {e["id"] for e in _entries()
            if json.loads(e.get("metadata") or "{}").get("catch_all")}


def _surface_forms(only=None):
    for entry in _entries():
        if only is None or entry["id"] in only:
            for raw in [entry.get("display_name"), *(entry.get("aliases") or [])]:
                if raw:
                    yield entry["id"], raw


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def catch_all():
    return _catch_all_ids()


def _producer_order(resolver, catch_all, raw):
    direct = resolver.resolve(raw, entity_type="metric").canonical_id
    if direct and direct not in catch_all:
        return direct
    extracted = extract_metric(raw)
    return resolver.resolve(extracted, entity_type="metric").canonical_id if extracted else None


@pytest.mark.parametrize("canonical,raw", sorted(_surface_forms()))
def test_producer_order_lands_on_its_own_canonical(resolver, catch_all, canonical, raw):
    got = _producer_order(resolver, catch_all, raw)
    assert got == canonical, f"{raw!r} -> {got!r} on the producer path, expected {canonical!r}"


def test_extraction_alone_introduces_no_new_misroute(resolver):
    """A fresh mis-route means a compound name the keyword table lacks: add
    a compound pattern before the generic one, not a new exemption."""
    fresh = sorted(
        (canonical, raw) for canonical, raw in _surface_forms()
        if (canonical, raw) not in KNOWN_EXTRACTION_MISROUTES
        and (resolver.resolve(extract_metric(raw), entity_type="metric").canonical_id
             if extract_metric(raw) else None) != canonical
    )
    assert fresh == [], f"new extraction-only mis-routes: {fresh}"
    actual = {
        (canonical, raw) for canonical, raw in _surface_forms()
        if (resolver.resolve(extract_metric(raw), entity_type="metric").canonical_id
            if extract_metric(raw) else None) != canonical
    }
    stale = sorted(KNOWN_EXTRACTION_MISROUTES - actual)
    assert stale == [], f"inventory entries the extractor now routes correctly; remove them: {stale}"


def test_extraction_inventory_names_real_entries():
    ids = {e["id"] for e in _entries()}
    stale = sorted(c for c, _ in KNOWN_EXTRACTION_MISROUTES if c not in ids)
    assert stale == [], f"inventory names metrics that no longer exist: {stale}"
