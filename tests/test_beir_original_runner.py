"""CPU-only protocol fixtures; not model execution evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "beir_runner", Path(__file__).resolve().parents[1] / "scripts/run_beir_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_pins_keep_native_framework_and_two_distinct_models():
    pins = json.loads(runner.PINS.read_text(encoding="utf-8"))
    assert pins["framework_version"] == "2.2.0"
    assert len(pins["models"]) == 2
    assert pins["protocol"]["rerank_top_k"] == 100
    assert pins["protocol"]["encoder_native_max_length"] == 512
    assert (
        pins["models"][0]["repo_id"]
        == "sentence-transformers/msmarco-distilbert-base-tas-b"
    )
    assert pins["models"][1]["repo_id"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert all(
        len(model["revision"]) == 40 and model["files"] for model in pins["models"]
    )


def test_full_query_and_known_document_contract():
    runner.validate_rankings({"q": {"d": 0.5}}, queries={"q"}, corpus={"d"})
    with pytest.raises(ValueError, match="every original test query"):
        runner.validate_rankings({}, queries={"q"}, corpus={"d"})
    with pytest.raises(ValueError, match="unknown original document"):
        runner.validate_rankings({"q": {"other": 1}}, queries={"q"}, corpus={"d"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.5"])
def test_nonfinite_or_wrong_score_types_fail(bad):
    with pytest.raises(ValueError, match="finite original retrieval scores"):
        runner.validate_rankings({"q": {"d": bad}}, queries={"q"}, corpus={"d"})


def test_rerank_pool_must_be_exact_dense_top_k_not_gold_candidates():
    dense = {"q": {"d1": 1.0, "d2": 0.7, "d3": 0.1}}
    runner.validate_rankings(
        {"q": {"d2": 5.0, "d1": 0.0}},
        queries={"q"},
        corpus={"d1", "d2", "d3"},
        dense=dense,
        top_k=2,
    )
    with pytest.raises(ValueError, match="exactly the original dense top-k"):
        runner.validate_rankings(
            {"q": {"d3": 5.0, "d1": 0.0}},
            queries={"q"},
            corpus={"d1", "d2", "d3"},
            dense=dense,
            top_k=2,
        )


def test_duplicate_result_ids_cannot_be_silently_dropped(tmp_path):
    path = tmp_path / "rankings.json"
    path.write_text('{"q":{"d":1,"d":2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate original ranking key"):
        runner.read_rankings(path)


def test_dense_ranking_cannot_omit_unjudged_documents_to_fake_completeness():
    with pytest.raises(ValueError, match="complete original dense top-1000"):
        runner.validate_rankings(
            {"q": {"positive": 1}}, queries={"q"}, corpus={"positive", "other"}
        )


def test_official_scorer_mutation_does_not_change_input_rankings():
    qrels, rankings = {"q": {"d": 1}}, {"q": {"d": 0.5}}

    def evaluate(refs, results, ks):
        assert refs == qrels and results == rankings and ks == [1]
        refs.clear()
        results.clear()
        return ({"NDCG@1": 1}, {}, {}, {})

    def custom(refs, results, ks, metric):
        assert refs == qrels and results == rankings and metric == "mrr"
        return {"MRR@1": 1}

    result = runner.score_original(
        SimpleNamespace(evaluate=evaluate, evaluate_custom=custom, k_values=[1]),
        qrels,
        rankings,
    )
    assert result["metrics"]["ndcg"] == {"NDCG@1": 1}
    assert qrels == {"q": {"d": 1}} and rankings == {"q": {"d": 0.5}}


def test_cpu_model_and_implicit_reuse_are_rejected():
    base = [
        "--source-id",
        "BEIR_SCIFACT",
        "--data-root",
        "data",
        "--output",
        "out",
        "--encoder-root",
        "model1",
        "--cross-encoder-root",
        "model2",
    ]
    with pytest.raises(SystemExit):
        runner.parse_args(base + ["--device", "cpu"])
    with pytest.raises(SystemExit):
        runner.parse_args(base + ["--dense-results", "old.json"])
    assert runner.parse_args(base + ["--device", "npu:1"]).mode == "run"


def test_score_requires_both_protocol_results():
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--source-id",
                "BEIR_NFCORPUS",
                "--data-root",
                "data",
                "--output",
                "out",
                "--mode",
                "score",
                "--dense-results",
                "one.json",
            ]
        )
