"""Dictionary compatibility fixtures, not native task or quality evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dureader_score", ROOT / "scripts/score_dureader_original.py"
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_reorders_keys_without_replacing_values_or_references():
    references = {"second": ["ref2"], "first": ["ref1"]}
    predictions = {"first": ["wrong"], "second": [""]}
    calls = []

    def original(pred, ref, bleu_order):
        assert list(pred) == list(ref)
        assert ref is references
        assert pred["first"] is predictions["first"]
        assert pred["second"] is predictions["second"]
        calls.append(bleu_order)
        return {"unchanged": True}

    module = SimpleNamespace(compute_bleu_rouge=original)
    with scorer.aligned_metric_keys(module):
        assert module.compute_bleu_rouge(predictions, references) == {"unchanged": True}
    assert module.compute_bleu_rouge is original
    assert list(predictions) == ["first", "second"]
    assert calls == [4]


def test_missing_keys_remain_failure_and_wrapper_restores_after_error():
    def original(*args):
        pytest.fail("must not call metric with mismatched keys")

    module = SimpleNamespace(compute_bleu_rouge=original)
    with pytest.raises(ValueError, match="key sets differ"):
        with scorer.aligned_metric_keys(module):
            module.compute_bleu_rouge(
                {"unexpected": ["answer"]}, {"original": ["answer"]}
            )
    assert module.compute_bleu_rouge is original


def test_score_calls_all_publisher_branches_and_retains_empty_predictions():
    calls = []
    predictions, references = {}, {"original": {"answers": ["reference"]}}

    def metric(pred, ref, task, source):
        assert pred is predictions and ref is references and source == "both"
        calls.append(task)
        return {"retained_bad_score": 0}

    module = SimpleNamespace(compute_bleu_rouge=lambda *a: None, get_metrics=metric)
    result = scorer.score(module, predictions, references)
    assert calls == ["main", "all", "description", "entity", "yesno"]
    assert result["all"] == {"retained_bad_score": 0}


def test_undefined_native_metric_is_null_not_fake_zero():
    import json

    original = {
        "yesno": {"Depends|Rouge-L": float("nan"), "Bleu-4": 0.0},
        "main": {"Rouge-L": 0.25},
    }
    report = scorer.metric_report(original)
    assert report["scores"]["yesno"]["Depends|Rouge-L"] is None
    assert report["scores"]["yesno"]["Bleu-4"] == 0.0
    assert report["scores"]["main"]["Rouge-L"] == 0.25
    assert report["undefined_metrics"] == ["yesno/Depends|Rouge-L"]
    assert "NaN" not in json.dumps(report, allow_nan=False)
