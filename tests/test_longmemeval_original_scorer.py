"""Software fixtures only: no real dataset, model or scoring claim."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "longmem_scorer", ROOT / "scripts/score_longmemeval_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_original_ids_and_saved_answers_are_required():
    refs = [{"question_id": "original", "question_type": "single-session-assistant"}]
    good = [{"question_id": "original", "hypothesis": "saved answer"}]
    assert runner.checked_inputs(refs, good) == {"single-session-assistant"}
    for bad in (
        [],
        good * 2,
        [{"question_id": "other", "hypothesis": "x"}],
        [{**good[0], "autoeval_label": {}}],
        [{**good[0], "hypothesis": None}],
    ):
        with pytest.raises(ValueError):
            runner.checked_inputs(refs, bad)


def controls():
    return {
        "model": "explicit-model",
        "status": "LOCAL_JUDGE_SELECTED_TYPE_SANITY_PASSED",
        "control_cases": [
            {
                "control_id": name,
                "expected": label,
                "actual_original_parse_label": label,
                "finish_reason": "stop",
            }
            for name, label in (
                ("positive-equivalent", True),
                ("negative-wrong", False),
                ("negative-empty", False),
            )
        ],
    }


def test_controls_do_not_qualify_another_model_or_question_type():
    runner.checked_controls(controls(), "explicit-model", {"single-session-assistant"})
    with pytest.raises(ValueError):
        runner.checked_controls(
            controls(), "another-model", {"single-session-assistant"}
        )
    with pytest.raises(ValueError):
        runner.checked_controls(controls(), "explicit-model", {"temporal-reasoning"})
    report = controls()
    report["control_cases"][-1]["actual_original_parse_label"] = True
    with pytest.raises(ValueError):
        runner.checked_controls(report, "explicit-model", {"single-session-assistant"})


@pytest.mark.parametrize("question_type", ["temporal-reasoning", "single-session-user"])
def test_explicit_type_controls_do_not_expand_to_other_types_or_abstention(
    question_type,
):
    report = {**controls(), "question_type": question_type, "abstention": False}
    runner.checked_controls(report, "explicit-model", {question_type})
    with pytest.raises(ValueError):
        runner.checked_controls(report, "explicit-model", {"single-session-assistant"})
    with pytest.raises(ValueError):
        runner.checked_controls(
            report, "explicit-model", {question_type}, abstention=True
        )


def test_selector_and_constructor_only_are_bound(monkeypatch):
    raw = b"model_zoo = {'official': ('official-model', 'local')}\nclient = OpenAI(api_key='fixture')\nresult = 'yes' in 'yesterday'\n"  # pragma: allowlist secret - inert source fixture
    monkeypatch.setattr(runner, "SCORER_SHA256", hashlib.sha256(raw).hexdigest())
    namespace = {"_BoundOpenAI": lambda **kwargs: kwargs}
    exec(runner.bound_program(raw, "explicit-model"), namespace)
    assert namespace["model_zoo"] == {
        "official": ("official-model", "local"),
        runner.ALIAS: ("explicit-model", "local"),
    }
    assert namespace["client"] == {
        "api_key": "fixture",  # pragma: allowlist secret - dummy fixture key
    }
    assert namespace["result"] is True  # Original substring parsing is not replaced.
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.bound_program(raw + b"# changed", "explicit-model")


def test_delivery_receipt_separates_native_scoring_and_consumer_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    source = ledger["delivery_inventory"]["LONGMEMEVAL"]
    receipt = json.loads((ROOT / source["delivery_receipt"]).read_text("utf-8"))
    assert source["checked"] and source["status"] == "AVAILABLE"
    assert receipt["source_checkbox_eligible"]
    assert receipt["native_execution"]["original_cases"] == 9
    assert (
        receipt["native_execution"][
            "reader_generations_repeated_for_scoring_or_handoff"
        ]
        == 0
    )
    assert receipt["qa_scoring"]["original_cases_scored"] == 9
    assert not receipt["qa_scoring"][
        "publisher_recommended_70b_or_gpt4_judge_reproduced"
    ]
    assert (
        receipt["public_and_research_handoff"]["original_complete_record_roundtrips"]
        == 9
    )
    assert receipt["public_and_research_handoff"]["model_calls"] == 0
