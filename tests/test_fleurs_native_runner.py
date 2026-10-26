"""Inert waveform/score fixtures; no original audio, model or accelerator run."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_fleurs_original")
scorer = load("score_fleurs_original")


def test_fixed_sampling_preserves_positions_and_explicit_full():
    assert list(runner.original_indices(264)) == [0, 1, 2]
    assert len(runner.original_indices(264, True)) == 264
    with pytest.raises(ValueError):
        runner.original_indices(0)


def test_bad_waveform_is_rejected_not_resampled():
    sf = SimpleNamespace(
        info=lambda _: SimpleNamespace(channels=2, frames=2),
        read=lambda *_args, **_kwargs: (SimpleNamespace(ndim=2), 8000),
    )
    with pytest.raises(ValueError, match="no repair"):
        runner.decode_original_audio(
            {"audio": {"bytes": b"inert"}, "num_samples": 2}, sf
        )


def fixture_run(tmp_path):
    cases = []
    for position in range(3):
        path = tmp_path / f"case-{position}.json"
        path.write_text(
            json.dumps(
                {
                    "locale": "af_za",
                    "position": position,
                    "original": {"transcription": "fixture reference"},
                    "result": {"text": ""},
                }
            ),
            encoding="utf-8",
        )
        cases.append(
            {
                "locale": "af_za",
                "position": position,
                "file": path.name,
                "sha256": scorer.sha(path),
            }
        )
    return {
        "source_id": "FLEURS",
        "status": "NATIVE_EXECUTION_COMPLETE_SCORE_PENDING",
        "failures": [],
        "full": False,
        "selected_locales": ["af_za"],
        "expected_cases_per_locale": {"af_za": 3},
        "cases": cases,
    }


def test_empty_native_predictions_remain_in_scoring(tmp_path):
    receipt = fixture_run(tmp_path)
    rows = scorer.collect_cases(tmp_path, receipt)
    assert rows["af_za"] == [("fixture reference", "")] * 3


@pytest.mark.parametrize(
    "change", ["failure", "missing_locale", "duplicate", "mutated", "incomplete"]
)
def test_invalid_or_incomplete_native_results_cannot_score(tmp_path, change):
    receipt = fixture_run(tmp_path)
    if change == "failure":
        receipt["failures"] = [{"position": 0}]
    elif change == "missing_locale":
        receipt["selected_locales"].append("am_et")
    elif change == "duplicate":
        receipt["cases"].append(receipt["cases"][0])
    elif change == "mutated":
        (tmp_path / receipt["cases"][0]["file"]).write_text("changed")
    else:
        receipt["cases"].pop()
    with pytest.raises(ValueError):
        scorer.collect_cases(tmp_path, receipt)
