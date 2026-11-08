"""Inert scoring-binding fixtures, no original tasks, judge calls or scores."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "discovery_judge_driver", ROOT / "scripts/run_discoverybench_original_judge.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_exact_original_marker_preserves_both_fields_without_rewrite():
    raw = "  original hypothesis\nWORKFLOW SUMMARY:\n original workflow "
    assert entry.prediction_fields(raw) == (
        "  original hypothesis\n",
        "\n original workflow ",
        "exact_original_workflow_marker",
    )


@pytest.mark.parametrize(
    "raw",
    [
        "whole original output",
        "A WORKFLOW SUMMARY: B WORKFLOW SUMMARY: C",
        "WORKFLOW SUMMARY: no hypothesis prefix",
    ],
)
def test_no_guessing_when_marker_absent_or_ambiguous(raw):
    assert entry.prediction_fields(raw) == (
        raw,
        None,
        "raw_original_output_no_unambiguous_workflow_marker",
    )


@pytest.mark.parametrize("raw", ["", None, {}])
def test_no_invented_prediction(raw):
    with pytest.raises(ValueError):
        entry.prediction_fields(raw)


def test_identity_mismatch_rejected(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.verified_json(path, "not-the-original-hash")
