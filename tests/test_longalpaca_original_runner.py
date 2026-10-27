"""Software-only instruction transport fixtures; no dataset/model inference."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "longalpaca_original", ROOT / "scripts/run_longalpaca_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_original_instruction_and_input_not_reference_or_provenance():
    row = {
        "instruction": " unchanged question\u2028including whitespace ",
        "input": " original context ",
        "output": "hidden reference",
        "file": "source-document-only.pdf",
    }
    assert runner.messages(row) == [
        {
            "role": "user",
            "content": " unchanged question\u2028including whitespace \n\n original context ",
        }
    ]
    assert (
        row["output"] == "hidden reference"
        and row["file"] == "source-document-only.pdf"
    )


@pytest.mark.parametrize("extra", [{}, {"input": ""}])
def test_no_input_does_not_invent_missing_context(extra):
    row = {"instruction": "original instruction", "output": "reference", **extra}
    assert runner.messages(row) == [{"role": "user", "content": "original instruction"}]


def test_fixed_positions_are_not_answer_filtered():
    rows = [{"instruction": str(i), "output": ""} for i in range(12000)]
    assert [i for i, _ in runner.select(rows)] == [0, 1, 2]
    assert [i for i, _ in runner.select(rows, start=11997)] == [11997, 11998, 11999]
    assert len(runner.select(rows, full=True)) == 12000
    for invalid in (
        {"start": True},
        {"count": 2},
        {"start": 11998},
        {"full": True, "count": 4},
    ):
        with pytest.raises(ValueError):
            runner.select(rows, **invalid)
