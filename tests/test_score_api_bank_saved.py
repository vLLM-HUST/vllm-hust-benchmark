"""Exercise the saved-output contract without upstream dependencies."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_api_bank_saved.py"
FILES = (
    "AddAgenda-AddAlarm-GetUserToken-level-2-1.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-2.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-3.jsonl",
)


@pytest.mark.parametrize(
    "mode",
    ["complete", "missing", "duplicate", "unexpected", "non_object", "no_output"],
)
def test_fixed_scoring_positions(tmp_path, mode):
    inputs = tmp_path / "lv1-lv2-samples/level-1-given-desc"
    inputs.mkdir(parents=True)
    (tmp_path / "api_call_extraction.py").write_text(
        "def get_api_call(output): return output\n"
        "def parse_api_call(call): return call, {}\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluator_by_json.py").write_text(
        "from types import SimpleNamespace\n"
        "class Sample:\n"
        "    @staticmethod\n"
        "    def from_chat_history(rows):\n"
        "        return [SimpleNamespace(ground_truth=r) for r in rows]\n"
        "class Evaluator:\n"
        "    def __init__(self, samples): pass\n"
        "    def evaluate(self, i, output):\n"
        "        return (True,) if i == 0 else 0.5\n",
        encoding="utf-8",
    )
    predictions = []
    for name in FILES:
        rows = [{"role": "API", "api_name": "AddAgenda"}, {"role": "AI"}]
        (inputs / name).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        predictions.extend(
            {"file": name, "id": i, "output": output}
            for i, output in enumerate(("AddAgenda", "reply"))
        )
    predictions[2]["output"] = "UnlistedTool"
    if mode == "missing":
        predictions.pop()
    elif mode == "duplicate":
        predictions.append(predictions[0])
    elif mode == "unexpected":
        predictions[-1]["id"] = 10
    elif mode == "non_object":
        predictions[-1] = []
    elif mode == "no_output":
        predictions[-1].pop("output")
    saved = tmp_path / "saved.jsonl"
    saved.write_text("\n".join(map(json.dumps, predictions)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--author-root",
            str(tmp_path),
            "--predictions",
            str(saved),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if mode == "complete":
        assert result.returncode == 0, result.stderr
        assert [row["score"] for row in json.loads(result.stdout)] == [
            True,
            0.5,
            False,
            0.5,
            True,
            0.5,
        ]
    else:
        assert result.returncode == 2
        assert not result.stdout
