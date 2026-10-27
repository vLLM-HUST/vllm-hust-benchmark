"""Original pair fidelity and declared model-adapter contracts; no real NPU."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
runner = importlib.import_module("run_paws_original")
auditor = importlib.import_module("verify_paws_original_saved")


def test_labels_and_identifiers_never_enter_input():
    row = {
        "id": "private-id",
        "sentence1": 'A "quoted" original sentence.',
        "sentence2": "原句不改写",
        "label": 1,
    }
    expected = runner.messages(row)
    row.update(id="different-id", label=0)
    assert runner.messages(row) == expected
    body = json.loads(expected[0]["content"][len(runner.INSTRUCTION) :])
    assert body == {"sentence1": row["sentence1"], "sentence2": row["sentence2"]}


@pytest.mark.parametrize(
    "answer,expected", [("0", 0), (" 1\n", 1), ("0 because...", None), ("", None)]
)
def test_strict_prediction_no_repair(answer, expected):
    value = {
        "choices": [{"message": {"content": answer}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 16, "total_tokens": 46},
    }
    assert runner.prediction(value) == expected
    value["usage"]["total_tokens"] = 0
    with pytest.raises(ValueError):
        runner.prediction(value)


def test_noisy_labels_are_not_gold_test_accuracy():
    common = {"label": 1, "prediction": None, "finish_reason": "length"}
    scored = runner.score(
        [
            {**common, "file": "labeled_final/test-a.parquet"},
            {**common, "file": "unlabeled_final/train-a.parquet"},
        ]
    )
    assert scored["labeled_final/test-a.parquet"]["accuracy"] == 0
    noisy = scored["unlabeled_final/train-a.parquet"]
    assert "accuracy" not in noisy
    assert noisy["noisy_label_agreement_not_accuracy"] == 0


def test_default_plan_no_endpoint_or_output(monkeypatch, tmp_path, capsys):
    output = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--data-root", "absent", "--output", str(output), "--full-test"],
    )
    monkeypatch.setattr(
        runner, "urlopen", lambda *a, **kw: pytest.fail("No model calls in plan")
    )
    runner.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


def test_fixture_full_test_and_all_auxiliary_files(monkeypatch, tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    files = []
    names = [
        "labeled_final/test",
        "labeled_final/train",
        "labeled_final/validation",
        "labeled_swap/train",
        "unlabeled_final/train",
        "unlabeled_final/validation",
    ]
    data = tmp_path / "data"
    for name in names:
        relative = name + "-00000-of-00001.parquet"
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "id": i,
                        "sentence1": "original first",
                        "sentence2": "original second",
                        "label": i % 2,
                    }
                    for i in range(4)
                ]
            ),
            path,
        )
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"dataset": {"revision": "fixture", "files": files}}))
    monkeypatch.setattr(runner, "PINS", pins)
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.payload

    def fake_open(request, **kwargs):
        if isinstance(request, str):
            return Response(
                b'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
            )
        body = json.loads(request.data)
        assert body["max_tokens"] == 16 and body["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        calls.append(body)
        return Response(
            json.dumps(
                {
                    "choices": [{"message": {"content": "1"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 1,
                        "total_tokens": 31,
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(runner, "urlopen", fake_open)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--data-root",
            str(data),
            "--output",
            str(output),
            "--full-test",
            "--disable-thinking",
            "--execute",
        ],
    )
    runner.main()
    receipt = json.loads((output / "receipt-private.json").read_text())
    assert receipt["completed_tasks"] == len(calls) == 19
    assert receipt["scores"][files[0]["path"]]["tasks"] == 4
    assert not receipt["source_checkbox_eligible"]
    monkeypatch.setattr(auditor, "PINS", pins)
    audit = tmp_path / "audit"
    monkeypatch.setattr(
        runner,
        "urlopen",
        lambda *a, **kw: pytest.fail("Saved audit must not call model"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--data-root",
            str(data),
            "--run",
            str(output),
            "--output",
            str(audit),
            "--execute",
        ],
    )
    auditor.main()
    proof = json.loads((audit / "receipt-public.json").read_text())
    assert proof["tasks"] == 19 and proof["new_model_calls"] == 0
    # A modified original pair cannot be hidden by reusing the stored score.
    request = output / "file-0/row-0.request-private.json"
    request.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--data-root",
            str(data),
            "--run",
            str(output),
            "--output",
            str(tmp_path / "audit-second"),
            "--execute",
        ],
    )
    with pytest.raises(ValueError):
        auditor.main()
    capsys.readouterr()
