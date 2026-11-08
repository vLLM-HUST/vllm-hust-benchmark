"""Paper metric and independent extraction fixtures; no native model evidence."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "original_codejudge", ROOT / "scripts/run_codejudge_original.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C", "C"),
        ("(A)", "A"),
        ("Analysis fixture. The answer is (B).", "B"),
        ("A or B", None),
        ("The answer is (A) or (B)", None),
        ("I", None),
        ("unformatted fixture", None),
    ],
)
def test_declared_strict_extractor_not_author_regex(text, expected):
    assert MODULE.extract_choice(text, list("ABC")) == expected


def test_macro_f1_keeps_original_class_set_and_failed_extraction():
    result = MODULE.metrics(["A", "B", "C"], ["A", None, "C"], list("ABC"))
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["unparsed_answers"] == 1
    # Sparse diagnostic sample must not quietly change original3-class metric.
    assert MODULE.metrics(["A"], ["A"], list("ABC"))["macro_f1"] == pytest.approx(1 / 3)


def test_published_prompt_used_verbatim_no_gold_or_format_rewrite():
    row = {
        "input": "original published fixture including original example",
        "answer": "C",
    }
    profile = {"model": "fixture", "max_output_tokens": 8192, "disable_thinking": True}
    body = MODULE.request_body(row, profile)
    assert body["messages"] == [{"role": "user", "content": row["input"]}]
    assert "answer" not in body and "tools" not in body
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 8192
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_configuration_label_spaces_kept_separate():
    assert MODULE.labels_for("CodeJudge_Eval_0shot_easy.json") == list("ABC")
    assert MODULE.labels_for("CodeJudge_Eval_1shot_middle.json") == list("ABCDEF")
    assert MODULE.labels_for("CodeJudge_Eval_1shot_hard.json") == list("ABCDEFGHI")
    with pytest.raises(ValueError):
        MODULE.labels_for("unknown.json")


def test_explicit_resume_does_not_repeat_saved_model_requests(tmp_path, monkeypatch):
    name = "CodeJudge_Eval_0shot_easy.json"
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {
            "input": f"original fixture{index}",
            "answer": "C",
            "data_id": index,
            "task_id": 77,
        }
        for index in range(3)
    ]
    (data / name).write_text(json.dumps(rows))
    monkeypatch.setattr(
        MODULE,
        "selection",
        lambda *args: ({"revision": "fixture"}, [{"path": name, "rows": 3}]),
    )
    monkeypatch.setattr(MODULE.IO, "verify", lambda *args: None)
    monkeypatch.setattr(MODULE, "idle", lambda *args: True)
    posts = []

    def transport(request, **kwargs):
        if isinstance(request, str):
            return io.BytesIO(json.dumps({"data": [{"id": "fixture"}]}).encode())
        posts.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "model": "fixture",
                    "choices": [{"message": {"content": "C"}, "finish_reason": "stop"}],
                }
            ).encode()
        )

    monkeypatch.setattr(MODULE, "urlopen", transport)
    args = SimpleNamespace(
        endpoint="http://127.0.0.1:19466/v1",
        files=None,
        data_root=data,
        output=tmp_path / "out",
        max_output_tokens=8192,
        model="fixture",
        disable_thinking=True,
        full=False,
        execute=True,
        resume=False,
        api_server_pid=1,
        npu_engine_pid=2,
    )
    assert MODULE.run(args)["saved_cases"] == 3
    assert len(posts) == 3
    args.resume = True
    result = MODULE.run(args)
    assert len(posts) == 3
    assert result["reused_cases"] == 3
    assert result["model_request_attempts_this_invocation"] == 0
    assert list(args.output.glob("previous-status-*.json"))
