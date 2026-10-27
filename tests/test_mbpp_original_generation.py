"""Inert task/transport fixtures; never executes Python predictions."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "mbpp_generate", ROOT / "scripts/run_mbpp_original_generation.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def task(index):
    return {
        "task_id": index,
        "text": f"fixture question {index}",
        "code": f"original solution {index}",
        "test_list": [f"fixture test {index} {i}" for i in range(3)],
        "test_setup_code": "fixture setup",
        "challenge_test_list": [],
    }


def examples():
    return [task(i) for i in (2, 3, 4)]


def test_author_template_keeps_all_tests_and_only_demonstration_solutions():
    prompt = RUNNER.original_prompt(task(11), examples())
    for i in (2, 3, 4):
        assert f"original solution {i}\n[DONE]" in prompt
    assert "original solution 11" not in prompt
    assert "fixture question 11" in prompt
    assert all(f"fixture test 11 {i}" in prompt for i in range(3))
    assert prompt.endswith("\n[BEGIN]\n")
    assert prompt.count("[DONE]") == 3


def test_selected_fixed_range_and_full_use_original_positions():
    rows = [task(i) for i in range(11, 511)]
    selected = RUNNER.selected_tasks(rows)
    assert [(i, row["task_id"]) for i, row in selected] == [(0, 11), (1, 12), (2, 13)]
    assert len(RUNNER.selected_tasks(rows, full=True)) == 500
    assert RUNNER.selected_tasks(rows, start=4)[0][1]["task_id"] == 15
    with pytest.raises(ValueError):
        RUNNER.selected_tasks(rows, count=2)
    with pytest.raises(ValueError):
        RUNNER.selected_tasks(rows, full=True, start=1)
    rows[-1] = task(11)
    with pytest.raises(ValueError):
        RUNNER.selected_tasks(rows)


@pytest.mark.parametrize("change", ["missing-test", "wrong-demo", "target-not-test"])
def test_prompt_never_falls_back_to_rewritten_tasks(change):
    item, demos = task(11), examples()
    if change == "missing-test":
        item["test_list"].pop()
    elif change == "wrong-demo":
        demos[0] = task(1)
    else:
        item["task_id"] = 2
    with pytest.raises(ValueError):
        RUNNER.original_prompt(item, demos)


@pytest.mark.parametrize(
    "finish,text", [("stop", "predicted code"), ("length", "incomplete"), ("stop", "")]
)
def test_raw_generations_are_retained_without_host_code_execution(
    tmp_path, finish, text
):
    requests = []

    def transport(url, body):
        requests.append((url, body))
        if url.endswith("/tokenize"):
            return {"count": 10}
        return {
            "choices": [{"text": text, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1 if text else 0},
        }

    result = RUNNER.generate(
        [(0, task(11))],
        examples(),
        output=tmp_path / "out",
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture",
        max_tokens=32,
        context=128,
        transport=transport,
    )
    assert requests[-1][0].endswith("/completions")
    assert "messages" not in requests[-1][1]
    assert requests[-1][1]["stop"] == ["[DONE]"]
    assert not result["functional_tests_executed"]
    assert not result["source_checkbox_eligible"]
    saved = json.loads((tmp_path / "out/task-11.private.json").read_text("utf-8"))
    assert saved["response"]["choices"][0]["text"] == text
    assert saved["original_task"] == task(11)


@pytest.mark.parametrize("failure", ["context", "transport", "response", "asset"])
def test_failures_preserve_attempt_without_changing_task(tmp_path, failure):
    calls = []

    def transport(url, body):
        calls.append(url)
        if failure == "transport":
            raise RuntimeError("fixture transport failure")
        if url.endswith("/tokenize"):
            return {"count": 127 if failure == "context" else 10}
        return {
            "choices": []
            if failure == "response"
            else [{"text": "prediction", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

    def verify():
        if failure == "asset":
            raise ValueError("fixture asset change")

    with pytest.raises((ValueError, RuntimeError)):
        RUNNER.generate(
            [(0, task(11)), (1, task(12))],
            examples(),
            output=tmp_path / "out",
            endpoint="http://127.0.0.1:19466/v1",
            model="fixture",
            max_tokens=32,
            context=128,
            transport=transport,
            final_verify=verify,
        )
    receipt = json.loads((tmp_path / "out/receipt-private.json").read_text("utf-8"))
    assert receipt["status"] == "FAILED_NOT_COMPLETE"
    assert (tmp_path / "out/task-11.private.json").is_file()
    if failure != "asset":
        assert not (tmp_path / "out/task-12.private.json").exists()
        assert len(calls) <= 2


def test_prepared_parquet_to_cli_request_chain_with_no_actual_model(
    tmp_path, monkeypatch, capsys
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "assets"
    files = []
    for split, ids in (("test", range(11, 511)), ("prompt", range(1, 11))):
        relative = f"full/{split}-00000-of-00001.parquet"
        path = root / "data" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([task(i) for i in ids]), path)
        raw = path.read_bytes()
        files.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    pins = tmp_path / "fixture-pins.json"
    pins.write_text(
        json.dumps(
            {
                "dataset": {
                    "repo_id": "fixture-not-publisher-data",
                    "revision": "fixture",
                    "files": files,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RUNNER, "PINS", pins)
    calls = []

    def transport(url, body):
        calls.append((url, body))
        if url.endswith("/tokenize"):
            return {"count": 10}
        return {
            "choices": [{"text": "fixture prediction", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

    actual = RUNNER.generate
    monkeypatch.setattr(
        RUNNER,
        "generate",
        lambda *args, **kwargs: actual(*args, **kwargs, transport=transport),
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--assets",
            str(root),
            "--output",
            str(output),
            "--model",
            "fixture-model",
            "--max-tokens",
            "32",
            "--context-capacity",
            "128",
        ],
    )
    RUNNER.main()
    assert json.loads(capsys.readouterr().out)["tasks"] == 3
    assert len(calls) == 6
    receipt = json.loads((output / "receipt-private.json").read_text("utf-8"))
    assert receipt["selected_task_ids"] == [11, 12, 13]
    assert receipt["source_identity"]["revision"] == "fixture"
    assert not receipt["functional_tests_executed"]
