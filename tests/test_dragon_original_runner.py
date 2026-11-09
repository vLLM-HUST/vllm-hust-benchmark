"""Protocol/transport fixtures only; no corpus, model import or device execution."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "dragon_run", SCRIPTS / "run_dragon_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_missing_separate_config_uses_original_model_config_only(tmp_path):
    class Model:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return "original-model-config"

    class Generation:
        @staticmethod
        def from_model_config(value):
            assert value == "original-model-config"
            return "derived-not-fake-publisher-file"

    value, source = runner.reader_generation_config(tmp_path, Generation, Model)
    assert (
        value == "derived-not-fake-publisher-file"
        and "no_separate_generation_file" in source
    )
    assert not list(tmp_path.iterdir())


def test_malformed_existing_generation_config_must_not_fallback(tmp_path):
    (tmp_path / "generation_config.json").write_text("broken original")

    class Generation:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("malformed original")

    with pytest.raises(ValueError, match="malformed"):
        runner.reader_generation_config(tmp_path, Generation, None)


def test_original_generation_fallbacks_and_model_values_are_not_shortened():
    config = SimpleNamespace(
        temperature=None, top_p=None, top_k=None, max_new_tokens=None
    )
    assert runner.generation_parameters(config, "reader") == {
        "model": "reader",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 512,
        "seed": 42,
    }
    config.max_new_tokens = 8192
    config.temperature, config.top_p, config.top_k = 0.9, 0.8, 50
    assert runner.generation_parameters(config, "reader")["max_tokens"] == 8192


def test_plan_has_no_model_import_or_output(tmp_path, capsys):
    missing = tmp_path / "not-created"
    before = set(sys.modules)
    runner.main(
        [
            "--author-root",
            str(missing),
            "--data-root",
            str(missing),
            "--embedding-root",
            str(missing),
            "--reader-root",
            str(missing),
            "--output",
            str(missing),
            "--model",
            "explicit-fallback-model",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == 0 and result["selected_questions"] == "first3"
    assert not missing.exists()
    assert not ({"torch", "torch_npu", "transformers"} & (set(sys.modules) - before))


def test_original_prompt_is_read_not_rewritten(tmp_path):
    (tmp_path / "examples").mkdir()
    source = tmp_path / "examples/calculate_baseline_metrics.py"
    source.write_text(
        "def main():\n    llm_system_prompt = ('Original ' 'Русский')\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    assert runner.original_system_prompt(tmp_path) == "Original Русский"
    assert source.read_bytes() == before
    source.write_text("llm_system_prompt = build_a_new_prompt()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unchanged"):
        runner.original_system_prompt(tmp_path)


def test_selected_references_keep_original_order_and_fields():
    rows = [
        {"public_id": 7, "answer": "not a model input"},
        {"public_id": 4, "answer": "original reference"},
        {"public_id": 2, "answer": "excluded"},
    ]
    assert runner.reference_subset(rows, [4, 7]) == rows[:2]
    with pytest.raises(ValueError, match="relations"):
        runner.reference_subset(rows + [rows[0]], [7])
    with pytest.raises(ValueError, match="relations"):
        runner.reference_subset(rows, [99])


def test_original_asset_verifier_checks_git_blobs_and_sha256(tmp_path):
    file = tmp_path / "config.json"
    file.write_bytes(b"{}")
    git = hashlib.sha1(b"blob 2\0{}").hexdigest()
    runner.verify_artifact(
        tmp_path, {"files": [{"path": "config.json", "bytes": 2, "git_blob": git}]}
    )
    runner.verify_artifact(
        tmp_path,
        {"files": [{"path": "config.json", "bytes": 2, "sha256": runner.sha(file)}]},
    )
    file.write_bytes(b"[]")
    with pytest.raises(ValueError, match="identity"):
        runner.verify_artifact(
            tmp_path, {"files": [{"path": "config.json", "bytes": 2, "git_blob": git}]}
        )
    (tmp_path / "unexpected.py").write_text("pass\n")
    with pytest.raises(ValueError, match="remote Python"):
        runner.verify_artifact(tmp_path, {"files": []})


def test_asset_path_must_remain_inside_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "outside").write_bytes(b"x")
    with pytest.raises(ValueError, match="path"):
        runner.verify_artifact(model, {"files": [{"path": "../outside", "bytes": 1}]})


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_original_completion_preserves_prompt_and_response(
    tmp_path, monkeypatch, finish
):
    body = {
        "model": "actual-model",
        "prompt": "original full template\nwith context",
        "max_tokens": 512,
    }
    response = {
        "choices": [{"text": "wrong answer still retained", "finish_reason": finish}]
    }
    seen = []

    @contextmanager
    def send(request, timeout):
        seen.append((request.full_url, json.loads(request.data), timeout))
        import io

        yield io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(runner, "urlopen", send)
    assert (
        runner.complete("http://127.0.0.1:123/v1", body, tmp_path, 0)
        == response["choices"][0]["text"]
    )
    assert seen == [("http://127.0.0.1:123/v1/completions", body, 1200)]
    assert (
        json.loads((tmp_path / "reader-0000-request-private.json").read_text()) == body
    )
    assert (
        json.loads((tmp_path / "reader-0000-response-private.json").read_text())
        == response
    )
    with pytest.raises(FileExistsError):
        runner.complete("http://127.0.0.1:123/v1", body, tmp_path, 0)
    assert len(seen) == 1


def test_bad_response_retained_without_automatic_retry(tmp_path, monkeypatch):
    @contextmanager
    def send(*args, **kwargs):
        import io

        yield io.BytesIO(b'{"choices": []}')

    monkeypatch.setattr(runner, "urlopen", send)
    with pytest.raises(ValueError, match="retained"):
        runner.complete("http://127.0.0.1:123/v1", {"prompt": "unchanged"}, tmp_path, 0)
    assert (tmp_path / "reader-0000-response-private.json").exists()


@pytest.mark.parametrize(
    "metric_lines, owners, valid",
    [
        (
            'vllm:num_requests_running{x="x"} 0\nvllm:num_requests_waiting{x="x"} 0',
            [123],
            True,
        ),
        (
            'vllm:num_requests_running{x="x"} 0\nvllm:num_requests_waiting{x="x"} 0',
            [123, os.getpid()],
            True,
        ),
        (
            'vllm:num_requests_running{x="x"} 0\nvllm:num_requests_waiting{x="x"} 0',
            [123, 99999],
            False,
        ),
        (
            'vllm:num_requests_running{x="x"} 1\nvllm:num_requests_waiting{x="x"} 0',
            [123],
            False,
        ),
        ('vllm:num_requests_running{x="x"} 0', [123], False),
        (
            'vllm:num_requests_running{x="x"} 0\nvllm:num_requests_waiting{x="x"} 0',
            [],
            False,
        ),
    ],
)
def test_no_mixed_service_or_unowned_npu(metric_lines, owners, valid, monkeypatch):
    @contextmanager
    def metrics(*args, **kwargs):
        import io

        yield io.BytesIO(metric_lines.encode())

    monkeypatch.setattr(runner, "urlopen", metrics)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: "\n".join(f"| 4 0 | {pid} | model |" for pid in owners),
    )
    if valid:
        runner.idle("http://127.0.0.1:123/v1", 123)
    else:
        with pytest.raises(RuntimeError, match="mixed"):
            runner.idle("http://127.0.0.1:123/v1", 123)
