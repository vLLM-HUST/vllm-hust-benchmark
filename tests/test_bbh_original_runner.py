"""CPU-only invented prompt fixtures; never model or task validation evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_bbh_boolean_original.py"
spec = importlib.util.spec_from_file_location("bbh_original_runner_cpu", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
GENERIC_SCRIPT = ROOT / "scripts/run_bbh_original_tasks.py"
generic_spec = importlib.util.spec_from_file_location(
    "bbh_original_tasks_runner_cpu", GENERIC_SCRIPT
)
generic_runner = importlib.util.module_from_spec(generic_spec)
generic_spec.loader.exec_module(generic_runner)
CLI_SCRIPT = ROOT / "src/vllm_hust_benchmark/workload_data.py"
cli_spec = importlib.util.spec_from_file_location("workload_data_cli_bbh", CLI_SCRIPT)
workload_cli = importlib.util.module_from_spec(cli_spec)
cli_spec.loader.exec_module(workload_cli)


def test_help_without_optional_evaluation_dependencies():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--data-root" in result.stdout


@pytest.mark.parametrize(
    "value",
    [
        "https://remote.example/v1",
        "http://127.0.0.1/v1?token=x",
        "http://user:pass@localhost/v1",  # pragma: allowlist secret - invented rejection-test credentials
        "http://localhost/not-v1",
        "http://localhost/v1#fragment",
    ],
)
def test_remote_or_credentialed_endpoints_rejected(value):
    with pytest.raises(ValueError):
        runner.checked_endpoint(value)


def test_publisher_prefix_unchanged_and_evaluation_target_never_read():
    prefix = "fixture task description\n\n" + "\n\n".join(
        f"Q: demo-{i}\nA: fixture demonstration." for i in range(3)
    )
    actual, demos = runner.publisher_prefix(
        ("fixture header\n-----\n" + prefix).encode()
    )
    assert actual == prefix
    assert demos == ["demo-0", "demo-1", "demo-2"]
    prompt = runner.task_prompt(
        actual, {"input": "  original\ninput ", "target": "REFERENCE_SENTINEL"}
    )
    assert (
        prompt == prefix + "\n\nQ:   original\ninput \nA: Let's think step by step.\n"
    )
    assert "REFERENCE_SENTINEL" not in prompt


@pytest.mark.parametrize(
    "raw", [b"no separator", b"a\n-----\nb\n-----\nc", b"a\n-----\nno demonstrations"]
)
def test_bad_publisher_boundary_rejected(raw):
    with pytest.raises(ValueError):
        runner.publisher_prefix(raw)


def test_bad_asset_identity_rejected(tmp_path):
    source = tmp_path / "fixture"
    source.write_bytes(b"not original")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.checked_asset(source, "0" * 64)


def test_generic_runner_help_without_optional_evaluation_dependencies():
    result = subprocess.run(
        [sys.executable, str(GENERIC_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--task-ids" in result.stdout
    assert "--full" in result.stdout


def test_root_cli_delegates_all_task_preparation_to_provider(
    tmp_path, monkeypatch, capsys
):
    seen = {}

    def prepare(**kwargs):
        seen.update(kwargs)
        return {"source_id": "BBH", "status": "fixture-only"}

    fake_provider = SimpleNamespace(prepare_bbh_original_tasks=prepare)
    monkeypatch.setattr(workload_cli, "load_provider", lambda: fake_provider)
    archive = tmp_path / "publisher.zip"
    output = tmp_path / "prepared"
    assert (
        workload_cli.main(
            [
                "prepare-bbh-original",
                "--data-archive",
                str(archive),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert seen == {"data_archive": archive, "output": output}
    assert json.loads(capsys.readouterr().out)["source_id"] == "BBH"


def test_generic_runner_supports_boolean_and_rejects_invalid_selection(monkeypatch):
    source = generic_runner.source_spec("BBH")
    monkeypatch.setattr(generic_runner, "source_spec", lambda source_id: source)
    _, tasks, prompts = generic_runner.selected_assets(["boolean_expressions"])
    assert tasks["boolean_expressions"]["path"] == "boolean_expressions.json"
    assert (
        prompts["boolean_expressions"]["path"] == "cot-prompts/boolean_expressions.txt"
    )
    for invalid in ([], ["unknown_task"], ["causal_judgement", "causal_judgement"]):
        with pytest.raises(ValueError, match="distinct tasks"):
            generic_runner.selected_assets(invalid)


@pytest.mark.parametrize(
    "full,expected_indices",
    [(False, [0, 1, 2]), (True, [0, 1, 2, 3])],
)
def test_generic_runner_uses_original_rows_and_never_sends_references(
    tmp_path, monkeypatch, full, expected_indices
):
    source = generic_runner.source_spec("BBH")
    manifest = json.loads(json.dumps(source))
    assets_by_path = {asset["path"]: asset for asset in manifest["files"]}
    task_ids = ["causal_judgement", "date_understanding"]
    raw_by_path = {}
    for task_id in task_ids:
        task_path = f"{task_id}.json"
        prompt_path = f"cot-prompts/{task_id}.txt"
        raw_by_path[task_path] = json.dumps(
            {
                "examples": [
                    {"input": f"ORIGINAL_INPUT_{index}", "target": f"REFERENCE_{index}"}
                    for index in range(4)
                ]
            }
        ).encode()
        raw_by_path[prompt_path] = (
            "fixture description\n-----\n"
            + "\n\n".join(f"Q: demo-{index}\nA: fixture answer" for index in range(3))
        ).encode()
        for path in (task_path, prompt_path):
            asset = assets_by_path[path]
            asset["sha256"] = hashlib.sha256(raw_by_path[path]).hexdigest()
            asset["bytes"] = len(raw_by_path[path])
            if asset["role"] == "data":
                asset["rows"] = 4
    data_root = tmp_path / "source"
    for name, raw in raw_by_path.items():
        path = data_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(generic_runner, "source_spec", lambda source_id: manifest)
    monkeypatch.setattr(
        generic_runner,
        "pinned_harness",
        lambda: (
            {"generation_kwargs": {"max_gen_toks": 10, "until": ["STOP"]}},
            lambda **kwargs: {"exact_match": 1.0},
            SimpleNamespace(fallback="FALLBACK", apply=lambda values, rows: values),
        ),
    )
    requests = []

    def fake_post(url, payload):
        if url.endswith("/tokenize"):
            return {"count": 5}
        requests.append(payload)
        return {
            "choices": [
                {
                    "message": {"content": "answer"},
                    "finish_reason": "stop",
                    "stop_reason": "STOP",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }

    monkeypatch.setattr(generic_runner, "post", fake_post)
    output = tmp_path / "private-output"
    receipt = generic_runner.run(
        SimpleNamespace(
            endpoint="http://127.0.0.1:19466/v1",
            task_ids=task_ids,
            output=output,
            data_root=data_root,
            model="fixture-model",
            max_model_len=100,
            full=full,
        )
    )
    assert len(requests) == len(task_ids) * len(expected_indices)
    assert receipt["selected_indices_by_task"] == {
        task_id: expected_indices for task_id in task_ids
    }
    serialized_prompts = json.dumps(requests)
    assert "ORIGINAL_INPUT_0" in serialized_prompts
    assert ("ORIGINAL_INPUT_3" in serialized_prompts) is full
    assert "REFERENCE_" not in serialized_prompts
    assert receipt["all_normal_stop"] is True
    assert receipt["all_answers_extracted"] is True
    assert (output / "receipt.json").exists()
