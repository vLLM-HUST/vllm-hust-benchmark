"""Offline fixtures only; no native judge or model claims."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "judge_recovery",
    Path(__file__).parents[1] / "scripts/recover_naturebench_original_judge.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def args(tmp_path, monkeypatch):
    for name in ("author", "runtime", "data", "saved"):
        (tmp_path / name).mkdir()
    (tmp_path / "saved/judge_verdict.json").write_text(
        json.dumps({"is_valid": None, "reason": "fixture failure"})
    )
    monkeypatch.setattr(runner, "verify_runtime", lambda a, r: r)
    monkeypatch.setattr(
        runner,
        "original_cpu_packages",
        lambda *a, **k: iter([{"source": {"revision": "fixture"}}]),
    )
    return SimpleNamespace(
        author_root=tmp_path / "author",
        runtime=tmp_path / "runtime",
        data_root=tmp_path / "data",
        saved_task=tmp_path / "saved",
        output=tmp_path / "fresh",
        task_id="original-id",
        model="actual-model",
        endpoint="http://127.0.0.1:12345",
    )


def test_plan_has_no_calls_or_output(args):
    result = runner.plan(args)
    assert result["new_agent_calls"] == 0
    assert result["source_checkbox_eligible"] is False
    assert not args.output.exists()


@pytest.mark.parametrize("value", [True, False])
def test_existing_valid_judge_is_not_rerun(args, value):
    (args.saved_task / "judge_verdict.json").write_text(json.dumps({"is_valid": value}))
    with pytest.raises(ValueError, match="reused"):
        runner.plan(args)


def test_output_overlap_rejected(args):
    args.output = args.saved_task / "new"
    with pytest.raises(ValueError, match="separate"):
        runner.plan(args)


def test_credential_url_rejected(args):
    args.endpoint = "http://user:fixture@localhost:12345"  # pragma: allowlist secret - rejected credential URL fixture
    with pytest.raises(ValueError, match="credentials"):
        runner.plan(args)


@pytest.mark.parametrize("input_tokens,expected_calls", [(9900, 0), (100, 1)])
def test_execute_keeps_budget_and_blocks_oversize(
    args, monkeypatch, input_tokens, expected_calls
):
    calls = []
    inputs = {"original": "full evidence"}
    original = SimpleNamespace(
        __file__=str(args.runtime / "judge_core/runner.py"),
        collect_judge_inputs_with_context=lambda *a, **k: inputs,
        build_user_prompt_with_context=lambda value: json.dumps(value),
        judge_task=lambda *a, **k: (
            calls.append(k) or {"is_valid": False, "reason": "valid negative verdict"}
        ),
    )
    policy = SimpleNamespace(
        __file__=str(args.runtime / "judge_core/policy.py"),
        MAX_JUDGE_OUTPUT_TOKENS=4096,
        JUDGE_SYSTEM="original system",
    )
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: original if name.endswith("runner") else policy,
    )
    requests = []

    def fetch(request, timeout):
        requests.append(request)
        payload = (
            {"input_tokens": input_tokens}
            if request.full_url.endswith("count_tokens")
            else {"data": [{"id": args.model, "max_model_len": 10000}]}
        )
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(runner, "urlopen", fetch)
    prepared = runner.plan(args)
    if expected_calls:
        assert runner.execute(args, prepared) == 0
    else:
        with pytest.raises(ValueError, match="no trimming"):
            runner.execute(args, prepared)
    assert len(calls) == expected_calls
    body = json.loads(requests[0].data)
    assert body["max_tokens"] == 4096
    assert body["messages"][0]["content"] == json.dumps(inputs)
    assert (
        json.loads((args.saved_task / "judge_verdict.json").read_text())["is_valid"]
        is None
    )
