"""Original Agent execution guards/protocol fixtures, not a native run."""

import importlib.util
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "mlagent_runner", ROOT / "scripts/run_mlagentbench_local_agent.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_plan_does_not_open_sources_or_call_model():
    result = MODULE.run(SimpleNamespace(execute=False))
    assert result["native_agent_tasks_run"] == 0
    assert result["source_checkbox_eligible"] is False


def test_original_default_task_and_full_budgets_retained():
    argv = MODULE.original_arguments(
        SimpleNamespace(
            model="Qwen/Qwen3-14B",
            work_root=Path("work"),
            log_root=Path("log"),
            task="debug",
        )
    )
    options = dict(zip(argv[1::2], argv[2::2]))
    assert options["--task"] == "debug"
    assert options["--agent-type"] == "ResearchAgent"
    assert options["--max-steps"] == options["--agent-max-steps"] == "50"
    assert options["--max-time"] == "18000"
    assert options["--edit-script-llm-max-tokens"] == "4000"
    assert options["--max-retries"] == "5"
    assert (
        options["--max-steps-in-context"]
        == options["--max-observation-steps-in-context"]
        == "3"
    )
    assert (
        options["--llm-name"]
        == options["--fast-llm-name"]
        == options["--edit-script-llm-name"]
        == "Qwen/Qwen3-14B"
    )


@pytest.mark.parametrize(
    "status",
    [
        "CapEff:\t0000000000000001\nNoNewPrivs:\t1",
        "CapEff:\t0000000000000000\nNoNewPrivs:\t0",
        "",
    ],
)
def test_unsafe_host_or_container_cannot_run_agent(status):
    with pytest.raises(ValueError, match="container"):
        MODULE.check_isolation(status)


def test_official_quickstart_task_can_be_selected_without_budget_changes():
    base = SimpleNamespace(
        model="Qwen/Qwen3-14B",
        work_root=Path("work"),
        log_root=Path("log"),
        task="debug",
    )
    default = MODULE.original_arguments(base)
    base.task = "cifar10"
    formal = MODULE.original_arguments(base)
    assert [(x, y) for x, y in zip(default, formal) if x != y] == [("debug", "cifar10")]


def test_zero_capability_no_new_privilege_control():
    MODULE.check_isolation("CapEff:\t0000000000000000\nNoNewPrivs:\t1")


def resume_fixture(tmp_path):
    root = tmp_path / "retained"
    (root / "env_log/traces/step_3_files").mkdir(parents=True)
    (root / "agent_log").mkdir()
    (root / "env_log/trace.json").write_text(json.dumps({"steps": [{}] * 4}))
    state = {
        "args": {
            "task": "cifar10",
            "max_steps": 50,
            "agent_max_steps": 50,
            "max_time": 18000,
            "max_retries": 5,
            "edit_script_llm_max_tokens": 4000,
            "max_steps_in_context": 3,
            "max_observation_steps_in_context": 3,
            **dict.fromkeys(
                ["llm_name", "fast_llm_name", "edit_script_llm_name"], "Qwen/Qwen3-14B"
            ),
        }
    }
    (root / "agent_log/agent_3_3.json").write_text(json.dumps(state))
    return SimpleNamespace(
        resume=root,
        resume_step=3,
        task="cifar10",
        model="Qwen/Qwen3-14B",
        work_root=tmp_path / "fresh-work",
        log_root=tmp_path / "fresh-log",
    )


def test_explicit_native_resume_preserves_checkpoint_and_original_budgets(tmp_path):
    args = resume_fixture(tmp_path)
    before = {p: p.read_bytes() for p in args.resume.rglob("*") if p.is_file()}
    argv = MODULE.original_arguments(args)
    assert argv[-4:] == ["--resume", str(args.resume), "--resume-step", "3"]
    assert dict(zip(argv[1::2], argv[2::2]))["--max-time"] == "18000"
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize(
    "change", ["model", "task", "overlap", "snapshot", "step", "budget"]
)
def test_resume_rejects_mismatched_or_incomplete_checkpoint(tmp_path, change):
    args = resume_fixture(tmp_path)
    if change == "model":
        args.model = "different-model"
    elif change == "task":
        args.task = "debug"
    elif change == "overlap":
        args.work_root = args.resume / "work"
    elif change == "snapshot":
        (args.resume / "env_log/traces/step_3_files").rmdir()
    elif change == "step":
        args.resume_step = -1
    else:
        path = args.resume / "agent_log/agent_3_3.json"
        saved = json.loads(path.read_text())
        saved["args"]["max_steps"] = 500
        path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        MODULE.resume_arguments(args)


def test_resume_step_without_checkpoint_is_not_silently_ignored():
    with pytest.raises(ValueError, match="explicit"):
        MODULE.resume_arguments(SimpleNamespace(resume=None, resume_step=3))


def test_formal_native_receipt_preserves_failure_recovery_and_remaining_scope():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/mlagentbench-formal-cifar10-native-audit-20261102.json"
        ).read_text("utf-8")
    )
    assert receipt["task"] == "cifar10"
    assert receipt["submitted_final_answer"] is False
    assert (
        receipt["original_agent_final_message"] == "No valid response after max_retries"
    )
    assert receipt["original_final_score"] == 0.4803
    assert receipt["this_formal_task_actual_npu_http200_jobs"] == 13
    assert (
        receipt["original_segment_jobs"] + receipt["resumed_new_inference_jobs"] == 13
    )
    assert receipt["exact_boundary_cached_replays_without_inference"] == 1
    assert (
        receipt["prior_grader_attempt_retained"]["not_accepted_as_scored_completion"]
        is True
    )
    assert receipt["remaining_fixed_task_ids"] == ["imdb", "ogbn-arxiv"]
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["quality_success_or_performance_improvement_claim"] is False
    assert (
        hashlib.sha256(
            (ROOT / receipt["entry"]).read_text("utf-8").encode()
        ).hexdigest()
        == receipt["entry_sha256"]
    )


def test_original_main_module_dispatch_preserves_python_m_semantics():
    tree = ast.parse(
        (ROOT / "scripts/run_mlagentbench_local_agent.py").read_text("utf-8")
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_module"
    ]
    assert len(calls) == 1
    keywords = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
    assert keywords == {"run_name": "__main__", "alter_sys": True}


def test_live_start_snapshot_is_bound_to_entry_not_counted_as_delivery():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/mlagentbench-original-agent-start-20261102.json"
        ).read_text("utf-8")
    )
    current = (ROOT / receipt["entry"]).read_text("utf-8")
    # The opt-in native resume branch is not entered by the historical run.
    start = current.index("def resume_arguments(args):\n")
    end = current.index("def original_arguments(args):\n")
    current = current[:start] + current[end:]
    assert current.count("] + resume_arguments(args)") == 1
    current = current.replace("] + resume_arguments(args)", "]")
    current = current.replace('    parser.add_argument("--resume", type=Path)\n', "")
    current = current.replace(
        '    parser.add_argument("--resume-step", type=int, default=0)\n', ""
    )
    # Reverse only the terminal metadata count fix. The full-file digest proves
    # all native execution logic is identical to the actual executed version.
    fixed = 'status="ORIGINAL_AGENT_RUNNER_RETURNED_REQUIRES_TRACE_AND_ORIGINAL_SCORE_AUDIT",\n        native_agent_tasks_run=1,\n'
    original = 'status="ORIGINAL_AGENT_RUNNER_RETURNED_REQUIRES_TRACE_AND_ORIGINAL_SCORE_AUDIT"\n'
    assert current.count(fixed) == 1
    assert current.count("        args.task,\n") == 1
    assert current.count('    parser.add_argument("--task", default="debug")\n') == 1
    current = current.replace("        args.task,\n", '        "debug",\n').replace(
        '    parser.add_argument("--task", default="debug")\n', ""
    )
    assert (
        receipt["entry_sha256"]
        == hashlib.sha256(current.replace(fixed, original).encode()).hexdigest()
    )
    assert receipt["task"] == "debug"
    assert receipt["max_time_seconds"] == 18000
    assert receipt["max_steps"] == receipt["agent_max_steps"] == 50
    assert receipt["completed_http200_model_jobs_at_snapshot"] >= 1
    assert receipt["earlier_loader_failure"]["exit_code"] == 1
    assert receipt["native_agent_delivery_claim"] is False
    assert receipt["source_checkbox_eligible"] is False
