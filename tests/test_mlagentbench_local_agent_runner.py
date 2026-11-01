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
            model="Qwen/Qwen3-14B", work_root=Path("work"), log_root=Path("log")
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


def test_zero_capability_no_new_privilege_control():
    MODULE.check_isolation("CapEff:\t0000000000000000\nNoNewPrivs:\t1")


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
    # Reverse only the terminal metadata count fix. The full-file digest proves
    # all native execution logic is identical to the actual executed version.
    fixed = 'status="ORIGINAL_AGENT_RUNNER_RETURNED_REQUIRES_TRACE_AND_ORIGINAL_SCORE_AUDIT",\n        native_agent_tasks_run=1,\n'
    original = 'status="ORIGINAL_AGENT_RUNNER_RETURNED_REQUIRES_TRACE_AND_ORIGINAL_SCORE_AUDIT"\n'
    assert current.count(fixed) == 1
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
