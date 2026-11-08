"""Scoring transport guards and original empty-score rejection, not task runs."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace
from typing import Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mlagentbench_original_eval as module


def test_plan_does_not_open_sources_or_run_scoring():
    assert module.run(SimpleNamespace(execute=False))["model_calls"] == 0


@pytest.mark.parametrize("value", [0, 0.5, 1, -0.1])
def test_finite_original_scores_need_not_be_quality_success(value):
    assert module.scored_outputs_valid(
        {"original": {"score": [value], "final_score": value}}
    )


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"score": [], "final_score": -1},
        {"score": [True], "final_score": True},
        {"score": [float("nan")], "final_score": float("nan")},
        {"score": [0.5], "final_score": 1},
    ],
)
def test_upstream_silent_empty_or_invalid_results_are_not_success(result):
    assert not module.scored_outputs_valid({"original": result})


def test_original_eval_namespace_arguments_and_caller_restoration(
    tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    for name in [
        "env_log/trace.json",
        "env_log/overall_time.txt",
        "log",
        "agent_log/main_log",
    ]:
        p = logs / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    (logs / "env_log/traces/step_final_files").mkdir(parents=True)
    port = tmp_path / "port"
    port.mkdir()
    marker = tmp_path / "docker-marker"
    marker.write_text("")
    status = tmp_path / "status"
    status.write_text("CapEff:\t0000000000000000\nNoNewPrivs:\t1")
    real_path = Path
    monkeypatch.setattr(
        module,
        "Path",
        lambda value: {
            "/.dockerenv": marker,
            "/proc/self/status": status,
        }.get(str(value), real_path(value)),
    )
    monkeypatch.setattr(module, "verify_port", lambda p: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    original_argv = sys.argv
    output = tmp_path / "fresh-score.json"

    def original_module(name, *, run_name, init_globals):
        assert name == "MLAgentBench.eval" and run_name == "__main__"
        assert init_globals == {"Dict": Dict}
        assert sys.argv == [
            "MLAgentBench.eval",
            "--task",
            "imdb",
            "--log-folder",
            str(logs),
            "--output-file",
            str(output),
        ]
        output.write_text(json.dumps({"fixture": {"score": [0], "final_score": 0}}))

    monkeypatch.setattr(module.runpy, "run_module", original_module)
    result = module.run(
        SimpleNamespace(
            execute=True, port_root=port, log_root=logs, output=output, task="imdb"
        )
    )
    assert result["scored_outputs_valid"] is True
    assert result["source_checkbox_eligible"] is False
    assert sys.argv is original_argv
