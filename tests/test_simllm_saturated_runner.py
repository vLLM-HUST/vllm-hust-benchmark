from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_simllm_saturated_throughput_warm_cache.sh"
WARM_CACHE_RUNNER = ROOT / "scripts" / "run_simllm_random_online_warm_cache.sh"


def run_dry(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "RESULT_DIR": str(tmp_path),
            "SIMLLM_SATURATED_DRY_RUN": "1",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_generates_fixed_saturated_contract(tmp_path: Path) -> None:
    result = run_dry(tmp_path)
    assert result.returncode == 0, result.stderr
    spec = json.loads((tmp_path / "saturated-same-spec.json").read_text())
    client = spec["client_parameters"]
    server = spec["server_parameters"]
    assert spec["scenario"] == "random-online"
    assert client["request_rate"] == "inf"
    assert client["max_concurrency"] == 16
    assert client["num_prompts"] == 32
    assert client["input_len"] == 4096
    assert client["output_len"] == 32
    assert client["temperature"] == 0
    assert server["gpu_memory_utilization"] == 0.6
    assert server["max_model_len"] == 32768
    assert server["max_num_batched_tokens"] == 4096


def test_token_budget_smaller_than_prompt_fails(tmp_path: Path) -> None:
    result = run_dry(
        tmp_path,
        SIMLLM_THROUGHPUT_INPUT_LEN="4096",
        SIMLLM_THROUGHPUT_MAX_NUM_BATCHED_TOKENS="2048",
    )
    assert result.returncode == 2
    assert "must be at least SIMLLM_THROUGHPUT_INPUT_LEN" in result.stderr


def test_runner_enforces_complete_ab_results_and_shared_seed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "SIMLLM_MEASURE_SEED=${SIMLLM_MEASURE_SEED:-$SIMLLM_WARMCACHE_SEED}" in text
    assert "completed $completed/$SIMLLM_THROUGHPUT_NUM_PROMPTS" in text
    assert '"$WARM_CACHE_RUNNER" "$SATURATED_SPEC_FILE"' in text
    assert 'if [[ "$RUN_BASELINE" != "1" || "$RUN_SIMLLM" != "1" ]]' in text
    assert "single-arm run complete; comparison not requested" in text


def test_baseline_with_no_rewrite_log_is_not_rejected_by_pipefail() -> None:
    text = WARM_CACHE_RUNNER.read_text(encoding="utf-8")
    assert "grep 'SimLLM rewrite_scheduler: skipped prefill'" in text
    assert "2>/dev/null || true; }" in text


def test_live_worker_binding_is_accepted_as_activation_evidence() -> None:
    text = WARM_CACHE_RUNNER.read_text(encoding="utf-8")
    assert "SimLLM worker patch state:" in text
    assert "_simllm_execute_model" in text
    assert "_simllm_model_forward" in text
