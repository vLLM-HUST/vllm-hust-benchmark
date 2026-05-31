from __future__ import annotations

from pathlib import Path

from vllm_hust_benchmark.official_runtime_inputs import SHAREGPT_DATASET_FILENAME
from vllm_hust_benchmark.official_runtime_inputs import normalize_client_parameters
from vllm_hust_benchmark.official_runtime_inputs import normalize_server_parameters
from vllm_hust_benchmark.official_runtime_inputs import resolve_runtime_dataset_path


def test_normalize_client_parameters_strips_unsupported_offline_flags(tmp_path: Path) -> None:
    worktree = tmp_path / "vllm"
    (worktree / "benchmarks").mkdir(parents=True)
    (worktree / "benchmarks" / "sonnet.txt").write_text("sonnet\n", encoding="utf-8")

    parameters = {
        "dataset_name": "sonnet",
        "dataset_path": "benchmarks/sonnet.txt",
        "num_prompts": 200,
        "num_warmups": 0,
        "ready_check_timeout_sec": 900,
    }

    normalized = normalize_client_parameters(
        parameters,
        benchmark_type="throughput",
        ready_check_timeout_sec=900,
        vllm_worktree=str(worktree),
    )

    assert "num_warmups" not in normalized
    assert "ready_check_timeout_sec" not in normalized
    assert normalized["dataset_path"] == str(worktree / "benchmarks" / "sonnet.txt")


def test_normalize_client_parameters_injects_ready_timeout_for_serve(tmp_path: Path) -> None:
    cache_root = tmp_path / "datasets"
    cache_root.mkdir()
    (cache_root / SHAREGPT_DATASET_FILENAME).write_text("[]\n", encoding="utf-8")

    normalized = normalize_client_parameters(
        {
            "dataset_name": "sharegpt",
            "dataset_path": SHAREGPT_DATASET_FILENAME,
            "num_prompts": 200,
        },
        benchmark_type="serve",
        ready_check_timeout_sec=900,
        dataset_cache_root=str(cache_root),
    )

    assert normalized["ready_check_timeout_sec"] == 900
    assert normalized["dataset_path"] == str(cache_root / SHAREGPT_DATASET_FILENAME)


def test_normalize_client_parameters_forces_eager_for_offline_when_requested() -> None:
    normalized = normalize_client_parameters(
        {
            "dataset_name": "random",
            "input_len": 1024,
            "output_len": 128,
            "batch_size": 8,
        },
        benchmark_type="latency",
        force_eager=True,
    )

    assert normalized["enforce_eager"] == ""


def test_normalize_server_parameters_parses_limit_mm_per_prompt() -> None:
    normalized = normalize_server_parameters({"limit_mm_per_prompt": "image=1,video=0"})

    assert normalized["limit_mm_per_prompt"] == {"image": 1, "video": 0}


def test_resolve_runtime_dataset_path_leaves_hf_dataset_id_unchanged(tmp_path: Path) -> None:
    resolved = resolve_runtime_dataset_path(
        "lmarena-ai/VisionArena-Chat",
        vllm_worktree=str(tmp_path / "vllm"),
        dataset_cache_root=str(tmp_path / "datasets"),
    )

    assert resolved == "lmarena-ai/VisionArena-Chat"