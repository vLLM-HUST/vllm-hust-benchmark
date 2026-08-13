"""Tests for the ``--paired-base-head`` mode of ``backfill_single_gpu.py``.

Covers milestone 3 (paired base/head reruns): the ``run`` subcommand accepts
``--paired-base-head BASE_REF HEAD_REF`` together with ``--pr-number`` and
``--workload``, runs the same workload against both vllm-hust commits with a
shared plugin commit / NPU, and writes two submission directories named
``historical-pr-backend-pr-<n>-{base,head}-<workload>-<hust>-<plugin>``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backfill_single_gpu.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_single_gpu_paired", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 40-char SHAs used as resolved commit values.
BASE_SHA = "1aa7cd10b7" + "0" * 30
HEAD_SHA = "e26ffd9063" + "0" * 30
PLUGIN_SHA = "cd29480d96" + "0" * 30


def _seed_state(tmp_path: Path) -> tuple[Path, Path]:
    """Redirect STATE_DIR/STATE_FILE into tmp_path and seed a minimal state."""
    state_dir = tmp_path / ".benchmarks" / "backfill-single-gpu"
    state_dir.mkdir(parents=True)
    state_file = state_dir / "state.json"
    state = {"hust_head": "orighead", "ascend_head": "origplugin", "cells": {}}
    state_file.write_text(json.dumps(state), encoding="utf-8")
    return state_dir, state_file


def _run_cmd_run_paired(
    mod,
    tmp_path: Path,
    *,
    extra_args: list[str],
    run_cell_mock,
) -> int:
    """Invoke ``cmd_run`` with the paired args and a fully mocked environment."""
    state_dir, state_file = _seed_state(tmp_path)
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "run",
        "--paired-base-head",
        "base",
        "head",
        "--pr-number",
        "131",
        "--workload",
        "random-online",
        "--ascend-commit",
        PLUGIN_SHA,
        "--npu-device",
        "0",
    ] + extra_args

    args = mod.build_parser().parse_args(argv)

    resolve_map = {"base": BASE_SHA, "head": HEAD_SHA}
    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "STATE_FILE", state_file),
        patch.object(mod, "REPO_ROOT", tmp_path),
        patch.object(
            mod,
            "_resolve_full_commit",
            side_effect=lambda ref: resolve_map[ref],
        ),
        patch.object(mod, "assert_plugin_commit_consistent", return_value=None),
        patch.object(mod, "commit_exists", return_value=True),
        patch.object(mod, "cell_already_present", return_value=False),
        patch.object(mod, "run_cell", side_effect=run_cell_mock),
        patch.object(mod, "current_head", return_value="orighead"),
        patch.object(mod, "_kill_port_process", return_value=None),
        patch.object(mod.subprocess, "run", return_value=MagicMock(returncode=0)),
    ):
        return mod.cmd_run(args)


# ---------------------------------------------------------------------------
# Naming helper
# ---------------------------------------------------------------------------


def test_build_paired_run_id_naming_base_and_head() -> None:
    mod = load_module()

    base_id = mod.build_paired_run_id(
        131, "base", "random-online", BASE_SHA, PLUGIN_SHA
    )
    head_id = mod.build_paired_run_id(
        131, "head", "random-online", HEAD_SHA, PLUGIN_SHA
    )

    assert base_id == (
        "historical-pr-backend-pr-131-base-random-online-1aa7cd10b7-cd29480d96"
    )
    assert head_id == (
        "historical-pr-backend-pr-131-head-random-online-e26ffd9063-cd29480d96"
    )
    # Both share the PR number, workload and plugin commit prefix.
    assert base_id.startswith("historical-pr-backend-pr-131-base-")
    assert head_id.startswith("historical-pr-backend-pr-131-head-")
    assert base_id.endswith("-cd29480d96")
    assert head_id.endswith("-cd29480d96")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def test_run_subcommand_parses_paired_base_head_and_pr_number() -> None:
    mod = load_module()

    args = mod.build_parser().parse_args(
        [
            "run",
            "--paired-base-head",
            "abc123",
            "def456",
            "--pr-number",
            "131",
            "--workload",
            "random-online",
        ]
    )

    assert args.paired_base_head == ["abc123", "def456"]
    assert args.pr_number == 131
    assert args.workload == "random-online"
    assert args.command == "run"


def test_pr_number_defaults_to_none_without_paired_mode() -> None:
    mod = load_module()

    args = mod.build_parser().parse_args(["run", "--commit", "abc123"])

    assert args.paired_base_head is None
    assert args.pr_number is None


# ---------------------------------------------------------------------------
# cmd_run paired orchestration (run_cell mocked)
# ---------------------------------------------------------------------------


def test_cmd_run_paired_mode_runs_base_then_head_with_shared_config(
    tmp_path: Path,
) -> None:
    mod = load_module()
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    calls: list[dict] = []

    def _run_cell(workload, hust_commit, ascend_commit, npu_id=0, **kwargs):
        calls.append(
            {
                "workload": workload,
                "hust_commit": hust_commit,
                "ascend_commit": ascend_commit,
                "npu_id": npu_id,
                "run_id_override": kwargs.get("run_id_override"),
            }
        )
        run_id = kwargs.get("run_id_override") or "fallback"
        sub_dir = submissions_dir / run_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "done",
            "hust_commit": hust_commit,
            "ascend_commit": ascend_commit,
            "run_id": run_id,
            "submission_dir": str(sub_dir),
        }

    rc = _run_cmd_run_paired(mod, tmp_path, extra_args=[], run_cell_mock=_run_cell)

    assert rc == 0
    # Exactly two runs, base first then head.
    assert len(calls) == 2
    assert calls[0]["hust_commit"] == BASE_SHA
    assert calls[1]["hust_commit"] == HEAD_SHA
    # Identical workload / plugin commit / NPU across both runs.
    assert calls[0]["workload"] == calls[1]["workload"] == "random-online"
    assert calls[0]["ascend_commit"] == calls[1]["ascend_commit"] == PLUGIN_SHA
    assert calls[0]["npu_id"] == calls[1]["npu_id"] == 0


def test_cmd_run_paired_mode_creates_two_named_submission_dirs(
    tmp_path: Path,
) -> None:
    mod = load_module()
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    def _run_cell(workload, hust_commit, ascend_commit, npu_id=0, **kwargs):
        run_id = kwargs.get("run_id_override") or "fallback"
        sub_dir = submissions_dir / run_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "done", "run_id": run_id, "submission_dir": str(sub_dir)}

    rc = _run_cmd_run_paired(mod, tmp_path, extra_args=[], run_cell_mock=_run_cell)

    assert rc == 0
    base_dir = submissions_dir / (
        "historical-pr-backend-pr-131-base-random-online-1aa7cd10b7-cd29480d96"
    )
    head_dir = submissions_dir / (
        "historical-pr-backend-pr-131-head-random-online-e26ffd9063-cd29480d96"
    )
    assert base_dir.is_dir()
    assert head_dir.is_dir()
    # Only two submission directories should have been produced.
    assert sorted(p.name for p in submissions_dir.iterdir()) == [
        base_dir.name,
        head_dir.name,
    ]


def test_cmd_run_paired_mode_requires_pr_number(tmp_path: Path) -> None:
    mod = load_module()
    state_dir, state_file = _seed_state(tmp_path)

    args = mod.build_parser().parse_args(
        [
            "run",
            "--paired-base-head",
            "base",
            "head",
            "--workload",
            "random-online",
            "--npu-device",
            "0",
        ]
    )

    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "STATE_FILE", state_file),
    ):
        rc = mod.cmd_run(args)

    assert rc == 1


def test_cmd_run_paired_mode_requires_workload(tmp_path: Path) -> None:
    mod = load_module()
    state_dir, state_file = _seed_state(tmp_path)

    args = mod.build_parser().parse_args(
        [
            "run",
            "--paired-base-head",
            "base",
            "head",
            "--pr-number",
            "131",
            "--npu-device",
            "0",
        ]
    )

    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "STATE_FILE", state_file),
    ):
        rc = mod.cmd_run(args)

    assert rc == 1


def test_cmd_run_paired_mode_rejects_explicit_commit(tmp_path: Path) -> None:
    mod = load_module()
    state_dir, state_file = _seed_state(tmp_path)

    args = mod.build_parser().parse_args(
        [
            "run",
            "--paired-base-head",
            "base",
            "head",
            "--pr-number",
            "131",
            "--workload",
            "random-online",
            "--commit",
            BASE_SHA,
            "--npu-device",
            "0",
        ]
    )

    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "STATE_FILE", state_file),
    ):
        rc = mod.cmd_run(args)

    assert rc == 1


# ---------------------------------------------------------------------------
# Single-commit mode unchanged
# ---------------------------------------------------------------------------


def test_cmd_run_single_commit_mode_passes_no_run_id_override(
    tmp_path: Path,
) -> None:
    mod = load_module()
    state_dir, state_file = _seed_state(tmp_path)
    calls: list[dict] = []

    def _run_cell(workload, hust_commit, ascend_commit, npu_id=0, **kwargs):
        calls.append(
            {
                "hust_commit": hust_commit,
                "run_id_override": kwargs.get("run_id_override"),
            }
        )
        return {"status": "done", "run_id": "single", "submission_dir": "x"}

    args = mod.build_parser().parse_args(
        [
            "run",
            "--commit",
            BASE_SHA,
            "--ascend-commit",
            PLUGIN_SHA,
            "--workload",
            "random-online",
            "--npu-device",
            "0",
        ]
    )

    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "STATE_FILE", state_file),
        patch.object(mod, "_resolve_full_commit", return_value=BASE_SHA),
        patch.object(mod, "assert_plugin_commit_consistent", return_value=None),
        patch.object(mod, "commit_exists", return_value=True),
        patch.object(mod, "cell_already_present", return_value=False),
        patch.object(mod, "run_cell", side_effect=_run_cell),
        patch.object(mod, "current_head", return_value="orighead"),
        patch.object(mod, "_kill_port_process", return_value=None),
        patch.object(mod.subprocess, "run", return_value=MagicMock(returncode=0)),
    ):
        rc = mod.cmd_run(args)

    assert rc == 0
    assert len(calls) == 1
    # Original behaviour: no override, run_cell builds the default run_id.
    assert calls[0]["run_id_override"] is None


# ---------------------------------------------------------------------------
# run_cell propagates run_id_override to the submission directory
# ---------------------------------------------------------------------------


def test_run_cell_uses_run_id_override_for_submission_dir(tmp_path: Path) -> None:
    mod = load_module()
    state_dir = tmp_path / ".benchmarks" / "backfill-single-gpu"
    state_dir.mkdir(parents=True)
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    override = "historical-pr-backend-pr-131-base-random-online-1aa7cd10b7-cd29480d96"

    def _submit_artifact(workload, hust_commit, ascend_commit, run_id, raw, **kwargs):
        # run_cell must pass the override through as the run_id.
        assert run_id == override
        sub_dir = submissions_dir / run_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    def _run_vllm_bench(workload, hust_commit, output_dir, npu_id=0, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / "raw.json"
        raw.write_text("{}", encoding="utf-8")
        return raw

    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "REPO_ROOT", tmp_path),
        patch.object(mod, "HUST_REPO", tmp_path / "hust"),
        patch.object(mod, "ASCEND_REPO", tmp_path / "ascend"),
        patch.object(mod, "assert_plugin_commit_consistent", return_value=None),
        patch.object(mod.subprocess, "run", return_value=MagicMock(returncode=0)),
        patch.object(mod, "_kill_port_process", return_value=None),
        patch.object(mod.time, "sleep", return_value=None),
        patch.object(mod, "git_checkout", return_value=None),
        patch.object(mod, "install_ascend_plugin", return_value=None),
        patch.object(mod, "run_vllm_bench", side_effect=_run_vllm_bench),
        patch.object(mod, "submit_artifact", side_effect=_submit_artifact),
        patch.object(mod, "_check_error_rate", return_value=None),
        patch.object(mod, "validate_submission", return_value=[]),
        patch.object(mod, "normalize_submission_artifact_file", return_value=None),
    ):
        result = mod.run_cell(
            "random-online",
            BASE_SHA,
            ascend_commit=PLUGIN_SHA,
            npu_id=0,
            run_id_override=override,
        )

    assert result["status"] == "done"
    assert result["run_id"] == override
    assert (submissions_dir / override).is_dir()


# ---------------------------------------------------------------------------
# _official_spec_path model guard
# ---------------------------------------------------------------------------


def test_official_spec_path_default_models() -> None:
    mod = load_module()
    with patch.object(mod, "REPO_ROOT", Path("/repo")):
        p = mod._official_spec_path("random-online")
        assert p.name == (
            "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"
        )
        p = mod._official_spec_path("instructcoder-online")
        assert p.name == (
            "official-ascend-jan-2026-v0180-instructcoder-online-"
            "qwen25-coder-14b-910b2.json"
        )


def test_official_spec_path_accepts_matching_model() -> None:
    mod = load_module()
    with patch.object(mod, "REPO_ROOT", Path("/repo")):
        # Full repo id, hf: id, bare short name, local path with matching basename.
        for m in (
            "Qwen/Qwen2.5-14B-Instruct",
            "hf:Qwen/Qwen2.5-14B-Instruct",
            "Qwen2.5-14B-Instruct",
            "/data/models/Qwen2.5-14B-Instruct",
        ):
            p = mod._official_spec_path("random-online", m)
            assert p.name.endswith("qwen25-14b-910b2.json")
        p = mod._official_spec_path(
            "instructcoder-online", "Qwen/Qwen2.5-Coder-14B-Instruct"
        )
        assert p.name.endswith("qwen25-coder-14b-910b2.json")


def test_official_spec_path_rejects_non_default_model() -> None:
    mod = load_module()
    with patch.object(mod, "REPO_ROOT", Path("/repo")):
        # Coder model on a base-14B workload, and base model on instructcoder.
        for wl, m in (
            ("random-online", "Qwen/Qwen2.5-Coder-14B-Instruct"),
            ("instructcoder-online", "Qwen/Qwen2.5-14B-Instruct"),
            ("random-online", "/data/models/Qwen2.5-Coder-14B-Instruct"),
        ):
            with pytest.raises(ValueError, match="--model"):
                mod._official_spec_path(wl, m)


# ---------------------------------------------------------------------------
# run_cell propagates temperature / additional-config / compilation-config
# ---------------------------------------------------------------------------


def test_run_cell_propagates_temperature_and_configs(tmp_path: Path) -> None:
    mod = load_module()
    state_dir = tmp_path / ".benchmarks" / "backfill-single-gpu"
    state_dir.mkdir(parents=True)
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    seen: dict = {}

    def _run_vllm_bench(workload, hust_commit, output_dir, npu_id=0, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        seen["bench_kwargs"] = kwargs
        raw = output_dir / "raw.json"
        raw.write_text("{}", encoding="utf-8")
        return raw

    def _submit_artifact(workload, hust_commit, ascend_commit, run_id, raw, **kwargs):
        seen["submit_kwargs"] = kwargs
        sub_dir = submissions_dir / run_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    with (
        patch.object(mod, "STATE_DIR", state_dir),
        patch.object(mod, "REPO_ROOT", tmp_path),
        patch.object(mod, "HUST_REPO", tmp_path / "hust"),
        patch.object(mod, "ASCEND_REPO", tmp_path / "ascend"),
        patch.object(mod, "assert_plugin_commit_consistent", return_value=None),
        patch.object(mod.subprocess, "run", return_value=MagicMock(returncode=0)),
        patch.object(mod, "_kill_port_process", return_value=None),
        patch.object(mod.time, "sleep", return_value=None),
        patch.object(mod, "git_checkout", return_value=None),
        patch.object(mod, "install_ascend_plugin", return_value=None),
        patch.object(mod, "run_vllm_bench", side_effect=_run_vllm_bench),
        patch.object(mod, "submit_artifact", side_effect=_submit_artifact),
        patch.object(mod, "_check_error_rate", return_value=None),
        patch.object(mod, "validate_submission", return_value=[]),
        patch.object(mod, "normalize_submission_artifact_file", return_value=None),
    ):
        result = mod.run_cell(
            "random-online",
            BASE_SHA,
            ascend_commit=PLUGIN_SHA,
            npu_id=0,
            model="Qwen/Qwen2.5-14B-Instruct",
            temperature=0.0,
            additional_config='{"full_graph_parallel": true}',
            compilation_config='{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        )

    assert result["status"] == "done"
    assert seen["bench_kwargs"]["temperature"] == 0.0
    assert seen["bench_kwargs"]["additional_config"] == '{"full_graph_parallel": true}'
    assert seen["bench_kwargs"]["compilation_config"] == (
        '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    )
    assert seen["submit_kwargs"]["temperature"] == 0.0
    assert seen["submit_kwargs"]["additional_config"] == '{"full_graph_parallel": true}'
    assert seen["submit_kwargs"]["compilation_config"] == (
        '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    )


# ---------------------------------------------------------------------------
# --spec-path: explicit specialty spec binding (issue #173)
# ---------------------------------------------------------------------------


def _write_910b3_spec(
    tmp_path: Path,
    workload: str = "random-online",
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    chip_model: str = "910B3",
    chip_count: int = 1,
) -> tuple[Path, dict]:
    spec = {
        "id": (
            f"specialty-ascend-full-graph-parallel-inplace-{workload}-qwen25-14b-910b3"
        ),
        "label": "Specialty Ascend full-graph-parallel inplace dual-stream",
        "scenario": workload,
        "model": model,
        "model_parameters": "14B",
        "model_precision": "FP16",
        "hardware_vendor": "Huawei",
        "hardware_chip_model": chip_model,
        "chip_count": chip_count,
        "node_count": 1,
        "server_parameters": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 32768,
        },
        "client_parameters": {
            "backend": "vllm",
            "endpoint": "/v1/completions",
            "dataset_name": "random",
            "num_prompts": 200,
            "request_rate": 1,
        },
    }
    spec_dir = tmp_path / "docs" / "official-baselines"
    spec_dir.mkdir(parents=True, exist_ok=True)
    path = spec_dir / (
        f"specialty-ascend-full-graph-parallel-inplace-{workload}-qwen25-14b-910b3.json"
    )
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_resolve_spec_path_explicit_overrides_default(tmp_path: Path) -> None:
    mod = load_module()
    spec_path, _ = _write_910b3_spec(tmp_path)
    with patch.object(mod, "REPO_ROOT", tmp_path):
        resolved = mod._resolve_spec_path(
            "random-online", "Qwen/Qwen2.5-14B-Instruct", str(spec_path)
        )
        assert resolved == spec_path.resolve()
        # 不传 spec_path 时回到默认 910B2 official spec
        default = mod._resolve_spec_path("random-online", "Qwen/Qwen2.5-14B-Instruct")
        assert default.name.endswith("qwen25-14b-910b2.json")


def test_resolve_spec_path_rejects_missing_file(tmp_path: Path) -> None:
    mod = load_module()
    with patch.object(mod, "REPO_ROOT", tmp_path):
        with pytest.raises(ValueError, match="--spec-path does not exist"):
            mod._resolve_spec_path("random-online", None, str(tmp_path / "nope.json"))


def test_validate_explicit_spec_accepts_matching(tmp_path: Path) -> None:
    mod = load_module()
    _, spec = _write_910b3_spec(tmp_path)
    mod._validate_explicit_spec(spec, "random-online", "Qwen/Qwen2.5-14B-Instruct")


def test_validate_explicit_spec_rejects_mismatched_scenario(tmp_path: Path) -> None:
    mod = load_module()
    _, spec = _write_910b3_spec(tmp_path, workload="random-online")
    with pytest.raises(ValueError, match="scenario"):
        mod._validate_explicit_spec(
            spec, "sharegpt-online", "Qwen/Qwen2.5-14B-Instruct"
        )


def test_validate_explicit_spec_rejects_mismatched_model(tmp_path: Path) -> None:
    mod = load_module()
    _, spec = _write_910b3_spec(tmp_path, model="Qwen/Qwen2.5-14B-Instruct")
    with pytest.raises(ValueError, match="model"):
        mod._validate_explicit_spec(
            spec, "random-online", "Qwen/Qwen2.5-Coder-14B-Instruct"
        )


def test_validate_explicit_spec_rejects_mismatched_chip_count(tmp_path: Path) -> None:
    mod = load_module()
    _, spec = _write_910b3_spec(tmp_path, chip_count=2)
    with pytest.raises(ValueError, match="chip_count"):
        mod._validate_explicit_spec(spec, "random-online", "Qwen/Qwen2.5-14B-Instruct")


def test_generate_same_spec_explicit_910b3_binds_910b3(tmp_path: Path) -> None:
    mod = load_module()
    spec_path, _ = _write_910b3_spec(tmp_path)
    with patch.object(mod, "REPO_ROOT", tmp_path):
        same_spec = mod._generate_same_spec(
            "random-online", "Qwen/Qwen2.5-14B-Instruct", str(spec_path)
        )
    assert same_spec["hardware_chip_model"] == "910B3"
    assert same_spec["chip_count"] == 1
    assert "910b3" in same_spec["spec_id"]
    assert "910b3" in same_spec["spec_source"]


def test_submit_artifact_explicit_spec_drives_910b3(tmp_path: Path) -> None:
    mod = load_module()
    spec_path, _ = _write_910b3_spec(tmp_path)
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")

    submit_cmds: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        submit_cmds.append(list(cmd))
        return MagicMock(returncode=0)

    with (
        patch.object(mod, "REPO_ROOT", tmp_path),
        patch.object(mod, "STATE_DIR", tmp_path / "state"),
        patch.object(mod, "_derive_engine_version", return_value="v0.0.0-test"),
        patch.object(mod.subprocess, "run", side_effect=_fake_run),
    ):
        mod.submit_artifact(
            "random-online",
            BASE_SHA,
            PLUGIN_SHA,
            "run-id",
            raw,
            model="Qwen/Qwen2.5-14B-Instruct",
            spec_path=str(spec_path),
        )

    assert len(submit_cmds) == 1
    cmd = submit_cmds[0]
    assert cmd[cmd.index("--hardware-chip-model") + 1] == "910B3"
    assert cmd[cmd.index("--chip-count") + 1] == "1"
    assert cmd[cmd.index("--spec-path") + 1] == str(spec_path.resolve())
    assert cmd[cmd.index("--model-name") + 1] == "Qwen/Qwen2.5-14B-Instruct"
