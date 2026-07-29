#!/usr/bin/env python3
"""为 issue #95 Layer 2 mock 模式生成 mock merge gate artifact。

在不依赖真实 NPU 的前提下，生成各场景的 base/head run_leaderboard.json，
让 merge-gate-check 的 CI 接线正确性可以被验证。

用法:
    python scripts/generate_mock_merge_gate_artifacts.py \
        --scenario pass --output-dir /tmp/mock/pass

生成的目录结构:
    <output-dir>/
        base/run_leaderboard.json
        head/run_leaderboard.json
        scenario_manifest.json   # 含场景元信息（期望 disposition、PR labels 等）

场景清单见 _SCENARIOS。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_SPEC_ID = "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
VALID_SPEC_HASH = "f8cc8fc26b4b9bb06d50f079174894a95d2bc0f49799374a652e6e04b75c8feb"
SHAREGPT_SPEC_ID = "official-ascend-jan-2026-v0.18.0-sharegpt-online-qwen25-14b-910b2"


def _make_run_leaderboard(
    *,
    data_source: str = "real-online-mock",
    model_repo_id: str = "Qwen/Qwen2.5-14B-Instruct",
    model_parameters: str = "14B",
    model_precision: str = "FP16",
    chip_model: str = "910B2",
    chip_count: int = 1,
    tensor_parallel_size: int = 1,
    workload_name: str = "random-online",
    gpu_memory_utilization: float = 0.6,
    max_model_len: int = 32768,
    spec_id: str = VALID_SPEC_ID,
    spec_hash: str = VALID_SPEC_HASH,
    same_spec_present: bool = True,
) -> dict:
    """构造一个 run_leaderboard.json payload。"""
    payload: dict = {
        "entry_id": f"mock-{model_parameters}-{chip_model}-{workload_name}",
        "engine": "vllm-hust",
        "engine_version": "v0.23.1rc0-mock",
        "config_type": "single_gpu" if chip_count == 1 else "multi_gpu",
        "hardware": {
            "vendor": "Huawei",
            "chip_model": chip_model,
            "chip_count": chip_count,
        },
        "model": {
            "repo_id": model_repo_id,
            "name": model_repo_id,
            "parameters": model_parameters,
            "precision": model_precision,
        },
        "workload": {"name": workload_name},
        "metrics": {"ttft_ms": 100.0, "tbt_ms": 30.0, "throughput_tps": 200.0},
        "metadata": {
            "submitted_at": "2026-07-29T00:00:00Z",
            "data_source": data_source,
        },
    }
    if same_spec_present:
        payload["same_spec"] = {
            "spec_id": spec_id,
            "resolved_spec_hash": spec_hash,
            "resolved_server_parameters": {
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            },
        }
    return payload


# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, dict] = {
    "pass": {
        "expected_disposition": "pass",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(),
        "head": lambda: _make_run_leaderboard(),
    },
    "fail_config_drift": {
        "expected_disposition": "fail",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(gpu_memory_utilization=0.9),
        "head": lambda: _make_run_leaderboard(gpu_memory_utilization=0.9),
    },
    "fail_config_drift_head_only": {
        "expected_disposition": "fail",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(),
        "head": lambda: _make_run_leaderboard(gpu_memory_utilization=0.9),
    },
    "fail_data_source": {
        "expected_disposition": "fail",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(data_source="smoke-test"),
        "head": lambda: _make_run_leaderboard(data_source="smoke-test"),
    },
    "fail_unpaired_spec": {
        "expected_disposition": "fail",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(spec_id=VALID_SPEC_ID),
        "head": lambda: _make_run_leaderboard(spec_id=SHAREGPT_SPEC_ID),
    },
    "fail_3b_not_14b": {
        "expected_disposition": "fail",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(
            model_repo_id="Qwen/Qwen2.5-3B-Instruct",
            model_parameters="3B",
        ),
        "head": lambda: _make_run_leaderboard(
            model_repo_id="Qwen/Qwen2.5-3B-Instruct",
            model_parameters="3B",
        ),
    },
    "fail_missing_artifact": {
        "expected_disposition": "fail",
        "expected_head_status": "missing",
        "base": lambda: _make_run_leaderboard(),
        "head": None,  # 不生成 head
    },
    "fail_missing_target_declaration": {
        "expected_disposition": "fail",
        # 不声明 target_id/version/profile_id → C3 fix 应 fail
        "base": lambda: _make_run_leaderboard(),
        "head": lambda: _make_run_leaderboard(),
    },
    "skip_docs_only": {
        "expected_disposition": "skip",
        "pr_labels": ["perf-skip:docs-only"],
        "skip_approver": "reviewer-bot",
        "base": None,
        "head": None,
    },
    "skip_docs_only_no_approver": {
        "expected_disposition": "fail",
        "pr_labels": ["perf-skip:docs-only"],
        # 不提供 skip_approver → M5 fix 应 fail
        "base": None,
        "head": None,
    },
    "specialty_valid": {
        "expected_disposition": "pass",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "multi-chip-2chip-text-14b",
        "specialty_spec": "multi-chip-2chip-text-14b",
        "specialty_reason": "验证 2-chip TP 并行扩展收益",
        "base": lambda: _make_run_leaderboard(
            chip_count=2,
            tensor_parallel_size=2,
        ),
        "head": lambda: _make_run_leaderboard(
            chip_count=2,
            tensor_parallel_size=2,
        ),
    },
    "specialty_no_reason": {
        "expected_disposition": "fail",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "multi-chip-2chip-text-14b",
        "specialty_spec": "multi-chip-2chip-text-14b",
        # specialty_reason 不传
        "base": lambda: _make_run_leaderboard(
            chip_count=2,
            tensor_parallel_size=2,
        ),
        "head": lambda: _make_run_leaderboard(
            chip_count=2,
            tensor_parallel_size=2,
        ),
    },
    "fail_registry_hash_mismatch": {
        "expected_disposition": "fail",
        "declared_target_id": "nonexistent-target",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
        "base": lambda: _make_run_leaderboard(),
        "head": lambda: _make_run_leaderboard(),
    },
}


def generate(scenario: str, output_dir: Path) -> dict:
    """生成指定场景的 mock artifact，返回 scenario_manifest。"""
    if scenario not in _SCENARIOS:
        raise ValueError(
            f"unknown scenario: {scenario!r}; supported: {sorted(_SCENARIOS)}"
        )
    config = _SCENARIOS[scenario]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_factory = config.get("base")
    head_factory = config.get("head")

    if base_factory is not None:
        base_path = output_dir / "base" / "run_leaderboard.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            json.dumps(base_factory(), indent=2) + "\n", encoding="utf-8"
        )

    if head_factory is not None:
        head_path = output_dir / "head" / "run_leaderboard.json"
        head_path.parent.mkdir(parents=True, exist_ok=True)
        head_path.write_text(
            json.dumps(head_factory(), indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "scenario": scenario,
        "expected_disposition": config["expected_disposition"],
    }
    if "expected_head_status" in config:
        manifest["expected_head_status"] = config["expected_head_status"]
    if "pr_labels" in config:
        manifest["pr_labels"] = config["pr_labels"]
    if "specialty_spec" in config:
        manifest["specialty_spec"] = config["specialty_spec"]
    if "specialty_reason" in config:
        manifest["specialty_reason"] = config["specialty_reason"]
    if "declared_target_id" in config:
        manifest["declared_target_id"] = config["declared_target_id"]
    if "declared_target_version" in config:
        manifest["declared_target_version"] = config["declared_target_version"]
    if "declared_profile_id" in config:
        manifest["declared_profile_id"] = config["declared_profile_id"]
    if "skip_approver" in config:
        manifest["skip_approver"] = config["skip_approver"]

    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate mock merge gate artifacts for issue #95 Layer 2 mock mode."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(_SCENARIOS),
        help="Mock scenario to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for mock artifacts.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print supported scenarios and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_scenarios:
        for name in sorted(_SCENARIOS):
            print(f"{name}: expected={_SCENARIOS[name]['expected_disposition']}")
        return 0

    manifest = generate(args.scenario, args.output_dir)
    print(f"Generated mock scenario '{args.scenario}' at {args.output_dir}")
    print(f"  expected_disposition: {manifest['expected_disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
