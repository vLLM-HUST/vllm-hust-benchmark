from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from vllm_hust_benchmark.same_spec import build_same_spec_payload

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_public_leaderboard_snapshots.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_public_snapshots", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_spec() -> dict:
    return build_same_spec_payload(
        {
            "id": "test-spec",
            "label": "test",
            "scenario": "random-online",
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "model_parameters": "14B",
            "model_precision": "FP16",
            "hardware_vendor": "Huawei",
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
            "server_parameters": {"tensor_parallel_size": 1},
            "client_parameters": {
                "backend": "vllm",
                "dataset_name": "random",
                "num_prompts": 1,
                "input_len": 16,
                "output_len": 8,
            },
        }
    )


def entry(entry_id: str, payload: dict) -> dict:
    return {
        "entry_id": entry_id,
        "engine": "test-engine",
        "engine_version": "1.0",
        "workload": {"name": "test"},
        "model": {"name": "test", "precision": "FP16"},
        "hardware": {"chip_model": "910B2"},
        "same_spec": payload,
    }


def test_rejects_one_recorded_hash_for_different_effective_parameters(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    first = same_spec()
    second = json.loads(json.dumps(first))
    second["resolved_server_parameters"]["enable_prefix_caching"] = "true"
    (tmp_path / "leaderboard_single.json").write_text(
        json.dumps([entry("first", first), entry("second", second)]),
        encoding="utf-8",
    )
    (tmp_path / "leaderboard_multi.json").write_text("[]", encoding="utf-8")
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_public_leaderboard_snapshots.py", "--snapshot-dir", str(tmp_path)],
    )

    assert module.main() == 1
    assert "maps to different effective parameters" in capsys.readouterr().out


def test_future_official_entry_requires_effective_config_contract() -> None:
    module = load_module()
    payload = same_spec()
    payload["spec_id"] = (
        "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
    )
    payload["scenario"] = "random-online"
    candidate = entry("future", payload)
    candidate["engine"] = "vllm-hust"
    candidate["workload"] = {
        "name": "random-online",
        "input_length": 1024,
        "output_length": 256,
        "batch_size": None,
        "concurrent_requests": None,
        "dataset": "random",
    }
    candidate["model"]["name"] = "Qwen/Qwen2.5-14B-Instruct"
    candidate["metadata"] = {"submitted_at": "2026-07-25T00:00:00Z"}

    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )

    assert any("workload config contract" in error for error in errors)
    assert any("workload_config_contract" in error for error in errors)


def test_accepts_only_attested_registered_production_trace_baseline() -> None:
    module = load_module()
    repo_root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (
            repo_root
            / "submissions"
            / "official-ascend-jan-2026-v0.22.1rc1-tracelab-coding-agent-replay-deepseek-r1-distill-qwen32b-2chip-910b2"
            / "run_leaderboard.json"
        ).read_text(encoding="utf-8")
    )

    assert module.validate_entry(
        artifact, source=Path("leaderboard_multi.json")
    ) == []

    spoofed = copy.deepcopy(artifact)
    spoofed["metadata"]["verified"] = False
    errors = module.validate_entry(spoofed, source=Path("leaderboard_multi.json"))
    assert any("public vllm baseline must be 0.18.0" in error for error in errors)
    assert any("retired public precision 'BF16'" in error for error in errors)


def test_compare_snapshot_rejects_mismatched_resolved_hashes(tmp_path: Path) -> None:
    module = load_module()
    left = entry("left", same_spec())
    right_payload = same_spec()
    right_payload["resolved_server_parameters"]["tensor_parallel_size"] = 2
    right_payload["resolved_spec_hash"] = "different-hash"
    right = entry("right", right_payload)
    (tmp_path / "leaderboard_compare.json").write_text(
        json.dumps(
            {
                "preferred_pair_count": 1,
                "preferred_pairs": [
                    {
                        "preferred_pair": {
                            "left": {
                                "entry_id": "left",
                                "same_spec": {
                                    "resolved_spec_hash": left["same_spec"][
                                        "resolved_spec_hash"
                                    ]
                                },
                            },
                            "right": {
                                "entry_id": "right",
                                "same_spec": {
                                    "resolved_spec_hash": "different-hash"
                                },
                            },
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    errors = module.validate_compare_snapshot(
        tmp_path, {"left": left, "right": right}
    )

    assert any("resolved_spec_hash mismatch" in error for error in errors)
