from __future__ import annotations

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


def test_quarantined_entry_ids_are_rejected() -> None:
    module = load_module()
    for entry_id in module.QUARANTINED_ENTRY_IDS:
        candidate = entry(entry_id, same_spec())
        errors = module.validate_entry(
            candidate,
            source=Path("leaderboard_single.json"),
        )
        assert any("quarantined entry" in error for error in errors), (
            f"entry_id {entry_id} should be rejected as quarantined"
        )
        assert any("issue #79" in error for error in errors)


def test_non_quarantined_entry_id_passes_quarantine_gate() -> None:
    module = load_module()
    candidate = entry("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", same_spec())
    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )
    assert not any("quarantined entry" in error for error in errors)
