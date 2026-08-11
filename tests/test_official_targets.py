from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.official_targets import (
    PUBLIC_CODE_MODEL,
    PUBLIC_TEXT_MODEL,
    PUBLIC_VISION_MODEL,
    build_registry,
    generated_outputs,
    render_active_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generated_outputs_are_current() -> None:
    for path, expected in generated_outputs(REPO_ROOT).items():
        assert path.read_text(encoding="utf-8") == expected


def test_registry_matches_schema() -> None:
    registry = build_registry(REPO_ROOT)
    schema = json.loads(
        (REPO_ROOT / "schemas" / "official_target_registry_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(registry)


def test_public_target_matrix_is_exact() -> None:
    registry = build_registry(REPO_ROOT)
    active = [target for target in registry["targets"] if target["status"] == "active"]
    assert len(active) == 9
    assert {target["model"]["id"] for target in active} == {
        PUBLIC_TEXT_MODEL,
        PUBLIC_CODE_MODEL,
        PUBLIC_VISION_MODEL,
    }
    assert {target["intended_use"] for target in active} == {"public-leaderboard"}
    for target in active:
        assert target["hardware"]["chip_model"] == "910B2"
        assert target["hardware"]["chip_count"] == 1
        assert target["model"]["precision"] == "FP16"
        assert target["server_parameters"]["tensor_parallel_size"] == 1
        assert target["server_parameters"]["gpu_memory_utilization"] == 0.6
        assert target["server_parameters"]["max_model_len"] == 32768


def test_perfgate_never_appears_as_active_public_target() -> None:
    registry = build_registry(REPO_ROOT)
    perfgate = [
        target for target in registry["targets"] if target["intended_use"] == "perfgate"
    ]
    assert perfgate
    assert {target["status"] for target in perfgate} == {"provisional"}
    assert all(target["model"]["parameters"] == "3B" for target in perfgate)


def test_source_spec_hashes_match() -> None:
    registry = build_registry(REPO_ROOT)
    for target in registry["targets"]:
        path = REPO_ROOT / target["source_spec"]["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert target["source_spec"]["sha256"] == digest


def test_target_ids_are_unique() -> None:
    registry = build_registry(REPO_ROOT)
    target_ids = [target["target_id"] for target in registry["targets"]]
    assert len(target_ids) == len(set(target_ids))


def test_active_target_output_excludes_provisional_profiles() -> None:
    output = render_active_targets(build_registry(REPO_ROOT))
    assert "Qwen/Qwen2.5-14B-Instruct" in output
    assert "Qwen/Qwen2.5-3B-Instruct" not in output


def test_public_target_drift_fails_closed(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"
    )
    spec_root = tmp_path / "docs" / "official-baselines"
    spec_root.mkdir(parents=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["server_parameters"]["gpu_memory_utilization"] = 0.9
    (spec_root / source.name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_memory_utilization=0.6"):
        build_registry(tmp_path)


def _classify(spec: dict) -> tuple[str, str, str]:
    from vllm_hust_benchmark.official_targets import _classify_spec

    return _classify_spec(Path("classify-spec.json"), spec)


def test_910b2_text_spec_remains_public_leaderboard_active() -> None:
    """The hardware gate must not regress the existing 910B2 public targets."""
    assert _classify(
        {
            "scenario": "random-online",
            "model": PUBLIC_TEXT_MODEL,
            "chip_count": 1,
            "hardware_chip_model": "910B2",
        }
    ) == ("public-leaderboard", "active", "core-text")
    assert _classify(
        {
            "scenario": "instructcoder-online",
            "model": PUBLIC_CODE_MODEL,
            "chip_count": 1,
            "hardware_chip_model": "910B2",
        }
    ) == ("public-leaderboard", "active", "code")
    assert _classify(
        {
            "scenario": "visionarena-online",
            "model": PUBLIC_VISION_MODEL,
            "chip_count": 1,
            "hardware_chip_model": "910B2",
        }
    ) == ("public-leaderboard", "active", "multimodal")


def test_910b3_spec_is_classified_as_specialty_provisional() -> None:
    """Non-910B2 hardware must never enter the public-leaderboard/active set,
    even when scenario/model/chip-count match a public target shape."""
    for scenario in ("random-online", "random-latency", "sharegpt-online"):
        assert _classify(
            {
                "scenario": scenario,
                "model": PUBLIC_TEXT_MODEL,
                "chip_count": 1,
                "hardware_chip_model": "910B3",
            }
        ) == ("specialty", "provisional", "specialty-text")
    assert _classify(
        {
            "scenario": "instructcoder-online",
            "model": PUBLIC_CODE_MODEL,
            "chip_count": 1,
            "hardware_chip_model": "910B3",
        }
    ) == ("specialty", "provisional", "specialty")


def test_built_registry_marks_910b3_targets_specialty() -> None:
    """Every 910B3 spec in docs/official-baselines must land as
    specialty/provisional in the built registry, never active."""
    registry = build_registry(REPO_ROOT)
    hardware_910b3 = [
        target
        for target in registry["targets"]
        if target["hardware"]["chip_model"] == "910B3"
    ]
    assert hardware_910b3
    for target in hardware_910b3:
        assert target["intended_use"] == "specialty"
        assert target["status"] == "provisional"
        assert target["profile"] == "specialty-text"


def test_public_target_validation_rejects_non_910b2_hardware() -> None:
    """_validate_public_target must reject hardware other than 910B2."""
    from vllm_hust_benchmark.official_targets import _validate_public_target

    spec = {
        "scenario": "random-online",
        "model": PUBLIC_TEXT_MODEL,
        "chip_count": 1,
        "node_count": 1,
        "model_precision": "FP16",
        "hardware_chip_model": "910B3",
        "server_parameters": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 32768,
        },
        "baseline_target": {"vllm_ref": "v0.18.0", "vllm_ascend_ref": "v0.18.0"},
    }
    with pytest.raises(ValueError, match="910B2"):
        _validate_public_target(spec, Path("specialty-910b3.json"))
