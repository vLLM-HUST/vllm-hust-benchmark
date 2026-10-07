from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from vllm_hust_benchmark.official_targets import _classify_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "materialize_issue136_dense_targets.py"
CORE_COMMIT = "c" * 40
PLUGIN_COMMIT = "d" * 40


def _load_script():
    spec = importlib.util.spec_from_file_location("issue136_targets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_repo(tmp_path: Path, module) -> Path:
    root = tmp_path / "repo"
    target_dir = root / "docs" / "official-baselines"
    target_dir.mkdir(parents=True)
    for workload in module.WORKLOADS:
        source = module._template_path(REPO_ROOT, workload)
        shutil.copy2(source, target_dir / source.name)
    return root


def test_materializes_directly_named_fixed_targets(tmp_path: Path) -> None:
    module = _load_script()
    repo = _fixture_repo(tmp_path, module)

    paths = module.materialize(
        repo,
        core_commit=CORE_COMMIT,
        plugin_commit=PLUGIN_COMMIT,
        load_profile="fixed-1-rps",
        rates={1: 1.0, 2: 1.0, 4: 1.0},
    )

    assert len(paths) == 12
    assert len({json.loads(path.read_text())["id"] for path in paths}) == 12
    for path in paths:
        payload = json.loads(path.read_text())
        tp = payload["server_parameters"]["tensor_parallel_size"]
        assert payload["chip_count"] == tp
        assert payload["client_parameters"]["request_rate"] == 1.0
        assert f"-tp{tp}-fixed-1rps-" in payload["id"]
        assert "vllm-0.23.0-vllm-ascend-0.25.1rc1" in payload["id"]
        assert payload["baseline_target"]["vllm_ref"] == CORE_COMMIT
        assert payload["baseline_target"]["vllm_ascend_ref"] == PLUGIN_COMMIT
        assert payload["server_parameters"]["compilation_config"] == {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        }
        assert payload["issue_136_contract"]["graph_capture_sizes"] == [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
        ]
        assert _classify_spec(path, payload) == (
            "specialty",
            "provisional",
            "dense-scaling",
        )


def test_fixed_profile_rejects_scaled_rate(tmp_path: Path) -> None:
    module = _load_script()
    repo = _fixture_repo(tmp_path, module)

    with pytest.raises(ValueError, match="fixed-1-rps"):
        module.materialize(
            repo,
            core_commit=CORE_COMMIT,
            plugin_commit=PLUGIN_COMMIT,
            load_profile="fixed-1-rps",
            rates={1: 1.0, 2: 2.0, 4: 4.0},
        )


def test_scaled_profile_rejects_tp_only_rates(tmp_path: Path) -> None:
    module = _load_script()
    repo = _fixture_repo(tmp_path, module)

    with pytest.raises(ValueError, match="workload-specific"):
        module.materialize(
            repo,
            core_commit=CORE_COMMIT,
            plugin_commit=PLUGIN_COMMIT,
            load_profile="scaled-load",
            rates={1: 1.5, 2: 3.0, 4: 6.0},
        )


def test_scaled_profile_uses_each_workloads_own_rates(tmp_path: Path) -> None:
    module = _load_script()
    repo = _fixture_repo(tmp_path, module)
    matrix = {
        workload: {
            1: float(index + 1),
            2: float(index + 2),
            4: float(index + 4),
        }
        for index, workload in enumerate(module.WORKLOADS)
    }

    paths = module.materialize(
        repo,
        core_commit=CORE_COMMIT,
        plugin_commit=PLUGIN_COMMIT,
        load_profile="scaled-load",
        rates=matrix,
    )

    assert len(paths) == 12
    for path in paths:
        payload = json.loads(path.read_text())
        workload = payload["scenario"]
        tp = payload["server_parameters"]["tensor_parallel_size"]
        expected = matrix[workload][tp]
        assert payload["client_parameters"]["request_rate"] == expected
        assert payload["issue_136_contract"]["request_rate"] == expected


def test_loads_versioned_workload_rate_matrix(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "rates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "issue-136-workload-rate-matrix/v1",
                "rates": {
                    workload: {"1": 1.0, "2": 2.0, "4": 4.0}
                    for workload in module.WORKLOADS
                },
            }
        )
    )

    matrix = module._load_rate_matrix(path)

    assert matrix["random-online"] == {1: 1.0, 2: 2.0, 4: 4.0}
