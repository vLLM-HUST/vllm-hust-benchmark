from __future__ import annotations

import importlib.util
import json
import math
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

    assert len(paths) == 15
    assert len({json.loads(path.read_text())["id"] for path in paths}) == 15
    for path in paths:
        payload = json.loads(path.read_text())
        tp = payload["server_parameters"]["tensor_parallel_size"]
        assert payload["chip_count"] == tp
        assert payload["client_parameters"]["request_rate"] == 1.0
        assert payload["client_parameters"]["temperature"] == 0
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
        assert payload["issue_136_contract"]["temperature"] == 0
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

    assert len(paths) == 15
    for path in paths:
        payload = json.loads(path.read_text())
        workload = next(
            workload for workload in module.WORKLOADS if f"-{workload}-" in path.name
        )
        tp = payload["server_parameters"]["tensor_parallel_size"]
        expected = matrix[workload][tp]
        assert payload["client_parameters"]["request_rate"] == expected
        assert payload["client_parameters"]["temperature"] == 0
        assert payload["issue_136_contract"]["request_rate"] == expected
        assert payload["issue_136_contract"]["temperature"] == 0


def test_loads_versioned_workload_rate_matrix(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "rates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "issue-136-workload-rate-matrix/v2",
                "rates": {
                    workload: {"1": 1.0, "2": 2.0, "4": 4.0}
                    for workload in module.WORKLOADS
                },
            }
        )
    )

    matrix = module._load_rate_matrix(path)

    assert matrix["random-online"] == {1: 1.0, 2: 2.0, 4: 4.0}


def test_communication_profile_uses_standard_decode_heavy_random_workload(
    tmp_path: Path,
) -> None:
    module = _load_script()
    repo = _fixture_repo(tmp_path, module)

    paths = module.materialize(
        repo,
        core_commit=CORE_COMMIT,
        plugin_commit=PLUGIN_COMMIT,
        load_profile="fixed-1-rps",
        rates={1: 1.0, 2: 1.0, 4: 1.0},
    )

    communication_paths = [
        path for path in paths if "communication-sensitive" in path.name
    ]
    assert len(communication_paths) == 3
    for path in communication_paths:
        payload = json.loads(path.read_text())
        tp = payload["server_parameters"]["tensor_parallel_size"]
        client = payload["client_parameters"]
        contract = payload["issue_136_contract"]
        assert payload["scenario"] == "random-online"
        assert client["dataset_name"] == "random"
        assert client["input_len"] == 128
        assert client["output_len"] == 1024
        assert client["random_range_ratio"] == 0.0
        assert client["ignore_eos"] is True
        assert client["temperature"] == 0
        assert client["seed"] == 0
        assert contract["workload_class"] == "decode-heavy-tp-collective-proxy/v1"
        assert contract["tp_role"] == (
            "no-cross-rank-control" if tp == 1 else "tp-collective"
        )


@pytest.mark.parametrize("invalid_rate", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_rate_matrix_rejects_non_finite_or_non_positive_rates(
    tmp_path: Path, invalid_rate: float
) -> None:
    module = _load_script()
    repo = _fixture_repo(tmp_path, module)
    matrix = {workload: {1: 1.0, 2: 2.0, 4: 4.0} for workload in module.WORKLOADS}
    matrix[module.COMMUNICATION_WORKLOAD][2] = invalid_rate

    with pytest.raises(ValueError, match="invalid|finite|greater than zero"):
        module.materialize(
            repo,
            core_commit=CORE_COMMIT,
            plugin_commit=PLUGIN_COMMIT,
            load_profile="scaled-load",
            rates=matrix,
        )


@pytest.mark.parametrize("invalid_rate", ["NaN", "Infinity", "-Infinity", "0"])
def test_cli_rate_parser_rejects_invalid_rates(invalid_rate: str) -> None:
    module = _load_script()

    with pytest.raises(Exception, match="invalid TP/rate"):
        module._parse_rates(f"1=1,2={invalid_rate},4=4")


@pytest.mark.parametrize("invalid_rate", [math.nan, math.inf, -math.inf])
def test_json_rate_matrix_rejects_non_finite_rates(
    tmp_path: Path, invalid_rate: float
) -> None:
    module = _load_script()
    path = tmp_path / "rates.json"
    rates = {workload: {"1": 1.0, "2": 2.0, "4": 4.0} for workload in module.WORKLOADS}
    rates[module.COMMUNICATION_WORKLOAD]["2"] = invalid_rate
    path.write_text(
        json.dumps({"schema_version": module.RATE_MATRIX_SCHEMA, "rates": rates})
    )

    with pytest.raises(ValueError, match="finite and greater than zero"):
        module._load_rate_matrix(path)
