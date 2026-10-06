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
