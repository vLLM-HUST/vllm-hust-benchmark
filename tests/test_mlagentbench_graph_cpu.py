"""Small software fixtures for context construction, not real native execution."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ml_graph_cpu", ROOT / "scripts/prepare_mlagentbench_graph_cpu.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def fixture(tmp_path, monkeypatch):
    software = tmp_path / "cache"
    software.mkdir()
    items = []
    for group in ["core", "scatter", "sparse"]:
        name = group + ".whl"
        raw = ("fixture " + group).encode()
        (software / name).write_bytes(raw)
        items.append(
            {
                "project": group,
                "version": "1.0",
                "group": group,
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"software": items}))
    monkeypatch.setattr(MODULE, "PROFILE", profile)
    return SimpleNamespace(
        action="prepare",
        software_root=software,
        output=tmp_path / "context",
        max_bytes=None,
        resume=False,
    )


def test_plan_does_not_open_assets():
    assert MODULE.prepare(SimpleNamespace(action="plan"))["model_calls"] == 0


def test_context_roundtrip_retains_offline_hashed_software(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    report = MODULE.prepare(args)
    assert report["software_files"] == 3 and report["build_network_required"] is False
    assert report["source_checkbox_eligible"] is False
    args.action = "verify"
    assert MODULE.prepare(args) == report
    for group in ["core", "scatter", "sparse"]:
        assert (
            "--hash=sha256:" in (args.output / f"requirements-{group}.txt").read_text()
        )


def test_corrupted_source_software_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    (args.software_root / "core.whl").write_bytes(b"bad")
    with pytest.raises(ValueError):
        MODULE.prepare(args)
    assert not args.output.exists()


def test_acquire_reuses_verified_cache_without_contacting_pypi(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.action = "acquire"

    def reject_network(*args, **kwargs):
        raise AssertionError("verified software must not be downloaded again")

    monkeypatch.setattr(MODULE, "urlopen", reject_network)
    result = MODULE.prepare(args)
    assert result["software_files"] == 3
    assert result["source_checkbox_eligible"] is False
    assert not args.output.exists()


def test_existing_context_not_overwritten_and_extra_files_rejected(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path, monkeypatch)
    MODULE.prepare(args)
    with pytest.raises(FileExistsError):
        MODULE.prepare(args)
    (args.output / "unexpected.txt").write_text("fixture")
    args.action = "verify"
    with pytest.raises(ValueError, match="unexpected"):
        MODULE.prepare(args)


def test_overlapping_context_and_cache_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.output = args.software_root / "context"
    with pytest.raises(ValueError, match="separate"):
        MODULE.prepare(args)


def test_real_profile_keeps_original_software_and_scientific_cpu_boundary():
    profile = json.loads(
        (
            ROOT / "configs/workload-native/mlagentbench-graph-cpu-software.json"
        ).read_text()
    )
    assert len(profile["software"]) == 7
    assert {item["group"] for item in profile["software"]} == {
        "core",
        "scatter",
        "sparse",
    }
    assert "LLM/encoder/judge remain on NPU" in profile["runtime"]
    assert profile["original_data_or_model_weights_bundled"] is False
    dockerfile = MODULE.DOCKERFILE.read_text()
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        assert name + "=2" in dockerfile


def test_actual_public_environment_receipt_is_bound_to_recipe_and_scope():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/mlagentbench-ogbn-arxiv-preparation-20261102.json"
        ).read_text("utf-8")
    )
    qualification = receipt["public_environment_qualification"]
    assert receipt["public_graph_environment_recipe_pending"] is False
    assert (
        qualification["dockerfile_sha256"]
        == hashlib.sha256(MODULE.DOCKERFILE.read_text("utf-8").encode()).hexdigest()
    )
    assert qualification["all_seven_original_software_identities_verified"] is True
    assert (
        qualification["original_score_record_equals_prior_qualified_environment"]
        is True
    )
    assert qualification["original_scorer_saved_prediction_score"] == 0.2571857704256939
    assert qualification["model_or_training_reruns"] == 0
    assert qualification["full_agent_execution_repeated_on_public_image"] is False
    assert qualification["source_checkbox_eligible"] is False
