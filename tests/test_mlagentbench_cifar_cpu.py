"""CPU image context/probe contracts, not Docker builds or native task runs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module(name, script):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / script)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PREPARE = module("mlagent_cpu_context", "prepare_mlagentbench_cifar_cpu.py")
PROBE = module("mlagent_cpu_probe", "check_mlagentbench_cifar_cpu.py")


def test_plan_without_software_does_not_create_or_import_neural_runtime(tmp_path):
    report = PREPARE.prepare(
        SimpleNamespace(
            action="plan",
            output=tmp_path / "absent",
            software_root=tmp_path / "missing",
        )
    )
    assert report["model_calls"] == report["native_tasks_run"] == 0
    assert not report["source_checkbox_eligible"]
    assert not list(tmp_path.iterdir())
    result = PROBE.probe(SimpleNamespace(profile=PROBE.PROFILE, execute=False))
    assert result["status"] == "PLAN_ONLY_NO_NEURAL_IMPORT"


def test_base_version_mismatch_is_not_silently_accepted():
    PROBE.validate_metadata(
        {"python": "3.10.22", "torch": "2.3.0"}, {"python": "3.10.22", "torch": "2.3.0"}
    )
    with pytest.raises(ValueError, match="torch"):
        PROBE.validate_metadata({"torch": "2.3.0"}, {"torch": "2.0.0"})


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    software = tmp_path / "software"
    software.mkdir()
    raw = b"explicit context fixture, not a real installable wheel"
    name = "torchvision-0.18.0-cp310-cp310-manylinux2014_aarch64.whl"
    (software / name).write_bytes(raw)
    profile = tmp_path / "fixture-profile.json"
    profile.write_text(
        json.dumps(
            {
                "software": {
                    "path": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(PREPARE, "PROFILE", profile)
    return SimpleNamespace(
        action="prepare", software_root=software, output=tmp_path / "context"
    )


def test_context_reuses_exact_wheel_without_build_or_install(fixture):
    result = PREPARE.prepare(fixture)
    assert result["software_files"] == 4
    assert (
        result["status"]
        == "TASK_CPU_BUILD_CONTEXT_VERIFIED_NOT_BUILT_OR_NATIVE_EXECUTION"
    )
    assert not result["source_checkbox_eligible"]
    assert "FROM ${BASE_IMAGE}" in (fixture.output / "Dockerfile").read_text()
    assert (
        "--no-index --no-deps --require-hashes"
        in (fixture.output / "Dockerfile").read_text()
    )
    with pytest.raises(FileExistsError):
        PREPARE.prepare(fixture)
    fixture.action = "verify"
    assert PREPARE.prepare(fixture)["context_sha256"] == result["context_sha256"]


def test_tampered_context_is_not_accepted(fixture):
    PREPARE.prepare(fixture)
    (fixture.output / "Dockerfile").write_text("wrong recipe", encoding="utf-8")
    fixture.action = "verify"
    with pytest.raises(ValueError, match="differs"):
        PREPARE.prepare(fixture)


def test_extra_context_files_cannot_be_copied_into_the_image(fixture):
    PREPARE.prepare(fixture)
    (fixture.output / "wheels/private-extra.txt").write_bytes(b"not an allowed asset")
    fixture.action = "verify"
    with pytest.raises(ValueError, match="only the declared"):
        PREPARE.prepare(fixture)


def test_context_must_not_overlap_the_original_cache(fixture):
    fixture.output = fixture.software_root / "unsafe-context"
    with pytest.raises(ValueError, match="separate"):
        PREPARE.prepare(fixture)
