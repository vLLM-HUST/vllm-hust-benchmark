"""Offline transfer fixtures are neither CIFAR data nor MLAgent task execution."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mlagent_cifar_assets", ROOT / "scripts/prepare_mlagentbench_cifar_assets.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)
RAW = b"explicit transfer fixture, not original CIFAR"


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def item(algorithm="md5"):
    return {
        "path": "cifar-10-python.tar.gz",
        "url": "https://example.invalid/original",
        "bytes": len(RAW),
        algorithm: hashlib.new(algorithm, RAW).hexdigest(),
    }


def test_default_plan_does_not_create_assets_or_execute_models(tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "absent", action="plan", parts=["software", "data"]
    )
    report = ENTRY.run(args)
    assert report["selected_bytes"] == 13995854 + 170498071
    assert report["model_calls"] == report["native_tasks_run"] == 0
    assert not report["source_checkbox_eligible"]
    assert not args.output.exists()


@pytest.mark.parametrize("algorithm", ["md5", "sha256"])
def test_whole_checksum_verification_and_final_reuse(tmp_path, algorithm):
    spec = item(algorithm)
    output = ENTRY.acquire(
        tmp_path, spec, max_bytes=len(RAW), opener=lambda *a, **k: Response(RAW)
    )
    assert output.read_bytes() == RAW
    assert ENTRY.verify(output, spec)["verified_algorithm"] == algorithm

    def forbidden(*args, **kwargs):
        raise AssertionError("valid final assets must not be redownloaded")

    assert ENTRY.acquire(tmp_path, spec, max_bytes=len(RAW), opener=forbidden) == output


def test_interrupted_original_bytes_resume_only_exact_owned_range(tmp_path):
    spec = item()
    with pytest.raises(ValueError, match="incomplete"):
        ENTRY.acquire(
            tmp_path, spec, max_bytes=len(RAW), opener=lambda *a, **k: Response(RAW[:4])
        )
    partial = tmp_path / (spec["path"] + ".partial")
    assert partial.read_bytes() == RAW[:4]

    def resumed(request, timeout):
        assert request.get_header("Range") == "bytes=4-"
        return Response(
            RAW[4:], 206, {"Content-Range": f"bytes 4-{len(RAW) - 1}/{len(RAW)}"}
        )

    assert (
        ENTRY.acquire(
            tmp_path, spec, max_bytes=len(RAW), resume=True, opener=resumed
        ).read_bytes()
        == RAW
    )


def test_owned_empty_partial_can_resume_without_overwrite(tmp_path):
    spec = item()
    with pytest.raises(ValueError, match="incomplete"):
        ENTRY.acquire(
            tmp_path, spec, max_bytes=len(RAW), opener=lambda *a, **k: Response(b"")
        )
    assert (
        ENTRY.acquire(
            tmp_path,
            spec,
            max_bytes=len(RAW),
            resume=True,
            opener=lambda *a, **k: Response(RAW),
        ).read_bytes()
        == RAW
    )


def test_wrong_resume_response_retains_partial(tmp_path):
    spec = item()
    with pytest.raises(ValueError, match="incomplete"):
        ENTRY.acquire(
            tmp_path, spec, max_bytes=len(RAW), opener=lambda *a, **k: Response(RAW[:4])
        )
    with pytest.raises(ValueError, match="exact original resume"):
        ENTRY.acquire(
            tmp_path,
            spec,
            max_bytes=len(RAW),
            resume=True,
            opener=lambda *a, **k: Response(RAW),
        )
    assert (tmp_path / (spec["path"] + ".partial")).read_bytes() == RAW[:4]


def test_wrong_official_digest_does_not_publish_final(tmp_path):
    spec = {**item(), "md5": "0" * 32}
    with pytest.raises(ValueError, match="checksum differs"):
        ENTRY.acquire(
            tmp_path, spec, max_bytes=len(RAW), opener=lambda *a, **k: Response(RAW)
        )
    assert not (tmp_path / spec["path"]).exists()
    assert (tmp_path / (spec["path"] + ".partial")).read_bytes() == RAW


def test_no_unowned_partial_adoption_or_budget_bypass(tmp_path):
    spec = item()
    with pytest.raises(ValueError, match="byte budget"):
        ENTRY.acquire(tmp_path, spec, max_bytes=0)
    (tmp_path / (spec["path"] + ".partial")).write_bytes(RAW[:4])
    with pytest.raises(ValueError, match="inspect existing"):
        ENTRY.acquire(tmp_path, spec, max_bytes=len(RAW))
    with pytest.raises(FileNotFoundError):
        ENTRY.acquire(tmp_path, spec, max_bytes=len(RAW), resume=True)


def test_registered_software_and_official_data_identity_are_not_model_weights():
    profile = json.loads(ENTRY.PROFILE.read_text("utf-8"))
    assert profile["task"] == "cifar10"
    assert profile["software"]["requires"][0] == "torch==2.3.0"
    assert profile["data"]["train_images"] == 50000
    assert profile["data"]["test_images"] == 10000
    assert profile["data"]["checksum_source"].startswith(
        "torchvision/datasets/cifar.py"
    )
    assert "model" not in profile


def test_actual_public_software_entry_evidence_stays_separate_from_native_tasks():
    report = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/mlagentbench-public-cifar-software-20261101.json"
        ).read_text("utf-8")
    )
    raw = (ROOT / report["entry"]).read_text("utf-8").encode()
    assert hashlib.sha256(raw).hexdigest() == report["entry_sha256"]
    assert report["exit_codes"] == [0, 0, 0]
    assert report["verified_original_wheel_bytes"] == 13995854
    assert report["old_wheel_unchanged"] and report["official_pypi_sha256_verified"]
    assert not report["torch_imported"] and not report["software_installed"]
    assert not report["dataset_or_native_task_verified"]
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
