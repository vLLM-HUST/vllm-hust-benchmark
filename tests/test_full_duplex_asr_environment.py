"""Environment-lock boundary tests; no installation, runtime or model execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duplex_asr_environment", ROOT / "scripts/prepare_full_duplex_asr_environment.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)
LOCK = ROOT / "docs/workload-data/validation/full-duplex-asr-overlay-lock-20261106.json"


def test_audited_overlay_has_complete_artifact_bindings_and_record_verification():
    lock = ENTRY.load_lock(LOCK)
    assert len(lock["private_packages"]) == 66
    assert lock["private_installed_record_hashes_verified"] == 8508
    assert (
        len(
            [
                p
                for p in lock["private_packages"]
                if p["artifact_kind"] == "source_distribution"
            ]
        )
        == 1
    )
    assert lock["fresh_environment_rebuild_verified"] is False


def test_default_plan_does_not_create_environment_or_download(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SPEC.origin),
            "--lock",
            str(LOCK),
            "--output",
            str(tmp_path / "env"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "change", ["duplicate", "core", "foreign-host", "newline", "hash", "missing-core"]
)
def test_invalid_lock_fails_before_any_install(tmp_path, change):
    lock = deepcopy(ENTRY.load_lock(LOCK))
    if change == "duplicate":
        lock["private_packages"].append(lock["private_packages"][0])
    elif change == "core":
        lock["private_packages"][0]["name"] = "torch"
    elif change == "foreign-host":
        lock["private_packages"][0]["url"] = "https://example.invalid/package.whl"
    elif change == "newline":
        lock["private_packages"][0]["url"] += "\n--unsafe"
    elif change == "hash":
        lock["private_packages"][0]["sha256"] = "not-a-hash"
    else:
        lock["protected_core_versions"].pop("numpy")
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        ENTRY.load_lock(path)


def test_wrong_platform_refuses_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(ENTRY.platform, "machine", lambda: "x86_64")
    output = tmp_path / "new-environment"
    with pytest.raises(ValueError, match="ARM64"):
        ENTRY.prepare(ENTRY.load_lock(LOCK), output)
    assert not output.exists()


@pytest.mark.parametrize("outside", [False, True])
def test_overlay_must_match_versions_and_be_installed_in_private_environment(
    tmp_path, monkeypatch, outside
):
    lock = ENTRY.load_lock(LOCK)
    env = tmp_path / "env"
    root = tmp_path / "global" if outside else env / "lib/python3.12/site-packages"
    payload = {
        row["name"]: {"version": row["version"], "location": str(root)}
        for row in lock["private_packages"]
    }
    monkeypatch.setattr(
        ENTRY.subprocess, "check_output", lambda command: json.dumps(payload).encode()
    )
    if outside:
        with pytest.raises(ValueError, match="location differs"):
            ENTRY.verify_overlay(lock, env)
    else:
        assert ENTRY.verify_overlay(lock, env) == 66


def test_existing_environment_verification_reuses_install_without_writes(
    tmp_path, monkeypatch
):
    lock = ENTRY.load_lock(LOCK)
    monkeypatch.setattr(
        ENTRY, "core_versions", lambda lock: lock["protected_core_versions"]
    )
    monkeypatch.setattr(ENTRY, "verify_overlay", lambda lock, env: 66)
    monkeypatch.setattr(
        ENTRY.subprocess,
        "check_output",
        lambda command: json.dumps(lock["protected_core_versions"]).encode(),
    )
    report = ENTRY.verify_existing(lock, tmp_path)
    assert report["downloads"] == report["model_calls"] == 0
    assert report["private_versions_and_locations_verified"] == 66
    assert list(tmp_path.iterdir()) == []
