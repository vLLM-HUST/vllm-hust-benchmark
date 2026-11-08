import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "duplex_model_environment",
    ROOT / "scripts/prepare_full_duplex_model_environment.py",
)
ENTRY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ENTRY)
LOCK = (
    ROOT / "docs/workload-data/validation/full-duplex-model-runtime-lock-20261107.json"
)


def test_default_plan_does_not_access_external_archive_or_install(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_full_duplex_model_environment.py"),
            "--lock",
            str(LOCK),
            "--publisher-archive-root",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["private_packages"] == 182
    assert report["external_source_bindings"] == 2
    assert not (tmp_path / "output").exists()


def archive_fixture(tmp_path):
    path = tmp_path / "publisher.zip"
    wheel = b"isolated test fixture, not a real installed package"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fixture.whl", wheel)
        archive.writestr("installer.run", b"must not extract or execute")
    sha = hashlib.sha256(wheel).hexdigest()
    lock = {
        "packages": [
            {
                "binding": "PUBLISHER_ARCHIVE_WHEEL",
                "archive": path.name,
                "name": "fixture",
                "filename": "fixture.whl",
                "member": "fixture.whl",
                "sha256": sha,
            }
        ],
        "publisher_archives": [
            {
                "name": path.name,
                "sha256": ENTRY.digest(path),
                "members": [
                    {"path": "fixture.whl", "bytes": len(wheel), "sha256": sha}
                ],
            }
        ],
    }
    return lock, path


def test_extracts_only_named_hash_bound_wheel_not_installer(tmp_path):
    lock, _ = archive_fixture(tmp_path)
    output = tmp_path / "wheels"
    output.mkdir()
    result = ENTRY.extract_wheels(lock, tmp_path, output)
    assert list(result) == ["fixture"]
    assert sorted(p.name for p in output.iterdir()) == ["fixture.whl"]


@pytest.mark.parametrize("mismatch", ["archive", "member", "size"])
def test_rejects_changed_archive_or_member(tmp_path, mismatch):
    lock, _ = archive_fixture(tmp_path)
    if mismatch == "archive":
        lock["publisher_archives"][0]["sha256"] = "0" * 64
    elif mismatch == "member":
        lock["packages"][0]["sha256"] = "0" * 64
    else:
        lock["publisher_archives"][0]["members"][0]["bytes"] = 1
    output = tmp_path / "wheels"
    output.mkdir()
    with pytest.raises(ValueError):
        ENTRY.extract_wheels(lock, tmp_path, output)


@pytest.mark.parametrize("mismatch", [None, "location", "version", "core"])
def test_overlay_verification_distinguishes_metadata_from_real_private_install(
    tmp_path, monkeypatch, mismatch
):
    lock = ENTRY.load_lock(LOCK)
    env = tmp_path / "env"
    packages = {
        row["name"]: {"version": row["version"], "location": str(env / "lib")}
        for row in lock["packages"]
        if row["binding"] != "EDITABLE_SOURCE"
    }
    core = dict(lock["protected_core_versions"])
    first = next(iter(packages))
    if mismatch == "location":
        packages[first]["location"] = str(tmp_path / "inherited")
    elif mismatch == "version":
        packages[first]["version"] = "0.0"
    elif mismatch == "core":
        core["numpy"] = "changed"
    results = iter([json.dumps(packages).encode(), json.dumps(core).encode()])
    monkeypatch.setattr(ENTRY.subprocess, "check_output", lambda *a, **k: next(results))
    if mismatch:
        with pytest.raises(ValueError):
            ENTRY.verify_overlay(lock, env)
    else:
        assert ENTRY.verify_overlay(lock, env) == 182


def test_supplement_is_hash_bound_private_and_never_overwrites(tmp_path):
    env = tmp_path / "env"
    package = env / "lib/python3.12/site-packages/sgl_kernel_npu/activation"
    package.mkdir(parents=True)
    source = tmp_path / "published.py"
    source.write_bytes(b"# non-neural isolated fixture\n")
    lock = {"sources": {"kernel-python-supplement": {"sha256": ENTRY.digest(source)}}}
    receipt = ENTRY.bind_kernel(lock, source, env)
    assert receipt["publisher_code_modified"] is False
    assert receipt["binary_extension_modified"] is False
    assert (package / "fused_sigmoid_mul.py").read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        ENTRY.bind_kernel(lock, source, env)


def test_changed_supplement_is_rejected_before_private_environment_access(tmp_path):
    source = tmp_path / "changed.py"
    source.write_bytes(b"changed")
    lock = {"sources": {"kernel-python-supplement": {"sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="hash differs"):
        ENTRY.bind_kernel(lock, source, tmp_path / "env")


def test_preparation_requires_explicit_supplement_before_creating_output(tmp_path):
    lock = ENTRY.load_lock(LOCK)
    with pytest.raises(ValueError, match="explicit hash-pinned"):
        ENTRY.prepare(lock, tmp_path, tmp_path / "output", None)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("collision", [False, True])
def test_core_isolation_copies_payloads_not_writable_symlinks(
    tmp_path, monkeypatch, collision
):
    base = tmp_path / "base"
    base.mkdir()
    (base / "fixture_core.py").write_bytes(b"# isolated non-neural fixture\n")
    env = tmp_path / "env"
    target = env / "lib/python3.12/site-packages"
    target.mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("include-system-site-packages = true\n")

    class Distribution:
        version = "1.0"
        files = [Path("fixture_core.py"), Path("../bin/fixture-wrapper")]

        def locate_file(self, relative):
            return base / relative

    monkeypatch.setattr(ENTRY.metadata, "distribution", lambda name: Distribution())
    lock = {"protected_core_versions": {"fixture-core": "1.0"}}
    if collision:
        (target / "fixture_core.py").write_bytes(b"preserve existing")
        with pytest.raises(FileExistsError):
            ENTRY.isolate_core_payloads(lock, env)
        assert (env / "pyvenv.cfg").read_text().endswith("= true\n")
        assert (target / "fixture_core.py").read_bytes() == b"preserve existing"
    else:
        report = ENTRY.isolate_core_payloads(lock, env)
        assert report["physical_core_files_copied"] == 1
        assert report["writable_core_payload_symlinks_created"] is False
        assert not (target / "fixture_core.py").is_symlink()
        (target / "fixture_core.py").write_bytes(b"private change")
        assert (base / "fixture_core.py").read_bytes() != b"private change"
        assert (env / "pyvenv.cfg").read_text().endswith("= false\n")
