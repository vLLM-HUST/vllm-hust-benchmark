"""Inert software archive fixtures; no original tasks, Docker or models."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "swe_source", SCRIPTS / "prepare_swebench_author_source.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, names=None, content=b"inert original software\n", kind=None):
    archive = tmp_path / "source.tar.gz"
    names = names or ["publisher-fixed/swebench/__init__.py"]
    with tarfile.open(archive, "w:gz") as handle:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            if kind is not None:
                info.type = kind
            handle.addfile(info, io.BytesIO(content))
    pins = {
        "files": [
            {
                "path": "swebench/__init__.py",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest(),
            }
        ]
    }
    return archive, pins


def test_original_software_preparation_and_identical_resume(tmp_path):
    archive, pins = fixture(tmp_path)
    output = tmp_path / "external"
    assert module.prepare(archive, output, pins) == 1
    assert module.prepare(archive, output, pins) == 1
    assert (
        output / "swebench/__init__.py"
    ).read_bytes() == b"inert original software\n"


@pytest.mark.parametrize(
    "names",
    [
        ["../swebench/__init__.py"],
        ["/publisher/swebench/__init__.py"],
        ["publisher-fixed/missing.py"],
        ["publisher-fixed/swebench/__init__.py"] * 2,
        ["publisher-fixed/swebench/__init__.py", "other-root/README.md"],
    ],
)
def test_unsafe_duplicate_missing_or_multiple_root_archives_rejected(tmp_path, names):
    archive, pins = fixture(tmp_path, names)
    with pytest.raises(ValueError):
        module.prepare(archive, tmp_path / "external", pins)


@pytest.mark.parametrize("field", ["sha256", "git_blob", "bytes"])
def test_identity_mismatch_cannot_publish_file(tmp_path, field):
    archive, pins = fixture(tmp_path)
    pins["files"][0][field] = (
        0 if field == "bytes" else "0" * len(pins["files"][0][field])
    )
    output = tmp_path / "external"
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert not (output / "swebench/__init__.py").exists()


def test_nonregular_selected_member_rejected(tmp_path):
    archive, pins = fixture(tmp_path, kind=tarfile.SYMTYPE)
    with pytest.raises(ValueError):
        module.prepare(archive, tmp_path / "external", pins)


def test_existing_different_file_never_overwritten(tmp_path):
    archive, pins = fixture(tmp_path)
    output = tmp_path / "external"
    target = output / "swebench/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing unrelated bytes")
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert target.read_bytes() == b"existing unrelated bytes"


def test_checkout_output_rejected_before_writes(tmp_path):
    archive, pins = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the checkout"):
        module.prepare(archive, module.ROOT / "never-create-swebench-source", pins)
