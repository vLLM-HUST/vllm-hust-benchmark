"""No real environment installation or model execution in these fixtures."""

import importlib.util
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "repobench_runtime", ROOT / "scripts/prepare_repobench_runtime.py"
)
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def test_explicit_inheritance_and_install_target(tmp_path):
    environment = tmp_path / "venv"
    create, install = RUNTIME.commands(
        Path("/tools/uv"), Path("/platform/python"), environment
    )
    assert "--system-site-packages" in create
    assert install[install.index("--python") + 1] == str(environment / "bin/python")
    assert not any(
        s.startswith(("torch==", "torch-npu==", "transformers==", "difflib"))
        for s in install
    )
    assert all("==" in spec for spec in RUNTIME.DEPENDENCIES)
    assert all(url in install for url in RUNTIME.GRAMMAR_SOURCES.values())
    assert all(
        "/archive/" in url and url.endswith(".zip")
        for url in RUNTIME.GRAMMAR_SOURCES.values()
    )


@pytest.mark.parametrize("inherit", ["false", "true"])
def test_qualified_profile_rejects_silent_inheritance_change(tmp_path, inherit):
    (tmp_path / "pyvenv.cfg").write_text(f"include-system-site-packages = {inherit}\n")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").touch()
    if inherit == "false":
        with pytest.raises(ValueError, match="explicit platform site inheritance"):
            RUNTIME.qualified_python(tmp_path)
    else:
        assert RUNTIME.qualified_python(tmp_path) == tmp_path / "bin/python"


def test_fixed_grammar_archive_reuse_and_tamper_rejection(tmp_path, monkeypatch):
    payload = b"original source archive fixture"
    path = tmp_path / "tree-sitter-python.zip"
    path.write_bytes(payload)
    monkeypatch.setattr(
        RUNTIME,
        "GRAMMAR_ARCHIVES",
        {
            "tree-sitter-python": (
                "fixed-revision",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        },
    )
    before = path.stat().st_mtime_ns
    RUNTIME.prepare_grammar_sources(tmp_path)
    assert path.stat().st_mtime_ns == before
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        RUNTIME.prepare_grammar_sources(tmp_path)
    _, install = RUNTIME.commands(
        Path("/uv"), Path("/python"), tmp_path / "venv", tmp_path
    )
    assert "tree-sitter-python @ " + path.as_uri() in install
