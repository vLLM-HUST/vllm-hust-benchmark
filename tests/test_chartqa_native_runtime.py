"""CPU setup safety fixtures only; no compilation or native benchmark calls."""

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "chartqa_runtime", ROOT / "scripts/prepare_chartqa_native_runtime.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_pins_include_actual_original_submodule_versions():
    assert set(runtime.PINS) == {"ffmpeg", "decord", "dlpack", "dmlc-core"}
    for repository, revision, tree in runtime.PINS.values():
        assert "/" in repository
        assert re.fullmatch("[0-9a-f]{40}", revision)
        assert re.fullmatch("[0-9a-f]{40}", tree)


def test_default_plan_makes_no_output(tmp_path):
    target = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_chartqa_native_runtime.py"),
            "--output",
            str(target),
            "--python",
            sys.executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY_NO_BUILD_OR_MODEL_CALL"
    assert not target.exists()


def test_existing_directory_is_not_overwritten(tmp_path):
    with pytest.raises(ValueError, match="Fresh"):
        runtime.run(tmp_path, Path(sys.executable), 4)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("jobs", [0, -1, 17])
def test_no_unbounded_build_parallelism(tmp_path, jobs):
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="CPU build jobs"):
        runtime.run(target, Path(sys.executable), jobs)
    assert not target.exists()


@pytest.fixture(scope="module")
def real_client_venv(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("chartqa-client-entry") / "client"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = prefix / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    return prefix, python


def test_real_venv_entry_is_preserved_not_resolved_to_base(real_client_venv):
    prefix, python = real_client_venv
    selected, actual_prefix = runtime.qualify_client_python(python)
    assert selected == python.absolute()
    assert actual_prefix == prefix.resolve()
    if python.is_symlink():
        assert selected != python.resolve()


def test_base_python_rejected_before_output_or_install(tmp_path):
    target = tmp_path / "must-not-exist"
    base_python = Path(getattr(sys, "_base_executable", sys.executable))
    with pytest.raises(ValueError, match="system Python installation is forbidden"):
        runtime.run(target, base_python, 4)
    assert not target.exists()


def test_inherited_system_packages_rejected(real_client_venv):
    prefix, python = real_client_venv
    cfg = prefix / "pyvenv.cfg"
    original = cfg.read_text(encoding="utf-8")
    try:
        cfg.write_text(
            original.replace(
                "include-system-site-packages = false",
                "include-system-site-packages = true",
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must not inherit"):
            runtime.qualify_client_python(python)
    finally:
        cfg.write_text(original, encoding="utf-8")


def test_client_probe_ignores_inherited_python_and_pip_redirects(
    real_client_venv, monkeypatch
):
    prefix, python = real_client_venv
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_CONFIG_FILE",
    ):
        monkeypatch.setenv(key, "must-not-inherit")
    selected, actual_prefix = runtime.qualify_client_python(python)
    assert selected == python.absolute() and actual_prefix == prefix.resolve()
    env = runtime.client_environment()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONHOME" not in env and "PIP_TARGET" not in env
