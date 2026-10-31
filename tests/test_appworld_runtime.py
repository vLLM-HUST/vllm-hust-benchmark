"""CPU fixtures only; never install AppWorld or execute its tasks in CI."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "appworld_runtime", ROOT / "scripts/prepare_appworld_runtime.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_does_not_write_or_unpack_protected_code(tmp_path):
    output = tmp_path / "not-created"
    result = MODULE.prepare(output, "/missing/uv", "/missing/python")
    assert result["status"] == "PLAN_ONLY" and not output.exists()
    assert result["model_calls"] == 0 and result["source_checkbox_eligible"] is False
    assert not any(command[-1] == "install" for command in result["steps"])
    assert not any("download" in command for command in result["steps"])


def test_protected_unpack_is_explicit_and_no_task_download():
    steps = MODULE.commands("/uv", "/python", "/external", True)
    assert steps[2] == [str(Path("/external") / "client/bin/appworld"), "install"]
    assert "appworld==0.1.3.post1" in steps[1]
    assert "https://pypi.org/simple" in steps[1]
    assert steps[1][steps[1].index("--link-mode") + 1] == "copy"
    assert not any("download" in command for command in steps)


@pytest.mark.parametrize("version,headers", [([3, 12, 0], True), ([3, 11, 6], False)])
def test_wrong_python_or_missing_arm_headers_fail_before_install(
    monkeypatch, version, headers
):
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=json.dumps(
                {"version": version, "machine": "aarch64", "headers_present": headers}
            )
        ),
    )
    with pytest.raises(ValueError):
        MODULE.check_python_runtime("/explicit/python")


def test_system_site_inheritance_is_rejected(tmp_path):
    client = tmp_path / "client"
    client.mkdir()
    (client / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="inheritance"):
        MODULE.isolated_python(tmp_path)


def test_observed_constraints_are_explicit_and_not_author_lock(tmp_path):
    constraints = (
        ROOT / "configs/workload-native/appworld-core-client-py311-observed.txt"
    )
    versions = [
        line
        for line in constraints.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert len(versions) == 83 and "appworld==0.1.3.post1" in versions
    assert all("==" in line and "://" not in line for line in versions)
    result = MODULE.prepare(
        tmp_path / "absent", "/uv", "/python", constraints=constraints
    )
    install = result["steps"][1]
    assert install[install.index("--constraint") + 1] == str(constraints.resolve())
    assert result["source_checkbox_eligible"] is False
