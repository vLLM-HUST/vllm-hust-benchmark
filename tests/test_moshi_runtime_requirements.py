import importlib.util
from pathlib import Path
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "moshi_preflight",
    Path(__file__).resolve().parents[1]
    / "scripts/inspect_moshi_runtime_requirements.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("version,compatible", [("2.10.0+cpu", False), ("2.9.0", True)])
def test_torch_constraint_is_not_native_qualification(tmp_path, version, compatible):
    wheel = tmp_path / "moshi.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "moshi-0.2.13.dist-info/METADATA",
            "Version: 0.2.13\nRequires-Dist: torch<2.10,>=2.2\n",
        )
        archive.writestr("moshi/server.py", "    torch.cuda.synchronize()\n")
    result = MODULE.inspect(wheel, version)
    assert result["torch_requirement_satisfied"] is compatible
    assert result["server_cuda_synchronize_lines"] == [1]
    assert result["npu_runtime_qualified"] is False
