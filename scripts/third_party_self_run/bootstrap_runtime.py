"""Restore the tested local runtime prerequisites without changing the model or results."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
profile = sys.argv[1]
if profile == "baseline":
    python = ROOT / "runtime-b0/bin/python"
    if not python.exists():
        subprocess.run(
            ["/usr/bin/python3.11", "-m", "venv", str(python.parent.parent)], check=True
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-r",
                str(ROOT / "baseline-python-requirements.lock"),
            ],
            check=True,
            env=dict(os.environ, PYTHONPATH=""),
        )
    if not Path("/usr/include/python3.11/Python.h").exists():
        subprocess.run(["dnf", "install", "-y", "python3-devel"], check=True)
    subprocess.run(
        [sys.executable, str(ROOT / "patch_baseline_runtime.py")], check=True
    )
    subprocess.run([sys.executable, str(ROOT / "patch_baseline_qwen35.py")], check=True)
    check = subprocess.run(
        [str(python), "-c", "import numba; import vllm_ascend.worker.worker"],
        capture_output=True,
        env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="2,3"),
    )
    if check.returncode:
        raise RuntimeError(
            "Baseline worker prerequisites failed: "
            + check.stderr.decode(errors="replace")[-4000:]
        )
elif profile == "hust":
    from version_gate import check

    identity = check(profiles=["B1"])
    if not identity["passed"]:
        raise RuntimeError(
            "B1 test-specific version/source admission failed before installation or patching: "
            + str(identity["arms"]["B1"]["errors"])
        )
    # The accepted runtime is recovered from its frozen source recipe. Never
    # substitute old checkouts, public wheels, or a moving main here.
