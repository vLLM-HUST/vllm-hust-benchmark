"""Check an existing isolated NPU auxiliary client; plan-only by default.

Does not create environments, install packages, download/load models, allocate
devices or execute tools/tasks. Use only on the platform's vendor NPU stack.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/marble-keybert-npu-client.txt"


def requirements():
    pins = {}
    for line in PINS.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)={2,3}([A-Za-z0-9.+_-]+)", line)
        if match is None or match[1] in pins:
            raise ValueError("unique exact auxiliary package pins required")
        pins[match[1]] = match[2]
    return pins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-python", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report = {
        "source_id": "MULTIAGENTBENCH",
        "status": "PLAN_ONLY",
        "model_loaded": False,
        "inference_executed": False,
        "packages_installed": False,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    # Preserve a venv interpreter symlink: resolving it would run the base env.
    python = args.client_python.absolute()
    if not python.is_file():
        raise ValueError("existing dedicated client interpreter required")
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    check = subprocess.run(
        [str(python), "-m", "pip", "check"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if check.returncode:
        report.update(
            status="AUXILIARY_DEPENDENCIES_FAILED", pip_check_exit_code=check.returncode
        )
        print(json.dumps(report, indent=2))
        return 1
    probe = """
import importlib.metadata,json,sys
assert sys.version_info[:2]==(3,12), 'Only Python3.12 profile qualified'
pins=PINS
actual={name:importlib.metadata.version(name) for name in pins}
assert actual==pins, 'Auxiliary dependency pin mismatch'
import torch
import torch_npu
import transformers
import sentence_transformers
assert not torch.npu.is_initialized(), 'Import must not initialize device'
print(json.dumps({'versions':actual,'npu_initialized':False}))
""".replace("PINS", repr(requirements()))
    result = subprocess.run(
        [str(python), "-IB", "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        diagnostic = next(
            (
                line[:300]
                for line in reversed(result.stderr.splitlines())
                if line.startswith(
                    (
                        "ModuleNotFoundError:",
                        "ImportError:",
                        "AssertionError:",
                        "RuntimeError:",
                    )
                )
            ),
            "Inspect dedicated-client library imports",
        )
        report.update(
            status="AUXILIARY_IMPORT_OR_VERSION_CHECK_FAILED",
            import_exit_code=result.returncode,
            diagnostic=diagnostic,
        )
        print(json.dumps(report, indent=2))
        return 1
    details = json.loads(result.stdout.splitlines()[-1])
    report.update(
        status="AUXILIARY_CLIENT_IMPORTS_VERIFIED_NOT_INFERENCE",
        pip_check_exit_code=0,
        import_exit_code=0,
        **details,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
