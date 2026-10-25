"""Prepare/verify the pinned Python3.12 HTTP client, never a model deployment.

Vendor CPU torch software is selectively linked, not whole system site-packages.
These symbolic links are not OS-enforced read-only mounts. Do not upgrade or
uninstall the vendor torch distribution through this client.
Inference remains on the separately deployed server; no model weights installed.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "configs/workload-native/nq-open-lmeval-client-constraints.txt"
VENDOR_NAMES = ("torch", "functorch", "torchgen", "torch-2.10.0+cpu.dist-info")


def plan(client, vendor_site, *, verify_only=False):
    requested = Path(client).absolute()
    if requested.is_symlink():
        raise ValueError("separate private client and vendor package roots required")
    client, vendor_site = requested.resolve(), Path(vendor_site).resolve()
    if (
        client.is_symlink()
        or client.is_relative_to(vendor_site)
        or vendor_site.is_relative_to(client)
    ):
        raise ValueError("separate private client and vendor package roots required")
    if client.exists() != verify_only:
        raise ValueError(
            "fresh client required for install; existing client required for verify-only"
        )
    if any(not (vendor_site / name).is_dir() for name in VENDOR_NAMES):
        raise ValueError("explicit vendor torch2.10.0+cpu software payloads required")
    return client, vendor_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--vendor-site", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise ValueError("this tested vendor-runtime profile requires Linux Python3.12")
    client, origin = plan(args.client, args.vendor_site, verify_only=args.verify_only)
    site = client / "lib/python3.12/site-packages"
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    if not args.verify_only:
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(client)
        for name in VENDOR_NAMES:
            (site / name).symlink_to(origin / name, target_is_directory=True)
        env["PIP_CACHE_DIR"] = str(client / "pip-cache")
        subprocess.run(
            [
                str(client / "bin/python"),
                "-I",
                "-m",
                "pip",
                "install",
                "-r",
                str(REQUIREMENTS),
            ],
            env=env,
            check=True,
        )
    for name in VENDOR_NAMES:
        if not (site / name).is_symlink() or (site / name).resolve() != origin / name:
            raise ValueError("client vendor binding differs from explicit runtime")
    check = subprocess.run(
        [str(client / "bin/python"), "-I", "-m", "pip", "check"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    print(
        json.dumps(
            {
                "status": "ISOLATED_HTTP_CLIENT_DEPENDENCIES_VERIFIED_NOT_NATIVE_EXECUTION",
                "pip_check": check.stdout.strip(),
                "model_calls": 0,
                "source_checkbox_eligible": False,
                "existing_vendor_or_model_service_modified": False,
                "verify_only": args.verify_only,
            }
        )
    )


if __name__ == "__main__":
    main()
