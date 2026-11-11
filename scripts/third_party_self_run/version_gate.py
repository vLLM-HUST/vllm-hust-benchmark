"""Check the named test's runtime before bootstrap, patching, or NPU use."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROBE = """import importlib.metadata as m, importlib.util as u, json, sys
packages = {}
for name in ['vllm', 'vllm-ascend', 'torch', 'torch-npu', 'triton-ascend']:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
origins = {}
for name in ['vllm', 'vllm_ascend']:
    spec = u.find_spec(name)
    origins[name] = spec.origin if spec else None
print(json.dumps({'python':sys.executable, 'packages':packages, 'origins':origins}))
"""


def compare_packages(observed, expected):
    errors = []
    for name, allowed in expected.items():
        if observed.get(name) not in allowed:
            errors.append(
                {"component": name, "expected": allowed, "observed": observed.get(name)}
            )
    return errors


def check_reconstruction(profile, observed):
    """Admit a new SDK reconstruction without relabelling it historical B1."""
    from rebuild_runtime import SDK_SHA, digest

    errors = []
    path = Path(
        profile.get("reconstruction_manifest", "missing-reconstruction-manifest")
    )
    if not path.is_file() or digest(path) != profile.get(
        "reconstruction_manifest_sha256"
    ):
        return [
            {
                "component": "reconstruction_manifest",
                "reason": "Missing or changed build inventory",
            }
        ]
    manifest = json.loads(path.read_text())
    sdk = Path(manifest["sdk"])
    if (
        manifest.get("sdk_sha256") != SDK_SHA
        or not sdk.is_file()
        or digest(sdk) != SDK_SHA
    ):
        errors.append(
            {
                "component": "reconstruction_archive",
                "reason": "SDK does not match the frozen donor digest",
            }
        )
    site = Path(manifest["site_packages"]).resolve()
    for package in ["vllm", "vllm_ascend"]:
        origin = observed.get("origins", {}).get(package)
        if origin is None or Path(origin).resolve() != site / package / "__init__.py":
            errors.append(
                {"component": "import_origin", "package": package, "observed": origin}
            )
    files = manifest.get("files", {})
    if not {"vllm/model_executor/models/qwen3_5.py", "vllm_ascend/ops/gdn.py"} <= set(
        files
    ):
        errors.append(
            {
                "component": "reconstruction_inventory",
                "reason": "Missing protected model sources",
            }
        )
    for name, expected in files.items():
        source = (site / name).resolve()
        if (
            not source.is_relative_to(site)
            or not source.is_file()
            or digest(source) != expected
        ):
            errors.append(
                {"component": "reconstructed_source_file", "path": str(source)}
            )
    return errors


def check(root=ROOT, profiles=None):
    authority = json.loads((root / "version-authority/tested-runtime.json").read_text())
    profile_path = root / "runtime-profile.json"
    settings = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    profiles = profiles or ["B0", "B1"]
    result = {
        "schema": "vllm-hust-test-version-check/v1",
        "authority": authority,
        "authority_sha256": hashlib.sha256(
            (root / "version-authority/tested-runtime.json").read_bytes()
        ).hexdigest(),
        "model_started": False,
        "patches_applied": False,
        "arms": {},
        "passed": True,
        "scope": "Package/source admission only; does not certify actual optimized execution or formal results.",
    }
    for arm in profiles:
        spec = next(a for a in authority["arms"] if a["role"] == arm)
        profile = settings.get(arm, {})
        python = Path(
            profile.get(
                "python",
                root / ("runtime-b0" if arm == "B0" else "runtime-b1") / "bin/python",
            )
        )
        expected = (
            {"vllm": ["0.18.0"], "vllm-ascend": ["0.18.0"]}
            if arm == "B0"
            else {
                "vllm": [
                    "0.28.1.post1.dev143+gf18cf803c",
                    "0.28.1.post1.dev143+gf18cf803c.empty",
                ],
                "vllm-ascend": [
                    "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909"
                ],
                "torch": ["2.13.0+cpu"],
                "torch-npu": ["2.13.0rc1", "2.13.0.rc1"],
            }
        )
        reconstructed = arm == "B1" and profile.get("identity_mode") == "reconstructed"
        if reconstructed:
            expected["vllm-ascend"].append(
                "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27"
            )
        errors = []
        observed = {}
        try:
            probe = subprocess.run(
                [str(python), "-c", PROBE],
                capture_output=True,
                text=True,
                timeout=30,
                env=dict(os.environ, PYTHONPATH=profile.get("pythonpath", "")),
            )
            if probe.returncode:
                raise RuntimeError(probe.stderr[-2000:])
            observed = json.loads(probe.stdout)
            errors.extend(compare_packages(observed["packages"], expected))
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            errors.append({"component": "runtime_probe", "error": str(error)})
        # A same-version upstream wheel is not the tested optimized implementation.
        if reconstructed:
            errors.extend(check_reconstruction(profile, observed))
        elif arm == "B1":
            capsule = Path(
                profile.get(
                    "source_capsule",
                    root / "full-answer-frozen-sources-20261030-r4.tar.gz",
                )
            )
            expected_hash = authority["private_originals"]["source_capsule_sha256"]
            actual = None
            if capsule.is_file():
                with capsule.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected_hash:
                errors.append(
                    {
                        "component": "tested_implementation_capsule",
                        "path": str(capsule),
                        "expected_sha256": expected_hash,
                        "observed_sha256": actual,
                        "reason": "B1 requires its tested worker/MOD/source combination; package labels or moving main are insufficient",
                    }
                )
            # This also prevents an unchecked source tree being injected through PYTHONPATH.
            manifest_path = Path(
                profile.get(
                    "installed_source_manifest",
                    root / "installed-b1-source-manifest.json",
                )
            )
            if not manifest_path.is_file():
                errors.append(
                    {
                        "component": "installed_source_manifest",
                        "reason": "No hash-bound installed implementation manifest; recover the frozen source capsule first",
                    }
                )
            else:
                manifest_bytes = manifest_path.read_bytes()
                if (
                    hashlib.sha256(manifest_bytes).hexdigest()
                    != spec["source_manifest_sha256"]
                ):
                    errors.append(
                        {
                            "component": "installed_source_manifest",
                            "reason": "Manifest does not match the tested source receipt",
                        }
                    )
                else:
                    # Recover original filename -> SHA mapping without accepting a newly invented manifest.
                    manifest = json.loads(manifest_bytes)
                    files = manifest.get("files", {})
                    if not isinstance(files, dict) or not files:
                        errors.append(
                            {
                                "component": "installed_source_manifest",
                                "reason": "Original manifest schema needs an explicit supported reader",
                            }
                        )
                    else:
                        for path, expected_hash in files.items():
                            installed = Path(profile.get("source_root", root)) / path
                            if (
                                not installed.is_file()
                                or hashlib.sha256(installed.read_bytes()).hexdigest()
                                != expected_hash
                            ):
                                errors.append(
                                    {
                                        "component": "installed_source_file",
                                        "path": str(installed),
                                    }
                                )
        result["arms"][arm] = {
            "identity_mode": "reconstructed" if reconstructed else "reference",
            "historical_implementation_reproduced": False if reconstructed else None,
            "qualification": profile.get("qualification"),
            "expected": expected,
            "observed": observed,
            "errors": errors,
            "passed": not errors,
        }
        result["passed"] = result["passed"] and not errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    body = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(body)
    print(body)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
