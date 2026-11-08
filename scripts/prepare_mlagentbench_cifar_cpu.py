"""Emit a fresh offline-build CPU task context using an already verified wheel."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mlagentbench_cifar_assets import PROFILE, verify

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_BUILD_OR_INSTALL",
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    profile = json.loads(PROFILE.read_text("utf-8"))
    wheel = args.software_root / profile["software"]["path"]
    verify(wheel, profile["software"])
    output, software = args.output.resolve(), args.software_root.resolve()
    if (
        output.is_relative_to(ROOT)
        or output.is_relative_to(software)
        or software.is_relative_to(output)
    ):
        raise ValueError(
            "fresh external build context separate from original software required"
        )
    sources = {
        "Dockerfile": ROOT
        / "configs/workload-native/mlagentbench-cifar10-cpu.Dockerfile",
        "torchvision-requirements.txt": ROOT
        / "configs/workload-native/mlagentbench-cifar10-torchvision-requirements.txt",
        "check_mlagentbench_cifar_cpu.py": ROOT
        / "scripts/check_mlagentbench_cifar_cpu.py",
        "profile.json": PROFILE,
    }
    files = {name: path.read_text("utf-8").encode() for name, path in sources.items()}
    if args.action == "prepare":
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, raw in files.items():
            with (output / name).open("xb") as stream:
                stream.write(raw)
        (output / "wheels").mkdir()
        target = output / "wheels" / wheel.name
        with wheel.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024**2)
    for name, raw in files.items():
        target = output / name
        if (
            target.is_symlink()
            or not target.resolve(strict=True).is_relative_to(output)
            or target.read_bytes() != raw
        ):
            raise ValueError("prepared build software differs from public entry")
    verify(output / "wheels" / wheel.name, profile["software"])
    expected_files = set(files) | {"wheels/" + wheel.name}
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("build context must contain only the declared task software")
    report.update(
        status="TASK_CPU_BUILD_CONTEXT_VERIFIED_NOT_BUILT_OR_NATIVE_EXECUTION",
        software_files=len(files),
        wheel_bytes=profile["software"]["bytes"],
        context_sha256={
            name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
        },
        required_base="Explicit qualified Python3.10.22/aarch64 CPU base from task asset profile; no default shared image mutation",
        network_required_for_pinned_wheel_install=False,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
