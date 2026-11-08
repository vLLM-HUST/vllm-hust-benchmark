"""Inspect an acquired Moshi wheel without importing it or downloading weights."""

import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import zipfile

from packaging.requirements import Requirement


def inspect(wheel, torch_version):
    wheel = Path(wheel)
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("expected exactly one wheel metadata record")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        requirements = metadata.get_all("Requires-Dist", [])
        torch_requirements = [
            Requirement(value)
            for value in requirements
            if Requirement(value).name == "torch"
        ]
        if len(torch_requirements) != 1:
            raise ValueError("expected exactly one torch requirement")
        server = archive.read("moshi/server.py").decode("utf-8")
        sync_lines = [
            number
            for number, line in enumerate(server.splitlines(), 1)
            if "torch.cuda.synchronize()" in line
        ]
    compatible = torch_requirements[0].specifier.contains(torch_version)
    return {
        "status": "STATIC_SOFTWARE_INSPECTION_NOT_NATIVE_EXECUTION",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "moshi_version": metadata["Version"],
        "torch_version": torch_version,
        "torch_requirement": str(torch_requirements[0]),
        "torch_requirement_satisfied": compatible,
        "server_cuda_synchronize_lines": sync_lines,
        "npu_runtime_qualified": False,
        "requirements": requirements,
        "next_checks": [
            "Qualify an isolated matching torch/torch-npu environment; do not replace a shared runtime.",
            "Qualify accelerator-specific warmup and original codec/model execution on NPU.",
            "Complete original duplex requests and original ASR scoring before source delivery.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--torch-version", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.wheel, args.torch_version), indent=2))


if __name__ == "__main__":
    main()
