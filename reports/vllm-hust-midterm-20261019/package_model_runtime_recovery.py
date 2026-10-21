"""Package portable run checks; retain full immutable data in release assets."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    destination = (
        Path(__file__).resolve().parent / "m01-recovery/model-runtime-20261021"
    )
    destination.mkdir(exist_ok=False)
    names = set()
    common = json.loads((root / "recovery-analysis-common-archive.json").read_text())
    names.add("recovery-analysis-common-archive.json")
    for name, metadata in common["members"].items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest() == metadata["sha256"]
        )
        names.add(name)
    runs = []
    for attempt in args.runs:
        assert re.fullmatch(r"r[0-9]+", attempt)
        run = f"m01-recovery-model-{attempt}"
        prefix = f"recovery_model_{attempt}"
        receipt = json.loads((root / run / "receipt.json").read_text())
        result = json.loads((root / run / "mmlu-pro.json").read_text())
        hardware_name = f"{prefix}-hardware.json"
        ledger_name = f"{prefix}-accounted-families.json"
        hardware = json.loads((root / hardware_name).read_text())
        ledger = json.loads((root / ledger_name).read_text())
        sampler_records = list(
            (root / run).glob("system/profiler/*/device_*/sample.json")
        )
        assert len(sampler_records) == 2
        periods = {
            json.loads(path.read_text())["profiling_period"] for path in sampler_records
        }
        assert len(periods) == 1
        sampling_seconds = periods.pop()
        names.update(str(path.relative_to(root)) for path in sampler_records)
        assert receipt["status"] == "measured" and receipt["all_devices_released"]
        assert (
            receipt["bench_exit_code"] == receipt["hardware_collector_exit_code"] == 0
        )
        assert result["completed"] == 200 and result["failed"] == 0
        assert (
            ledger["model_effective_flops"] is None and ledger["m01_attainment"] is None
        )
        names.update(
            f"{run}/{name}"
            for name in [
                "receipt.json",
                "mmlu-pro.json",
                "launch-command.json",
                "compiler-cache-initialization.json",
                "npu-before.txt",
                "npu-after.txt",
                "system/receipt.json",
                "system-ready.json",
                "kvmat-import.json",
                "service.log",
            ]
        )
        names.update(
            str(path.relative_to(root))
            for path in (root / run).glob("microflow-*.jsonl")
        )
        for pattern in [
            f"{prefix}-*.json",
            f"*{prefix}*.py",
            f"vocab-projection-summary-{prefix}.json",
        ]:
            names.update(path.name for path in root.glob(pattern))
        manifest_name = f"completed-model-{attempt}-archive.json"
        names.add(manifest_name)
        manifest = json.loads((root / manifest_name).read_text())
        runs.append(
            {
                "run": run,
                "hardware_sampling_seconds": sampling_seconds,
                "completed": result["completed"],
                "failed": result["failed"],
                "output_tokens": result["total_output_tokens"],
                "client_duration_seconds": result["duration"],
                "instrumented_output_tokens_per_second": result["output_throughput"],
                "hardware": hardware_name,
                "hardware_status": hardware["status"],
                "operation_ledger": ledger_name,
                "basic_operation_subtotal": ledger[
                    "accounted_basic_operation_subtotal"
                ],
                "raw_archive": {
                    "url": f"https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-{attempt}/{manifest['archive']}",
                    "bytes": manifest["bytes"],
                    "sha256": manifest["sha256"],
                    "member_manifest": manifest_name,
                },
            }
        )
    for name in [
        "model-recovery-archive-r5.json",
        "persistent-triton-recovery-r2.json",
        "mamba-import-recovery-r3.json",
        "model-recovery-preflight-r5.xml",
        "recovery-sampling-r6-tests.xml",
        "test_gdn_recovery_evidence_r5.py",
        "test_recovery_sampling_r6.py",
        "prepare_model_recovery_r6.py",
    ]:
        names.add(name)
    for directory in ["state-commit-recovery-r5", "gdn-recovery-probe-r5"]:
        names.update(
            str(path.relative_to(root)) for path in (root / directory).rglob("*.json")
        )
    members = []
    archive_path = destination / "evidence.tar.xz"
    with tarfile.open(archive_path, "w:xz") as archive:
        for name in sorted(names):
            path = root / name
            assert path.resolve().is_relative_to(root)
            assert path.is_file() and not path.is_symlink()
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
            members.append(
                {
                    "name": name,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    packed = archive_path.read_bytes()
    index = {
        "schema": "m01-model-runtime-recovery/v1",
        "archive": archive_path.name,
        "archive_bytes": len(packed),
        "archive_sha256": hashlib.sha256(packed).hexdigest(),
        "members": members,
        "runs": runs,
        "compiler_distribution": "triton-ascend 3.6.0+dev20261007182438",
        "runtime_archive_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r5/m01-model-recovery-r5.tar.gz",
        "runtime_archive_manifest": "model-recovery-archive-r5.json",
        "full_model_inference_completed": True,
        "original_runtime_binary_identity_proven": False,
        "formal_b1_admission": False,
        "model_effective_flops": None,
        "m01_attainment": None,
    }
    (destination / "archive.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps({"members": len(members), "bytes": len(packed), "runs": len(runs)})
    )


if __name__ == "__main__":
    main()
