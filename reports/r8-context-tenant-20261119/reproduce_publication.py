"""Project the sealed r8 pair into public scalar, window and tenant evidence.

Pass the already SHA256-verified raw ZIP and a new output directory. This is
CPU reconstruction of existing measurements and never starts model inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

RAW_SHA256 = "650bcfc53532bd29b6a5c38a2549a2a7057f1c7df3b0a62cce98f64a5e4405fe"  # pragma: allowlist secret -- fixed public archive SHA256
RAW_BYTES = 54075318
RAW_URL = "https://github.com/vLLM-HUST/vllm-hust-engine/releases/download/performance-exploration-20261115-r1/r8-context-tenant-pair-20261119-r6-raw.zip"
SUMMARY_SHA256 = "7629035f84441dde80293fda25b939072fc052914389e3ccc364bf4620af0308"  # pragma: allowlist secret -- fixed public summary SHA256
PHASES = ("4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def project(pair, *, chart_sha256, chart_bytes):
    contract = pair["contract"]
    if (
        pair["schema"] != "vllm-hust-paired-context-tenant-supplement/v1"
        or pair["execution_complete"] is not True
        or [row["phase"] for row in pair["comparison"]] != list(PHASES)
        or contract["output_tokens"] != 256
        or contract["ignore_eos"] is not True
        or contract["temperature"] != 0
        or contract["seed"] != 17
        or contract["task_count"] != 5
        or contract["automatic_retries"] != 0
        or contract["quality_measured"] is not False
    ):
        raise ValueError("Frozen r8 context/tenant source scope differs")
    for row in pair["comparison"]:
        for role in ("B0", "B1"):
            arm = row[role]
            metrics = arm["metrics"]
            if (
                arm["status"] != "completed"
                or metrics["failed"] != 0
                or metrics["successful"] != metrics["requests"]
                or metrics["output_tokens"] != 256 * metrics["successful"]
                or len(arm["windows"]) != 5
                or abs(
                    metrics["output_tokens"] / metrics["duration_seconds"]
                    - metrics["output_tokens_per_second"]
                )
                > 1e-9
            ):
                raise ValueError(
                    "Incomplete or inconsistent paired context measurements"
                )
            if (
                sum(tenant["requests"] for tenant in arm["per_tenant"].values())
                != metrics["requests"]
            ):
                raise ValueError("Per-tenant counts do not conserve formal requests")
        ratio = (
            row["B1"]["metrics"]["output_tokens_per_second"]
            / row["B0"]["metrics"]["output_tokens_per_second"]
        )
        if abs(ratio - row["throughput_ratio"]) > 1e-12:
            raise ValueError(
                "Paired throughput ratio does not match raw source metrics"
            )
    return dict(
        schema="r8-context-tenant-publication/v1",
        engine_release="full-answer-b1-20261117-r8",
        baseline_id="original018-cann8.5.1",
        report_date="2026-11-18",
        execution_complete=True,
        model="Qwen3.5-35B-A3B",
        precision="BF16",
        tensor_parallel_size=2,
        chip_model="Ascend 910B2",
        physical_cards=[1, 6],
        logical_devices=[0, 1],
        source_summary_sha256=SUMMARY_SHA256,
        arm_seals=pair["arm_seals"],
        raw_asset=dict(
            url=RAW_URL,
            sha256=RAW_SHA256,
            bytes=RAW_BYTES,
            asset_id=764965300,
            redownload_verified=True,
        ),
        protocol=contract,
        comparisons=pair["comparison"],
        chart=dict(
            path="data/r8-context-tenant-20261119/paired-curves.svg",
            sha256=chart_sha256,
            bytes=chart_bytes,
        ),
        workload_label_zh="SWE / Terminal 公开真实历史请求回放",
        workload_label_en="Published real SWE / Terminal history request replay",
        quality_measured=False,
        effective_flops_certified=False,
        efficiency_percent=None,
        original_five_profile_inference_rerun=False,
        limits=[
            "Five distinct histories; repetitions and sixty-second windows are not independent task samples.",
            "The formal submission window is300 seconds and all requests drain afterward. Each window reports its own drain.",
            "Four logical tenant queues share one engine and device pair; this does not measure security isolation.",
            "Utilization is a time-weighted device sample estimate, separate from effective compute.",
            "Preserve the original qualitative context/tenant requirements; no new numerical pass threshold is imposed.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.read_bytes()
    if len(raw) != RAW_BYTES or digest(raw) != RAW_SHA256:
        raise ValueError("Raw context/tenant archive identity differs")
    with zipfile.ZipFile(args.raw) as archive:
        manifest = json.loads(archive.read("RAW-SHA256.json"))
        if set(archive.namelist()) != set(manifest) | {"RAW-SHA256.json", "README.txt"}:
            raise ValueError("Raw archive inventory differs")
        for name, expected in manifest.items():
            if digest(archive.read(name)) != expected:
                raise ValueError("Raw archive member changed: " + name)
        for role in ("B0", "B1"):
            prefix = "run/native/" + role + "/"
            seal = json.loads(archive.read(prefix + "checksums.json"))
            actual = {
                name[len(prefix) :]: value
                for name, value in manifest.items()
                if name.startswith(prefix) and name != prefix + "checksums.json"
            }
            if seal != actual:
                raise ValueError("Complete raw arm seal differs")
        source = archive.read("run/native/paired-summary.json")
        if digest(source) != SUMMARY_SHA256:
            raise ValueError("Paired summary source changed")
        chart = archive.read("run/native/paired-report/paired-curves.svg")
        result = project(
            json.loads(source), chart_sha256=digest(chart), chart_bytes=len(chart)
        )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary-public-v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (args.out / "paired-curves.svg").write_bytes(chart)
    print(
        json.dumps(
            dict(
                execution_complete=True,
                source_summary_sha256=SUMMARY_SHA256,
                phases=len(result["comparisons"]),
                new_inference_executed=False,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
