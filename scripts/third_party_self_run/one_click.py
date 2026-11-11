#!/usr/bin/env python3
"""Freeze inputs, execute native inference, and generate an auditable report."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import statistics
import fcntl
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def review(out):
    if (out / "paired-summary.json").exists():
        manifest = json.loads((out / "checksums.json").read_text())
        actual_files = {
            p.relative_to(out).as_posix()
            for p in out.rglob("*")
            if p.is_file()
            and not any(s in p.parts for s in ["cache", "__pycache__"])
            and p != out / "checksums.json"
        }
        if actual_files != set(manifest):
            raise ValueError("Paired report inventory changed")
        for name, expected in manifest.items():
            path = (out / name).resolve()
            if not path.is_relative_to(out.resolve()) or sha(path) != expected:
                raise ValueError(f"Paired report file changed: {name}")
        for arm in ["B0", "B1"]:
            if (out / arm / "results").exists():
                review(out / arm / "results")
            elif not (out / f"{arm}-bootstrap-error.json").is_file():
                raise ValueError(f"Missing source result or bootstrap error: {arm}")
        combined = json.loads((out / "paired-summary.json").read_text())
        for arm in ["B0", "B1"]:
            source = out / arm / "results/diagnostic-summary.json"
            if source.exists() and combined["data"][arm] != json.loads(
                source.read_text()
            ):
                raise ValueError(f"Paired source summary differs: {arm}")

        def percentage(a, b, reduction=False):
            if a is None or b is None or a <= 0:
                return None
            return 100 * (1 - b / a) if reduction else 100 * (b / a - 1)

        for group, expected in combined["paired"].items():
            arms = {
                arm: combined["data"][arm]["groups"].get(group, {})
                for arm in ["B0", "B1"]
            }
            a, b = arms["B0"], arms["B1"]
            actual = {
                "arms": arms,
                "throughput_gain_pct": percentage(
                    a.get("output_tps"), b.get("output_tps")
                ),
            }
            for metric in ["ttft_ms", "tpot_ms"]:
                actual[metric + "_improvement_pct"] = percentage(
                    a.get(metric, {}).get("mean"), b.get(metric, {}).get("mean"), True
                )
            if a.get("output_tps") and b.get("output_tps"):
                actual["equal_rate_card_time_reduction_pct"] = 100 * (
                    1 - a["output_tps"] / b["output_tps"]
                )
            if actual != expected:
                raise ValueError(f"Paired metrics do not recompute: {group}")
        if combined["formal_complete"] is not False or any(
            v["status"] != "unassessable" for v in combined["formal_targets"].values()
        ):
            raise ValueError(
                "Diagnostic archive cannot certify the unexecuted formal suite"
            )
        print("PAIRED REVIEW OK: source reports and combined inventory verified")
        return 0
    summary = json.loads((out / "summary.json").read_text())
    plan_path = out.parent / "execution-plan.json"
    if sha(plan_path) != summary["plan_sha256"]:
        raise ValueError("Frozen plan hash changed")
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["files"].items():
        p = (plan_path.parent / name).resolve()
        if not p.is_relative_to(plan_path.parent.resolve()) or sha(p) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    manifest = json.loads((out / "checksums.json").read_text())
    for name, expected in manifest.items():
        path = (out / name).resolve()
        if (
            not path.is_relative_to(out.resolve())
            or not path.is_file()
            or sha(path) != expected
        ):
            raise ValueError(f"Output changed: {name}")
    actual_files = {
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.name != "checksums.json" and "cache" not in p.parts
    }
    if actual_files != set(manifest):
        raise ValueError("Output file inventory changed")
    data = json.loads((out / "diagnostic-summary.json").read_text())
    path = out / "native/requests.jsonl"
    records = (
        [json.loads(s) for s in path.read_text().splitlines() if s]
        if path.exists()
        else []
    )
    successful = [r for r in records if r["status"] == "success"]
    receipt_path = out / "native/receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["config_sha256"] != sha(out.parent / "diagnostic-config.json")
            or receipt["contract_sha256"]
            != sha(out.parent / "diagnostic-contract.json")
            or receipt["requests_completed"] != len(successful)
        ):
            raise ValueError("Native receipt identity or counts changed")
    if data["attempted"] != len(records) or data["completed"] != len(successful):
        raise ValueError("Request counts do not recompute")
    for group, expected in data["groups"].items():
        subset = [r for r in successful if r["group"] == group]
        actual = {
            "completed": len(subset),
            "output_tokens": sum(r["output_tokens"] for r in subset),
            "generate_wall_s": sum(r["elapsed_s"] for r in subset),
            "length_stops": sum(r["finish_reason"] == "length" for r in subset),
        }
        if "output_tps" in expected:
            seconds = actual["generate_wall_s"]
            actual["output_tps"] = (
                actual["output_tokens"] / seconds if seconds else None
            )
            for key in ["ttft_ms", "tpot_ms"]:
                samples = sorted(r[key] for r in subset if r.get(key) is not None)
                quantiles = {}
                for label, q in [("p95", 0.95), ("p99", 0.99)]:
                    x = (len(samples) - 1) * q
                    quantiles[label] = (
                        (
                            samples[math.floor(x)]
                            + (samples[math.ceil(x)] - samples[math.floor(x)])
                            * (x - math.floor(x))
                        )
                        if samples
                        else None
                    )
                actual[key] = {
                    "samples": len(samples),
                    "mean": statistics.mean(samples) if samples else None,
                    **quantiles,
                }
        if actual != expected:
            raise ValueError(f"Group metrics do not recompute: {group}")
    print(
        f"REVIEW OK: {len(manifest)} files verified; {len(successful)}/{data['planned']} native requests completed"
    )
    print(
        "Formal target states:",
        json.loads((out / "summary.json").read_text())["targets"],
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="New campaign directory (default: timestamped runs/ subdirectory)",
    )
    parser.add_argument(
        "--review",
        type=Path,
        help="Verify and recompute an existing report without using NPU",
    )
    parser.add_argument(
        "--profile", choices=["stock", "baseline", "hust", "paired"], default="paired"
    )
    parser.add_argument(
        "--retry",
        type=Path,
        help="Retry a prepared frozen campaign into a new results directory",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check the test-specific version/source lock without loading a model",
    )
    parser.add_argument(
        "--rebuild-runtime",
        type=Path,
        help="Explicitly reconstruct the matching SDK into a new directory before version admission",
    )
    args = parser.parse_args()
    if args.review:
        return review(args.review.resolve())
    if args.rebuild_runtime:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "rebuild_runtime.py"),
                "--out",
                str(args.rebuild_runtime),
            ],
            check=True,
        )
    from version_gate import check

    if args.verify_only:
        identity = check()
        print(json.dumps(identity, ensure_ascii=False, indent=2))
        return 0 if identity["passed"] else 2
    if args.retry:
        work = args.retry.resolve()
        plan = json.loads((work / "execution-plan.json").read_text())
        config = json.loads((work / "diagnostic-config.json").read_text())
        if config.get("profile") == "hust" and not check(profiles=["B1"])["passed"]:
            raise ValueError(
                "The frozen HUST campaign is not the test-qualified B1 version; use --review for its historical records"
            )
        for name, expected in plan["files"].items():
            if sha(work / name) != expected:
                raise ValueError(f"Cannot retry changed frozen input: {name}")
        # Earlier campaigns fail immediately on a busy loading lock. Queue at the
        # entry point so their immutable adapter source need not be rewritten.
        deadline = time.monotonic() + 1800
        while True:
            try:
                with Path(
                    "/tmp/gennova-device-leases/container-model-loading.lock"
                ).open("a+") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Resource queue exceeded 1800 seconds")
                time.sleep(5)
        name = "retry-" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y%m%d-%H%M%S"
        )
        env = dict(
            os.environ,
            VLLM_HUST_KIT=str(ROOT),
            VLLM_HUST_REPORT_PYTHON=str(ROOT / ".report-venv/bin/python"),
        )
        result = subprocess.run(
            [sys.executable, str(work / "run_all.py"), "--out", name], cwd=work, env=env
        )
        if result.returncode:
            return result.returncode
        review(work / name)
        data = json.loads((work / name / "diagnostic-summary.json").read_text())
        return (
            0
            if data["completed"] == data["planned"] and data["receipt_exported"]
            else 1
        )
    report_python = ROOT / ".report-venv/bin/python"
    clean = dict(os.environ, PYTHONPATH="")
    if not report_python.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", str(report_python.parent.parent)],
            check=True,
            env=clean,
        )
    version = subprocess.run(
        [
            str(report_python),
            "-c",
            'import matplotlib; assert matplotlib.__version__ == "3.10.8"',
        ],
        capture_output=True,
        env=clean,
    )
    if version.returncode:
        subprocess.run(
            [
                str(report_python),
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements-report.txt"),
            ],
            check=True,
            env=clean,
        )
    if args.profile in ["baseline", "hust"]:
        identity = check(profiles=["B0" if args.profile == "baseline" else "B1"])
        if not identity["passed"]:
            print(json.dumps(identity, ensure_ascii=False, indent=2))
            return 2
        subprocess.run(
            [sys.executable, str(ROOT / "bootstrap_runtime.py"), args.profile],
            check=True,
        )
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    work = args.out.resolve() if args.out else ROOT / "runs" / now
    work.mkdir(parents=True, exist_ok=False)
    if args.profile == "paired":
        identity = check()
        (work / "version-check.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n"
        )
        if not identity["passed"]:
            for arm in ["B0", "B1"]:
                (work / f"{arm}-bootstrap-error.json").write_text(
                    json.dumps(
                        {
                            "stage": "version_preflight",
                            "native_requests": 0,
                            "reason": "Test-specific version/source check failed before bootstrap or NPU use",
                            "details": "version-check.json",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            (work / "execution-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "vllm-hust-version-rejected-execution/v1",
                        "native_requests_per_arm": 200,
                        "model_started": False,
                        "authority_sha256": identity["authority_sha256"],
                    },
                    indent=2,
                )
                + "\n"
            )
            subprocess.run(
                [str(report_python), str(ROOT / "paired_report.py"), str(work)],
                check=True,
            )
            review(work)
            print("VERSION CHECK FAILED: see", work / "version-check.json")
            return 2
        manifest = {
            "schema": "vllm-hust-paired-execution/v1",
            "created_at": now,
            "order": ["B0", "B1"],
            "profiles": {"B0": "baseline", "B1": "hust"},
            "entry_sha256": sha(Path(__file__)),
            "reporter_sha256": sha(ROOT / "paired_report.py"),
            "scope": "real two-card paired diagnostic plus four strict formal dispositions",
            "official_thresholds": {
                "compute_efficiency_pct": 50,
                "throughput_gain_pct": 10,
                "ttft_improvement_pct": 5,
                "tpot_improvement_pct": 5,
                "cost_reduction_pct": 10,
            },
            "quality_and_five_dataset_acceptance": "not replaced by these diagnostic samples",
        }
        manifest["failure_policy"] = (
            "Stop subsequent native profiles after the first execution/bootstrap failure; render all four dispositions."
        )
        (work / "execution-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        codes = {}
        for arm, profile in manifest["profiles"].items():
            if any(code != 0 for code in codes.values()):
                codes[arm] = None
                (work / f"{arm}-bootstrap-error.json").write_text(
                    json.dumps(
                        {
                            "stage": "not_started",
                            "reason": "Earlier profile failed; no fallback or hidden retry",
                            "native_requests": 0,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                continue
            print(f"RUN {arm}: {profile}", flush=True)
            with (work / f"{arm}-driver.log").open("w") as log:
                run = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__)),
                        "--profile",
                        profile,
                        "--out",
                        str(work / arm),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            codes[arm] = run.returncode
            if not (work / arm / "results").exists():
                (work / f"{arm}-bootstrap-error.json").write_text(
                    json.dumps(
                        {
                            "returncode": run.returncode,
                            "stage": "bootstrap_or_prepare",
                            "log": f"{arm}-driver.log",
                            "native_requests": 0,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        (work / "execution-codes.json").write_text(json.dumps(codes, indent=2) + "\n")
        subprocess.run(
            [str(report_python), str(ROOT / "paired_report.py"), str(work)], check=True
        )
        review(work)
        return 0 if all(code == 0 for code in codes.values()) else 1
    for name in ["prepare.py", "native_probe.py", "run_all.py", "measurements.py"]:
        shutil.copyfile(ROOT / "pipeline" / name, work / name)
    env = dict(
        os.environ,
        VLLM_HUST_KIT=str(ROOT),
        VLLM_HUST_REPORT_PYTHON=str(report_python),
        VLLM_HUST_PROFILE=args.profile,
    )
    invocation = {
        "argv": sys.argv,
        "created_at": now,
        "entry_sha256": sha(Path(__file__)),
        "report_matplotlib": "3.10.8",
        "protocol": "real local TP2 diagnostic plus strict formal readiness assessment",
    }
    (work / "entry-invocation.json").write_text(
        json.dumps(invocation, indent=2, ensure_ascii=False) + "\n"
    )
    command = subprocess.run(
        [sys.executable, str(work / "run_all.py"), "--out", "results"],
        cwd=work,
        env=env,
    )
    out = work / "results"
    if command.returncode or not (out / "report.html").exists():
        return command.returncode or 1
    review(out)
    data = json.loads((out / "diagnostic-summary.json").read_text())
    print("Report:", out / "report.html")
    print(
        "Formal acceptance is reported separately from successful diagnostic execution."
    )
    return 0 if data["completed"] == data["planned"] and data["receipt_exported"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("FAILED:", error, file=sys.stderr)
        sys.exit(2)
