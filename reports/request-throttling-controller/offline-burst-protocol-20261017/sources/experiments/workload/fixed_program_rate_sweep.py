#!/usr/bin/env python3
"""Prepare or execute a fixed-program arrival-rate sweep in a recorded Ray job."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def prepare(source, destination, source_rate, rate):
    if source_rate <= 0 or rate <= 0:
        raise ValueError("rates must be positive")
    with (source / "sessions.csv").open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError("empty workload")
    first = min(float(row["arrival_time"]) for row in rows)
    content = [{k: v for k, v in row.items() if k != "arrival_time"} for row in rows]
    content_hash = hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()
    ).hexdigest()
    arrivals = {}
    for row in rows:
        arrival = float(row["arrival_time"])
        previous = arrivals.setdefault(row["session_id"], arrival)
        if previous != arrival:
            raise ValueError("all rounds of a program must have the same arrival time")
        row["arrival_time"] = str(round((arrival - first) * source_rate / rate))
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "sessions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["fixed_workload"] = {
        "program_count": len(arrivals),
        "round_count": len(rows),
        "program_content_sha256": content_hash,
        "source_sessions_sha256": hashlib.sha256(
            (source / "sessions.csv").read_bytes()
        ).hexdigest(),
        "source_rate_programs_per_second": source_rate,
        "rate_programs_per_second": rate,
        "arrival_scaling": "source_rate / rate scaling; milliseconds rounded",
    }
    manifest.setdefault("knobs", {}).update(
        program_rate=rate,
        arrival_duration_seconds=max(float(row["arrival_time"]) for row in rows) / 1000,
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["fixed_workload"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", required=True, type=Path)
    parser.add_argument("--source-rate", required=True, type=float)
    parser.add_argument("--rates", required=True, nargs="+", type=float)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evidence-type", choices=["live", "smoke"], default="live")
    parser.add_argument("--wait-for-job", default="")
    parser.add_argument("--safety-drain-seconds", type=int, default=3600)
    arguments = parser.parse_args()
    root = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
    if "/" in arguments.experiment_name or len(arguments.experiment_name) > 65:
        parser.error(
            "use a readable experiment name of at most 65 characters without /"
        )
    if len(set(arguments.rates)) != len(arguments.rates):
        parser.error("duplicate rates")
    output = root / "data/results/workload" / arguments.experiment_name
    output.mkdir(parents=True, exist_ok=False)
    if arguments.wait_for_job and arguments.execute:
        from ray.job_submission import JobSubmissionClient

        client = JobSubmissionClient(os.environ["RAY_DASHBOARD_ADDRESS"])
        started = time.monotonic()
        while client.get_job_status(arguments.wait_for_job).value in (
            "RUNNING",
            "PENDING",
        ):
            if time.monotonic() - started > 9000:
                raise TimeoutError(
                    "preceding experiment did not finish; no sweep launched"
                )
            print(
                "waiting for preceding experiment: " + arguments.wait_for_job,
                flush=True,
            )
            time.sleep(30)
        if client.get_job_status(arguments.wait_for_job).value != "SUCCEEDED":
            raise RuntimeError("preceding experiment failed; no sweep launched")
    records = []
    for index, rate in enumerate(arguments.rates):
        label = f"{rate:g}"
        name = arguments.experiment_name + "-rate-" + label
        destination = root / "data/workloads" / name
        record = prepare(
            arguments.source_workload, destination, arguments.source_rate, rate
        )
        record.update(experiment_name=name)
        records.append(record)
        (output / "sweep_manifest.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        if not arguments.execute:
            continue
        environment = dict(
            os.environ,
            DRAIN_AFTER_ARRIVALS="1",
            CONCURRENCY="0",
            DRAIN_TIMEOUT_S=str(arguments.safety_drain_seconds),
            EVIDENCE_TYPE=arguments.evidence_type,
            FIXED_PROGRAM_CONTENT="1",
        )
        configurations = ["vllm-direct", "thunderagent-with-throttling"]
        if index % 2:
            configurations.reverse()
        subprocess.run(
            [
                "bash",
                str(root / "experiments/workload/program_configurations.sh"),
                name,
                name,
                *configurations,
            ],
            env=environment,
            check=True,
        )
        rows = []
        reference_input_hash = None
        for completed_record in records:
            for path in (
                root / "data/results/workload" / completed_record["experiment_name"]
            ).glob("*/summary.json"):
                summary = json.loads(path.read_text())
                result = summary["result"]
                inputs = [
                    (
                        r["session_id"],
                        r["round_idx"],
                        r.get("prompt_token_ids_sha256", ""),
                    )
                    for r in (
                        json.loads(line)
                        for line in (path.parent / "rounds.jsonl")
                        .read_text()
                        .splitlines()
                    )
                ]
                input_hash = hashlib.sha256(
                    json.dumps(sorted(inputs)).encode()
                ).hexdigest()
                if reference_input_hash is None:
                    reference_input_hash = input_hash
                valid = (
                    result["sessions_unfinished"] == 0
                    and result["rounds_failed"] == 0
                    and input_hash == reference_input_hash
                    and all(item[2] for item in inputs)
                    and all(
                        item["count"] == 0 for item in result["mismatches"].values()
                    )
                )
                rows.append(
                    {
                        "rate_programs_per_second": completed_record[
                            "rate_programs_per_second"
                        ],
                        "configuration": path.parent.name,
                        "all_programs_completed": result["sessions_unfinished"] == 0,
                        "valid_comparison": valid,
                        "programs_completed": result["sessions_completed"],
                        "programs_unfinished": result["sessions_unfinished"],
                        "makespan_seconds": result.get("makespan_seconds"),
                        "program_latency_mean_seconds": result[
                            "program_end_to_end_latency_seconds"
                        ].get("mean"),
                        "program_latency_p90_seconds": result[
                            "program_end_to_end_latency_seconds"
                        ].get("p90"),
                        "steps_per_minute": result["rounds_completed_per_minute"],
                        "summary_path": str(path),
                        "prompt_inputs_sha256": input_hash,
                    }
                )
        with (output / "comparison.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if not all(row["valid_comparison"] for row in rows):
            raise RuntimeError("incomplete programs: comparison saved, sweep stopped")
    print("sweep output: " + str(output), flush=True)


if __name__ == "__main__":
    main()
