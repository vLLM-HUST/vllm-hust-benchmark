"""Export aggregate results and payload-free outcomes from the private run archive."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def export(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    runs = archive / "evidence/runs"
    report = json.loads((archive / "comparison.json").read_text())
    report.update(
        measurement_date="2026-10-15",
        formal_acceptance=False,
        repeats=1,
        hardware="Ascend 910B2, two separate TP2 services on four physical devices",
        scope="MMLU-Pro full test; FrontierScience full Olympiad and Research; HLE deferred",
        source_scope="BetterScale source snapshot and qualified existing CANN9.1 runtime; no later multimodal patch",
        engine_configuration={
            "dtype": "bfloat16",
            "tensor_parallel_size": 2,
            "speculative_tokens": 2,
            "cudagraph_mode": "FULL",
            "execution_seats": 16,
            "resident_seats": 36,
            "state_bytes_per_rank": 20 << 30,
            "host_bytes_per_rank": 8 << 30,
            "max_model_len": 65536,
            "max_num_batched_tokens": 4096,
            "prefix_caching": True,
        },
        baseline_reports={
            "mmlu": "../mmlu-pro-qwen35-b0-precheck-20261012",
            "frontier": "../frontierscience-qwen35-b0-precheck-20261013",
        },
    )
    outcomes = {}
    score = json.loads((runs / "mmlu/full/score.json").read_text())
    outcomes["mmlu"] = []
    for row in score["outcomes"]:
        record = runs / "mmlu/full/requests" / (row["case_id"] + ".json")
        keep = {
            k: row[k]
            for k in (
                "case_id",
                "category",
                "valid",
                "correct",
                "transport_success",
                "length_limited",
            )
        }
        keep["record_sha256"] = sha(record)
        outcomes["mmlu"].append(keep)
    outcomes["frontier"] = []
    with (runs / "frontier/full/report/per_case.csv").open() as handle:
        for row in csv.DictReader(handle):
            for key in ("solver_truncated", "judge_truncated", "correct"):
                row[key] = {"True": True, "False": False, "": None}[row[key]]
            row["score"] = float(row["score"]) if row["score"] else None
            row["audit_flag_count"] = int(row["audit_flag_count"])
            for role in ("solver", "judge"):
                record = (
                    runs / f"frontier/full/{role}/requests" / (row["case_id"] + ".json")
                )
                row[role + "_record_sha256"] = sha(record)
            outcomes["frontier"].append(row)
    packed = json.dumps(outcomes, ensure_ascii=False, separators=(",", ":")).encode()
    (output / "per-case-outcomes.json.gz").write_bytes(gzip.compress(packed, mtime=0))
    evidence = {}
    for dataset, server, bench in (
        ("mmlu", "server-mmlu", "mmlu-pro"),
        ("frontier", "server-text", "frontierscience"),
    ):
        rels = [
            f"{dataset}/plan/contract.json",
            f"{dataset}/bench/outputs/{bench}.json",
            f"{server}/argv.json",
            f"{server}/server.log",
            f"{server}/deployment.json",
            f"{server}/finished.json",
            f"{server}/release.json",
        ]
        evidence[dataset] = {name: sha(runs / name) for name in rels}
        report[dataset + "_release"] = json.loads(
            (runs / server / "finished.json").read_text()
        )
        raw_bench = json.loads(
            (runs / dataset / f"bench/outputs/{bench}.json").read_text()
        )
        report[dataset + "_bench"].update(
            {
                key: raw_bench[key]
                for key in ("total_input_tokens", "total_output_tokens")
            }
        )
    report["mmlu_categories"] = score["categories"]
    save(output / "report.json", report)
    save(output / "private-evidence-index.json", evidence)
    save(
        output / "verification.json",
        json.loads((archive / "verification.json").read_text()),
    )
    save(
        output / "publication.json",
        {
            "scope": "Aggregate summaries, outcome flags and hashes only. No questions, images, answers, environment dumps or raw server commands.",
            "verification_boundary": "Public verifier checks internal counts and file integrity. Independent regrading requires omitted raw responses. Hashes are not signatures.",
            "private_archives": {
                name: sha(archive / name)
                for name in (
                    "mmlu-complete-evidence.tar.gz",
                    "frontier-complete-evidence.tar.gz",
                )
            },
            "original_mmlu_cases_sha256": sha(runs / "mmlu/plan/cases.json"),
            "original_frontier_client_code": json.loads(
                (runs / "frontier/plan/contract.json").read_text()
            )["client_code"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.archive, args.output)
