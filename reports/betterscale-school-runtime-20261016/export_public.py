"""Project completed school-runtime aggregates; never export request payloads."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def export(archive, release):
    bindings = {}
    with tarfile.open(archive) as capsule:

        def read(name):
            raw = capsule.extractfile(name).read()
            bindings[name] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)

        measured = read("report.json")
        assert measured["status"] == "completed"
        runtime = measured["runtime"]
        performance = measured["performance"]
        for scope, perf in performance.items():
            if scope in ("mmlu", "frontier"):
                name = "mmlu-pro" if scope == "mmlu" else "frontierscience"
                raw = read(f"runs/{scope}/bench/outputs/{name}.json")
            else:
                raw = read(f"bench/{scope}/outputs/{scope}.json")
                accounting = read(f"bench/{scope}/token-accounting.json")
                assert accounting["exact_match"]
                assert accounting["server_tokenize_counts"] == raw["input_lens"]
                perf["input_token_accounting_verified"] = True
                perf["input_token_accounting"] = "Verified against server /tokenize"
            assert raw["completed"] == perf["num_prompts"] and raw["failed"] == 0
            assert raw["output_throughput"] == perf["output_throughput"]
            for key in ("total_input_tokens", "total_output_tokens"):
                perf[key] = raw[key]

    with archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert archive_hash == release["archive_sha256"]
    return {
        "schema": 1,
        "date": "2026-10-16",
        "status": "completed",
        "accuracy": measured["accuracy"],
        "performance": performance,
        "runtime": {
            "source_commit": release["commit"],
            "source_tag": "v1",
            "tested_base_commit": runtime["base_commit"],
            "release_delta": "Documentation, version and qualification metadata only",
            "model": "Qwen3.5-35B-A3B",
            "hardware": "Ascend 910B2; two concurrent TP2 lanes on four devices",
            **{
                k: runtime[k]
                for k in (
                    "packages",
                    "python",
                    "precision",
                    "tp",
                    "mtp",
                    "graph",
                    "state_bytes_per_rank",
                    "host_cache_bytes_per_rank",
                )
            },
            "cann": "9.1.0",
            "max_model_len": 65536,
            "max_num_batched_tokens": 4096,
            "lanes": {
                lane: {
                    k: values[k]
                    for k in ("execution_seats", "resident_seats", "images")
                }
                for lane, values in runtime["lanes"].items()
            },
            "validation": "23/23 basic smoke; full dataset scopes and six separate load tests completed",
        },
        "limitations": measured["limitations"]
        + [
            "School-runtime compatibility acceptance, not formal fixed-target leaderboard admission.",
            "Aggregate-only public projection; independent regrading requires private raw responses.",
            "HLE unresolved judges: Gold 18, Revision 28; Frontier 3 per track. Retained as nonsuccess, not confirmed incorrect.",
        ],
        "evidence": {
            "archive_sha256": archive_hash,
            "source_files_sha256": bindings,
            "tested_code_sha256": runtime["source_files"],
            "native_sha256": {Path(k).name: v for k, v in runtime["native"].items()},
            "note": "Integrity bindings, not signatures. No questions, images, responses or environment dumps published.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = export(args.archive, json.loads(args.release.read_text()))
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
