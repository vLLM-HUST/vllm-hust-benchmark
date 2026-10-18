"""Verify Revision text/image performance independently of collection."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root, archive=None):
    root = Path(root)
    if (root / "SHA256SUMS").exists():
        for line in (root / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert Path(name).name == name
            assert sha((root / name).read_bytes()) == expected
    proof = json.loads((root / "checkpoint.json").read_text())
    assert proof["state"] == "THREE_REPEAT_PERFORMANCE_COMPLETE"
    rows = proof["records"]
    assert len(rows) == 16 and len({r["stage"] for r in rows}) == 16
    for modality, count in [("text", 200), ("image", 118)]:
        selected = [r for r in rows if r["modality"] == modality]
        assert len(selected) == 8
        assert {r["input_sha256"] for r in selected} == {
            proof["input_sha256"][modality]
        }
        for arm in ["native", "controller"]:
            arm_rows = [r for r in selected if r["arm"] == arm]
            assert {r["repetition"] for r in arm_rows} == {"warmup", 1, 2, 3}
            for key, value in proof["aggregate"][modality][arm].items():
                assert value == statistics.median(
                    r[key] for r in arm_rows if r["repetition"] != "warmup"
                )
        for row in selected:
            assert (
                row["completed"] == count and row["failed"] == 0 and row["duration"] > 0
            )
            assert 0 < row["total_output_tokens"] <= count * 256
            for key, total in [
                ("output_throughput", row["total_output_tokens"]),
                ("request_throughput", count),
                (
                    "total_token_throughput",
                    row["total_input_tokens"] + row["total_output_tokens"],
                ),
            ]:
                assert math.isclose(row[key], total / row["duration"], rel_tol=1e-9)
    if archive:
        assert sha(Path(archive).read_bytes()) == proof["private_raw_sha256"]
        with tarfile.open(archive) as tar:

            def raw(name):
                return tar.extractfile(name).read()

            queue_bytes = raw("queue-plan.json")
            assert sha(queue_bytes) == proof["source_queue_sha256"]
            assert json.loads(queue_bytes)["run_id"] == proof["run_id"]
            for row in rows:
                label = "revision-" + row["modality"]
                base = "performance/" + row["stage"]
                data = raw(base + "/outputs/" + label + ".json")
                assert sha(data) == row["raw_sha256"]
                actual = json.loads(data)
                assert actual["num_prompts"] == row["completed"]
                assert (
                    actual["max_concurrency"] is None
                    and actual["request_rate"] == "inf"
                )
                inputs = raw(base + "/inputs/" + label + ".materialized.jsonl")
                assert (
                    sha(inputs) == row["input_sha256"]
                    and len(inputs.splitlines()) == row["completed"]
                )
                command = raw(base + "/outputs/" + label + ".cmd.txt").decode()
                for flag in [
                    "--custom-output-len 256",
                    "--temperature 0",
                    "--seed 0",
                    "--disable-shuffle",
                ]:
                    assert flag in command
                for key, value in row.items():
                    if key in actual:
                        assert actual[key] == value
                summary = json.loads(raw(base + "/outputs/" + label + ".summary.json"))
                for key in [
                    "completed",
                    "failed",
                    "duration",
                    "output_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                ]:
                    assert summary[key] == actual[key]
    return {
        "performance": "PASS",
        "private_archive_verified": bool(archive),
        "formal_points": 12,
        "warmups_excluded": 4,
        "aggregate": proof["aggregate"],
        "delay_provenance": "Original vllm aggregates verified; detailed per-request timing is retained privately but was not independently recomputed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive")
    arguments = parser.parse_args()
    print(json.dumps(verify(Path(__file__).parent, arguments.archive)))
