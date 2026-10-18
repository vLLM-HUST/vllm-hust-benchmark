#!/usr/bin/env python3
"""Independently recompute bounded-window statistics from public raw records."""

import csv
import gzip
import hashlib
import json
from pathlib import Path


def verify(root):
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = root / name
        assert root.resolve() in path.resolve().parents
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    config = json.loads((root / "config.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert (
        hashlib.sha256((root / "sessions.csv").read_bytes()).hexdigest()
        == config["expected_sessions_sha256"]
    )
    assert config["concurrency"] >= 32 and report["finite_makespan_claim"] is False
    with (root / "sessions.csv").open() as stream:
        inputs = {
            (r["session_id"], int(r["round_idx"])): r for r in csv.DictReader(stream)
        }
    signatures = []
    for point in report["points"]:
        assert point["status"] == "MEASURED_WINDOW"
        with gzip.open(root / point["label"] / "rounds.jsonl.gz", "rt") as stream:
            rr = [json.loads(line) for line in stream]
        assert len(rr) == point["completed_calls"] and len(
            {(r["session_id"], r["round_idx"]) for r in rr}
        ) == len(rr)
        for r in rr:
            assert (
                r["finished_seconds"] <= config["window_seconds"]
                and (r["session_id"], r["round_idx"]) in inputs
            )
            assert (
                r["previous_history_sha256"] == r["inherited_history_sha256"]
                and r["prompt_token_ids_sha256"]
            )
            if r["status"] == "ok":
                assert (
                    r["engine_completion_tokens"] == r["output_len"]
                    and r["engine_prompt_tokens"] == r["expected_engine_prompt_tokens"]
                )
        assert sum(r["status"] != "ok" for r in rr) == point["request_errors"]
        output = sum(r.get("engine_completion_tokens") or 0 for r in rr)
        assert output == point["completed_output_tokens"]
        assert (
            abs(
                output / config["window_seconds"]
                - point["completed_output_tokens_per_second"]
            )
            < 1e-8
        )
        samples = [
            json.loads(line)
            for line in (root / point["label"] / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        assert (
            samples[0]["t"] < 2
            and config["window_seconds"] - 2
            <= samples[-1]["t"]
            <= config["window_seconds"]
        )
        elapsed = samples[-1]["t"] - samples[0]["t"]
        for field, counter in [
            ("engine_output_tokens_per_second", "vllm:generation_tokens_total"),
            ("engine_input_tokens_per_second", "vllm:prompt_tokens_total"),
        ]:
            assert (
                abs(
                    (samples[-1][counter] - samples[0][counter]) / elapsed
                    - point[field]
                )
                < 1e-8
            )
        signatures.append(
            {
                (r["session_id"], r["round_idx"]): (
                    r["prompt_token_ids_sha256"],
                    r["engine_prompt_tokens"],
                    r["engine_completion_tokens"],
                )
                for r in rr
                if r["status"] == "ok"
            }
        )
    common = set(signatures[0]).intersection(*[set(s) for s in signatures[1:]])
    assert common and all(
        all(s[k] == signatures[0][k] for k in common) for s in signatures[1:]
    )
    assert (
        report["overlapping_inputs_equal"]
        and len(common) == report["overlapping_input_rounds"]
    )
    assert report["qualified_for_matrix"] is False
    print(
        json.dumps(
            dict(
                integrity="PASS",
                windows=len(report["points"]),
                concurrency=config["concurrency"],
                seconds_per_window=config["window_seconds"],
                common_rounds=len(common),
                qualification="LIMITED_SINGLE_PAIR",
            )
        )
    )


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
