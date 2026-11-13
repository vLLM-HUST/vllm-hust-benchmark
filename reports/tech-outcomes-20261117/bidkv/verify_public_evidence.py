"""Verify public numeric derivatives without claiming access to controlled SSE."""

import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def verify():
    manifest = read(ROOT / "manifest.json")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        raw = path.read_bytes()
        assert len(raw) == item["bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    frozen_sources = {}
    source_directory = ROOT / "driver-mod-v2"
    if (source_directory / "source-manifest.json").is_file():
        frozen_sources = read(source_directory / "source-manifest.json")["files"]
        for path, checksum in frozen_sources.items():
            candidate = source_directory / (
                path + ".source" if path.endswith(".py") else path
            )
            assert hashlib.sha256(candidate.read_bytes()).hexdigest() == checksum
    folder = ROOT / "c8-results-v1"
    custody = read(folder / "policy-custody-index.json")
    count = 0
    for arm, expected in [("native", 342), ("bidkv", 343)]:
        directory = folder / arm
        local = read(directory / "manifest.json")
        assert {item["path"] for item in local["files"]} == {
            path.name for path in directory.iterdir() if path.name != "manifest.json"
        }
        for item in local["files"]:
            raw = (directory / item["path"]).read_bytes()
            assert len(raw) == item["bytes"]
            assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        rows = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == expected
        assert len({row["case_id"] for row in rows}) == expected
        allowed = {
            "case_id",
            "phase",
            "lane",
            "status",
            "http_status",
            "start",
            "first_generated_event",
            "last_generated_event",
            "first_content_event",
            "last_content_event",
            "end",
            "elapsed_s",
            "input_tokens",
            "output_tokens",
            "finish_reason",
            "cached_tokens_reported",
            "controlled_record_sha256",
            "controlled_sse_sha256",
        }
        assert all(set(row) == allowed for row in rows)
        phases = read(directory / "phases.json")
        assert [phase["name"] for phase in phases] == [
            "4096-warmup",
            "4096-formal",
            "16384-warmup",
            "16384-formal",
        ]
        admission = read(directory / "independent-verification.json")
        assert admission["passed"] and admission["total_requests"] == expected
        for phase, admitted in zip(phases, admission["phases"], strict=True):
            selected = [row for row in rows if row["phase"] == phase["name"]]
            assert len(selected) == phase["requests"] == admitted["requests"]
            assert phase["failed"] == 0
            assert phase["seconds"] == (60 if phase["name"].endswith("warmup") else 300)
            assert admitted["zero_reuse"]["passed"]
            assert all(
                value == 0
                for value in admitted["zero_reuse"][
                    "prefix_hit_counter_deltas"
                ].values()
            )
            assert admitted["zero_reuse"]["unique_salts"] == len(selected)
            for row in selected:
                assert row["status"] == "ok" and row["http_status"] == 200
                assert row["finish_reason"] == "length" and row["output_tokens"] == 256
                assert row["input_tokens"] == (
                    4115 if row["phase"].startswith("4096") else 16407
                )
                assert row["cached_tokens_reported"] is None
                assert (
                    phase["start"]["monotonic"]
                    <= row["start"]["monotonic"]
                    < phase["submission_deadline_monotonic"]
                )
                assert (
                    row["start"]["monotonic"]
                    <= row["first_generated_event"]["monotonic"]
                    <= row["end"]["monotonic"]
                    <= phase["end"]["monotonic"]
                )
            values = [
                row["first_generated_event"]["monotonic"] - row["start"]["monotonic"]
                for row in selected
            ]
            tpot = [
                (row["end"]["monotonic"] - row["first_generated_event"]["monotonic"])
                / 255
                for row in selected
            ]
            for key, numbers in [
                ("ttft_generated_seconds", values),
                ("tpot_completion_seconds", tpot),
            ]:
                assert phase[key]["samples"] == len(numbers)
                assert math.isclose(phase[key]["mean"], statistics.mean(numbers))
                for field, quantile in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]:
                    assert math.isclose(
                        phase[key][field], percentile(numbers, quantile)
                    )
        policy = read(directory / "policy-summary.json")
        original = next(item for item in custody["arms"] if item["arm"] == arm)
        assert original["service_final_phase"] == "preflight-complete"
        assert [item["last_stats"] for item in original["journal_originals"]] == policy
        for item in policy:
            assert item["enabled"] == (arm == "bidkv")
            assert (
                item["calls"]
                == item["decisions"]
                == item["failures"]
                == item["invalid_selections"]
                == 0
            )
        count += len(rows)
    return {
        "public_files_verified": len(manifest["files"]),
        "c8_numeric_requests_recomputed": count,
        "default_mod_frozen_source_files_verified": len(frozen_sources),
        "original_sse_publicly_read_back": False,
        "controlled_sse_independent_verification_receipts_indexed": True,
        "five_dataset_formal_completion_established_by_this_capsule": False,
        "selector_mechanism_gain_established": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
