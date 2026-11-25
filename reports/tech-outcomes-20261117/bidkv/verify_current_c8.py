"""Verify current public C8 numeric derivatives and controlled-custody references."""

import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_bytes())


def percentile(values, q):
    values = sorted(values)
    point = (len(values) - 1) * q
    low, high = math.floor(point), math.ceil(point)
    return values[low] + (values[high] - values[low]) * (point - low)


def verify(root, driver):
    result = {}
    for arm in ("native", "bidkv"):
        directory = root / arm
        if not directory.is_dir():
            continue
        manifest = read(directory / "manifest.json")
        for item in manifest["files"]:
            raw = (directory / item["path"]).read_bytes()
            assert len(raw) == item["bytes"]
            assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        rows = [
            json.loads(line)
            for line in (directory / "requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == len({row["case_id"] for row in rows})
        assert all(row["status"] == "ok" and row["http_status"] == 200 for row in rows)
        identity = read(directory / "execution-identity.json")
        assert identity["common_enable_cpu_binding"] is True
        assert (
            identity["source_manifest_sha256"]
            == hashlib.sha256(
                (driver / "source-manifest.json").read_bytes()
            ).hexdigest()
        )
        assert (
            identity["copied_carrier_inventory_sha256"]
            == hashlib.sha256(
                (driver / "carrier-source-admission.json").read_bytes()
            ).hexdigest()
        )
        checked = read(directory / "independent-verification.json")
        assert checked["passed"] and checked["total_requests"] == identity[
            "requests_verified"
        ] == len(rows)
        phases = read(directory / "phases.json")
        assert [item["name"] for item in phases] == [
            "4096-warmup",
            "4096-formal",
            "16384-warmup",
            "16384-formal",
        ]
        for phase, proof in zip(phases, checked["phases"], strict=True):
            selected = [row for row in rows if row["phase"] == phase["name"]]
            assert len(selected) == phase["requests"] == proof["requests"]
            assert phase["failed"] == 0 and phase["seconds"] == (
                60 if phase["name"].endswith("warmup") else 300
            )
            assert proof["zero_reuse"]["passed"]
            assert all(
                value == 0
                for value in proof["zero_reuse"]["prefix_hit_counter_deltas"].values()
            )
            assert proof["zero_reuse"]["unique_salts"] == len(selected)
            generated, tpot = [], []
            for row in selected:
                assert row["output_tokens"] == 256 and row["finish_reason"] == "length"
                assert row["input_tokens"] == (
                    4115 if phase["name"].startswith("4096") else 16407
                )
                assert row["cached_tokens_reported"] is None
                start = row["start"]["monotonic"]
                first = row["first_generated_event"]["monotonic"]
                end = row["end"]["monotonic"]
                assert (
                    phase["start"]["monotonic"]
                    <= start
                    < phase["submission_deadline_monotonic"]
                )
                assert start <= first <= end <= phase["end"]["monotonic"]
                generated.append(first - start)
                tpot.append((end - first) / 255)
            assert (
                sum(row["output_tokens"] for row in selected)
                == phase["output_tokens_accepted_including_drain"]
            )
            for key, values in (
                ("ttft_generated_seconds", generated),
                ("tpot_completion_seconds", tpot),
            ):
                assert phase[key]["samples"] == len(values)
                assert math.isclose(phase[key]["mean"], statistics.mean(values))
                for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                    assert math.isclose(phase[key][name], percentile(values, q))
        custody = read(directory / "policy-custody-index.json")
        journals = [item["last_stats"] for item in custody["journals"]]
        assert journals and all(
            item["enabled"] == (arm == "bidkv") for item in journals
        )
        assert all(
            item["failures"] == item["invalid_selections"] == 0 for item in journals
        )
        result[arm] = {
            "numeric_requests_recomputed": len(rows),
            "original_sse_publicly_read_back": False,
            "all_first_generated_equal_first_content": all(
                row["first_generated_event"] == row["first_content_event"]
                for row in rows
            ),
            "policy_last_observed_stats": journals,
            "scope": "Original60/300 synthetic qualification, including response drain; not full-five completion or measured fixed-window token throughput",
        }
    pair_path = root / "paired-summary.json"
    if pair_path.is_file():
        pair = read(pair_path)
        assert (
            pair["native_requests"] == result["native"]["numeric_requests_recomputed"]
        )
        assert pair["bidkv_requests"] == result["bidkv"]["numeric_requests_recomputed"]
        for arm in ("native", "bidkv"):
            assert pair[arm + "_policy_calls"] == sum(
                item["calls"] for item in result[arm]["policy_last_observed_stats"]
            )
            for phase in read(root / arm / "phases.json"):
                observed = next(
                    item for item in pair["phases"] if item["phase"] == phase["name"]
                )
                rate = phase["output_tokens_accepted_including_drain"] / (
                    phase["end"]["monotonic"] - phase["start"]["monotonic"]
                )
                assert math.isclose(rate, observed[arm + "_window_tokens_per_second"])
                assert phase["requests"] == observed[arm + "_requests"]
                assert phase["failed"] == observed[arm + "_failures"] == 0
        for phase in pair["phases"]:
            ratio = 100 * (
                phase["bidkv_window_tokens_per_second"]
                / phase["native_window_tokens_per_second"]
                - 1
            )
            assert math.isclose(ratio, phase["throughput_change_percent"])
    lifecycle_path = root / "c8-owned-lifecycle.json"
    if lifecycle_path.is_file():
        lifecycle = read(lifecycle_path)
        assert lifecycle["no_new_model_or_benchmark_requests"] is True
        assert {item["arm"] for item in lifecycle["arms"]} == {"native", "bidkv"}
        for item in lifecycle["arms"]:
            assert item["controller_child"]["actual_returncode"] == 0
            assert item["client_actual_wait_returncode"] == 0
            assert item["preflight"]["actual_wait_returncode"] == 0
            assert item["api_model"]["actual_wait_returncode"] == 0
            assert item["client_pid"] is None and item["client_start_end_unix"] is None
            result[item["arm"]]["actual_owned_wait_receipts_verified"] = True
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(verify(root / "c8-results-v2", root / "driver-mod-v4"), indent=2))
