"""Recompute published metrics and check historical custody, never import a model."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def raw(path):
    return (
        path.read_bytes()
        if path.is_file()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def read(path):
    return json.loads(raw(path))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def distribution(values):
    values = sorted(values)
    result = {
        "samples": len(values),
        "mean": statistics.mean(values) if values else None,
    }
    for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        p = (len(values) - 1) * q
        result[name] = (
            values[math.floor(p)]
            + (values[math.ceil(p)] - values[math.floor(p)]) * (p - math.floor(p))
            if values
            else None
        )
    return result


def verify(root=None):
    root = root or Path(__file__).resolve().parent / "five-results-v1"
    for item in read(root / "manifest.json")["files"]:
        path = root / item["path"]
        assert path.resolve().is_relative_to(root.resolve())
        value = raw(path)
        assert len(value) == item["bytes"]
        assert hashlib.sha256(value).hexdigest() == item["sha256"]
    for item in read(root / "compressed-members.json")["files"]:
        value = raw(root / item["logical_path"])
        assert len(value) == item["original_bytes"]
        assert hashlib.sha256(value).hexdigest() == item["original_sha256"]
        assert (
            hashlib.sha256((root / item["stored_path"]).read_bytes()).hexdigest()
            == item["stored_sha256"]
        )
    summary = read(root / "summary.json")
    assert summary["formal_complete"] is True and summary["status"] == "completed"
    assert summary["planned_solver_requests_per_arm"] == 672
    assert summary["planned_judge_inputs"] == 720
    closure = read(root / "audit/closure.json")
    assert closure["all_stages_passed"] is True
    assert closure["raw_sse_verified"] == closure["verified_records"] == 2064
    assert closure["original_manifest_files_verified"] == 4585
    assert closure["original_inventory_unchanged"] is True
    assert closure["model_startups"] == closure["model_generation_requests"] == 0
    for item in closure["quality_recomputation"]:
        assert item["byte_identical"] is True
        assert item["original_sha256"] == item["recomputed_sha256"]
        assert (
            hashlib.sha256(raw(root / (item["name"] + "-summary.json"))).hexdigest()
            == item["original_sha256"]
        )
    custody = read(root / "controlled-originals-index.json")
    assert custody["originals_not_redistributed"] is True
    hashes = {x["sha256"] for x in custody["files"]}
    totals = {"solver": 0, "judge": 0}
    expected = {"mmlu": 200, "hle": 200, "frontier": 160, "swe": 64, "terminal": 48}
    stages = summary["stages"]
    assert len(stages) == len({x["name"] for x in stages}) == 14
    calculated = {}
    for stage in stages:
        n = stage["planned"]
        assert stage["status"] == "completed" and stage["exit_code"] == 0
        assert stage["recorded"] == stage["sse_capture_files"] == n
        assert stage["unbound_sse_capture_files"] == 0
        proof = stage["independent_verification"]
        assert proof == read(root / "audit/verification" / (stage["name"] + ".json"))
        assert proof["recorded"] == proof["raw_sse_verified"] == n
        assert proof["zero_reuse"]["passed"] is True
        rows = read(root / stage["numeric_records_file"])
        assert len(rows) == len({r["case_id"] for r in rows}) == n
        assert all(
            r["controlled_record_sha256"] in hashes
            and r["controlled_sse_sha256"] in hashes
            for r in rows
        )
        totals[stage["stage"]] += n
        if stage["stage"] != "solver":
            continue
        assert expected[stage["dataset"]] == n
        successful = [r for r in rows if r["status"] == "ok"]
        assert len(successful) == n
        start = min(r["start"]["monotonic"] for r in rows)
        end = max(r["end"]["monotonic"] for r in rows)
        ttft, tpot = [], []
        for r in successful:
            a, b, c = (r[k]["monotonic"] for k in ("start", "first_token", "end"))
            assert math.isfinite(a) and a <= b <= c and math.isfinite(c)
            ttft.append(b - a)
            if r["usage"]["completion_tokens"] > 1:
                tpot.append((c - b) / (r["usage"]["completion_tokens"] - 1))
        tokens = sum(r["usage"]["completion_tokens"] for r in successful)
        close(stage["summary"]["solver_batch_wall_seconds"], end - start)
        assert tokens == stage["summary"]["accepted_output_tokens"]
        close(stage["summary"]["output_tokens_per_second"], tokens / (end - start))
        assert (
            sum(r["finish_reason"] == "length" for r in rows)
            == stage["summary"]["length_stops"]
        )
        for name, values in (("ttft_seconds", ttft), ("tpot_seconds", tpot)):
            observed = distribution(values)
            recorded = stage["summary"][name]
            assert observed["samples"] == recorded["samples"]
            for key in ("mean", "p50", "p95", "p99"):
                close(observed[key], recorded[key])
        assert n - len(tpot) == stage["summary"]["tpot_excluded_output_tokens_le_one"]
        calculated[stage["name"]] = tokens / (end - start)
    assert totals == {"solver": 1344, "judge": 720}
    mechanism = read(root / "connector-mechanism.json")
    saves = []
    for stage in mechanism["stages"]:
        admission = stage["cache_admission"]
        assert admission["passed"] is True
        assert all(v == 0 for v in admission["prefix"]["deltas"].values())
        if stage["arm"] == "pegaflow":
            values = admission["engine_connector"]["deltas"]
            assert all(
                values[k] == 0
                for k in (
                    "vllm:pega_load_success_total",
                    "vllm:pega_load_failure_total",
                    "vllm:pega_save_failure_total",
                )
            )
            saves.append(values["vllm:pega_save_success_total"])
    assert saves == [12, 8, 0, 402, 390]
    history = read(root / "audit/historical-receipts-supplement-v2.json")
    for name in (
        "outer-campaign-actual-wait.json",
        "c8-controller-wait.json",
        "formal-controller-wait.json",
    ):
        value = history["actual_wait_receipts"][name]["observed"]
        assert (
            value["phase"] == "exited"
            and value["child_returncode"] == value["supervisor_returncode"] == 0
        )
    assert len(history["historical_service_lifecycle"]) == 5
    for service in history["historical_service_lifecycle"]:
        assert service["stop.json"]["data"]["server_exit_code"] == 0
        assert service["stop.json"]["data"]["sidecar_exit_code"] in (None, -15)
        assert service["cleanup"]["empty_allocated_devices"] is True
    c8 = read(root / "c8/paired-summary.json")
    assert c8["all_records_independently_replayed"] is True
    assert (c8["native_requests"], c8["pegaflow_requests"], c8["failed"]) == (
        343,
        341,
        0,
    )
    for phase in c8["phases"]:
        for arm in ("native", "pegaflow"):
            close(
                phase[arm + "_window_tokens_per_second"],
                phase[arm + "_requests"]
                * 256
                / phase[arm + "_phase_elapsed_seconds_including_drain"],
            )
        close(
            phase["throughput_change_percent"],
            100
            * (
                phase["pegaflow_window_tokens_per_second"]
                / phase["native_window_tokens_per_second"]
                - 1
            ),
        )
    request_root = root / "c8/requests"
    for item in read(request_root / "manifest.json")["files"]:
        value = raw(request_root / item["path"])
        assert (
            len(value) == item["bytes"]
            and hashlib.sha256(value).hexdigest() == item["sha256"]
        )
    indexed = {
        (x["arm"], x["sha256"], x["sse_sha256"])
        for x in read(request_root / "controlled-record-index.json")["files"]
    }
    for arm, count in (("native", 343), ("pegaflow", 341)):
        rows = read(request_root / (arm + "-requests.json"))
        phases = read(request_root / (arm + "-phases.json"))["phases"]
        assert len(rows) == len({r["case_id"] for r in rows}) == count
        assert all(
            r["status"] == "ok" and r["usage"]["completion_tokens"] == 256 for r in rows
        )
        assert all(
            (arm, r["controlled_record_sha256"], r["stream_sha256"]) in indexed
            for r in rows
        )
        for phase in phases:
            selected = [r for r in rows if r["phase"] == phase["name"]]
            assert len(selected) == phase["requests"]
            tokens = sum(r["usage"]["completion_tokens"] for r in selected)
            assert tokens == phase["output_tokens_accepted_including_drain"]
            assert phase["seconds"] == (60 if phase["name"].endswith("warmup") else 300)
            start, end = (phase[k]["monotonic"] for k in ("start", "end"))
            assert all(
                start
                <= r["start"]["monotonic"]
                <= phase["submission_deadline_monotonic"]
                and r["end"]["monotonic"] <= end
                for r in selected
            )
            summary_phase = next(p for p in c8["phases"] if p["phase"] == phase["name"])
            close(
                tokens / (end - start), summary_phase[arm + "_window_tokens_per_second"]
            )
    return {
        "formal_complete": True,
        "public_numeric_identities_checked": 2064,
        "public_solver_metrics_recomputed": 1344,
        "controlled_original_SSE_replay_receipts": 2064,
        "public_original_SSE_readback": False,
        "original_quality_byte_identical_recomputations": 4,
        "actual_formal_saves": sum(saves),
        "actual_formal_restores": 0,
        "throughput_changes_percent": {
            d: 100 * (calculated["pegaflow-" + d] / calculated["native-" + d] - 1)
            for d in expected
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
