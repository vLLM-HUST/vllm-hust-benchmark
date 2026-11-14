"""Verify public numeric derivatives without claiming access to controlled SSE."""

import hashlib
import json
import math
from pathlib import Path
import runpy
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
    aborted_raw = (ROOT / "aborted-cohort01/correction-and-counts.json").read_bytes()
    assert hashlib.sha256(aborted_raw).hexdigest() == (
        "e81bec941b94bd9cb1a40b8df5223a776ec12516269d5f3ecffc3458df32aa98"
    )
    aborted = json.loads(aborted_raw)
    assert aborted["status"] == "aborted_by_parent_pre_audit"
    assert aborted["records_unchanged"] is True
    assert aborted["native"] == {
        "planned": 672,
        "submitted": 200,
        "completed_success": 44,
        "returned_failures": 0,
        "cancelled_unfinished": 156,
        "not_executed": 472,
        "returned_length_stops": 2,
        "accepted_output_tokens_in_completed_records": 113977,
        "accepted_output_tokens_unfinished": None,
    }
    assert aborted["bidkv"] == {
        "planned": 672,
        "submitted": 0,
        "completed_success": 0,
        "not_executed": 672,
    }
    assert aborted["judge"] == {
        "planned_inputs": 720,
        "submitted": 0,
        "not_executed": 720,
    }
    assert aborted["standard_batch_throughput"] is None
    assert aborted["formal_quality_score"] is None
    assert aborted["controller_exit"]["actual_os_returncode"] is None
    assert aborted["client_exit"]["actual_os_returncode"] is None
    assert aborted["api_exit"]["exit_code"] == 0
    captures = aborted["unbound_cancelled_streams"]
    assert len(captures) == len({item["file"] for item in captures}) == 156
    for item in captures:
        assert item["bytes"] == item["observed_wire_lines"] == 0
        assert item["observed_done"] is False
        assert item["observed_final_usage"] is None
        assert item["sha256"] == hashlib.sha256(b"").hexdigest()
    frozen_sources = {}
    for version in (2, 3, 4):
        source_directory = ROOT / f"driver-mod-v{version}"
        if not (source_directory / "source-manifest.json").is_file():
            continue
        sources = read(source_directory / "source-manifest.json")["files"]
        for path, checksum in sources.items():
            candidate = source_directory / (
                path + ".source" if path.endswith(".py") else path
            )
            assert hashlib.sha256(candidate.read_bytes()).hexdigest() == checksum
        frozen_sources[source_directory.name] = len(sources)
    analysis_sources = read(ROOT / "analysis-v2/analysis-manifest.json")["files"]
    for name, checksum in analysis_sources.items():
        stored_name = "original-README.md.source" if name == "README.md" else name
        assert (
            hashlib.sha256(
                (ROOT / "analysis-v2" / stored_name).read_bytes()
            ).hexdigest()
            == checksum
        )
    current_c8 = runpy.run_path(str(ROOT / "verify_current_c8.py"))["verify"](
        ROOT / "c8-results-v2", ROOT / "driver-mod-v4"
    )
    pressure_sources = read(ROOT / "pressure-admission-v1/source-manifest.json")[
        "files"
    ]
    for name, checksum in pressure_sources.items():
        stored_name = name + ".source" if name.endswith(".py") else name
        assert (
            hashlib.sha256(
                (ROOT / "pressure-admission-v1" / stored_name).read_bytes()
            ).hexdigest()
            == checksum
        )
    pressure_driver_sources = read(ROOT / "pressure-driver-v1/source-manifest.json")[
        "files"
    ]
    assert len(pressure_driver_sources) == 12
    for name, checksum in pressure_driver_sources.items():
        stored_name = name + ".source" if name.endswith(".py") else name
        assert (
            hashlib.sha256(
                (ROOT / "pressure-driver-v1" / stored_name).read_bytes()
            ).hexdigest()
            == checksum
        )
    pressure_plan = read(ROOT / "pressure-driver-v1/execution-plan.json")
    assert pressure_plan["fixed_workload"]["planned_requests"] == 48
    assert pressure_plan["fixed_workload"]["max_tokens"] == 4096
    assert pressure_plan["time"]["original_operation_seconds"] == 3600
    assert (
        pressure_plan["candidate_conditional_go"][
            "requires_exact_native_metadata_fingerprint"
        ]
        is True
    )
    assert (
        pressure_plan["candidate_conditional_go"]["second_publication_park_required"]
        is False
    )
    pressure_v2_sources = read(ROOT / "pressure-driver-v2/source-manifest.json")[
        "files"
    ]
    assert len(pressure_v2_sources) == 12
    for name, checksum in pressure_v2_sources.items():
        stored_name = name + ".source" if name.endswith(".py") else name
        assert (
            hashlib.sha256(
                (ROOT / "pressure-driver-v2" / stored_name).read_bytes()
            ).hexdigest()
            == checksum
        )
    plan_v2 = read(ROOT / "pressure-driver-v2/execution-plan.json")
    assert plan_v2["fixed_workload"]["planned_requests"] == 16
    assert plan_v2["fixed_workload"]["repeats_per_arm"] == 1
    assert plan_v2["fixed_workload"]["max_output_tokens"] == 65536
    assert plan_v2["time"]["original_operation_seconds"] == 3600
    operation_root = ROOT / "pressure-driver-v2"
    operation = read(operation_root / "operation-launcher-manifest.json")
    assert (
        operation["fixed_controller_manifest_sha256"]
        == hashlib.sha256(
            (operation_root / "source-manifest.json").read_bytes()
        ).hexdigest()
    )
    for name, checksum in operation["files"].items():
        assert (
            hashlib.sha256(
                (operation_root / (name + ".source")).read_bytes()
            ).hexdigest()
            == checksum
        )
    pressure_capacity = None
    pressure_capacity_v2 = None
    capacity_root = ROOT / "pressure-capacity-v1"
    if capacity_root.is_dir():
        for item in read(capacity_root / "manifest.json")["files"]:
            raw = (capacity_root / item["path"]).read_bytes()
            assert len(raw) == item["bytes"]
            assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        pressure_capacity = runpy.run_path(
            str(ROOT / "pressure-driver-v1/capacity_rules.py.source")
        )["evaluate"](read(capacity_root / "native-cache-groups-rpc.json"))
        assert pressure_capacity == read(
            capacity_root / "native-capacity-go-no-go.json"
        )
        pressure_capacity_v2 = runpy.run_path(
            str(ROOT / "pressure-driver-v2/capacity_rules.py.source")
        )["evaluate"](read(capacity_root / "native-cache-groups-rpc.json"))
        assert pressure_capacity_v2 == read(
            capacity_root / "native-capacity-go-no-go-v2.json"
        )
        assert (
            pressure_capacity_v2["metadata_fingerprint_sha256"]
            == pressure_capacity["metadata_fingerprint_sha256"]
        )
        clock = read(capacity_root / "operation-clock.json")
        assert clock["operation_limit_seconds"] == 3600
        assert clock["deadline_monotonic"] - clock["started_monotonic"] == 3600
        admission = read(capacity_root / "native-admission-status.json")
        assert admission["phase"] == "admitted-awaiting-public-capacity-binding"
        assert admission["preflight_submitted_generation_requests"] == 0
        token_ids = read(capacity_root / "native-token-id-admission.json")
        assert token_ids["passed"] is True and len(token_ids["checked"]) == 8
        assert all(row["tokens"] == 16407 for row in token_ids["checked"])
    progress_path = ROOT / "solver-progress-v1/solver-progress.json"
    closed_solver_audits = 0
    if progress_path.is_file():
        progress = read(progress_path)
        assert progress["formal_complete"] is False
        assert progress["private_content_not_included"] is True
        assert progress["planned_solver_requests_per_arm"] == 672
        assert progress["planned_judge_inputs"] == 720
        expected = {"mmlu": 200, "hle": 200, "frontier": 160, "swe": 64, "terminal": 48}
        stages = progress["solver_stages"]
        assert len(stages) == 10
        assert {(item["arm"], item["dataset"]) for item in stages} == {
            (arm, dataset) for arm in ("native", "bidkv") for dataset in expected
        }
        for item in stages:
            count = expected[item["dataset"]]
            summary = item["summary"]
            proof = item["independent_verification"]
            assert (
                summary["planned"]
                == summary["executed"]
                == summary["successful"]
                == count
            )
            assert summary["failed"] == summary["not_executed"] == 0
            assert (
                proof["planned"]
                == proof["recorded"]
                == proof["raw_sse_verified"]
                == count
            )
            assert proof["metrics_recomputed"] and proof["zero_reuse"]["passed"]
            assert math.isclose(
                summary["output_tokens_per_second"],
                summary["accepted_output_tokens"]
                / summary["solver_batch_wall_seconds"],
            )
            closed_solver_audits += count
        assert len(progress["actual_audit_waits"]) == 10
        assert all(
            item["actual_returncode"] == 0 for item in progress["actual_audit_waits"]
        )
        for item in progress["solver_policy_custody"]:
            assert item["owned_api_stop"]["exit_code"] == 0
            for journal in item["journals"]:
                stats = journal["last_stats"]
                assert stats["enabled"] == (item["arm"] == "bidkv")
                assert (
                    stats["calls"]
                    == stats["decisions"]
                    == stats["failures"]
                    == stats["invalid_selections"]
                    == 0
                )
    failure_raw = (ROOT / "native04-failure-v1/failure-capsule.json").read_bytes()
    assert hashlib.sha256(failure_raw).hexdigest() == (
        "39b6fc5658aa892b16ee511cb963e87351f4eaf5f4acad7f54237664a4a5888b"
    )
    failure = json.loads(failure_raw)
    assert failure["attempt"] == "native-04"
    assert failure["model_generation_requests"] == 0
    assert failure["owned_preflight_wait_receipt"]["actual_returncode"] == -2
    assert failure["owned_api_wait_receipt"]["exit_code"] == -9
    assert len(failure["fatal_extract"]) == 5
    assert len(failure["raw_originals_manifest"]) == 9
    assert len(failure["installed_operator_asset_inventory"]) == 804
    assert any(
        "binary_info_config.json" in item["path"] and not item["present_in_failed_copy"]
        for item in failure["installed_operator_asset_inventory"]
    )
    original_manifest = (
        ROOT / "v4-pre-generation-correction/source-manifest.json"
    ).read_bytes()
    assert hashlib.sha256(original_manifest).hexdigest() == (
        "291cac4d77050c7f90e5f4da2d6a296af3a979813cd7066abff9288e27a0e804"
    )
    correction = read(ROOT / "v4-pre-generation-correction/correction-receipt.json")
    assert correction["generation_requests_before_correction"] == 0
    assert correction["startup_model_reloaded"] is False
    original_campaign = (
        ROOT / "v4-pre-generation-correction/campaign81.py.source"
    ).read_bytes()
    assert (
        hashlib.sha256(original_campaign).hexdigest()
        == (json.loads(original_manifest)["files"]["campaign81.py"])
        == correction["prior_campaign_sha256"]
    )
    admission_path = ROOT / "native05-admission-v1/actual-admission.json"
    if admission_path.is_file():
        admission_raw = admission_path.read_bytes()
        assert hashlib.sha256(admission_raw).hexdigest() == (
            "d6d46e552b8b60e1d5450f04be1c062ebf6db213fb3a32cd594f5c7d40e0e9a6"
        )
        actual = json.loads(admission_raw)
        assert actual["model_generation_requests"] == 0
        assert actual["tokenizations_passed"] == 128
        assert actual["binding_requested_default_true"] is True
        assert actual["binding_warning_or_skip_seen"] is False
        assert len(actual["actual_worker_thread_affinity"]) == 2
        assert len(actual["controlled_originals_manifest"]) == 139
        assert len(actual["telemetry_numeric"]) == 57
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
    full = (
        runpy.run_path(str(ROOT / "verify_full_five.py"))["verify"](
            ROOT / "five-results-v1"
        )
        if (ROOT / "five-results-v1").is_dir()
        else None
    )
    return {
        "public_files_verified": len(manifest["files"]),
        "c8_numeric_requests_recomputed": count,
        "default_mod_frozen_source_files_verified": frozen_sources,
        "zero_generation_native04_failure_receipt_verified": True,
        "actual_native05_zero_generation_admission_verified": admission_path.is_file(),
        "analysis_only_frozen_members_verified": len(analysis_sources),
        "zero_generation_pressure_admission_source_members_verified": len(
            pressure_sources
        ),
        "supplemental_fixed_pressure_source_members_verified": len(
            pressure_driver_sources
        ),
        "actual_zero_generation_pressure_capacity": pressure_capacity,
        "actual_zero_generation_pressure_capacity_smaller_v2": pressure_capacity_v2,
        "smaller_pressure_v2_source_members_verified": len(pressure_v2_sources),
        "default_true_c8": current_c8,
        "closed_solver_controlled_sse_audits_indexed": closed_solver_audits,
        "original_sse_publicly_read_back": False,
        "controlled_sse_independent_verification_receipts_indexed": True,
        "aborted_cohort_receipt_counts_verified": True,
        "aborted_cohort_controlled_answers_publicly_reverified": False,
        "five_dataset_formal_completion_established_by_this_capsule": bool(full),
        "full_five_numeric_and_custody_verification": full,
        "selector_mechanism_gain_established": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
