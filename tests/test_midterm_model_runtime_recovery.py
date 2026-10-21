"""Portable consistency checks for recovered same-workload measurements."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/model-runtime-20261021"
)


@pytest.fixture(scope="module")
def evidence():
    index = json.loads((BASE / "archive.json").read_text())
    packed = (BASE / index["archive"]).read_bytes()
    assert len(packed) == index["archive_bytes"]
    assert hashlib.sha256(packed).hexdigest() == index["archive_sha256"]
    expected = {row["name"]: row for row in index["members"]}
    assert len(expected) == len(index["members"])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in result
            data = archive.extractfile(member).read()
            assert len(data) == expected[member.name]["size"]
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            result[member.name] = data
    assert result.keys() == expected.keys()
    return index, result


def test_requests_configuration_and_fresh_caches(evidence):
    index, files = evidence
    assert len(index["runs"]) == 2
    commands, caches = [], []
    for row in index["runs"]:
        run = row["run"]
        receipt = json.loads(files[f"{run}/receipt.json"])
        result = json.loads(files[f"{run}/mmlu-pro.json"])
        cache = json.loads(files[f"{run}/compiler-cache-initialization.json"])
        commands.append(json.loads(files[f"{run}/launch-command.json"])[2:])
        caches.append(cache["destination"])
        assert (
            cache["initial_entries"] == []
            and cache["startup_excluded_from_measurement"]
        )
        assert receipt["status"] == "measured" and receipt["all_devices_released"]
        assert (
            receipt["bench_exit_code"] == receipt["hardware_collector_exit_code"] == 0
        )
        assert receipt["allocated_cards"] == 2 and receipt["reserved_cards"] == 4
        assert receipt["simultaneous_other_services"] == 0
        assert (
            receipt["input_hashes"]["materialized.200.jsonl"]
            == "fb79b6fa98f232282a948e4a17f404f80c03dc2bae05af57b95b9cc0651b3506"  # pragma: allowlist secret
        )
        assert result["completed"] == row["completed"] == 200
        assert result["failed"] == row["failed"] == 0
        assert result["total_output_tokens"] == row["output_tokens"] == 51200
        assert row["instrumented_output_tokens_per_second"] == pytest.approx(
            result["total_output_tokens"] / result["duration"]
        )
    assert commands[0] == commands[1]
    assert len(set(caches)) == len(caches)
    assert [row["hardware_sampling_seconds"] for row in index["runs"]] == [240, 360]
    for row in index["runs"]:
        samples = [
            json.loads(data)
            for name, data in files.items()
            if name.startswith(f"{row['run']}/system/profiler/")
            and name.endswith("/sample.json")
        ]
        assert len(samples) == 2
        assert {sample["devices"] for sample in samples} == {"0", "1"}
        assert all(
            sample["profiling_period"] == row["hardware_sampling_seconds"]
            for sample in samples
        )


def test_interval_coverage_and_no_false_admission(evidence):
    index, files = evidence
    assert index["full_model_inference_completed"]
    assert not index["original_runtime_binary_identity_proven"]
    assert not index["formal_b1_admission"]
    assert index["model_effective_flops"] is None and index["m01_attainment"] is None
    for row in index["runs"]:
        hardware = json.loads(files[row["hardware"]])
        ledger = json.loads(files[row["operation_ledger"]])
        assert (
            ledger["accounted_basic_operation_subtotal"]
            == row["basic_operation_subtotal"]
        )
        assert (
            sum(item["global_basic_operations"] for item in ledger["families"])
            == row["basic_operation_subtotal"]
        )
        assert len(ledger["families"]) == 10
        assert (
            ledger["model_effective_flops"] is None and ledger["m01_attainment"] is None
        )
        for window in hardware["windows"]:
            covered = all(
                device["minimum_channel_coverage"] == 1
                for device in window["hbm_devices"]
            )
            if not covered:
                assert hardware["status"] == "partial_interval_coverage_not_full_m01"
                assert window["hbm_bytes"] is None
            else:
                assert window["hbm_bytes"] is not None
        if row["run"].endswith("r7"):
            assert (
                hardware["status"]
                == "same_run_named_interval_counter_coverage_verified"
            )
            assert all(
                device["minimum_channel_coverage"] == 1
                for window in hardware["windows"]
                for device in window["hbm_devices"]
            )


def test_recovery_compiler_and_raw_archive_locators(evidence):
    index, files = evidence
    compiler = json.loads(files["mamba-import-recovery-r3.json"])
    assert compiler["status"] == "mamba_abi_import_passed"
    assert compiler["vllm_has_triton"] and compiler["backends"]["ascend"]["active"]
    assert "m01-triton-recovery-dev20261007-r2" in compiler["selected_path"]
    for row in index["runs"]:
        archive = row["raw_archive"]
        manifest = json.loads(files[archive["member_manifest"]])
        assert archive["bytes"] == manifest["bytes"]
        assert archive["sha256"] == manifest["sha256"]
        assert manifest["member_verification"] == "all_members_passed"
        assert archive["url"].startswith(
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
        )
        assert f"{row['run']}/mmlu-pro.json" in manifest["members"]


def test_actual_graph_and_microflow_paths(evidence):
    index, files = evidence
    for row in index["runs"]:
        run = row["run"]
        log = files[f"{run}/service.log"].decode()
        assert "m01-triton-recovery-dev20261007-r2/triton/" in log
        assert "Graph capturing finished" in log and "runtime_mode=FULL" in log
        assert (
            "KV_MATERIALIZATION_PLUGIN_REGISTERED host_api=school028-recompute-only"
            in log
        )
        ranks = set()
        journals = [
            name
            for name in files
            if name.startswith(f"{run}/microflow-") and name.endswith(".jsonl")
        ]
        assert len(journals) == 2
        for name in journals:
            events = [json.loads(line) for line in files[name].splitlines()]
            loaded = [event for event in events if event["event"] == "model_loaded"]
            assert len(loaded) == 1 and loaded[0]["active"]
            ranks.add(loaded[0]["rank"])
            checks = [
                event for event in events if event["event"] == "partition_equivalence"
            ]
            assert {event["tokens"] for event in checks} == {1024, 2048, 4096}
            assert all(all(event["exact"]) for event in checks)
            assert any(
                event["event"] == "prefix" and event["split"] and event["capturing"]
                for event in events
            )
        assert ranks == {0, 1}
