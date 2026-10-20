"""Verify r9 device rows and same-request-interval hardware from archived data."""

from collections import Counter
import hashlib
import importlib.util
import json
import sqlite3

import pytest

from test_midterm_vocab_prefill import BASE, archive_members

DIRECTORY = BASE / "device-routing-r9"


@pytest.fixture(scope="module")
def members():
    return archive_members(DIRECTORY)


def test_graph_recorder_qualification(members):
    receipt = json.loads(members["moe-capture-control-r1/receipt.json"])
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"]
    for rank in (0, 1):
        raw = json.loads(members[f"moe-capture-r1/rank-{rank}.json"])
        assert raw["status"] == "qualified_routing_graph_recorder"
        assert len(raw["cases"]) == 6 and len(raw["records"]) == 4
        expected_records = []
        for index, case in enumerate(raw["cases"]):
            assert len(case["topk_ids"]) == 32
            assert all(len(row) == len(set(row)) == 8 for row in case["topk_ids"])
            counts = Counter(n for row in case["topk_ids"] for n in row)
            expected = [counts[i] for i in range(256)]
            assert expected == case["actual_counts"] == case["expected_counts"]
            assert sum(expected) == 256 and case["cursor"] == index + 1
            expected_records.append(expected)
        assert raw["records"] == expected_records[:4] and raw["overflow_detected"]


def test_actual_model_rows_are_not_graph_padding(members):
    summary = json.loads((DIRECTORY / "routing.json").read_text())
    catalog = json.loads((DIRECTORY / "same-run-catalog.json").read_text())
    receipt = json.loads(members["microflow-token-accounting-r9/receipt.json"])
    result = json.loads(members["microflow-token-accounting-r9/mmlu-pro.json"])
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert result["completed"] == 200 and result["failed"] == 0
    assert summary["status"] == "device_rows_audited"
    phase = summary["request_phase"]
    total_semantics = 0
    for rank in summary["ranks"]:
        data = members["microflow-token-accounting-r9/" + rank["file"]]
        assert hashlib.sha256(data).hexdigest() == rank["source_sha256"]
        events = [json.loads(line) for line in data.splitlines()]
        installed = next(
            e for e in events if e["event"] == "moe_device_capture_installed"
        )
        assert (
            installed["capacity_per_context"] == 8 and installed["group_list_type"] == 1
        )
        assert (
            installed["kernel_sha256"]
            == hashlib.sha256(
                members["token-observer-v8/hust_token_observer/moe_device_capture.py"]
            ).hexdigest()
        )
        assert (
            installed["runtime_source_sha256"]
            == hashlib.sha256(
                members["source/runtime/vllm_ascend/ops/fused_moe/routed_experts.py"]
            ).hexdigest()
        )
        counts, modes, histogram = Counter(), Counter(), Counter()
        selected = [
            e
            for e in events
            if e["event"] == "moe_device_counts"
            and phase["start_wall"] <= e["wall"] <= phase["end_wall"]
        ]
        for event in selected:
            role = event["role"]
            expected_layers = {
                row["name"] for row in installed["layers"] if row["role"] == role
            }
            assert {layer["name"] for layer in event["layers"]} == expected_layers
            assert len(expected_layers) == (40 if role == "target" else 1)
            counts[role + "_contexts"] += 1
            valid = (
                [event["actual_tokens"]]
                if role == "target"
                else [s["actual_tokens"] for s in event["draft_steps"]]
            )
            for layer in event["layers"]:
                assert layer["calls"] == len(layer["group_counts"]) == len(valid)
                for rows, n in zip(layer["group_counts"], valid):
                    assert len(rows) == 256 and all(
                        isinstance(v, int) and v >= 0 for v in rows
                    )
                    device_rows = sum(rows)
                    assert device_rows in (n * 8, event["padded_tokens"] * 8)
                    mode = "valid_rows" if device_rows == n * 8 else "padded_rows"
                    modes[role + ":" + mode] += 1
                    counts[role + "_gmm1_calls"] += 1
                    counts[role + "_device_rows"] += device_rows
                    counts[role + "_host_valid_rows"] += n * 8
                    histogram.update(rows)
        assert dict(counts) == rank["totals"] and dict(modes) == rank["row_modes"]
        assert dict(histogram) == {
            int(k): v for k, v in rank["per_expert_rows_histogram"].items()
        }
        per_row = 2 * (2048 * 512 + 256 * 2048)
        useful = (
            sum(counts[r + "_host_valid_rows"] for r in ("target", "draft")) * per_row
        )
        physical = (
            sum(counts[r + "_device_rows"] for r in ("target", "draft")) * per_row
        )
        assert useful == rank["host_valid_routed_projection_flops"] < physical
        assert physical == rank["device_routed_projection_flops_including_padding"]
        total_semantics += useful
    assert total_semantics == catalog["routed_projection_host_valid_semantic_flops"]
    assert (
        sum(catalog["components"].values())
        == catalog["covered_basic_semantic_subtotal"]
    )
    assert (
        catalog["model_effective_flops"] is None
        and catalog["attainment_percent"] is None
    )


def test_same_interval_hbm_and_communication(members, monkeypatch, tmp_path):
    report = BASE.parent
    monkeypatch.syspath_prepend(str(report))
    spec = importlib.util.spec_from_file_location(
        "r9_hardware", report / "recover_model_hardware.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hbm = json.loads((DIRECTORY / "hbm.json").read_text())
    hccs = json.loads((DIRECTORY / "hccs.json").read_text())
    catalog = json.loads((DIRECTORY / "same-run-catalog.json").read_text())
    receipt = json.loads(members["microflow-token-accounting-r9/receipt.json"])
    result = json.loads(members["microflow-token-accounting-r9/mmlu-pro.json"])
    phase = module.request_phase(receipt, result)
    assert phase == hbm["phase"] == hccs["request_phase"] == catalog["request_phase"]
    assert hbm["status"] == "complete_hbm_coverage"
    for device in hbm["devices"]:
        index = device["channels"][0]["device"]
        name = next(
            name for name in members if name.endswith(f"device_{index}/sqlite/hbm.db")
        )
        assert hashlib.sha256(members[name]).hexdigest() == device["sha256"]
        path = tmp_path / f"hbm-{index}.db"
        path.write_bytes(members[name])
        with sqlite3.connect(path) as connection:
            samples = connection.execute(
                "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
            ).fetchall()
        calculated = module.integrate(
            samples, phase["start_wall"] * 1e9, phase["end_wall"] * 1e9
        )
        assert module.has_complete_hbm_coverage(calculated)
        assert calculated["prorated_bytes"] == pytest.approx(
            device["prorated_bytes"], rel=1e-12
        )
    assert catalog["hardware"]["hbm_prorated_bytes"] == hbm["total_prorated_bytes"]
    raw = [
        json.loads(line)
        for line in members["r9-system-control/hccs64.jsonl"].splitlines()
    ]
    for device in hccs["devices"]:
        samples = [r for r in raw if r["card"] == device["card"]]
        for bound, interval in device["intervals"].items():
            start = next(
                r
                for r in samples
                if r["before"]["wall_ns"] == interval["start_before_wall_ns"]
            )
            end = next(
                r
                for r in samples
                if r["before"]["wall_ns"] == interval["end_before_wall_ns"]
            )
            if bound == "lower":
                assert start["before"]["wall_ns"] >= phase["start_wall"] * 1e9
                assert end["after"]["wall_ns"] <= phase["end_wall"] * 1e9
            else:
                assert start["after"]["wall_ns"] <= phase["start_wall"] * 1e9
                assert end["before"]["wall_ns"] >= phase["end_wall"] * 1e9
            for direction in ("tx", "rx"):
                assert interval[direction + "_flits"] == [
                    y - x
                    for x, y in zip(
                        start[direction + "_counts"], end[direction + "_counts"]
                    )
                ]
    for bound in ("lower", "upper"):
        assert (
            hccs["aggregate_tx_link_bytes"][bound]
            == sum(sum(d["intervals"][bound]["tx_flits"]) for d in hccs["devices"]) * 20
        )
    assert catalog["hardware"]["tx_link_byte_bounds"] == hccs["aggregate_tx_link_bytes"]
