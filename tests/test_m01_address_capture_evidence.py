from collections import Counter
import json
from pathlib import Path
import tarfile

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/diagnostic-capture-20261023"
)


def test_stopped_capture_is_not_a_completed_workload():
    with tarfile.open(BASE / "m01-capture-r5-stopped-20261023.tar.gz") as bundle:
        receipt = json.loads(
            bundle.extractfile("m01-capture-b1-r5/receipt.json").read()
        )
        assert receipt["status"] == "failed" and receipt["error"].startswith(
            "KeyboardInterrupt"
        )
        assert "client_pid" not in receipt
        assert receipt["all_devices_released"] and receipt["sessions_drained"]
        assert not any(m.name.endswith("/mmlu-pro.json") for m in bundle)


def test_sixth_capture_raw_position_differences_remain_visible():
    with tarfile.open(BASE / "m01-capture-r6-evidence-20261023.tar.gz") as bundle:

        def read(name):
            return json.loads(bundle.extractfile("m01-capture-b1-r6/" + name).read())

        receipt = read("receipt.json")
        assert receipt["status"] == "diagnostic_capture_joined"
        assert receipt["client_exit_code"] == 0
        assert receipt["all_devices_released"] and receipt["sessions_drained"]
        assert (
            receipt["workload"]["completed"] == 200
            and receipt["workload"]["failed"] == 0
        )
        assert receipt["effective_flops"] is None and receipt["m01_attainment"] is None
        scheduler = read("scheduler-961957.json")
        frames = {
            e["frame"]: e["identity"]["wave"]
            for e in scheduler["events"]
            if e["event"] == "delivery"
        }
        previous, rejections = {}, {}
        for e in scheduler["events"]:
            if e["event"] not in ("output_update", "not_updated"):
                continue
            key = e["request_key"]
            previous[frames[e["frame"]], key] = rejections.get(key, 0)
            if e["event"] == "output_update":
                rejections[key] = (
                    e["drafts"] + 1 - e["observation"]["raw_count"]
                    if e["drafts"]
                    else 0
                )
        workers = [read(f"worker-{pid}.json") for pid in (962244, 962245)]
    left, right = [w["draft_device_inputs"]["records"] for w in workers]
    changes = Counter()
    for a, b in zip(left, right, strict=True):
        assert a["metadata"] == b["metadata"]
        for field in a["values"]:
            if a["values"][field] != b["values"][field]:
                changes[field] += 1
    assert changes == {"first_positions": 70, "step1_slots": 70}
    for worker, expected_rows, expected_tokens in zip(
        workers, (4018, 3997), (12054, 11991), strict=True
    ):
        data = worker["draft_device_inputs"]
        assert data["complete"] and not data["errors"]
        assert len(data["records"]) == 2649 and data["integers_used"] == 2656292
        assert data["stream_order_qualified"] is False
        tables = {
            (r["metadata"]["identity"]["wave"], r["metadata"]["draft_step"]): r[
                "values"
            ]["block_table"]
            for r in data["records"]
            if r["metadata"]["kind"] == "owned_read_table"
        }
        lengths = {
            (r["metadata"]["identity"]["wave"], r["metadata"]["draft_step"]): r
            for r in worker["draft_device_lengths"]["records"]
        }
        assert len(lengths) == 1766
        nonzero_rows = nonzero_tokens = 0
        for record in data["records"]:
            m, value = record["metadata"], record["values"]
            if m["kind"] != "draft_inputs":
                continue
            wave = m["identity"]["wave"]
            axes = value["first_positions"]
            assert all(axis == axes[0] for axis in axes)
            positions, start = axes[0], 0
            for row, key in enumerate(m["request_keys"]):
                end = lengths[wave, 0]["device_query_lengths"][row]
                kv = lengths[wave, 0]["device_kv_lengths"][row]
                blocks = tables[wave, 0][row][: (kv + 127) // 128]
                inverse = {block: i for i, block in enumerate(blocks)}
                assert len(inverse) == len(blocks)
                delta = set()
                for index in range(start, end):
                    slot = value["step0_slots"][index]
                    logical = inverse[slot // 128] * 128 + slot % 128
                    delta.add(positions[index] - logical)
                assert len(delta) == 1
                if delta != {0}:
                    assert delta == {previous[wave, key]}
                    nonzero_rows += 1
                    nonzero_tokens += end - start
                position = positions[value["sample_indices"][row]] + 1
                assert (
                    value["step1_slots"][row]
                    == tables[wave, 1][row][position // 128] * 128 + position % 128
                )
                start = end
        assert (nonzero_rows, nonzero_tokens) == (expected_rows, expected_tokens)
