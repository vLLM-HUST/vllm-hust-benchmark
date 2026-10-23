from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/diagnostic-capture-20261023"
)


def test_fourth_capture_previous_batch_producer_mapping():
    with tarfile.open(BASE / "m01-r4-draft-producers-20261023.tar.gz") as bundle:
        report = json.loads(bundle.extractfile("m01-r4-draft-producers.json").read())
    with tarfile.open(BASE / "m01-capture-r4-evidence-20261023.tar.gz") as bundle:

        def read(name):
            return json.loads(bundle.extractfile("m01-capture-b1-r4/" + name).read())

        scheduler = read("scheduler-894022.json")
        frames = {
            e["frame"]: e["identity"]["wave"]
            for e in scheduler["events"]
            if e["event"] == "delivery"
        }
        updates = {
            (frames[e["frame"]], e["request_key"]): e
            for e in scheduler["events"]
            if e["event"] in ("output_update", "not_updated")
        }
        for pid in (894394, 894395):
            worker = read(f"worker-{pid}.json")
            waves = [
                e["identity"]["wave"]
                for e in worker["events"]
                if e["event"] == "sample_begin"
            ]
            assert waves == sorted(set(waves))
            previous = dict(zip(waves[1:], waves[:-1]))
            slots = {}
            for record in worker["draft_device_lengths"]["records"]:
                m = record["metadata"]
                for row, key in enumerate(m["request_keys"]):
                    identity = (m["identity"]["wave"], key, m["draft_step"])
                    assert identity not in slots
                    slots[identity] = row
            used, consumers = set(), set()
            for record in report["records"]:
                producer, consumer, key = (
                    record["producer_wave"],
                    record["consumer_wave"],
                    record["request_key"],
                )
                assert producer == previous[consumer]
                assert (consumer, key) not in consumers
                consumers.add((consumer, key))
                assert updates[producer, key]["event"] == "output_update"
                event = updates[consumer, key]
                assert record["scheduled_drafts"] == event["drafts"] == 2
                assert record["consumer_status"] == event["event"]
                assert record["retained_drafts"] == event.get("observation", {}).get(
                    "retained_drafts", 0
                )
                for step in range(2):
                    identity = (producer, key, step)
                    assert (
                        slots[identity] == record["producer_row"]
                        and identity not in used
                    )
                    used.add(identity)
            assert consumers == {
                identity for identity, event in updates.items() if event["drafts"]
            }
            assert len(used) == 38428 and len(set(slots) - used) == 440
    assert len(report["records"]) == 19214
    assert sum(r["retained_drafts"] for r in report["records"]) == 32042
    assert report["effective_flops"] is None and report["reference_seconds"] is None


def test_fourth_capture_device_lengths_match_request_progress():
    with tarfile.open(BASE / "m01-capture-r4-evidence-20261023.tar.gz") as bundle:

        def read(name):
            return json.loads(bundle.extractfile("m01-capture-b1-r4/" + name).read())

        receipt = read("receipt.json")
        assert receipt["status"] == "diagnostic_capture_joined"
        assert receipt["client_exit_code"] == 0
        assert receipt["sessions_drained"] and receipt["all_devices_released"]
        scheduler = read("scheduler-894022.json")
        maps = []
        for pid in (894394, 894395):
            worker = read(f"worker-{pid}.json")
            lengths = worker["draft_device_lengths"]
            assert (
                lengths["complete"]
                and not lengths["errors"]
                and not lengths["binding_errors"]
            )
            assert lengths["stream_order_qualified"] is False
            assert len(lengths["records"]) == 1758
            rows = {}
            for r in lengths["records"]:
                m = r["metadata"]
                assert m["identity"]["run_id"] == receipt["run_id"]
                prior = 0
                for i, key in enumerate(m["request_keys"]):
                    identity = (m["identity"]["wave"], key, m["draft_step"])
                    assert identity not in rows
                    end = r["device_query_lengths"][i]
                    rows[identity] = (end - prior, r["device_kv_lengths"][i])
                    prior = end
            maps.append(rows)
        assert maps[0] == maps[1]
        frames = {
            e["frame"]: e["identity"]["wave"]
            for e in scheduler["events"]
            if e["event"] == "delivery"
        }
        prefill, output, counts = Counter(), Counter(), Counter()
        seen = set()
        retained = 0
        for e in scheduler["events"]:
            if e["event"] not in ("output_update", "not_updated"):
                continue
            key, wave = e["request_key"], frames[e["frame"]]
            if e["drafts"] == 0:
                prefill[key] += e["scheduled_tokens"]
            first, second = (wave, key, 0), (wave, key, 1)
            q0, kv0 = maps[0][first]
            q1, kv1 = maps[0][second]
            seen.update((first, second))
            assert q0 == e["scheduled_tokens"] and q1 == 1 and kv1 == kv0 + 1
            delta = kv0 - (prefill[key] + output[key] + e["drafts"])
            if e["event"] == "output_update":
                assert delta == 0
                counts["updated"] += 1
                output[key] += e["observation"]["appended_count"]
                retained += e["observation"]["retained_drafts"]
            else:
                assert delta in (0, 1)
                if delta:
                    assert output[key] == 256
                counts[f"skipped_delta_{delta}"] += 1
        assert seen == set(maps[0])
        assert counts == {
            "updated": 19292,
            "skipped_delta_0": 81,
            "skipped_delta_1": 61,
        }
        assert len(output) == 200 and set(output.values()) == {256}
        assert sum(prefill.values()) == 282265 and retained == 32042
        assert receipt["effective_flops"] is None and receipt["m01_attainment"] is None


def test_partial_arithmetic_remains_separate_from_acceptance():
    with tarfile.open(BASE / "m01-r3-accounting-progress-20261023.tar.gz") as outer:
        draft = json.loads(
            outer.extractfile("m01-r3-draft-input-candidates.json").read()
        )
        nested = outer.extractfile("m01-r3-target-catalog-20261023.tar.gz").read()
    assert draft["effective_flops"] is None
    assert len(draft["records"]) == draft["counts"]["output_update"] == 19281
    assert sum(r["rejected_rows"] for r in draft["records"]) == 6033
    for row in draft["records"]:
        assert (
            row["local_sample_index"]
            == row["scheduled_rows"] - 1 - row["rejected_rows"]
        )
        assert 0 <= row["local_sample_index"] < row["scheduled_rows"]
    with tarfile.open(fileobj=io.BytesIO(nested)) as bundle:
        target = json.loads(bundle.extractfile("m01-r3-target-catalog.json").read())
    assert (
        sum(target["parts"].values())
        == target["listed_target_subtotal"]
        == 1768655972286925
    )
    assert target["retained_draft_vocabulary_subset"] == 32610860400640
    assert target["effective_flops"] is None
    assert target["reference_time"] is None and target["efficiency_percent"] is None


def test_failed_capture_preserves_identity_and_recovered_counts():
    manifest = json.loads((BASE / "manifest.json").read_text())
    records = {}
    for item in manifest["archives"]:
        archive = BASE / item["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["sha256"]
        members = {}
        with tarfile.open(archive) as bundle:
            for member in bundle:
                assert (
                    not Path(member.name).is_absolute()
                    and ".." not in Path(member.name).parts
                )
                assert member.isfile() or member.isdir()
                if member.isfile():
                    raw = bundle.extractfile(member).read()
                    members[member.name] = hashlib.sha256(raw).hexdigest()
                    if member.name.startswith("m01-capture-b1-r1/") or member.name in (
                        "capture-r1-host-count-recovery.json",
                        "capture-r1-skipped-row-classification.json",
                    ):
                        records[member.name] = raw
        assert members == item["members"]
    prefix = "m01-capture-b1-r1/"
    receipt = json.loads(records[prefix + "receipt.json"])
    assert receipt["status"] == "failed" and receipt["client_exit_code"] == 1
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert prefix + "mmlu-pro.json" not in records
    ledger_raw = records[prefix + "scheduler-712941.json"]
    ledger = json.loads(ledger_raw)
    recovery = json.loads(records["capture-r1-host-count-recovery.json"])
    boundary = json.loads(records[prefix + "boundary.jsonl"])
    assert (
        recovery["low_perf_ns"] == boundary["calls"][2]["clock_after"]["perf_after_ns"]
    )
    assert (
        recovery["high_perf_ns"]
        == boundary["calls"][3]["clock_before"]["perf_before_ns"]
    )
    assert (
        recovery["source_sha256"]["/workspace/m01-capture-b1-r1/scheduler-712941.json"]
        == hashlib.sha256(ledger_raw).hexdigest()
    )
    events = [
        e
        for e in ledger["events"]
        if recovery["low_perf_ns"] <= e["perf_ns"] <= recovery["high_perf_ns"]
    ]
    updates = [e for e in events if e["event"] == "output_update"]
    per_request = Counter()
    for event in updates:
        per_request[event["request_key"]] += event["observation"]["appended_count"]
    assert len(per_request) == 200 and set(per_request.values()) == {256}
    assert sum(per_request.values()) == recovery["appended_output_tokens"] == 51200
    for field in ("raw_accepted_drafts", "retained_drafts"):
        assert sum(e["observation"][field] for e in updates) == recovery[field]
    assert recovery["raw_accepted_drafts"] - recovery["retained_drafts"] == 56
    skipped = json.loads(records["capture-r1-skipped-row-classification.json"])
    assert skipped["source_sha256"] == hashlib.sha256(ledger_raw).hexdigest()
    assert skipped["original_retained_count_complete"] is False
    assert skipped["classes"] == {
        "before_first_output_no_drafts": 16,
        "after_complete_output_budget": 124,
    }
    prior, classification = Counter(), Counter()
    for event in ledger["events"]:
        if event["event"] == "output_update":
            prior[event["request_key"]] += event["observation"]["appended_count"]
        elif event["event"] == "not_updated":
            state = (event["drafts"], prior[event["request_key"]])
            assert state in ((0, 0), (2, 256))
            classification[
                "before_first_output_no_drafts"
                if state == (0, 0)
                else "after_complete_output_budget"
            ] += 1
    assert dict(classification) == skipped["classes"]
    for record in (manifest, recovery, skipped):
        assert record["effective_flops"] is None and record["m01_attainment"] is None
    assert manifest["performance_acceptance"] is False


def test_retry_preserves_responses_without_accepting_invalid_metadata():
    archive = BASE / "m01-capture-r2-evidence-20261023.tar.gz"
    with tarfile.open(archive) as bundle:

        def read(name):
            return json.loads(bundle.extractfile("m01-capture-b1-r2/" + name).read())

        receipt = read("receipt.json")
        checkpoint = read("request-checkpoint.json")
        assert receipt["status"] == "failed"
        assert receipt["client_exit_code"] == 1
        assert receipt["sessions_drained"] and receipt["all_devices_released"]
        assert checkpoint["run_id"] == receipt["run_id"] == "capture-b1-r2"
        requests = checkpoint["requests"]
        assert len(requests) == 200
        assert all(row["output"]["success"] is True for row in requests)
        assert sum(row["output"]["output_tokens"] for row in requests) == 51200
        assert all(isinstance(row["output"]["itl"], list) for row in requests)
        assert checkpoint["m01_attainment"] is None
        for rank, pid in enumerate((747955, 747956)):
            worker = read(f"worker-{pid}.json")
            assert worker["rank"] == rank and worker["run_id"] == receipt["run_id"]
            events = Counter(event["event"] for event in worker["events"])
            assert events["execute_begin"] == events["execute_end"] == 882
            assert events["sample_begin"] == events["sample_end"] == 881
            assert events["host_gdn_metadata_invalid"] == worker["errors"] == 881
            assert worker["complete"] is False and worker["dropped"] == 0
            assert worker["pending"] is False
        assert not any(
            Path(m.name).name.startswith("scheduler-") for m in bundle.getmembers()
        )


def test_third_capture_closes_metadata_not_m01_acceptance():
    with tarfile.open(BASE / "m01-capture-r3-derived-20261023.tar.gz") as bundle:
        summary = json.loads(
            bundle.extractfile("m01-capture-r3-derived-summary.json").read()
        )
    with tarfile.open(BASE / "m01-capture-r3-evidence-20261023.tar.gz") as bundle:

        def read(name):
            raw = bundle.extractfile("m01-capture-b1-r3/" + name).read()
            assert hashlib.sha256(raw).hexdigest() == summary["source_sha256"][name]
            return json.loads(raw)

        receipt = read("receipt.json")
        assert receipt["status"] == "diagnostic_capture_joined"
        assert receipt["client_exit_code"] == 0
        assert receipt["sessions_drained"] and receipt["all_devices_released"]
        result = read("mmlu-pro.json")
        assert result["completed"] == 200 and result["failed"] == 0
        assert result["total_input_tokens"] == 282265
        assert result["total_output_tokens"] == 51200
        boundary = read("boundary.jsonl")
        assert boundary["performance_acceptance"] is False
        low = boundary["calls"][2]["clock_after"]["perf_after_ns"]
        high = boundary["calls"][3]["clock_before"]["perf_before_ns"]
        scheduler = read("scheduler-805337.json")
        assert scheduler["retained_count_complete"] is False
        events = [e for e in scheduler["events"] if low <= e["perf_ns"] <= high]
        outputs, skipped = Counter(), Counter()
        updates = []
        for e in events:
            if e["event"] == "output_update":
                obs = e["observation"]
                assert obs["status"] == "accepted_prefix_then_bonus_host_counts"
                assert not obs["is_stale"]
                outputs[e["request_key"]] += obs["appended_count"]
                updates.append(obs)
            elif e["event"] == "not_updated":
                skipped[(e["drafts"], outputs[e["request_key"]])] += 1
        assert len(outputs) == 200 and set(outputs.values()) == {256}
        assert skipped == {(0, 0): 18, (2, 256): 110}
        for key in ("raw_accepted_drafts", "retained_drafts"):
            assert sum(o[key] for o in updates) == summary[key]
        assert summary["raw_accepted_drafts"] - summary["retained_drafts"] == 67
        for rank, pid in enumerate((805910, 805911)):
            worker = read(f"worker-{pid}.json")
            assert worker["rank"] == rank
            assert worker["run_id"] == scheduler["run_id"] == receipt["run_id"]
            assert worker["complete"] and worker["errors"] == worker["dropped"] == 0
            counts = Counter(e["event"] for e in worker["events"])
            assert counts["execute_begin"] == counts["execute_end"] == 874
            assert (
                counts["host_gdn_metadata"]
                == counts["sample_begin"]
                == counts["sample_end"]
                == 873
            )
            for e in worker["events"]:
                if e["event"] == "host_gdn_metadata":
                    layers = [name for g in e["groups"] for name in g["layers"]]
                    assert len(layers) == len(set(layers)) == 30
                    assert all(g["live"] == len(e["request_keys"]) for g in e["groups"])
        assert summary["effective_flops"] is None
        assert summary["reference_time"] is None and summary["m01_attainment"] is None
