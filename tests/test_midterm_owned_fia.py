"""Recompute owned-FIA coverage and logical pairs from sealed device records."""

from collections import Counter, defaultdict
import hashlib
import json
import types

import pytest

from test_midterm_vocab_prefill import BASE, archive_members


@pytest.fixture(scope="module")
def evidence():
    return archive_members(BASE / "owned-fia-r13-r14")


def test_two_stream_qualification_and_overflow(evidence):
    plan_data = evidence["source/owned-lengths-plan-r1.json"]
    plan = json.loads(plan_data)
    receipt = json.loads(
        evidence["qualification/owned-lengths-control-r1/receipt.json"]
    )
    assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
    assert receipt["all_devices_released"] and not receipt["sampler_errors"]
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256(evidence["source/" + name]).hexdigest() == digest
    for rank in (0, 1):
        raw = json.loads(evidence[f"qualification/owned-lengths-r1/rank-{rank}.json"])
        assert raw["rank"] == rank
        assert raw["plan_sha256"] == hashlib.sha256(plan_data).hexdigest()
        for case, capacity in zip(raw["cases"], plan["capacities"], strict=True):
            capture = case["capture"]
            assert case["capacity"] == capture["capacity"] == capacity
            assert len(capture["records"]) == min(capacity, plan["iterations"])
            assert (
                case["overflow"]
                == len(capture["errors"])
                == max(0, plan["iterations"] - capacity)
            )
            assert all(e["error_type"] == "OverflowError" for e in capture["errors"])
            for i, record in enumerate(capture["records"]):
                q = [i * 100 + j + 1 for j in range(plan["rows"])]
                kv = [v + 1000 + rank * 10000 for v in q]
                assert record["slot"] == i
                assert record["device_query_lengths"] == q
                assert record["device_kv_lengths"] == kv


def test_failed_standard_hook_is_not_accepted(evidence):
    summary = json.loads(evidence["source/r13-fia-capture-gap.json"])
    assert summary["performance_publication"] is False
    assert summary["model_effective_flops"] is None
    assert summary["m01_attainment"] is None
    traces = [
        data
        for name, data in evidence.items()
        if name.startswith("microflow-token-accounting-r13/token-accounting-")
    ]
    assert len(traces) == 2
    for data in traces:
        events = [json.loads(line) for line in data.splitlines()]
        snapshots = [e for e in events if e["event"] == "fia_graph_snapshot"]
        assert len(snapshots) == 882
        assert all(e["role"] == "draft" and e["templates"] == [] for e in snapshots)


def test_actual_lengths_match_publications_and_recompute_pairs(evidence):
    prefix = "microflow-token-accounting-r14/"
    receipt = json.loads(evidence[prefix + "receipt.json"])
    summary = json.loads(evidence["source/r14-device-attention-pairs.json"])
    assert receipt["all_devices_released"]
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None
    helper = types.ModuleType("sealed_owned_fia_pairs")
    exec(
        compile(evidence["source/owned_fia_pairs.py"], "owned_fia_pairs.py", "exec"),
        helper.__dict__,
    )
    sequences = []
    for export in receipt["owned_fia_exports"]:
        pid = export["pid"]
        data = evidence[prefix + f"owned-fia-{pid}.json"]
        raw = json.loads(data)
        ready = json.loads(evidence[prefix + f"owned-fia-{pid}.ready"])
        assert ready["pid"] == pid
        assert ready["sha256"] == hashlib.sha256(data).hexdigest()
        assert ready["records"] == len(raw["records"])
        assert not raw["errors"]
        for name, digest in raw["source_sha256"].items():
            source = "token-observer-v12/hust_token_observer/" + name
            assert hashlib.sha256(evidence["source/" + source]).hexdigest() == digest
            assert receipt["input_hashes"][source] == digest
        events = [
            json.loads(line)
            for line in evidence[prefix + f"token-accounting-{pid}.jsonl"].splitlines()
        ]
        assert not any(e["event"] == "owned_fia_snapshot_error" for e in events)
        rank = next(e["rank"] for e in events if e["event"] == "observer_installed")
        published = [e for e in events if e["event"] == "owned_fia_frame_published"]
        assert len(published) == len(raw["records"]) == 2643
        totals = defaultdict(Counter)
        sequence = []
        for entry, event in zip(raw["records"], published, strict=True):
            meta = entry["metadata"]
            assert all(event[k] == v for k, v in meta.items())
            assert entry["slot"] == event["device_capture_slot"]
            assert (
                entry["host_enqueue_begin_wall"]
                <= entry["host_enqueue_end_wall"]
                <= event["wall"]
            )
            role = (
                "target"
                if meta["role"] == "target"
                else f"draft_step{meta['draft_step']}"
            )
            q = meta["host_planning_query_lengths"]["host_values"]
            kv = meta["host_planning_kv_lengths"]["host_values"]
            assert entry["device_query_lengths"][: len(q)] == q
            totals[role]["publications"] += 1
            for label, query, lengths in (
                ("device_", entry["device_query_lengths"], entry["device_kv_lengths"]),
                ("host_planning_", q, kv),
            ):
                values = helper.count_device_pairs(
                    query, lengths, len(q), meta["frame"]["tokens"]
                )
                totals[role].update({label + k: v for k, v in values.items()})
            sequence.append(
                (
                    role,
                    meta["bank"],
                    meta["tokens"],
                    entry["device_query_lengths"],
                    entry["device_kv_lengths"],
                )
            )
        expected = next(r["roles"] for r in summary["ranks"] if r["rank"] == rank)
        assert dict(totals) == expected
        assert all(values["publications"] == 881 for values in totals.values())
        sequences.append(sequence)
    assert len(sequences) == 2
    assert sequences[0] == sequences[1]
