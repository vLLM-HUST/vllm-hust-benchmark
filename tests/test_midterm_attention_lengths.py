"""Validate length equality and causal pair totals from raw archived events."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/attention-length-r5"
)


def test_length_archive_and_independent_pair_reconstruction():
    manifest = json.loads((ROOT / "archive.json").read_text())
    parts = []
    for part in manifest["parts"]:
        data = (ROOT / part["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        parts.append(data)
    payload = b"".join(parts)
    assert hashlib.sha256(payload).hexdigest() == manifest["concatenated_sha256"]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert set(archive.getnames()) == set(manifest["members"])
        members = {
            name: archive.extractfile(name).read() for name in archive.getnames()
        }
    for name, digest in manifest["members"].items():
        assert hashlib.sha256(members[name]).hexdigest() == digest
    lengths = json.loads((ROOT / "length-summary.json").read_text())
    pairs = json.loads((ROOT / "pairs-summary.json").read_text())
    partial = json.loads((ROOT / "partial-operators.json").read_text())
    for name, summary in [
        ("attention-length-summary-r5.json", lengths),
        ("attention-pairs-summary-r5.json", pairs),
        ("partial-operator-accounting-r5.json", partial),
    ]:
        assert summary == json.loads(members["source/" + name])
        assert summary["model_effective_flops"] is None
    prefix = "microflow-token-accounting-r5/"
    receipt = json.loads(members[prefix + "receipt.json"])
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert lengths["completed"] == 200 and lengths["failed"] == 0
    failed = json.loads(members["microflow-token-accounting-r4/receipt.json"])
    assert failed["status"] == "failed" and failed["child_exit_codes"] == []
    assert failed["all_devices_released"]
    assert {rank["rank"] for rank in lengths["ranks"]} == {0, 1}
    for rank in lengths["ranks"]:
        events = [
            json.loads(line) for line in members[prefix + rank["file"]].splitlines()
        ]
        gate = rank["runtime_gate"]
        assert gate["needs_seq_lens_cpu_sync"] and gate["use_async_spec_decode"]
        assert (
            gate["source_sha256"]
            == receipt["input_hashes"][
                "token-observer-v4/hust_token_observer/lengths.py"
            ]
        )
        selected = [
            e
            for e in events
            if lengths["request_phase"]["start_wall"]
            <= e["wall"]
            <= receipt["bench_ended_wall"]
        ]
        snapshots = [e for e in selected if e["event"] == "attention_length_snapshot"]
        assert len(snapshots) == rank["snapshots"]
        assert (
            sum(e["synchronization_seconds"] for e in snapshots)
            == rank["synchronization_seconds"]
        )
        comparisons = 0
        pending, totals = {}, {}
        for event in selected:
            if event["event"] == "attention_length_snapshot":
                assert event["role"] not in pending
                pending[event["role"]] = event
                groups = event["metadata"] + [
                    group for step in (event["draft_steps"] or []) for group in step
                ]
                for group in groups:
                    assert group["cpu_seq_lens"] == group["device_seq_lens"]
                    assert (
                        group["cpu_cumulative_query_lengths"]
                        == group["device_query_start_loc"][1:]
                    )
                    comparisons += 2
            elif event["event"] == "forward_enter":
                snapshot = pending.pop(event["role"])
                if event["role"] == "target":
                    cases = [("target", snapshot["metadata"], event["actual_tokens"])]
                else:
                    cases = [
                        (
                            f"draft_step{i}",
                            groups,
                            event["draft_steps"][i]["actual_tokens"],
                        )
                        for i, groups in enumerate(snapshot["draft_steps"])
                    ]
                for role, groups, actual in cases:
                    assert len(groups) == 1
                    group = groups[0]
                    q_ends, kvs = (
                        group["cpu_cumulative_query_lengths"],
                        group["cpu_seq_lens"],
                    )
                    assert len(q_ends) == len(kvs)
                    counted = padded = pair_count = start = 0
                    for end, kv in zip(q_ends, kvs, strict=True):
                        assert end >= start
                        q = max(0, min(end, actual) - min(start, actual))
                        assert kv >= q
                        pair_count += sum(range(kv - q + 1, kv + 1))
                        counted += q
                        padded += end - start - q
                        start = end
                    assert counted == actual
                    entry = totals.setdefault(
                        role,
                        {
                            "contexts": 0,
                            "causal_pairs": 0,
                            "valid_query_tokens": 0,
                            "excluded_padded_query_tokens": 0,
                        },
                    )
                    entry["contexts"] += 1
                    entry["causal_pairs"] += pair_count
                    entry["valid_query_tokens"] += counted
                    entry["excluded_padded_query_tokens"] += padded
        assert not pending
        expected = next(r for r in pairs["ranks"] if r["rank"] == rank["rank"])
        assert totals == expected["roles"]
        assert (
            comparisons
            == rank["comparison_groups"]
            == sum(rank["comparison_counts"].values())
        )
        assert all(key.endswith(":equal") for key in rank["comparison_counts"])
    assert partial["covered_semantic_subtotal"] == (
        sum(e["semantic_flops"] for e in partial["projections"])
        + sum(partial["causal_qk_av_semantic_flops"].values())
    )
    assert partial["excluded"] and partial["attainment_percent"] is None
