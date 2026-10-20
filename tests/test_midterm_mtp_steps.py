"""Reconstruct host MTP extents from archived events, not summary constants."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/token-accounting-r2"
)


def test_archived_step_counts_and_lifecycle():
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
    summary = json.loads((ROOT / "step-summary.json").read_text())
    assert summary == json.loads(
        members["source/token-accounting-step-summary-r2.json"]
    )
    prefix = "microflow-token-accounting-r2/"
    receipt = json.loads(members[prefix + "receipt.json"])
    assert receipt["all_devices_released"] and receipt["status"] == "measured"
    assert receipt["bench_exit_code"] == 0
    assert summary["completed"] == 200 and summary["failed"] == 0
    assert summary["model_effective_flops"] is None
    assert summary["attainment_percent"] is None
    assert {rank["rank"] for rank in summary["ranks"]} == {0, 1}
    assert summary["ranks"][0]["roles"] == summary["ranks"][1]["roles"]
    tail = summary["request_phase"]["end_wall"]
    for rank in summary["ranks"]:
        events = [
            json.loads(line) for line in members[prefix + rank["file"]].splitlines()
        ]
        opened, selected = {}, []
        for event in events:
            if event["event"] == "observer_installed":
                assert event["rank"] == rank["rank"]
                assert (
                    event["observer_sha256"]
                    == receipt["input_hashes"][
                        "token-observer-v2/hust_token_observer/__init__.py"
                    ]
                )
            elif event["event"] == "forward_enter":
                assert event["role"] not in opened
                opened[event["role"]] = event
            elif event["event"] == "forward_exit":
                entry = opened.pop(event["role"])
                assert entry["wall"] <= event["wall"]
                assert entry["actual_tokens"] == event["actual_tokens"]
                if (
                    summary["request_phase"]["start_wall"]
                    <= entry["wall"]
                    <= receipt["bench_ended_wall"]
                ):
                    selected.append(entry)
                    tail = max(tail, event["wall"])
        assert not opened
        for role in ("target", "draft"):
            entries = [entry for entry in selected if entry["role"] == role]
            recorded = rank["roles"][role]
            assert len(entries) == recorded["contexts"]
            assert (
                sum(entry["actual_tokens"] for entry in entries)
                == recorded["context_actual_tokens"]
            )
            assert (
                sum(entry["padded_tokens"] for entry in entries)
                == recorded["context_padded_tokens"]
            )
            if role == "draft":
                assert all(len(entry["draft_steps"]) == 2 for entry in entries)
                for index, step in enumerate(recorded["steps"]):
                    counts = []
                    for entry in entries:
                        observed = entry["draft_steps"][index]
                        values = [
                            layer["scalars"]["num_actual_tokens"]
                            for layer in observed["layers"]
                        ]
                        assert values and all(value == values[0] for value in values)
                        assert observed["actual_tokens"] == values[0]
                        counts.append(values[0])
                    assert step["unknown_contexts"] == 0
                    assert sum(counts) == step["actual_tokens"]
                assert (
                    recorded["steps"][0]["actual_tokens"]
                    != recorded["steps"][1]["actual_tokens"]
                )
    assert tail == summary["compute_tail_end_wall"]
    assert tail <= receipt["bench_ended_wall"]
    assert summary["compute_tail_seconds"] > 0
