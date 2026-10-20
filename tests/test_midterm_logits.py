"""Recompute published logits observations from archived runtime events."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/token-accounting-r3"
)


def test_logits_archive_and_reconstruction():
    manifest = json.loads((ROOT / "archive.json").read_text())
    parts = []
    for part in manifest["parts"]:
        payload = (ROOT / part["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == part["sha256"]
        parts.append(payload)
    payload = b"".join(parts)
    assert hashlib.sha256(payload).hexdigest() == manifest["concatenated_sha256"]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert set(archive.getnames()) == set(manifest["members"])
        members = {
            name: archive.extractfile(name).read() for name in archive.getnames()
        }
    for name, digest in manifest["members"].items():
        assert hashlib.sha256(members[name]).hexdigest() == digest
    summary = json.loads((ROOT / "logits-summary.json").read_text())
    assert summary == json.loads(members["source/logits-accounting-summary-r3.json"])
    prefix = "microflow-token-accounting-r3/"
    receipt = json.loads(members[prefix + "receipt.json"])
    result = json.loads(members[prefix + "mmlu-pro.json"])
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    assert result["completed"] == summary["completed"] == 200
    assert result["failed"] == summary["failed"] == 0
    assert summary["model_effective_flops"] is None
    assert summary["attainment_percent"] is None
    assert {rank["rank"] for rank in summary["ranks"]} == {0, 1}
    for rank in summary["ranks"]:
        events = [
            json.loads(line) for line in members[prefix + rank["file"]].splitlines()
        ]
        installed = [e for e in events if e["event"] == "observer_installed"]
        assert len(installed) == 1 and installed[0]["rank"] == rank["rank"]
        assert (
            installed[0]["observer_sha256"]
            == receipt["input_hashes"][
                "token-observer-v3/hust_token_observer/__init__.py"
            ]
        )
        selected = [
            e
            for e in events
            if summary["request_phase"]["start_wall"]
            <= e["wall"]
            <= receipt["bench_ended_wall"]
        ]
        target = [e for e in selected if e["event"] == "target_logits_input"]
        draft = [e for e in selected if e["event"] == "draft_replay_input"]
        contexts = [e for e in selected if e["event"] == "forward_enter"]
        assert len(target) == rank["target_logits_calls"]
        assert sum(e["rows"] for e in target) == rank["target_logits_input_rows"]
        assert len(draft) == rank["draft_replay_calls"]
        assert (
            sum(e["first_step_sample_indices_rows"] for e in draft)
            == rank["draft_first_step_sample_indices_rows"]
        )
        target_contexts = [e for e in contexts if e["role"] == "target"]
        draft_contexts = [e for e in contexts if e["role"] == "draft"]
        assert len(target_contexts) == len(target) == len(draft_contexts) == len(draft)
        assert (
            sum(e["actual_tokens"] for e in target_contexts)
            == rank["target_forward_actual_tokens"]
        )
        for index in range(2):
            assert (
                sum(e["draft_steps"][index]["actual_tokens"] for e in draft_contexts)
                == rank["draft_step_actual_tokens"][index]
            )
        for entry, replay in zip(draft_contexts, draft, strict=True):
            assert entry["wall"] <= replay["wall"]
            assert entry["actual_tokens"] == replay["num_tokens"]
            assert entry["padded_tokens"] == replay["num_input_tokens"]
            assert entry["draft_steps"][1]["actual_tokens"] == replay["batch_size"]
            assert replay["batch_size"] == replay["first_step_sample_indices_rows"]
        assert rank["target_logits_input_rows"] < rank["target_forward_actual_tokens"]
