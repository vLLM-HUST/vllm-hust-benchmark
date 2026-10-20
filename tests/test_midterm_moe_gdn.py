"""Verify sealed MoE/GDN evidence and independently reconstruct GDN totals."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/moe-gdn-r6-r7"
)


def test_moe_gdn_archive_and_live_schedule():
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
    for local, source in [
        ("loaded-moe.json", "loaded-moe-summary-r6.json"),
        ("moe-projections.json", "moe-projection-summary-r6.json"),
        ("gdn-schedule.json", "gdn-host-summary-r7.json"),
    ]:
        assert (ROOT / local).read_bytes() == members["source/" + source]
    for run in (6, 7):
        prefix = f"microflow-token-accounting-r{run}/"
        receipt = json.loads(members[prefix + "receipt.json"])
        result = json.loads(members[prefix + "mmlu-pro.json"])
        assert receipt["status"] == "measured" and receipt["all_devices_released"]
        assert result["completed"] == 200 and result["failed"] == 0
    summary = json.loads((ROOT / "gdn-schedule.json").read_text())
    assert summary["model_effective_flops"] is None
    receipt = json.loads(members["microflow-token-accounting-r7/receipt.json"])
    for rank in summary["ranks"]:
        matching = [
            data
            for name, data in members.items()
            if name.startswith("microflow-token-accounting-r7/token-accounting-")
            and hashlib.sha256(data).hexdigest() == rank["source_sha256"]
        ]
        assert len(matching) == 1
        counts, chunks = Counter(), Counter()
        for line in matching[0].splitlines():
            event = json.loads(line)
            if event["event"] != "forward_enter" or not (
                summary["request_phase"]["start_wall"]
                <= event["wall"]
                <= receipt["bench_ended_wall"]
            ):
                continue
            if event["role"] == "draft":
                assert event["owned_gdn"] == []
                continue
            assert len(event["owned_gdn"]) == 1
            group = event["owned_gdn"][0]
            assert len(group["layers"]) == 30
            q = [b - a for a, b in zip(group["cu"], group["cu"][1:])]
            assert all(n > 0 for n in q)
            assert sum(q) == event["actual_tokens"]
            ids = group["verify_ids"]
            assert len(set(ids)) == group["verify_count"] == len(ids)
            assert all(0 <= i < len(q) and q[i] <= group["width"] for i in ids)
            prefill = [n for i, n in enumerate(q) if i not in ids]
            counts["contexts"] += 1
            counts["actual_tokens"] += sum(q)
            counts["prefill_tokens"] += sum(prefill)
            counts["verify_tokens"] += sum(q[i] for i in ids)
            if group["decode_only"]:
                assert not prefill
                counts["decode_only_contexts"] += 1
            else:
                counts["mixed_contexts"] += 1
                for size, descriptors in group["chunk_indices"].items():
                    expected = [
                        [i, j]
                        for i, n in enumerate(prefill)
                        for j in range((n + int(size) - 1) // int(size))
                    ]
                    assert descriptors == expected
                    chunks[size] += len(expected)
        assert counts == rank["counts"]
        assert chunks == rank["prefill_chunks_by_size"]
