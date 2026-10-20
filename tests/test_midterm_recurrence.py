"""Verify r8 hashes, raw slot eligibility and same-run component boundaries."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/recurrence-r8"
)


def test_recurrence_archive_and_independent_slot_reconstruction():
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
    summary = json.loads((ROOT / "recurrence.json").read_text())
    assert summary == json.loads(members["source/gdn-recurrence-summary-r8.json"])
    prefix = "microflow-token-accounting-r8/"
    receipt = json.loads(members[prefix + "receipt.json"])
    result = json.loads(members[prefix + "mmlu-pro.json"])
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert result["completed"] == 200 and result["failed"] == 0
    for rank in summary["ranks"]:
        data = members[prefix + rank["file"]]
        assert hashlib.sha256(data).hexdigest() == rank["sha256"]
        counts = Counter()
        for line in data.splitlines():
            event = json.loads(line)
            if (
                event["event"] != "forward_enter"
                or event["role"] != "target"
                or not (
                    summary["request_phase"]["start_wall"]
                    <= event["wall"]
                    <= receipt["bench_ended_wall"]
                )
            ):
                continue
            counts["target_contexts"] += 1
            layers = []
            for group in event["owned_gdn"]:
                layers.extend(group["layers"])
                q = [b - a for a, b in zip(group["cu"], group["cu"][1:])]
                assert sum(q) == event["actual_tokens"]
                active = skipped = 0
                assert len(group["verify_ids"]) == len(group["verify_slots"])
                for index, row in zip(
                    group["verify_ids"], group["verify_slots"], strict=True
                ):
                    assert len(row) == group["width"] == 3
                    if row[0] < 0:
                        skipped += q[index]
                    else:
                        active += q[index]
                n = len(group["layers"])
                counts["eligible_token_layers"] += active * n
                counts["skipped_token_layers"] += skipped * n
                kind = "decode" if group["decode_only"] else "mixed"
                counts[kind + "_eligible_token_layers"] += active * n
            assert len(layers) == len(set(layers)) == 30
        assert counts == rank["counts"]
        head_tokens = counts["eligible_token_layers"] * 16
        assert rank["semantic_basic_flops"] == head_tokens * (7 * 128 * 128 + 128)
        assert rank["source_tiled_basic_flops"] == head_tokens * (
            7 * 128 * 128 + 4 * 128
        )
        assert rank["semantic_exp_evaluations"] == head_tokens
        assert rank["source_tiled_exp_evaluations"] == 4 * head_tokens
    catalog = json.loads((ROOT / "same-run-catalog.json").read_text())
    for name, digest in catalog["input_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    pairs = json.loads((ROOT / "attention-pairs.json").read_text())
    moe = json.loads((ROOT / "moe-projections.json").read_text())
    assert (
        catalog["request_phase"]
        == summary["request_phase"]
        == pairs["request_phase"]
        == moe["request_phase"]
    )
    assert (
        {r["rank"]: r["sha256"] for r in summary["ranks"]}
        == {r["rank"]: r["source_sha256"] for r in pairs["ranks"]}
        == {r["rank"]: r["source_sha256"] for r in moe["ranks"]}
    )
    assert catalog["covered_basic_semantic_subtotal"] == sum(
        catalog["components"].values()
    )
    assert (
        catalog["model_effective_flops"] is None
        and catalog["attainment_percent"] is None
    )
    assert catalog["excluded"] and catalog["limitations"]
