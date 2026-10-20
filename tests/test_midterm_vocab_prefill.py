"""Verify vocabulary counts and native prefill shapes from sealed r8 evidence."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery"
)


def archive_members(directory):
    manifest = json.loads((directory / "archive.json").read_text())
    parts = []
    for part in manifest["parts"]:
        data = (directory / part["file"]).read_bytes()
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
    return members


def test_vocabulary_and_prefill_from_same_run_raw_events():
    root = BASE / "vocabulary-prefill-r8"
    members = archive_members(root)
    raw = archive_members(BASE / "recurrence-r8")
    vocab = json.loads((root / "vocabulary.json").read_text())
    prefill = json.loads((root / "prefill-shapes.json").read_text())
    catalog = json.loads((root / "same-run-catalog.json").read_text())
    receipt = json.loads(raw["microflow-token-accounting-r8/receipt.json"])
    source_receipt = json.loads(members["gdn-native-source-r8/source-receipt.json"])
    manifest = json.loads(
        members["gdn-native-source-r8/" + source_receipt["manifest_file"]]
    )
    assert manifest["sha256"] == source_receipt["deployed_library_sha256"]
    for path, digest in source_receipt["source_sha256"].items():
        assert (
            hashlib.sha256(members["gdn-native-source-r8/" + path]).hexdigest()
            == digest
        )
    assert (
        vocab["request_phase"] == prefill["request_phase"] == catalog["request_phase"]
    )
    for rank in vocab["ranks"]:
        data = raw["microflow-token-accounting-r8/" + rank["file"]]
        assert hashlib.sha256(data).hexdigest() == rank["source_sha256"]
        events = [json.loads(line) for line in data.splitlines()]
        inventory = next(e for e in events if e["event"] == "loaded_model_inventory")
        for role in ("target", "draft"):
            head = [
                m for m in inventory[role]["modules"] if m["name"].endswith("lm_head")
            ]
            assert len(head) == 1
            assert (
                head[0]["parameters"]["weight"]["shape"]
                == rank["loaded_head_shapes"][role]
            )
        selected = [
            e
            for e in events
            if vocab["request_phase"]["start_wall"]
            <= e["wall"]
            <= receipt["bench_ended_wall"]
        ]
        rows = {
            "target": sum(
                e["rows"] for e in selected if e["event"] == "target_logits_input"
            ),
            "draft_step0": sum(
                e["first_step_sample_indices_rows"]
                for e in selected
                if e["event"] == "draft_replay_input"
            ),
            "draft_step1": sum(
                e["draft_steps"][1]["actual_tokens"]
                for e in selected
                if e["event"] == "forward_enter" and e["role"] == "draft"
            ),
        }
        assert rows == rank["rows"]
        for role, count in rows.items():
            assert vocab["semantic_projection_flops"][role] == 2 * count * 248320 * 2048
        hist = Counter()
        for event in selected:
            if event["event"] != "forward_enter" or event["role"] != "target":
                continue
            for group in event["owned_gdn"]:
                if group["decode_only"]:
                    continue
                for start, end in zip(group["prefill_cu"], group["prefill_cu"][1:]):
                    for offset in range(start, end, 64):
                        hist[min(64, end - offset)] += len(group["layers"])
        shape_rank = next(r for r in prefill["ranks"] if r["rank"] == rank["rank"])
        assert shape_rank["source_sha256"] == rank["source_sha256"]
        assert hist == {
            int(k): v for k, v in shape_rank["chunk_token_layers_histogram"].items()
        }
        assert shape_rank["dense_mnk_arithmetic_subtotal"] == sum(
            count * 16 * (6 * n * 128 * 128 + 2 * n * n * 256)
            for n, count in hist.items()
        )
        assert (
            sum(n * count for n, count in hist.items())
            == shape_rank["prefill_token_layers"]
        )
    assert vocab["semantic_projection_subtotal"] == sum(
        vocab["semantic_projection_flops"].values()
    )
    assert catalog["covered_basic_semantic_subtotal"] == sum(
        catalog["components"].values()
    )
    assert (
        catalog["components"]["vocabulary_projection_semantics"]
        == vocab["semantic_projection_subtotal"]
    )
    assert catalog["native_h_o_dense_arithmetic_included"] is False
    for summary in (vocab, prefill, catalog):
        assert (
            summary["model_effective_flops"] is None
            and summary["attainment_percent"] is None
        )
