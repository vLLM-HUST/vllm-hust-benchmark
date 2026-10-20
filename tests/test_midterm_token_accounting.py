"""Validate qualified host extents without treating them as effective FLOPs."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/token-accounting-r1"
)


def test_archive_integrity():
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
        for name, digest in manifest["members"].items():
            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            )


def test_complete_context_extent_accounting():
    summary = json.loads((ROOT / "drained-summary.json").read_text())
    assert summary["status"] == "complete_context_pairs_with_explicit_compute_tail"
    assert summary["rank_histograms_match"]
    assert summary["completed"] == 200 and summary["failed"] == 0
    ranks = summary["ranks"]
    assert {r["rank"] for r in ranks} == {0, 1}
    assert ranks[0]["roles"] == ranks[1]["roles"]
    for rank in ranks:
        for role in rank["roles"].values():
            hist = role["histogram"]
            assert sum(h["contexts"] for h in hist) == role["contexts"]
            assert (
                sum(h["contexts"] * h["actual_tokens"] for h in hist)
                == role["sum_context_actual_tokens"]
            )
            assert (
                sum(h["contexts"] * h["padded_tokens"] for h in hist)
                == role["sum_context_padded_tokens"]
            )
            assert role["padding_tokens"] == (
                role["sum_context_padded_tokens"] - role["sum_context_actual_tokens"]
            )
        target = rank["roles"]["target"]
        assert target["contexts"] == 874
        assert target["sum_context_actual_tokens"] == 339775
        assert target["sum_context_padded_tokens"] == 405676


def test_tail_is_separate_and_no_flops_claim():
    strict = json.loads((ROOT / "request-window-summary.json").read_text())
    drained = json.loads((ROOT / "drained-summary.json").read_text())
    assert strict["status"] == "rank_extent_mismatch"
    assert drained["compute_tail_seconds"] > 0
    assert drained["compute_tail_end_wall"] < drained["client_lifecycle_end_wall"]
    assert (
        drained["request_plus_compute_tail_seconds"]
        > (drained["request_phase"]["duration_seconds"])
    )
    assert drained["model_effective_flops"] is None
    assert drained["attainment_percent"] is None
    assert any("multiple MTP steps" in text for text in drained["limitations"])
