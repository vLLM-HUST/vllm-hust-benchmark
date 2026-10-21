"""Portable raw-evidence checks; no runtime or full-model admission."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/state-commit-recovery"
)


@pytest.fixture(scope="module")
def evidence():
    index = json.loads((BASE / "archive.json").read_text())
    packed = (BASE / index["archive"]).read_bytes()
    assert len(packed) == index["archive_bytes"]
    assert hashlib.sha256(packed).hexdigest() == index["archive_sha256"]
    expected = {row["name"]: row for row in index["members"]}
    assert len(expected) == len(index["members"])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in result
            data = archive.extractfile(member).read()
            assert len(data) == expected[member.name]["size"]
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            result[member.name] = data
    assert result.keys() == expected.keys()
    return result


def test_lifecycle_failures_and_scope(evidence):
    for attempt in (1, 2, 3):
        prefix = f"state-commit-recovery-r{attempt}"
        receipt = json.loads(evidence[f"{prefix}/receipt.json"])
        assert receipt["all_four_idle_after"]
        assert not receipt["protocol"]["prior_cache_imported"]
        assert not receipt["performance_publication"]
        assert (
            receipt["exit_code"]
            == receipt["cleanup_exit_code"]
            == (0 if attempt == 3 else 1)
        )
        for name, digest in receipt["source_sha256"].items():
            relative = name.split("/20261019-midterm-aligned/", 1)[1]
            assert hashlib.sha256(evidence[relative]).hexdigest() == digest
        if attempt < 3:
            assert receipt["status"] == "failed"
            assert (
                b"No module named 'triton.experimental.gluon.nvidia'"
                in evidence[f"{prefix}/child.log"]
            )
    audit = json.loads(evidence["state-commit-recovery-audit-r3.json"])
    assert not audit["model_qualification"] and not audit["performance_publication"]
    assert not audit["original_runtime_identity_proven"]


@pytest.mark.parametrize("rank", [0, 1])
def test_reconstructed_arrays_and_events(evidence, rank):
    raw = json.loads(evidence[f"state-commit-recovery-r3/rank-{rank}.json"])
    assert raw["rank"] == rank
    assert raw["producer_stream"] != raw["consumer_stream"]
    assert len(raw["cases"]) == len(raw["records"]) == 24
    groups = {}
    for case, event in zip(raw["cases"], raw["records"], strict=True):
        rows, width = case["rows"], case["width"]
        assert event["rows"] == rows and event["token_width"] == width
        assert event["stream_before"] == event["stream_after"] == raw["consumer_stream"]
        assert 0 <= event["start_ms"] <= event["end_ms"]
        expected = dict(
            recurrent=[-7] * 32,
            conv=[-8] * 32,
            remaining=case["initial_remaining"].copy(),
            accepted=[-9] * 32,
        )
        for i, seat in enumerate(case["seats"]):
            count = sum(token != -1 for token in case["tokens"][i])
            remaining = case["initial_remaining"][seat]
            committed = min(count, remaining)
            if remaining > 0:
                if case["verify"][i]:
                    expected["recurrent"][seat] = committed + 2
                expected["conv"][seat] = committed + 2 if case["verify"][i] else 1
            if case["sampling"][i]:
                expected["remaining"][seat] -= committed
            expected["accepted"][i] = count
        assert case["actual"] == case["expected"] == expected
        groups.setdefault((rows, width), []).append(case)
    assert set(groups) == {(n, w) for n in (0, 1, 24, 32) for w in (1, 2, 3)}
    for (rows, _), pair in groups.items():
        assert len(pair) == 2 and {case["variant"] for case in pair} == {0, 1}
        if rows:
            assert pair[0]["actual"]["accepted"] != pair[1]["actual"]["accepted"]
