"""Recompute published fixed-subset scores and retain truncation boundaries."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from vllm_hust_benchmark.mmlu_pro_pyramidkv import extract_answer

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/quality-regression"
)


@pytest.fixture(scope="module")
def evidence():
    manifest = json.loads((ROOT / "raw/archive.json").read_text())
    parts = []
    for part in manifest["parts"]:
        data = (ROOT / "raw" / part["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        parts.append(data)
    payload = b"".join(parts)
    assert hashlib.sha256(payload).hexdigest() == manifest["concatenated_sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert set(archive.getnames()) == set(manifest["members"])
        for name, digest in manifest["members"].items():
            data = archive.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest() == digest
            if name.endswith(".json"):
                files[name] = json.loads(data)
    return files


@pytest.mark.parametrize("filename", ["microflow-pair.json", "kvmat-pair.json"])
def test_pair_is_complete_and_rescores(evidence, filename):
    pair = json.loads((ROOT / filename).read_text())
    labels = evidence["source/verified-mmlu-quality-inputs.json"]["cases"]
    baseline, candidate = pair["engines"]
    assert (
        evidence[f"{baseline}/quality-tokenize-preflight.json"]
        == evidence[f"{candidate}/quality-tokenize-preflight.json"]
    )
    assert (
        evidence[f"{baseline}/quality-request-plan.json"]
        == evidence[f"{candidate}/quality-request-plan.json"]
    )
    recomputed = []
    for run, key in ((baseline, "b0"), (candidate, "b1")):
        receipt = evidence[f"{run}/receipt.json"]
        result = evidence[f"{run}/mmlu-pro.json"]
        assert receipt["status"] == "measured" and receipt["all_devices_released"]
        assert receipt["bench_exit_code"] == 0
        assert result["completed"] == result["total"] == 200
        assert result["failed"] == 0
        assert len(result["cases"]) == 200
        scores = []
        for i, case in enumerate(result["cases"]):
            assert case == evidence[f"{run}/quality-case-{i:03d}.json"]
            assert case["case_id"] == labels[i]["case_id"]
            answer = extract_answer(case["output"])
            assert answer == case["answer"]
            correct = answer == labels[i]["answer"]
            assert correct == case["correct"]
            scores.append(correct)
        assert sum(scores) == pair[key]["correct"] == result["correct"]
        assert pair[key]["accuracy"] == sum(scores) / 200
        assert pair[key]["truncated"] == sum(
            c["finish_reason"] == "length" for c in result["cases"]
        )
        recomputed.append(scores)
    gained = sum(not b0 and b1 for b0, b1 in zip(*recomputed, strict=True))
    lost = sum(b0 and not b1 for b0, b1 in zip(*recomputed, strict=True))
    assert gained == pair["paired_transitions"].get("incorrect_to_correct", 0)
    assert lost == pair["paired_transitions"].get("correct_to_incorrect", 0)
    assert pair["delta_percentage_points"] == (gained - lost) / 2
    assert pair["noninferiority_claim"] is None
    assert pair["not_full_dataset"] is True


def test_natural_end_and_truncation_are_separate(evidence):
    finish = json.loads((ROOT / "finish-breakdown.json").read_text())
    assert len(finish["runs"]) == 3
    for row in finish["runs"]:
        cases = evidence[f"{row['run']}/mmlu-pro.json"]["cases"]
        for reason, values in row["finish_breakdown"].items():
            selected = [c for c in cases if c["finish_reason"] == reason]
            assert values["count"] == len(selected)
            assert values["scorer_matches"] == sum(c["correct"] for c in selected)
        assert row["natural_end_and_correct_fraction_of_all_cases"] == (
            row["finish_breakdown"]["stop"]["scorer_matches"] / 200
        )


def test_failed_attempts_are_not_promoted(evidence):
    for run in ("mmlu-quality-b0-r1", "mmlu-quality-b0-r2", "mmlu-quality-b0-r3"):
        assert evidence[f"{run}/receipt.json"]["status"] == "failed"
    failed = evidence["mmlu-quality-b0-r3/mmlu-pro.json"]
    assert failed["failed"] > 0
    assert any("SocketTimeoutError" in c.get("error", "") for c in failed["cases"])
