import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "reports/mmlu-quality018-20261022"


def read(name):
    return json.loads((DATA / name).read_text())


def digest(name):
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()


def test_native_quality_and_all_reused_candidates_recompute():
    audit = read("quality018-completed-audit-r1.json")
    rows = audit["cases"]
    assert len(rows) == len({row["case_id"] for row in rows}) == 200
    counts = audit["counts"]
    for key, value in {
        "correct": sum(row["correct"] for row in rows),
        "failed": sum(not row["success"] for row in rows),
        "completed": sum(row["success"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "unparsed": sum(row["unparsed"] for row in rows),
    }.items():
        assert counts[key] == value
    assert counts["correct"] == 151 and counts["failed"] == 0
    assert counts["truncated"] == 18 and audit["accuracy"] == 151 / 200
    for comparison in audit["comparisons"]:
        name = comparison["candidate_run"]
        candidates = [row["candidates"][name] for row in rows]
        assert (
            sum(row["correct"] for row in candidates) == comparison["candidate_correct"]
        )
        assert (
            sum(row["truncated"] for row in candidates)
            == comparison["candidate_truncated"]
        )
        improved = sum(
            not row["correct"] and row["candidates"][name]["correct"] for row in rows
        )
        regressed = sum(
            row["correct"] and not row["candidates"][name]["correct"] for row in rows
        )
        assert (improved, regressed) == (
            comparison["improved_cases"],
            comparison["regressed_cases"],
        )
        assert (
            comparison["accuracy_delta_percentage_points"]
            == (improved - regressed) / 200 * 100
        )
        assert (
            comparison["requests_byte_content_equivalent"]
            and comparison["scorer_identical"]
        )
    assert [row["candidate_correct"] for row in audit["comparisons"]] == [146, 152, 149]
    assert audit["highest_observed_quality_candidate"] == "mmlu-quality-kvmat-r1"


def test_runtime_and_original_release_receipt_are_preserved():
    audit = read("quality018-completed-audit-r1.json")
    identity = read("runtime-identity.json")
    assert identity["vllm_version"] == identity["ascend_version"] == "0.18.0"
    receipt = read("receipt.json")
    assert receipt["status"] == "completed"
    assert receipt["child_exit_codes_reverse_order"] == [0, 0]
    assert receipt["all_devices_released"] is False
    release = read("quality018-release-followup-r1.json")
    assert release["original_controller_exit_code"] == 1
    assert release["original_receipt_sha256"] == digest("receipt.json")
    assert release["all_devices_released"] and release["service_process_group_empty"]
    assert release["signals_sent"] == []
    assert audit["release_followup_sha256"] == digest(
        "quality018-release-followup-r1.json"
    )
    final = read("quality018-finalization-r1.json")
    assert audit["finalization_sha256"] == digest("quality018-finalization-r1.json")
    assert final["changed_files"] == ["service.log"]
    assert (
        final["final_log_bytes"] - final["original_log_prefix_bytes"]
        == final["appended_bytes"]
        == 302
    )
    for name in ("receipt.json", "runtime-identity.json", "cache-policy.json"):
        assert digest(name) == final["final_files"][name]
    assert not read("archive.json")["public_raw_answers"]
