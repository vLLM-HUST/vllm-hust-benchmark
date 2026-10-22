import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/frontierscience-combination-quality-20261022"


def read(name):
    return json.loads((DATA / name).read_text())


def test_complete_combination_quality_matches_case_records():
    summary = read("summary.json")
    with (DATA / "per_case.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["case_id"] for r in rows}) == 160
    for track, total in (("olympiad", 100), ("research", 60)):
        selected = [r for r in rows if r["track"] == track]
        observed = summary["tracks"][track]
        assert len(selected) == observed["total"] == total
        assert observed["successes"] == sum(
            r["judge_status"] == "ok" and r["correct"] == "True" for r in selected
        )
        assert observed["graded"] == sum(r["judge_status"] == "ok" for r in selected)
        assert observed["judge_unresolved"] == sum(
            r["judge_status"] != "ok" for r in selected
        )
        assert observed["solver_truncated"] == sum(
            r["solver_truncated"] == "True" for r in selected
        )
        assert observed["judge_truncated"] == sum(
            r["judge_truncated"] == "True" for r in selected
        )
        assert observed["success_rate"] == observed["successes"] / total
        assert observed["solver_failures"] == observed["judge_transport_failures"] == 0
    assert [summary["tracks"][t]["successes"] for t in ("olympiad", "research")] == [
        62,
        2,
    ]


def test_runtime_custody_and_unresolved_results_are_not_relabelled():
    audit = read("completed-r3-audit.json")
    assert audit["retained_records_byte_identical"] == 25
    assert audit["designated_018_baseline"] is False
    for name, key in (
        ("summary.json", "summary_sha256"),
        ("per_case.csv", "per_case_sha256"),
        ("contract.json", "contract_sha256"),
    ):
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == audit[key]
    assert read("solver-runtime.json")["modules"]["vllm"]["version"].startswith(
        "0.28.1."
    )
    assert read("judge-runtime.json")["modules"]["vllm"][
        "version_attribute"
    ].startswith("0.28.1.")
    assert read("solver-receipt.json")["service_exit_code"] == 130
    assert read("judge-receipt.json")["service_exit_code"] == 0
    assert read("judge-receipt.json")["record_status_counts"] == {
        "invalid_judge": 8,
        "ok": 152,
    }
    for name in ("solver-receipt.json", "judge-receipt.json"):
        assert read(name)["all_devices_released"] is True
