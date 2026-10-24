import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "reports/vllm-hust-midterm-20261019/m01-recovery/storage-reference-supplement-20261024"
)


def test_archives_and_selected_results_are_byte_identical():
    index = json.loads((BASE / "archive.json").read_text())
    assert index["archives"]
    for row in index["archives"]:
        assert row["in_git"] is False
        assert row["raw_url"].endswith("/" + row["name"])
        readback = json.loads(
            (BASE / row["name"].replace(".tar.gz", "-readback.json")).read_text()
        )
        assert readback["archive_bytes"] == row["bytes"]
        assert readback["archive_sha256"] == row["sha256"]
        assert readback["all_members_read_back"] and readback["members_verified"] == 262
        manifest = json.loads(
            (BASE / row["name"].replace(".tar.gz", "-manifest.json")).read_text()
        )
        assert len(manifest["members"]) == 262
    for name, digest in index["selected_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest


def test_completed_measurements_do_not_promote_unqualified_efficiency():
    summary = json.loads((BASE / "summary.json").read_text())
    assert summary["unique_original_cases_completed"] == 281
    assert summary["conditional_cases_skipped_by_reuse"] == 8
    assert summary["authorized_metadata_revision_cases"] == 18
    totals = summary["totals"]
    assert totals["numerical_checks"] == 1402
    assert totals["stable_graph_groups"] == 478
    assert totals["unstable_graph_groups"] == 60
    assert all(row["released"] for row in summary["batches"].values())
    for family, budget in {"A": 900, "B": 600, "C": 1200, "D": 1800}.items():
        elapsed = sum(
            row["elapsed_seconds"]
            for name, row in summary["batches"].items()
            if name.split("-")[0] == family
        )
        assert 0 < elapsed <= budget
    validation = summary["model_validation"]
    assert validation["fit_count"] == 1
    assert validation["passed"] == 6 and validation["failed"] == 26
    measured = summary["formal_result_unchanged"]
    assert math.isclose(
        measured["C_TFLOPS_per_chip"],
        measured["F"] / (measured["chips"] * measured["T_seconds"] * 1e12),
        rel_tol=1e-14,
    )
    assert [s["node_coverage"]["missing_duration"] for s in summary["scenarios"]] == [
        18111,
        18081,
        18051,
    ]
    for scenario in summary["scenarios"]:
        assert scenario["structural"]["acyclic"]
        assert (
            sum(scenario["node_coverage"].values())
            == scenario["structural"]["nodes"]
            == 361096
        )
        for field in (
            "reference_seconds",
            "reference_tflops_per_chip",
            "efficiency_percent",
            "attainment",
        ):
            assert scenario[field] is None and summary[field] is None
    assert not summary["replaces_best_b1_performance"]
