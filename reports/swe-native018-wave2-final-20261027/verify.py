"""Verify three original outcomes without converting missing tests into failures."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(here=HERE, root=ROOT):
    spec = importlib.util.spec_from_file_location(
        "final_catalog", here / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build(root, here)
    assert result == read(here / "catalog.json")
    prior = read(root / module.PRIOR)
    assert result["entries"][:26] == prior["entries"]
    assert (
        result["attempted_unique_tasks"],
        result["recorded_official_resolved"],
        result["unattempted_tasks"],
    ) == (29, 4, 702)
    for name in ("protonmail", "qutebrowser", "tutanota"):
        summary = read(here / f"summary-{name}.json")
        review = read(here / f"review-{name}.json")
        whitelist = read(here / f"storage-whitelist-{name}.json")
        custody = read(here / f"custody-{name}.json")
        assert sha(here / f"review-{name}.json") == whitelist["files"]["summary.json"]
        assert (
            sha(here / f"summary-{name}.json")
            == custody["summary_sha256"]
            == review["CPU_summary_sha256"]
        )
        assert custody["manifest_sha256"] == review["CPU_seal_sha256"]
        assert custody["files_verified"] == review["CPU_files_verified"]
        assert (
            sha(here / f"derive_private_{name}.py.source")
            == summary["evidence_sha256"]["derivation_source"]
        )
        grade, application = summary["grading"], review["patch_application"]
        assert grade["apply_exit_code"] == application["exit_code"] == 0
        assert grade["state_capture"] == application["state_capture"] == "complete"
        assert grade["patch_application_attested"] is True
        assert (
            grade["after_graded_file_states_equal"]
            is application["after_graded_equal"]
            is True
        )
        assert (
            summary["agent"]["patch_sha256"]
            == application["raw_patch_sha256"]
            == application["effective_patch_sha256"]
        )
        assert summary["agent"]["patch_bytes"] == application["raw_patch_bytes"]
        assert grade["official_result"] == review["official_grading"]["resolved"]
        assert grade["parsed_test_counts"] == review["official_grading"]["test_counts"]
        for kind in ("fail_to_pass", "pass_to_pass"):
            assert all(
                grade["official_test_coverage"][kind][k]
                == review["official_grading"][kind][k]
                for k in ("expected", "passed", "missing")
            )
        assert (
            summary["agent"]["model_calls"]
            == review["request_audit"]["successful_returns"]
        )
        assert (
            review["request_audit"]["rejected"]
            == review["request_audit"]["http_retries"]
            == 0
        )
    proton = read(here / "review-protonmail.json")
    assert (
        proton["raw_execution"]["ConversationView"]["tests"]
        == proton["raw_execution"]["ConversationView"]["failed"]
        == 10
    )
    assert proton["raw_execution"]["hook_F2P"]["execution_established"] is False
    assert proton["parser_boundary"]["original_parsed_entries"] == 4
    assert (
        proton["parser_boundary"]["kind"]
        == "file-level test execution FAILED placeholders"
    )
    assert (
        proton["parser_boundary"]["actual_Jest_failure_glyph"]
        != proton["parser_boundary"]["parser_failure_glyph"]
    )
    qute = read(here / "review-qutebrowser.json")
    assert (
        qute["raw_test_execution"]["passed"]
        == qute["raw_test_execution"]["collected"]
        == 10
    )
    assert qute["raw_test_execution"]["missing"] == 0
    assert "XIO" in qute["raw_test_execution"]["retained_post_test_log"]
    assert qute["official_grading"]["resolved"] is True
    tuta = read(here / "review-tutanota.json")
    raw = tuta["raw_test_execution"]
    assert raw["type_build_completed"] is True
    assert raw["missing_dependency"] == "esbuild-linux-arm64"
    assert (
        raw["F2P_missing"] == 107
        and raw["functional_assertion_failures_not_established"] is True
    )
    assert (
        result["entries"][-1]["raw_test_coverage"]["target_test_assertions_failed"]
        is False
    )
    current = read(here / "current-aggregate.json")
    assert current["attempted_tasks"] == current["recovery_denominator"] == 29
    assert (
        current["original_official_positives"] == 4
        and current["unattempted_tasks"] == 702
    )
    recovery = read(root / current["recovery_source"])
    assert current["prior_recovery_snapshot"] == recovery["aggregate_distinction"]
    delta = (
        current["prior_recovery_snapshot"]["positives_after_disclosed_recovery"]
        - current["prior_recovery_snapshot"]["original_official_positives"]
    )
    assert current["positives_after_disclosed_recovery"] == 4 + delta == 5
    assert current["additional_Agent_attempts_for_recovery"] == 0
    return dict(
        status="passed",
        attempts=29,
        original_positive=4,
        same_answer_recovery_positive=5,
        unattempted=702,
    )


if __name__ == "__main__":
    print(json.dumps(verify()))
