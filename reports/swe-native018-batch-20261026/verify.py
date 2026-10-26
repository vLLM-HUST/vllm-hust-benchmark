"""Offline verification of public aggregates, not reproduction of private grading."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def verify():
    spec = importlib.util.spec_from_file_location(
        "batch_catalog", HERE / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.build()
    assert actual == json.loads((HERE / "catalog.json").read_text())
    assert actual["attempted_unique_tasks"] + actual["unattempted_tasks"] == 731
    assert actual["official_unresolved_grading"] == 0
    assert actual["full_dataset_acceptance"] is False
    review = json.loads((HERE / "five-independent-review.json").read_text())
    assert review["completed"] == 5
    assert review["grading_reexecutions"] == review["new_model_calls"] == 0
    assert review["patch_modified"] is False
    for item in review["items"]:
        entry = next(e for e in actual["entries"] if e["run_id"] == item["run_id"])
        summary = json.loads((ROOT / entry["source"]).read_text())
        assert item["task_identity_sha256"] == entry["task_identity_sha256"]
        assert (
            item["quality"]["official_resolved"]
            == summary["grading"]["official_result"]
        )
        assert item["quality"]["independent_rule_equal"] is True
        assert item["quality"]["unresolved"] == []
        for key in ("model_calls", "elapsed_seconds", "patch_sha256"):
            assert item["agent"][key] == summary["agent"][key]
        assert (
            item["application"]["apply_exit_code"]
            == summary["grading"]["apply_exit_code"]
        )
        for key in ("fail_to_pass", "pass_to_pass"):
            reviewed = item["quality"][key]
            counted = summary["grading"]["official_test_coverage"][key]
            assert all(
                reviewed[k] == counted[k] for k in ("expected", "passed", "missing")
            )
            assert reviewed["failed"] + reviewed["missing"] == counted["not_passed"]
    custody = json.loads((HERE / "private-custody.json").read_text())
    assert custody["private_contents_included"] is False
    assert len(custody["runs"]) == 6
    for item in custody["runs"]:
        entry = next(e for e in actual["entries"] if e["run_id"] == item["run_id"])
        assert entry["source_sha256"] == item["summary_sha256"]
        assert item["files_verified"] > 0 and item["old_seals_preserved"] is True
    manifest = HERE / "public-files.json"
    files = json.loads(manifest.read_text())
    assert len({item["path"] for item in files}) == len(files)
    for item in files:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == item["bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    return {
        "status": "passed",
        "files": len(files),
        "attempted": actual["attempted_unique_tasks"],
        "official_resolved": actual["recorded_official_resolved"],
        "application_unproven": actual["application_evidence_limited_tasks"],
        "confirmed_application_failed": actual[
            "confirmed_patch_application_failure_tasks"
        ],
        "new_model_calls": 0,
        "private_grading_reexecuted": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
