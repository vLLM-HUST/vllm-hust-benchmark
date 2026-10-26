"""Verify public aggregates and retained private-evidence bindings offline."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def verify():
    spec = importlib.util.spec_from_file_location(
        "next_catalog", HERE / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.build()
    assert data == read(HERE / "catalog.json")
    prior = read(ROOT / module.PRIOR)
    assert data["entries"][: len(prior["entries"])] == prior["entries"]
    inputs = read(HERE / "inputs.json")
    for item in inputs["observations"]:
        value = read(HERE / item["summary"])
        review = read(HERE / item["review"])
        release = read(HERE / item["release"])
        custody = read(HERE / item["custody"])
        assert review["official_result"] == value["grading"]["official_result"]
        assert review["unresolved"] == value["official_unresolved_grading"] == 0
        assert review["request_calls"] == value["agent"]["model_calls"]
        assert review["agent_seconds"] == value["agent"]["elapsed_seconds"]
        assert (
            review["patch_application"]["patch_sha256"]
            == value["agent"]["patch_sha256"]
        )
        assert (
            review["patch_application"]["exit_code"]
            == value["grading"]["apply_exit_code"]
        )
        for name in ("fail_to_pass", "pass_to_pass"):
            a, b = review[name], value["grading"]["official_test_coverage"][name]
            assert all(a[key] == b[key] for key in ("expected", "passed", "missing"))
            failed = a.get("failed", a.get("failed_in_official_output"))
            assert failed + a["missing"] == b["not_passed"]
        assert custody["run_id"] == value["run_id"]
        assert custody["files_verified"] == review["seal"]["cpu_files_verified"]
        assert custody["private_contents_included"] is False
        assert (
            release["sealed_files_verified"] == review["seal"]["service_files_verified"]
        )
        assert (
            release["original_receipt_sha256"]
            == value["evidence_sha256"]["service_release"]
        )
        assert release["service"]["status"] == "completed"
        assert release["service"]["service_exit_code"] == 0
        assert all(
            release["service"][key]
            for key in ("sessions_drained", "all_devices_released", "seal_allowed")
        )
        assert review["no_reruns"] and review["new_model_calls"] == 0
        source = HERE / item["derivation_source"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == value["evidence_sha256"]["derivation_source"]
        )
    for item in read(HERE / "public-files.json"):
        path = Path(item["path"])
        assert not path.is_absolute() and ".." not in path.parts
        raw = (ROOT / path).read_bytes()
        assert (
            len(raw) == item["bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        )
    return {
        "status": "passed",
        "attempted": data["attempted_unique_tasks"],
        "official_resolved": data["recorded_official_resolved"],
        "unattempted": data["unattempted_tasks"],
        "model_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
