"""Offline aggregate checks; private raw evidence is indexed, not bundled."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    def read(n):
        return json.loads((HERE / n).read_text())

    s, r, p, t = (
        read(n)
        for n in (
            "summary.json",
            "independent-review.json",
            "profiling-review.json",
            "tests.json",
        )
    )
    assert (
        hashlib.sha256((HERE / "summary.json").read_bytes()).hexdigest()
        == p["summary_sha256"]
    )
    assert s["runtime"] == {"vllm": "0.18.0", "vllm_ascend": "0.18.0"}
    assert (
        s["model_task_attempts"]
        == s["scored_model_tasks"]
        == s["passed_model_tasks"]
        == 1
    )
    assert s["dataset_tasks"] == 89 and s["unattempted_tasks"] == 88
    assert s["agent_turns"] == r["original_trial"]["original_model_HTTP_responses"] == 5
    assert s["original_verifier"]["reward"] is None
    assert read("original-trial.json")["reward"] is None
    a = s["recovery"]
    assert a["reward"] == r["original_test_results"]["reward"] == 1
    assert (
        a["tests"]
        == a["passed"]
        == t["summary"]["tests"]
        == t["summary"]["passed"]
        == 6
    )
    assert a["failed"] == t["summary"]["failed"] == 0
    assert a["seconds"] == r["verifier_elapsed_seconds"] < a["timeout_seconds"] == 900
    assert (
        a["additional_Agent_attempts"]
        == a["additional_model_calls"]
        == a["automatic_retries"]
        == 0
    )
    assert a["answer_file_hashes_unchanged"] and a["original_tests_unchanged"]
    assert a["answer_files_checked"] == 5 and a["all_owned_containers_released"]
    assert a["process_exit_code"] == r["release"]["process_exit_code"] == 0
    assert a["cpus"] * 10**9 == r["resource_identity"]["NanoCpus"]
    assert a["memory_bytes"] == r["resource_identity"]["Memory"] == 2147483648
    names = {x["name"].split("::")[-1] for x in t["tests"]}
    assert len(names) == 6 and names == set(r["original_test_results"]["names"])
    assert all(x["status"] == "passed" for x in t["tests"])
    assert s["evidence"]["recovery_plan_sha256"] == r["recovery_plan_sha256"]
    assert (
        hashlib.sha256((HERE / "recovery.py.source").read_bytes()).hexdigest()
        == s["evidence"]["recovery_source_sha256"]
    )
    assert s["evidence"]["recovery_seal_sha256"] == r["recovery_seal_sha256"]
    assert (
        s["evidence"]["original_receipt_sha256"]
        == r["original_trial"]["receipt_sha256"]
    )
    assert (
        s["script_provenance"]["sha256"]
        == r["preserved_answer"]["derived_check_cert"]["derived_sha256"]
    )
    assert not s["script_provenance"]["original_filesystem_sha_captured"]
    assert s["script_provenance"]["original_mode"] is None
    assert not s["script_provenance"]["complete_original_answer_snapshot"]
    assert not s["transport"]["verified_after_execution"]
    assert s["transport"]["verified_before_execution"]
    assert not s["full_dataset_acceptance"] and not s["paired_B0_B1"]
    assert (
        not s["amd64_image_equivalence_claimed"]
        and not s["raw_questions_answers_or_keys_included"]
    )
    assert r["release"]["all_Docker_containers_absent"]
    assert not r["original_trial"]["original_records_overwritten"]
    index = read("restricted-evidence-index.json")
    assert len(index["files"]) == r["recovery_sealed_files_verified"] == 17
    assert len(index["source_bindings"]) == r["CPU_bindings_verified"] == 30
    receipt = read("recovery-receipt.json")
    assert (
        receipt["recovered_reward"] == a["reward"]
        and receipt["original_reward"] is None
    )
    return {
        "status": "passed",
        "unique_model_trials": 1,
        "recovered_score": 1,
        "tests_passed": 6,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
