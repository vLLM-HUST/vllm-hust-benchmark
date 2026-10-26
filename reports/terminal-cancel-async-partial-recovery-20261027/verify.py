"""Offline public cross-checks only; no private files, model or verifier execution."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    def read(name):
        return json.loads((HERE / name).read_text())

    s = read("summary.json")
    p = read("storage-public/summary.json")
    review = read("independent-review.json")
    for name, digest in review["public_projection_files"].items():
        assert (
            hashlib.sha256((HERE / "storage-public" / name).read_bytes()).hexdigest()
            == digest
        )
    for role in ("original", "recovery"):
        raw = (HERE / f"{role}-profiling-review.json").read_bytes()
        assert (
            hashlib.sha256(raw).hexdigest()
            == s["evidence"][f"{role}_profiling_audit_sha256"]
        )
    original = s["original_attempt"]
    r = s["recovery"]
    assert (
        original["exception"]
        == p["model_trial_result"]["exception"]
        == "ContextLengthExceededError"
    )
    assert original["reward"] is p["model_trial_result"]["reward"] is None
    assert original["task_complete"] is original["verifier_started"] is False
    assert original["successful_steps"] == original["HTTP200"] == 17
    assert original["episode_counter"] == original["HTTP_requests"] == 18
    assert original["HTTP400"] == 1
    assert (
        original["rejected_input_tokens_lower_bound"]
        + original["requested_output_tokens"]
        > original["context_window"]
        == 32768
    )
    assert original["exact_rejected_input_tokens"] is None
    assert original["budget"]["turns"] == 100
    assert (
        original["budget"]["agent_timeout_sec"]
        == original["budget"]["verifier_timeout_sec"]
        == 900
    )
    assert original["budget"]["max_tokens"] == 4096 and original["budget"]["seed"] == 17
    for key, value in (
        ("reward", 0),
        ("tests", 6),
        ("passed", 1),
        ("failed", 5),
        ("missing", 0),
        ("process_exit_code", 0),
    ):
        assert r[key] == p["recovery_result"][key] == review["result"][key] == value
    ctrf = read("storage-public/ctrf-name-status.json")
    tests = read("tests.json")
    mapped = {row["name"]: row["status"] for row in ctrf["tests"]}
    assert len(mapped) == 6 and tests == mapped
    assert sum(v == "passed" for v in mapped.values()) == 1
    assert mapped["test_outputs.py::test_run_py_file_exists"] == "passed"
    for key in (
        "additional_Agent_attempts",
        "additional_model_calls",
        "control_repeats",
        "automatic_retries",
    ):
        assert r[key] == 0
    a = s["artifact_provenance"]
    assert a["original_filesystem_sha256"] is a["original_mode"] is None
    assert a["chosen_mode"] == "0644" and a["answer_repaired"] is False
    assert a["derived_bytes"] == p["answer_provenance"]["bytes"] == 1103
    assert a["derived_sha256"] == p["answer_provenance"]["derived_sha256"]
    assert (
        r["answer_before_after_equal"]
        and r["transport_before_after_equal"]
        and r["all_owned_resources_released"]
    )
    assert (
        r["cpus"] == 1
        and r["memory_bytes"] == 2147483648
        and r["cpuset"] == "24-31"
        and r["mtu"] == 1400
    )
    assert (
        r["timeout_seconds"] == 900
        and r["source_bindings_verified"] == 37
        and r["recovery_files_verified"] == 22
    )
    assert s["coverage"] == dict(
        dataset_tasks=89,
        attempted_unique_tasks=2,
        unattempted_tasks=87,
        completed_answer_scored_tasks=1,
        completed_answer_passed_tasks=1,
        partial_artifact_scored_tasks=1,
        partial_artifact_reward=0,
        completed_Agent_score_for_this_task=None,
    )
    for key in (
        "full_dataset_acceptance",
        "paired_B0_B1",
        "amd64_image_equivalence_claimed",
        "raw_questions_answers_or_request_ids_included",
    ):
        assert s[key] is False
    assert (
        p["source_hashes"]["original_receipt"]
        == s["evidence"]["original_trial_hashes"]["receipt"]
    )
    for key, path in (
        ("recovery_receipt", "receipt.json"),
        ("CTRF", "trial/verifier/ctrf.json"),
        ("stdout", "trial/verifier/test-stdout.txt"),
        ("recovery_seal", "SHA256SUMS"),
        ("plan", "plan.json"),
    ):
        assert p["source_hashes"][key] == s["evidence"]["recovery"][path]
    return {
        "status": "passed",
        "original_reward": None,
        "partial_artifact_reward": 0,
        "passed": 1,
        "tests": 6,
        "new_model_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
