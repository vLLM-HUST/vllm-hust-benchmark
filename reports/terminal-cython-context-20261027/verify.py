"""Check public control and original Agent evidence without running a task."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def verify():
    root = HERE.parents[1]
    for item in read("public-files.json"):
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        raw = (root / relative).read_bytes()
        assert len(raw) == item["bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    for prefix in ("controls", "model"):
        manifest = read(prefix + "-whitelist.json")
        for name, digest in manifest["files"].items():
            assert (
                hashlib.sha256((HERE / (prefix + "-" + name)).read_bytes()).hexdigest()
                == digest
            )
    controls = read("controls-summary.json")
    model = read("model-summary.json")
    assert controls["original_control_preserved"]["oracle_passed"] == 10
    assert controls["original_control_preserved"]["oracle_reward"] == 0
    current = controls["new_environment_controls"]
    assert current["nop"]["passed"] == 1 and current["nop"]["failed"] == 10
    assert current["nop"]["reward"] == 0
    assert current["oracle"]["passed"] == 11 and current["oracle"]["failed"] == 0
    assert current["oracle"]["reward"] == 1
    assert all(role["all_owned_containers_released"] for role in current.values())
    environment = controls["environment_change"]
    assert (
        environment["planarity_repaired"]
        == model["environment"]["planarity_version"]
        == "0.6"
    )
    assert environment["native_sha256"] == next(
        iter(model["environment"]["native_sha256"].values())
    )
    assert (
        environment["original_pyknotid_commit"]
        == model["environment"]["pyknotid_actual_commit"]
    )
    assert environment["other24_original_versions_retained"] is True
    assert model["environment"]["other24_present_versions_match"] is True
    assert (
        model["HTTP_requests"] == model["HTTP_success"] + model["HTTP_rejected"] == 30
    )
    assert model["HTTP_success"] == model["successful_Agent_steps"] == 29
    assert model["Harbor_episode_counter"] == 30
    assert model["trial_exception"] == "ContextLengthExceededError"
    assert model["original_budget"]["context"] == 32768
    assert model["original_budget"]["turns"] == 100
    assert (
        model["agent_elapsed_seconds"]
        < model["original_budget"]["agent_seconds"]
        == 900
    )
    assert model["verifier"] is model["reward"] is model["CTRF"] is None
    assert model["official_scored"] is False
    assert model["Terminal_attempts"] == 3 and model["Terminal_unattempted"] == 86
    assert model["prior_2_attempt_records_unchanged"] is True
    assert model["recovery_boundary"]["artifact_snapshot"] == "empty manifest only"
    assert (
        model["recovery_boundary"][
            "compiled_pyknotid_binaries_original_bytes_SHA_not_captured"
        ]
        is True
    )
    boundary = read("source-boundary.json")
    whitelist = read("source-boundary-whitelist.json")
    assert (
        hashlib.sha256((HERE / "source-boundary.json").read_bytes()).hexdigest()
        == whitelist["files"]["source-boundary.json"]
    )
    assert boundary["verified_upstream_blobs"] == 144
    assert boundary["original_recorded_edits_replayed"] == 11
    assert boundary["modified_tracked_files"] == 9
    assert boundary["derived_diff_matches_original_capture"] is True
    assert (
        boundary["derived_diff_sha256"] == model["environment"]["tracked_diff_sha256"]
    )
    assert (
        boundary["original_reward"]
        is boundary["original_verifier"]
        is boundary["original_CTRF"]
        is None
    )
    assert boundary["new_Agent_model_build_verifier_calls"] == 0
    release = model["resource_release"]
    assert all(
        release[k] is True
        for k in (
            "CPU_owned_containers_released",
            "CPU_controller_and_child_exited",
            "sessions_drained",
            "all_devices_released",
            "inflight_observed_zero",
        )
    )
    assert release["service_exit_code"] == 0
    return {
        "status": "passed",
        "model_attempts": 3,
        "unattempted": 86,
        "new_task_reward": None,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
