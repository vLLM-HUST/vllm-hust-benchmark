"""Offline public consistency verification; private diagnostics are not rerun."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(s):
    assert s["schema"] == "terminal-single-trace-diagnostic/v1"
    assert s["original_control"] == {
        "nop_passed": 1,
        "oracle_passed": 10,
        "total": 11,
        "oracle_reward": 0,
        "unchanged": True,
    }
    assert s["diagnostic"]["test_exit"] == 1 and s["diagnostic"]["setup_exit"] == 0
    assert (
        s["diagnostic"]["exception"] == "KeyError"
        and s["diagnostic"]["missing_attribute"] == "pos"
    )
    assert s["diagnostic"]["observed_node_attributes"] == [
        "vertex_end",
        "vertex_position",
        "vertex_start",
    ]
    assert (
        s["new_model_calls"] == s["new_Agent_attempts"] == s["new_control_scores"] == 0
    )
    assert s["task_reward"] is None and s["all_containers_released"]
    for key in (
        "model_qualification_passed",
        "original_trace_recovered",
        "original_environment_bytes_reproduced",
        "original_git_commit_object_recreated",
        "test_or_oracle_modified",
        "architecture_causation_or_upgrade_history_proven",
        "private_task_or_answer_bodies_included",
    ):
        assert s[key] is False, key
    assert s["no_build_isolation_disclosed"] and s["source_blob_count"] == 144
    assert len(s["versions"]) == 25
    assert s["restricted_evidence"]["r2_files"] == 179
    return True


def verify():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        assert not Path(rel).is_absolute() and ".." not in Path(rel).parts
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h, rel
    s = json.loads((ROOT / "summary.json").read_text())
    validate(s)
    review = json.loads((ROOT / "independent-review.json").read_text())
    assert review["trace_sha256"] == s["restricted_evidence"]["r2_trace_sha256"]
    assert review["sealed_files_verified"] == s["restricted_evidence"]["r2_files"]
    return {
        "status": "passed",
        "scope": "public hashes and semantic consistency; not rerunning private diagnostics",
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
