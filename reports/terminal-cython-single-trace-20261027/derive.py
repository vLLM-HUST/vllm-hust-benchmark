"""Rebuild the public diagnostic summary from controlled, sealed CPU evidence."""

import hashlib
import json
from pathlib import Path

W = Path("/swe-persist/swe-pro-shuhao-20261025")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sealed(root, name="SHA256SUMS"):
    result = {}
    for line in (root / name).read_text().splitlines():
        h, rel = line.split("  ", 1)
        assert not Path(rel).is_absolute() and ".." not in Path(rel).parts
        assert sha(root / rel) == h, rel
        result[rel] = h
    return result


def derive():
    old = W / "terminal-build-cython-arm64-r1"
    r1 = W / "terminal-cython-single-trace-r1"
    r2 = W / "terminal-cython-single-trace-r2"
    original = sealed(old, "SHA256SUMS-controls-r1")
    first, final = sealed(r1), sealed(r2)

    def read(p):
        return json.loads(p.read_text())

    failed, result, plan = (
        read(r1 / "receipt.json"),
        read(r2 / "receipt.json"),
        read(r2 / "plan.json"),
    )
    assert failed["status"] == "failed" and failed["setup_exit"] == 128
    assert failed["all_containers_released"] and result["all_containers_released"]
    assert result["status"] == "diagnostic_completed"
    assert result["setup_exit"] == 0 and result["test_exit"] == 1
    trace = (r2 / "single-test.stdout").read_text()
    assert "representation.py:338: KeyError" in trace and "KeyError: 'pos'" in trace
    assert "{'vertex_end': 20, 'vertex_position': 0, 'vertex_start': 0}" in trace
    return {
        "schema": "terminal-single-trace-diagnostic/v1",
        "task": "build-cython-ext",
        "kind": "reconstructed_single_test_diagnostic_not_control_score",
        "original_control": {
            "nop_passed": 1,
            "oracle_passed": 10,
            "total": 11,
            "oracle_reward": 0,
            "unchanged": True,
        },
        "diagnostic": {
            "setup_exit": 0,
            "test_exit": 1,
            "test_node": plan["test_node"],
            "exception": "KeyError",
            "missing_attribute": "pos",
            "observed_node_attributes": [
                "vertex_end",
                "vertex_position",
                "vertex_start",
            ],
            "source_location": "pyknotid/representations/representation.py:338",
            "preceding_call": "planarity.networkx_graph(pg)",
            "conclusion": "Attribute interface mismatch observed in this reconstruction.",
        },
        "source_commit": plan["git_commit"],
        "source_blob_count": 144,
        "versions": plan["versions"],
        "resources": {
            "cpus": 1,
            "memory_bytes": 2147483648,
            "cpuset": "24-31",
            "network_MTU": 1400,
        },
        "budgets": {"setup_seconds": 600, "single_test_seconds": 900},
        "prior_setup_failure": {
            "exit": 128,
            "stage": "Git HTTP2 transport",
            "test_executed": False,
            "preserved": True,
        },
        "new_model_calls": 0,
        "new_Agent_attempts": 0,
        "new_control_scores": 0,
        "model_qualification_passed": False,
        "task_reward": None,
        "all_containers_released": True,
        "original_trace_recovered": False,
        "original_environment_bytes_reproduced": False,
        "original_git_commit_object_recreated": False,
        "no_build_isolation_disclosed": True,
        "test_or_oracle_modified": False,
        "architecture_causation_or_upgrade_history_proven": False,
        "private_task_or_answer_bodies_included": False,
        "restricted_evidence": {
            "controlled_location": "Authorized CPU persistent workspace",
            "original_seal_sha256": sha(old / "SHA256SUMS-controls-r1"),
            "r1_seal_sha256": sha(r1 / "SHA256SUMS"),
            "r2_seal_sha256": sha(r2 / "SHA256SUMS"),
            "original_files": len(original),
            "r1_files": len(first),
            "r2_files": len(final),
            "r2_plan_sha256": sha(r2 / "plan.json"),
            "r2_receipt_sha256": sha(r2 / "receipt.json"),
            "r2_trace_sha256": sha(r2 / "single-test.stdout"),
            "r2_files_sha256": final,
        },
    }


if __name__ == "__main__":
    print(json.dumps(derive(), ensure_ascii=False, indent=2))
