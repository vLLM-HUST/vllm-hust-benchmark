"""Verify original circuit controls without starting a task."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    whitelist = json.loads((HERE / "whitelist.json").read_text())
    for name, expected in whitelist["files"].items():
        assert Path(name).name == name
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    value = json.loads((HERE / "summary.json").read_text())
    assert value["files_verified"] == 81
    assert value["original_budget"] == {"build": 600, "agent": 3600, "verifier": 3600}
    assert value["model_Agent_attempts"] == 0
    assert value["original3tests_and_32000gate_limit_unchanged"] is True
    assert value["two_distinct_fresh_containers"] is True
    assert value["CPU_containers_absent"] and value["controller_exited"]
    names = {
        "test_outputs.py::test_gates_file_exists",
        "test_outputs.py::test_gates_file_size",
        "test_outputs.py::test_sqrt_fib",
    }
    for name, reward, passed, failed in (("nop", 0, 2, 1), ("oracle", 1, 3, 0)):
        row = value["results"][name]
        assert (row["reward"], row["passed"], row["failed"]) == (reward, passed, failed)
        assert set(row["test_statuses"]) == names
        assert list(row["test_statuses"].values()).count("passed") == passed
        assert list(row["test_statuses"].values()).count("failed") == failed
        assert row["released"] and row["resources_and_four_mounts_verified"]
    return {
        "status": "passed",
        "nop": "2/3 reward0",
        "oracle": "3/3 reward1",
        "new_model_attempts": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
