"""Verify public original controls without adding model coverage."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = (3, 1, 900)


def verify():
    whitelist = json.loads((HERE / "whitelist.json").read_text())
    assert set(whitelist["files"]) == {"summary.json"}
    raw = (HERE / "summary.json").read_bytes()
    expected = whitelist["files"]["summary.json"]
    assert hashlib.sha256(raw).hexdigest() == expected
    s = json.loads(raw)
    count, nop_pass, timeout = EXPECTED
    assert s["model_Agent_attempts"] == 0
    assert s["original_budget"] == dict(build=600, agent=timeout, verifier=timeout)
    assert (
        s["two_distinct_fresh_containers"]
        and s["CPU_containers_absent"]
        and s["controller_exited"]
    )
    assert s["results"]["nop"]["reward"] == 0
    assert s["results"]["nop"]["passed"] == nop_pass
    assert s["results"]["oracle"]["reward"] == 1
    assert s["results"]["oracle"]["passed"] == count
    for value in s["results"].values():
        assert value["passed"] + value["failed"] == len(value["test_statuses"]) == count
        assert list(value["test_statuses"].values()).count("passed") == value["passed"]
        assert value["resources_and_four_mounts_verified"] and value["released"]
    return dict(status="passed", original_controls_only=True, model_attempts=0)


if __name__ == "__main__":
    print(json.dumps(verify()))
