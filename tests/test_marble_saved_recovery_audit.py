"""Stopped native consumers never authorize orphaned queue inference."""

import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
entry = importlib.import_module("audit_marble_original_research_saved")


def fixture(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    state = {
        "source_id": "MULTIAGENTBENCH",
        "status": "RUNNING_ORIGINAL_RESEARCH_NATIVE_CLI",
        "fixed_original_rows": [0, 1, 2],
        "completed": [{"original_row": 0, "cli_exit_code": 1}],
        "active_child": {"pid": 18, "birth": "125"},
    }
    (root / "state-private.json").write_text(json.dumps(state))
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"pid": 17, "birth": "123", "stage": str(root)}))
    queue = root / "model-queue"
    queue.mkdir()
    (queue / ("a" * 32 + ".request.json")).write_text("fixture only")
    return root, launch


def test_stale_running_state_does_not_authorize_orphan_requests(tmp_path):
    root, launch = fixture(tmp_path)
    report = entry.audit(root, launch, proc_root=tmp_path / "proc")
    assert report["status"] == "STOPPED_WITH_INCOMPLETE_ORIGINAL_PROTOCOL"
    assert report["reported_historical_state"].startswith("RUNNING")
    assert not report["pending_requests_may_be_served"]
    assert report["model_queue"] == {"requests": 1, "responses": 0, "pending": 1}
    assert report["new_model_calls"] == 0


def test_pid_reuse_and_missing_birth_are_not_live(tmp_path):
    process = tmp_path / "17"
    process.mkdir()
    (process / "stat").write_text(
        "17 (name with spaces) " + " ".join(["S"] + ["0"] * 18 + ["999"])
    )
    assert not entry.process_live({"pid": 17, "birth": "123"}, proc_root=tmp_path)
    assert not entry.process_live({"pid": 17}, proc_root=tmp_path)
    assert entry.process_live({"pid": 17, "birth": "999"}, proc_root=tmp_path)


def test_orphan_response_rejected(tmp_path):
    root, launch = fixture(tmp_path)
    (root / "model-queue" / ("b" * 32 + ".response.json")).write_text("fixture")
    with pytest.raises(ValueError, match="without original request"):
        entry.audit(root, launch, proc_root=tmp_path / "proc")


def test_live_parent_is_not_request_scoped_authorization(tmp_path):
    root, launch = fixture(tmp_path)
    process = tmp_path / "proc/17"
    process.mkdir(parents=True)
    (process / "stat").write_text(
        "17 (python) " + " ".join(["S"] + ["0"] * 18 + ["123"])
    )
    report = entry.audit(root, launch, proc_root=tmp_path / "proc")
    assert report["live_consumer_exists"]
    assert not report["pending_requests_may_be_served"]
