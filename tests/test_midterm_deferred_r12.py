"""Reconstruct successful capture and retain post-response device work."""

import json

from test_midterm_vocab_prefill import BASE, archive_members


def test_r12_capture_and_request_boundaries(tmp_path, monkeypatch):
    root = BASE / "deferred-device-r12"
    members = archive_members(root)
    for name, data in members.items():
        relative = name.removeprefix("source/")
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace = {
        "__name__": "archived_r12_audit",
        "__file__": str(tmp_path / "audit_deferred_request_window.py"),
    }
    exec(
        compile(
            members["source/audit_deferred_request_window.py"],
            "audit_deferred_request_window.py",
            "exec",
        ),
        namespace,
    )
    result = namespace["audit"](tmp_path / "microflow-token-accounting-r12")
    assert result == json.loads((root / "summary.json").read_text())
    assert result["completed"] == 200 and result["failed"] == 0
    assert result["whole_model_effective_flops"] is result["m01_attainment"] is None
    for rank in result["ranks"]:
        counts = rank["request_boundary_frame_counts"]
        assert counts["inside"] == {"target": 888, "draft": 888}
        assert counts["after"]["draft"] == 1
        assert counts["after"]["target"] + counts["crosses_boundary"]["target"] == 1
        assert (
            rank["last_capture_event_end_wall_bounds"][0]
            > result["request_phase"]["end_wall"]
        )
    system = json.loads(members["system/r12-system-control/receipt.json"])
    assert system["profiler_exit_code"] == system["profiler_cleanup_exit_code"] == 0
    assert system["sampler_errors"] == []
    assert system["workload_status"] == "measured"
