"""Re-run the raw recorder audit from the committed archive."""

import hashlib
import json

import pytest

from test_midterm_vocab_prefill import BASE, archive_members


def test_deferred_recorder_archive(tmp_path):
    archive = BASE / "deferred-recorder-r1"
    members = archive_members(archive)
    for name, data in members.items():
        relative = name.removeprefix("source/")
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    namespace = {
        "__name__": "archived_audit",
        "__file__": str(tmp_path / "audit_moe_deferred_capture.py"),
    }
    exec(
        compile(
            members["source/audit_moe_deferred_capture.py"],
            "audit_moe_deferred_capture.py",
            "exec",
        ),
        namespace,
    )
    result = namespace["audit"](tmp_path)
    assert result == json.loads((archive / "summary.json").read_text())
    assert result["production_model_coverage_proven"] is False
    receipt = json.loads(members["moe-deferred-capture-control-r1/receipt.json"])
    assert receipt["reserved_devices"] == [4, 5, 6, 7]
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    raw_path = tmp_path / "moe-deferred-capture-r1/rank-0.json"
    original = raw_path.read_bytes()
    for key in ("cursor", "overflow"):
        raw = json.loads(original)
        row = raw["cases"][1]["roles"][1]
        row[key] = 0 if key == "cursor" else False
        raw_path.write_text(json.dumps(raw))
        with pytest.raises(AssertionError):
            namespace["audit"](tmp_path)
