import json
import re
from pathlib import Path


def test_incremental_public_protection_receipt_is_explicit_and_verifiable():
    path = (
        Path(__file__).resolve().parents[1]
        / "archive/mod-releases-20261021/repository-protection.json"
    )
    receipt = json.loads(path.read_text())
    assert receipt["all_repositories_complete"] is False
    assert receipt["all_evidence_migrated"] is False
    assert receipt["audit_conclusion"] == "success"
    assert re.fullmatch(r"[a-f0-9]{40}", receipt["policy_revision"])
    rows = receipt["repositories"]
    assert rows
    assert len({row["repository"] for row in rows}) == len(rows)
    for row in rows:
        prefix = f"https://github.com/vLLM-HUST/{row['repository']}/"
        assert row["visibility"] == "public"
        assert row["strict"] and row["enforce_admins"]
        assert row["trusted_gate_conclusion"] == "success"
        assert row["trusted_gate_run"].startswith(prefix + "actions/runs/")
        assert row["pr"].startswith(prefix + "pull/")
        assert re.fullmatch(r"[a-f0-9]{40}", row["merge_commit"])
        assert row["required_check"] == {
            "context": "Benchmark-only data (organization)",
            "app_id": 15368,
        }
