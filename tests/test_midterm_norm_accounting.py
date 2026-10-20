"""Verify portable norm arithmetic, source hashes and same-run provenance."""

import hashlib
import json
import subprocess
import sys

from test_midterm_vocab_prefill import BASE, archive_members


def test_archived_scalar_rms_math(tmp_path):
    members = archive_members(BASE / "norm-accounting-r9")
    for name in ("audit_norm_accounting.py", "test_norm_accounting.py"):
        (tmp_path / name).write_bytes(members["source/" + name])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_norm_accounting.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_norm_source_counts_and_same_run_extension():
    root = BASE / "norm-accounting-r9"
    members = archive_members(root)
    summary = json.loads((root / "accounting.json").read_text())
    for name, record in summary["sources"].items():
        assert (
            hashlib.sha256(members["source/model-sources/" + name]).hexdigest()
            == record["sha256"]
        )
    counts_data = members["source/host-attention-pairs-r9.json"]
    assert hashlib.sha256(counts_data).hexdigest() == summary["counts_source_sha256"]
    counts = json.loads(counts_data)
    assert counts["ranks"][0]["roles"] == counts["ranks"][1]["roles"]
    roles = counts["ranks"][0]["roles"]
    target = roles["target"]["valid_query_tokens"]
    draft = sum(roles[f"draft_step{i}"]["valid_query_tokens"] for i in range(2))
    assert summary["target_valid_tokens"] == target
    assert summary["draft_valid_tokens"] == draft
    expected_rows = {
        "target.hidden_no_residual": target,
        "target.hidden_with_residual": target * 80,
        "draft.embedding_hidden_and_first_norm": draft * 3,
        "draft.post_attention_and_final_norm": draft * 2,
        "target.full_attention_qk": target * 180,
        "draft.full_attention_qk": draft * 18,
        "target.gdn_output": target * 960,
    }
    assert {e["name"]: e["rows"] for e in summary["entries"]} == expected_rows
    for entry in summary["entries"]:
        n, d = entry["rows"], entry["width"]
        assert entry["basic_arithmetic_subtotal"] == n * (
            4 * d + 1 + d * entry["residual"] + d * entry["gated"]
        )
        assert entry["operations"]["inverse_sqrt_evaluations"] == n
        assert entry["operations"]["silu_evaluations"] == n * d * entry["gated"]
    assert summary["basic_arithmetic_subtotal"] == sum(
        e["basic_arithmetic_subtotal"] for e in summary["entries"]
    )
    prior_data = members["source/same-run-partial-accounting-r9-v5.json"]
    combined = json.loads((root / "same-run-catalog.json").read_text())
    assert hashlib.sha256(prior_data).hexdigest() == combined["previous_catalog_sha256"]
    prior = json.loads(prior_data)
    for key, value in prior["components"].items():
        assert combined["components"][key] == value
    assert (
        combined["components"]["model_rms_and_residual_basic_ops"]
        == summary["basic_arithmetic_subtotal"]
    )
    assert (
        sum(combined["components"].values())
        == combined["covered_basic_semantic_subtotal"]
    )
    assert combined["model_effective_flops"] is combined["attainment_percent"] is None
