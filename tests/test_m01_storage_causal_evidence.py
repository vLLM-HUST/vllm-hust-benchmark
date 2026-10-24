import hashlib
import json
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "vllm-hust-midterm-20261019"
    / "m01-recovery"
    / "storage-causal-20261024"
)


def test_storage_causal_qualification_preserves_scope_and_identity():
    reference = json.loads((ROOT / "release.json").read_text())
    raw = (ROOT / "qualification-summary.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference["qualification_summary_sha256"]
    summary = json.loads(raw)
    assert summary["run_id"] == "capture-b1-storage-r6"
    assert summary["formal_measurement"] is False
    assert summary["measurement_role"] == "diagnostic_qualification"
    assert summary["verified_sha_entries"] == reference["raw_entries"] == 23
    assert summary["raw_manifest_sha256"] == reference["raw_manifest_sha256"]
    assert summary["workload"]["completed"] == 200
    assert summary["workload"]["failed"] == 0
    assert summary["workload"]["total_input_tokens"] == 282265
    assert summary["workload"]["total_output_tokens"] == 51200
    assert summary["released"] and summary["sessions_drained"]
    assert summary["correction"]["causal_endpoint_mismatches"] == 0
    assert summary["correction"]["second_draft_rows_per_rank"] == 19354
    assert summary["correction"]["full_slot_address_mismatches"] == 0
    assert summary["logical_reconstruction_matches_both_rank_physical_dependencies"]
    for key in ("effective_flops", "reference_seconds", "m01_attainment"):
        assert summary[key] is None
        assert reference[key] is None
    assert reference["bytes"] == 233523751
    assert reference["download_url"].startswith(
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
    )
    assert reference["download_url"].endswith(reference["archive"])
