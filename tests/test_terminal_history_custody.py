"""Validate the committed custody locator, not experimental performance."""

import json
from pathlib import Path
import re


def test_terminal_history_release_locator():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "archive/terminal-history-20261021/release.json").read_text()
    )
    assert receipt["schema"] == "release-custody-receipt/v1"
    assert receipt["repository"] == "vLLM-HUST/vllm-hust-benchmark"
    assert receipt["release_tag"] == "evidence-terminal-history-20261021-v1"
    assert receipt["release_id"] > 0
    assert receipt["indexed_paths"] == 1196 and receipt["terminal_runs"] == 93
    assert (
        receipt["download_back_verified"] and receipt["all_new_members_sha256_verified"]
    )
    assert receipt["source_files_deleted"] and receipt["deleted_file_count"] == 1103
    assert receipt["deleted_logical_bytes"] == 5485630183
    assert receipt["local_receipts_retained"] and not receipt["performance_admission"]
    assets = {row["name"]: row for row in receipt["assets"]}
    assert len(assets) == len(receipt["assets"])
    assert {
        "terminal-history.tar.gz",
        "manifest.json",
        "verification.json",
        "cleanup-receipt.json",
        "archive_terminal_history_v1.py",
        "verify_history_custody_v1.py",
        "test_history_custody_safety.py",
        "history-custody-safety-tests.xml",
    } <= assets.keys()
    for row in assets.values():
        assert row["bytes"] > 0
        assert re.fullmatch("[0-9a-f]{64}", row["sha256"])
