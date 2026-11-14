"""Ensure the archive is an index of immutable, versioned report sources."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_history_covers_all_midterm_revisions_and_preserves_sources():
    catalog = json.loads(
        (ROOT / "website-publication/data/report-history-v1.json").read_text()
    )
    assert catalog["schema"] == "benchmark-report-history/v1"
    assert catalog["timezone"] == "Asia/Shanghai"
    entries = catalog["entries"]
    assert len({r["id"] for r in entries}) == len(entries)
    assert [r["date"] for r in entries] == sorted(
        (r["date"] for r in entries), reverse=True
    )
    current = [r for r in entries if r["status"] == "current"]
    assert len(current) == 1 and current[0]["id"] == catalog["current_report_id"]
    archived = {r["id"]: r for r in entries if r["date_kind"] == "source_revision_date"}
    sources = list((ROOT / "website-publication/data").glob("midterm-v*.json"))
    assert set(archived) == {p.stem for p in sources}
    for p in sources:
        source = json.loads(p.read_bytes())
        row = archived[p.stem]
        assert (row["source_commit"], row["report_url"], row["report_sha256"]) == (
            source["source_commit"],
            source["report_url"],
            source["report_sha256"],
        )
        assert row["source_revision_at"].endswith("Z")
    for row in entries:
        prefix = f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{row['source_commit']}/"
        assert row["report_url"].startswith(prefix)
        if row["id"].startswith("midterm-"):
            continue
        report = ROOT / row["report_url"].removeprefix(prefix)
        assert hashlib.sha256(report.read_bytes()).hexdigest() == row["report_sha256"]
        if row.get("artifact_path"):
            assert (ROOT / "website-publication" / row["artifact_path"]).is_file()
