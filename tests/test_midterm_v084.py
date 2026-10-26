import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_hle_packet_bytes():
    folder = ROOT / "reports/hle-native018-completed-20261026"
    lines = (folder / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 12
    for line in lines:
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest


def test_hle_and_single_agent_projection(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v084 as builder

    data = builder.build("fixture", lambda path: (ROOT / path).read_bytes())
    old = json.loads((ROOT / "website-publication/data/midterm-v083.json").read_text())
    assert data["wholechain_reference"] == old["wholechain_reference"]
    assert data["m01_method"] == old["m01_method"]
    assert (
        data["frontierscience_designated_pair"]
        == old["frontierscience_designated_pair"]
    )
    for track, total, correct, unresolved in [
        ("gold", 668, 157, 23),
        ("revision", 1143, 316, 31),
    ]:
        summary = data["hle_designated_baseline"][track]
        with (ROOT / builder.HLE / f"b0-{track}-per-case.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == total == summary["total"]
        assert (
            sum(r["correct"] == "True" for r in rows) == correct == summary["successes"]
        )
        assert (
            sum(r["judge_status"] != "ok" for r in rows)
            == unresolved
            == summary["unresolved"]
        )
    swe = data["swe_single_agent"]
    assert swe["executed_tasks"] == 1 and swe["resolved"] == 0
    assert swe["total_dataset_tasks"] == 731 and not swe["full_dataset_acceptance"]
    assert swe["agent"]["status"] == "LimitsExceeded"
    assert swe["agent"]["model_calls"] == 100 and not swe["agent"]["agent_submitted"]
    assert swe["grading"]["model_calls"] == 0 and swe["unresolved_grading"] == 0
    assert not swe["private_task_patch_and_trajectory_included"]
    sections = {s["id"]: s for s in data["sections"]}
    assert "hle-designated-baseline" in sections
    assert "swe-single-agent-observation" in sections
    for key in ("tables", "tables_en"):
        rows = {row[0]: row for row in sections["v15-remaining"][key][0]["rows"]}
        assert "23/31" in rows["HLE-Verified"][1]
        assert "0/1" in rows["SWE-bench-Pro"][1]
        assert "128/128/122" in rows["SWE-bench-Pro"][2]
    assert len(sections) == len(data["sections"])
    assert "仍在补测" not in sections["dataset-quality-inventory"]["scope"]
