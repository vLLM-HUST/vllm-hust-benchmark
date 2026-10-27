"""Offline source acquisition contracts; not native NarrativeQA validation."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/prepare_narrativeqa_original_metric.py"


def test_metric_plan_is_offline_and_creates_nothing(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--output", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["files"] == 2 and plan["bytes"] == 13904
    assert plan["model_calls"] == 0 and not plan["source_checkbox_eligible"]
    assert not target.exists()


def test_metric_pins_keep_provider_and_known_metric_semantic_boundary():
    pins = json.loads(
        (ROOT / "configs/workload-native/narrativeqa-metric-pins.json").read_text(
            "utf-8"
        )
    )
    assert pins["source_id"] == "NARRATIVEQA" and pins["references"] == 2
    assert "not dataset-author" in pins["metric_provider"]
    assert "weights(0,0,0,1)" in pins["limitations"]
    assert "not pltrdy" in pins["dependency_boundary"]
    for item in pins["scoring_files"]:
        assert len(item["sha256"]) == 64 and len(item["git_blob"]) == 40
        assert item["url"].endswith(item["git_blob"])


def test_metric_resume_requires_explicit_execution(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--output", str(tmp_path / "unused"), "--resume"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires --execute" in result.stderr
