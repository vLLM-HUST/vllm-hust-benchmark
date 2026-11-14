"""Synthetic figure metadata must describe the current input,not old outcomes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_mixed_provenance_and_changed_goal_outcomes_reach_the_figures(tmp_path):
    python = os.environ.get("PLOT_TEST_PYTHON", sys.executable)
    version = subprocess.run([python, "-c", "import matplotlib;print(matplotlib.__version__)"], capture_output=True, text=True)
    if version.returncode or version.stdout.strip() != "3.10.8":
        pytest.skip("Figure verification requires pinned matplotlib3.10.8")
    data = json.loads((ROOT / "reports/five-datasets-standard-20261113/summary-public-v2.json").read_bytes())
    data.update(reused_stages=2, new_inference_stages=13)
    for row in data["project_final_assessment"]["targets"]["3.4.2"]["numeric_results_by_profile"]:
        row["numeric_threshold_met"] = True
    path = tmp_path / "synthetic-summary.json"
    path.write_text(json.dumps(data))
    subprocess.run([python, str(ROOT / "scripts/plot_five_dataset_results.py"), "--summary", str(path), "--out", str(tmp_path)], check=True)
    subprocess.run([python, str(ROOT / "scripts/plot_project_final_goals.py"), "--summary", str(path), "--output", str(tmp_path)], check=True)
    assert "2 verified reused stages, 13 new native stages" in (tmp_path / "throughput-latency.svg").read_text()
    final = (tmp_path / "project-final-targets.svg").read_text()
    assert "All profiles meet numeric thresholds" in final
    assert "HLE misses" not in final
