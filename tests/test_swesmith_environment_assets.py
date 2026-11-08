import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_original_environment_plan_preserves_source_and_no_execution(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "swesmith-env",
            "--author-root",
            str(target),
            "--action",
            "plan",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["source_id"] == "SWESMITH"
    assert report["files"] == 4 and report["bytes"] == 3302
    assert report["model_calls"] == 0
    assert report["source_checkbox_eligible"] is False
    assert not target.exists()
    pins = json.loads(
        (ROOT / "configs/workload-native/swesmith-env-author-source.json").read_text()
    )
    assert pins["repository"] == "SWE-bench/SWE-smith-envs"
    assert len({item["path"] for item in pins["files"]}) == 4
    assert all(".." not in Path(item["path"]).parts for item in pins["files"])
