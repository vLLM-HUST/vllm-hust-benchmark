"""Original software plan only; never download or execute benchmark tasks in CI."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_streamingbench_plan_preserves_all_native_software_modes(tmp_path):
    profile = json.loads(
        (ROOT / "configs/workload-native/streamingbench-author-source.json").read_text()
    )
    paths = {item["path"] for item in profile["files"]}
    assert {
        "LICENSE",
        "scripts/preprocess.sh",
        "scripts/eval.sh",
        "scripts/stats.sh",
        "src/benchmark/StreamingBench.py",
        "src/benchmark/StreamingBenchSQA.py",
        "src/benchmark/StreamingBenchProactive.py",
        "src/benchmark/StreamingOpenStreamText.py",
        "src/model/Qwen2VL.py",
    } <= paths
    assert not any(path.endswith((".json", ".mp4", ".png")) for path in paths)
    target = tmp_path / "author"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "streamingbench",
            "--author-root",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["source_id"] == "STREAMINGBENCH"
    assert plan["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert plan["files"] == len(paths)
    assert plan["model_calls"] == 0
    assert not plan["source_checkbox_eligible"]
    assert not target.exists()
