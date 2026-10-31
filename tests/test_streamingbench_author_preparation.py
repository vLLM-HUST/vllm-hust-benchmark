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
    media = profile["media_archive_inventory"]
    assert len(media["archives"]) == 8
    members = media["selected_original_video_members"]
    assert len({item["original_task_media_path"] for item in members}) == 10
    assert sum(item["bytes"] for item in members) == 2823253471
    assert media["directory_range_bytes_total"] == 179576
    assert media["video_payload_acquired"] is True
    assert media["fixed_video_selection_complete"] is False
    assert len(media["qualified_original_members"]) == 9
    assert (
        sum(item["bytes"] for item in media["qualified_original_members"]) == 2421684080
    )
    assert (
        sum(item["video_frames"] for item in media["qualified_original_members"])
        == 232219
    )
    assert media["whole_archives_rehashed"] is False
    assert media["media_rights_not_inferred_from_code_license"] is True
    assert media["audio_paths_require_original_preprocessing"] == 45
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
