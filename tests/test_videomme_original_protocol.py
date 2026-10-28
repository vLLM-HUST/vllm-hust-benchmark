"""Frozen metadata/protocol constraints only; no dataset acquisition or inference."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_original_profiles_keep_subtitle_and_no_subtitle_distinct():
    spec = json.loads(
        (ROOT / "configs/workload-native/videomme-original-protocol.json").read_text(
            "utf-8"
        )
    )
    assert [p["official_name"] for p in spec["profiles"]] == [
        "Video-MME_8frame",
        "Video-MME_8frame_subs",
    ]
    assert [p["use_subtitle"] for p in spec["profiles"]] == [False, True]
    assert all(p["frames"] == 8 and p["fps"] == -1 for p in spec["profiles"])
    assert not spec["source_checkbox_eligible"] and spec["completed_native_calls"] == 0
    assert spec["remaining"] and spec["usage_boundary"]


def test_frozen_selection_keeps_nine_videos_and_all_original_questions():
    spec = json.loads(
        (ROOT / "configs/workload-native/videomme-original-protocol.json").read_text(
            "utf-8"
        )
    )
    assert spec["original_tasks"] == 2700 and spec["original_videos"] == 900
    assert (
        len(spec["selected_original_positions"])
        == spec["selected_tasks_per_profile"]
        == 27
    )
    assert spec["selected_original_positions"] == [
        start + offset for start in (0, 900, 1800) for offset in range(9)
    ]
    assert all(len(ids) == 3 for ids in spec["selected_original_video_ids"].values())
    assert spec["planned_native_calls"] == 54
