"""Invented CPU fixtures only; no neural or video-model qualification."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location(
        "ovo_npu_entry", ROOT / "scripts/run_ovobench_original_npu.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import score_ovobench_original_saved as saved_score
finally:
    sys.path.pop(0)


def test_video_input_keeps_video_not_image_frames():
    body = entry.video_messages("/private/clip.mp4", "unchanged prompt", 64)
    assert body == [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "/private/clip.mp4",
                    "max_pixels": 151200,
                    "nframes": 64,
                },
                {"type": "text", "text": "unchanged prompt"},
            ],
        }
    ]
    for value in (True, 0, 1, 65, 64.0):
        with pytest.raises(ValueError):
            entry.video_messages("x", "y", value)


def test_scalar_text_adapter_is_lossless_not_answer_extraction():
    text = "  The answer may be A or B.\n"
    assert entry.scorer_text([text]) == text
    assert entry.scorer_text(text) == text
    for value in (None, [], ["A", "B"], [["A"]], 1):
        with pytest.raises(ValueError):
            entry.scorer_text(value)


def test_full_media_requires_all_clips_and_labels_observed_hashes_honestly(tmp_path):
    selection = {"required_native_clips": [{"path": "0.mp4"}, {"path": "1.mp4"}]}
    registered = {
        "all_required_clips_registered": False,
        "files": [{"path": "0.mp4", "bytes": 3, "sha256": "fixture-original"}],
    }
    with pytest.raises(ValueError):
        entry.media_input_snapshot(selection, registered, tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        entry.media_input_snapshot(selection, registered, tmp_path, full=True)
    (tmp_path / "1.mp4").write_bytes(b"invented non-video fixture")
    result = entry.media_input_snapshot(selection, registered, tmp_path, full=True)
    assert result[0]["registered_original_identity"] is True
    assert result[1]["registered_original_identity"] is False
    assert (
        result[1]["sha256"] == hashlib.sha256(b"invented non-video fixture").hexdigest()
    )


def test_scoring_copy_preserves_labels_failures_and_raw_prediction():
    original = {
        "backward": [{"response": [" A\n"], "ground_truth": "B"}],
        "realtime": [{"response": None}],
        "forward": [
            {"task": "REC", "test_info": [{"response": ["count=12"], "count": 4}]}
        ],
    }
    result, counts = saved_score.projection(original)
    assert counts == {"response_slots": 3, "singleton_lists_unwrapped": 2}
    assert result["backward"][0] == {"response": " A\n", "ground_truth": "B"}
    assert result["realtime"][0]["response"] is None
    assert result["forward"][0]["test_info"][0] == {"response": "count=12", "count": 4}
    assert original["backward"][0]["response"] == [" A\n"]


def test_original_groups_keep_forward_stages_and_input_order():
    forward = {"task": "SSR", "test_info": [{"step": "one"}, {"step": "two"}]}
    backward = {"task": "ASI"}
    realtime = {"task": "OCR"}
    groups = entry.group_annotations([forward, backward, realtime])
    assert groups == {
        "backward": [backward],
        "realtime": [realtime],
        "forward": [forward],
    }
    assert groups["forward"][0] is forward
    with pytest.raises(ValueError):
        entry.group_annotations([{"task": "invented"}])


@pytest.mark.parametrize("lfs", [False, True])
def test_model_identity_rejects_wrong_revision_or_same_size_changes(tmp_path, lfs):
    raw = b"invented fixture, not model weights"
    path = tmp_path / "fixture"
    path.write_bytes(raw)
    item = {
        "path": path.name,
        "bytes": len(raw),
        "git_blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
    }
    if lfs:
        item["lfs_sha256"] = hashlib.sha256(raw).hexdigest()
    pins = {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "revision": "fixture-revision",
        "files": [item],
    }
    entry.verify_model(tmp_path, "fixture-revision", pins)
    with pytest.raises(ValueError):
        entry.verify_model(tmp_path, "wrong-revision", pins)
    path.write_bytes(b"x" * len(raw))
    with pytest.raises(ValueError):
        entry.verify_model(tmp_path, "fixture-revision", pins)
