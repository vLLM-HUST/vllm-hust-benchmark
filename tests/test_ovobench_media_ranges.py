"""Invented TAR fixtures only; not publisher media acquisition evidence."""

import base64
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location(
        "ovo_media_ranges", ROOT / "scripts/plan_ovobench_original_media_ranges.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)


def fixture():
    header = tarfile.TarInfo("chunked_videos/0_0.mp4")
    header.size = 700
    raw = header.tobuf(format=tarfile.USTAR_FORMAT)
    index = {
        "entries": [
            {
                "header_base64": base64.b64encode(raw).decode(),
                "header_sha256": hashlib.sha256(raw).hexdigest(),
                "path": header.name,
                "bytes": header.size,
                "directory": False,
                "header_offset": 0,
                "data_offset": 512,
                "part": "chunked_videos.a",
            }
        ],
        "etags": {"chunked_videos.a": "observed-not-publisher-hash"},
        "next_offset": 1536,
        "zero_headers": 0,
        "complete": False,
    }
    inventory = {
        "parts": [
            {"path": "chunked_videos.a", "bytes": 1024},
            {"path": "chunked_videos.b", "bytes": 4096},
        ]
    }
    selection = {
        "required_native_clips": [{"path": "0_0.mp4", "case_id": 0, "stage_index": 0}],
        "required_native_clip_count": 1,
    }
    return index, selection, inventory


def test_original_member_crosses_parts_without_shortening():
    index, selection, inventory = fixture()
    report = entry.plan(index, selection, inventory)
    assert report["located_original_bytes"] == 700
    assert report["members"][0]["ranges"] == [
        {"part": "chunked_videos.a", "start": 512, "end": 1023},
        {"part": "chunked_videos.b", "start": 0, "end": 187},
    ]
    assert report["captured_archive_index_complete"] is False
    assert report["media_acquired"] is False and report["model_calls"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("header_offset", 512),
        ("data_offset", 0),
        ("bytes", 699),
        ("path", "replacement.mp4"),
        ("header_sha256", "0" * 64),
    ],
)
def test_changed_header_relations_fail(field, value):
    index, selection, inventory = fixture()
    index["entries"][0][field] = value
    with pytest.raises(ValueError):
        entry.plan(index, selection, inventory)


def test_missing_selected_clip_does_not_select_other_member():
    index, selection, inventory = fixture()
    selection["required_native_clips"][0]["path"] = "original-other-stage.mp4"
    report = entry.plan(index, selection, inventory)
    assert (
        report["located_original_clips"] == 0
        and not report["fixed_selection_fully_located"]
    )
    assert report["missing_original_members"] == [
        "chunked_videos/original-other-stage.mp4"
    ]


def test_captured_end_marker_state_is_distinct_from_partial_checkpoint():
    index, selection, inventory = fixture()
    original = deepcopy(index)
    index.update(zero_headers=1, next_offset=2048)
    assert (
        entry.plan(index, selection, inventory)["captured_archive_index_complete"]
        is False
    )
    index.update(zero_headers=2, next_offset=2560, complete=True)
    assert (
        entry.plan(index, selection, inventory)["captured_archive_index_complete"]
        is True
    )
    assert original["zero_headers"] == 0
    index["next_offset"] = 0
    with pytest.raises(ValueError, match="checkpoint/end-marker"):
        entry.plan(index, selection, inventory)


def test_out_of_archive_original_member_span_fails():
    with pytest.raises(ValueError, match="exceeds registered"):
        entry.part_ranges(3, 2, [{"path": "p", "bytes": 4}])
