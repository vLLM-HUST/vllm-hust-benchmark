"""Tiny invented selection fixtures, never original video or model validation."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "stream_selection",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_streamingbench_native_selection.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_fixed_selection_preserves_entire_nested_sequence_and_media():
    group = {
        "video_path": "./videos/original.mp4",
        "questions": [
            {
                "question": "query",
                "time_stamp": "00:01:00",
                "answer": "A",
                "audio_path": "./audio/original.wav",
            }
        ],
    }
    rows = [[deepcopy(group), deepcopy(group)] for _ in range(4)]
    snapshot = deepcopy(rows)
    selected, media, groups, questions = runner.select_groups(rows)
    assert selected == snapshot[:3]
    assert rows == snapshot
    assert groups == questions == 6
    assert media == [
        ("audio", "./audio/original.wav"),
        ("video", "./videos/original.mp4"),
    ]
    assert len(runner.MODES) == 7
    assert "proactive_50" not in runner.MODES


def test_insufficient_originals_fail_instead_of_filling_with_new_tasks():
    with pytest.raises(ValueError, match="at least three"):
        runner.select_groups([{}])


def test_prepare_is_explicit_external_and_never_overwrites(tmp_path, monkeypatch):
    data = tmp_path / "originals"
    data.mkdir()
    files = []
    row = {
        "video_path": "./videos/original.mp4",
        "questions": [{"question": "query", "time_stamp": "00:00:01", "answer": "A"}],
    }
    for mode in runner.MODES:
        relative = f"src/data/questions_{mode}.json"
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps([row] * 4).encode()
        path.write_bytes(raw)
        files.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": 4,
            }
        )
    monkeypatch.setattr(runner, "source_spec", lambda _: {"files": files})
    output = tmp_path / "selected"
    plan = runner.prepare(data)
    assert not output.exists()
    assert sum(job["selected_outer_rows"] for job in plan["jobs"]) == 21
    assert plan["model_calls"] == 0 and not plan["source_checkbox_eligible"]
    runner.prepare(data, output)
    assert len(list(output.glob("questions_*.json"))) == 7
    assert all(
        json.loads(path.read_text()) == [row] * 3
        for path in output.glob("questions_*.json")
    )
    with pytest.raises(FileExistsError):
        runner.prepare(data, output)
    with pytest.raises(ValueError, match="external and separate"):
        runner.prepare(data, data / "new-subset")
