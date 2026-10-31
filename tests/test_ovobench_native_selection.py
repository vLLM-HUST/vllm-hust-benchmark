"""Invented selection fixtures, not original videos or inference evidence."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ovo_selection", ROOT / "scripts/prepare_ovobench_native_selection.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def cases():
    result = []
    for task in sorted(entry.TASKS):
        for _ in range(4):
            row = {
                "id": len(result),
                "task": task,
                "video": "unchanged.mp4",
                "question": "unchanged",
            }
            if task in entry.FORWARD:
                row["test_info"] = [
                    {"realtime": 2, "count": 1},
                    {"realtime": 4, "count": 2},
                ]
            result.append(row)
    return result


def test_every_first_three_keeps_all_fields_and_forward_stages():
    rows = cases()
    original = deepcopy(rows)
    selected, clips = entry.fixed_selection(rows)
    assert rows == original and len(selected) == 36
    assert [row["id"] for row in selected] == [
        row["id"] for i, row in enumerate(rows) if i % 4 < 3
    ]
    assert len(clips) == 27 + 9 * 2
    for row in selected:
        assert row == original[row["id"]]
        actual = [item for item in clips if item["case_id"] == row["id"]]
        assert len(actual) == (
            len(row["test_info"]) if row["task"] in entry.FORWARD else 1
        )


def test_duplicate_or_missing_original_tasks_fail_not_replaced():
    rows = cases()
    with pytest.raises(ValueError, match="unique original"):
        entry.fixed_selection(rows + [rows[0]])
    with pytest.raises(ValueError, match="all twelve"):
        entry.fixed_selection(rows[:-4])
    forward = next(row for row in rows if row["task"] in entry.FORWARD)
    forward["test_info"] = []
    with pytest.raises(ValueError, match="complete original forward"):
        entry.fixed_selection(rows)


def test_explicit_subset_output_preserves_original_cases_and_never_overwrites(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "original.json"
    raw = json.dumps(cases()).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(
        entry,
        "source_spec",
        lambda _: {
            "files": [
                {
                    "path": path.name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ]
        },
    )
    plan = entry.prepare(data)
    assert plan["selected_original_cases"] == 36
    output = tmp_path / "fresh-output"
    report = entry.prepare(data, output)
    expected, clips = entry.fixed_selection(cases())
    assert (
        json.loads((output / "original-first3-each-task.json").read_text()) == expected
    )
    assert report["required_native_clips"] == clips
    assert path.read_bytes() == raw
    with pytest.raises(FileExistsError):
        entry.prepare(data, output)
