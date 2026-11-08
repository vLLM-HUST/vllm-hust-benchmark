"""Preparation guards only; no original data, task execution or neural calls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "discovery_case_prepare",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_discoverybench_original_cases.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def record(blockers):
    return SimpleNamespace(
        task={"task_id": "fixture/0/0"}, runtime={"blockers": blockers}
    )


def test_selection_preserves_explicit_original_order():
    rows = [record([]), record([])]
    assert entry.selected_records(rows, [1, 0]) == [(1, rows[1]), (0, rows[0])]


@pytest.mark.parametrize("positions", [[], [0, 0], [-1], [2]])
def test_invalid_sampling_not_repaired(positions):
    with pytest.raises(ValueError):
        entry.selected_records([record([])], positions)


def test_blocked_case_is_not_replaced_by_another():
    with pytest.raises(ValueError, match="do not replace"):
        entry.selected_records([record(["ORIGINAL_ASSET_BLOCKED"]), record([])], [0])


def test_preparation_never_overwrites_or_writes_into_original_data(tmp_path):
    data = tmp_path / "original"
    data.mkdir()
    with pytest.raises(ValueError):
        entry.output_path(data, data / "new-run")
    with pytest.raises(FileExistsError):
        entry.output_path(data, data)
    assert entry.output_path(data, tmp_path / "new-run")[1] == tmp_path / "new-run"
