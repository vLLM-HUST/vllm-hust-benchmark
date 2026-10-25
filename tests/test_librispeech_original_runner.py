import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "librispeech_original_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_librispeech_original.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_selection_contains_first_three_and_longest():
    assert runner.select_indices([1, 2, 3, 35, 4]) == [0, 1, 2, 3]
    assert runner.select_indices([35, 2, 3]) == [0, 1, 2]
    assert runner.select_indices([1, 2, 3], full=True) == [0, 1, 2]
    assert runner.select_indices([1, 2, 3], indices=[2, 1]) == [1, 2]


@pytest.mark.parametrize("indices", [[1, 1], [-1], [3]])
def test_invalid_selection_is_not_replaced(indices):
    with pytest.raises(ValueError, match="unique in-range"):
        runner.select_indices([1, 2, 3], indices=indices)
