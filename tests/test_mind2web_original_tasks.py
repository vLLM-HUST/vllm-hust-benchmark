import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mind2web_tasks", ROOT / "scripts/prepare_mind2web_original_tasks.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selection_preserves_all_fields_actions_and_order():
    tasks = [
        {
            "annotation_id": str(i),
            "extra_original_field": {"value": i},
            "actions": [
                {"action_uid": f"{i}-{j}", "raw_html": "fixture"} for j in range(i + 1)
            ],
            "action_reprs": [str(j) for j in range(i + 1)],
        }
        for i in range(4)
    ]
    before = copy.deepcopy(tasks)
    selected = module.select_original_tasks(tasks)
    assert selected == before[:3]
    assert tasks == before
    assert [len(task["actions"]) for task in selected] == [1, 2, 3]


@pytest.mark.parametrize(
    "tasks", [[], [{"annotation_id": "x", "actions": [], "action_reprs": []}] * 3]
)
def test_missing_complete_original_tasks_are_not_replaced(tasks):
    with pytest.raises(ValueError):
        module.select_original_tasks(tasks)
