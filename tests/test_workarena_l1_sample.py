"""Offline controls for WorkArena sample selection, not native execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "workarena_l1_plan",
    Path(__file__).resolve().parents[1] / "scripts/plan_workarena_l1_sample.py",
)
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


class Task:
    def __init__(self, task_id):
        self.task_id = task_id

    def get_task_id(self):
        return self.task_id


def test_keeps_first_publisher_seeds_without_reordering():
    a, b, c, d, e = (Task(key) for key in "abcde")
    pairs = [(c, 8), (c, 9), (b, 2), (d, 3), (e, 5), (a, 1)]
    labels = dict(a="read", b="form", c="form", d="menu", e="list")
    planned = PLANNER.fixed_l1_sample(
        pairs, labels.get, {f"browsergym/{k}" for k in labels}
    )
    assert [(r["task_id"], r["seed"]) for r in planned] == [
        ("c", 8),
        ("d", 3),
        ("e", 5),
    ]


@pytest.mark.parametrize("registered,seed", [(set(), 1), ({"browsergym/a"}, True)])
def test_does_not_skip_invalid_selected_pair(registered, seed):
    with pytest.raises(ValueError, match="not registered"):
        PLANNER.fixed_l1_sample([(Task("a"), seed)], lambda _: "form", registered)


def test_refuses_to_backfill_with_unclassified_tasks():
    with pytest.raises(ValueError, match="three registered categories"):
        PLANNER.fixed_l1_sample([(Task("a"), 1)], lambda _: None, {"browsergym/a"})


def original_benchmark():
    return SimpleNamespace(
        name="workarena_l1",
        backends=["workarena"],
        is_multi_tab=False,
        env_args_list=[
            SimpleNamespace(task_name="a", task_seed=19, max_steps=15, task_kwargs=None)
        ],
        high_level_action_set_args=SimpleNamespace(
            subsets=("workarena",),
            multiaction=False,
            strict=False,
            retry_with_force=True,
            demo_mode="off",
        ),
    )


def test_binds_actual_env_args_without_reconstructing_it():
    benchmark = original_benchmark()
    bound, actions = PLANNER.bind_l1_environment(
        benchmark, [{"task_id": "a", "seed": 19}], 1
    )
    assert bound[0] is benchmark.env_args_list[0]
    assert actions["subsets"] == ("workarena",)


@pytest.mark.parametrize(
    "field,value",
    [("task_seed", 20), ("max_steps", 5), ("task_kwargs", {"changed": True})],
)
def test_refuses_changes_to_published_task_configuration(field, value):
    benchmark = original_benchmark()
    setattr(benchmark.env_args_list[0], field, value)
    with pytest.raises(ValueError, match="original WorkArena environment"):
        PLANNER.bind_l1_environment(benchmark, [{"task_id": "a", "seed": 19}], 1)


def test_refuses_changed_action_set():
    benchmark = original_benchmark()
    benchmark.high_level_action_set_args.multiaction = True
    with pytest.raises(ValueError, match="action profile changed"):
        PLANNER.bind_l1_environment(benchmark, [{"task_id": "a", "seed": 19}], 1)
