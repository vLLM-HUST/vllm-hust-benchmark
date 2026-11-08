"""Offline controls for WorkArena sample selection, not native execution."""

import importlib.util
import json
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


@pytest.fixture
def agentlab_entry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_workarena_agentlab")


def test_dedicated_instance_guard_does_not_use_pool(agentlab_entry):
    with pytest.raises(ValueError, match="no pool fallback"):
        agentlab_entry.require_dedicated_instance({"SNOW_INSTANCE_POOL": "pool.json"})
    credentials = {
        name: "fixture" for name in ("SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD")
    }
    agentlab_entry.require_dedicated_instance(
        {**credentials, "SNOW_INSTANCE_URL": "https://fixture.service-now.com"}
    )
    with pytest.raises(ValueError, match="dedicated HTTPS"):
        agentlab_entry.require_dedicated_instance(
            {
                **credentials,
                "SNOW_INSTANCE_URL": "https://fixture.service-now.com.invalid",
            }
        )


def test_native_runner_checks_the_existing_inventory(tmp_path, agentlab_entry):
    row = {"task_id": "a", "seed": 19, "category": "fixture"}
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "delivery_inventory": {
                    "WORKARENA": {
                        "fixed_l1_sample_preflight_20261107": {"selected": [row]}
                    }
                }
            }
        )
    )
    agentlab_entry.verify_selected_sample(
        inventory, [{**row, "gym_id": "browsergym/a"}]
    )
    with pytest.raises(ValueError, match="differs"):
        agentlab_entry.verify_selected_sample(inventory, [{**row, "seed": 20}])


@pytest.mark.parametrize("steps", [15, 16])
def test_native_terminal_result_checks_original_budget(tmp_path, agentlab_entry, steps):
    (tmp_path / "summary_info.json").write_text(
        json.dumps(
            {
                "cum_reward": 0.0,
                "n_steps": steps,
                "terminated": False,
                "truncated": True,
                "err_msg": None,
            }
        ),
        encoding="utf-8",
    )
    row = {"task_id": "fixture", "seed": 19, "category": "fixture"}
    run = SimpleNamespace(
        err_msg=None, exp_dir=tmp_path, env_args=SimpleNamespace(max_steps=15)
    )
    if steps == 16:
        with pytest.raises(ValueError, match="original step budget"):
            agentlab_entry.fixed_terminal_result(row, run)
    else:
        result = agentlab_entry.fixed_terminal_result(row, run)
        assert result == {
            **row,
            "steps": 15,
            "reward": 0.0,
            "terminated": False,
            "truncated": True,
        }
