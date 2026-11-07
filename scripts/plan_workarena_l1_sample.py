"""Select a fixed L1 slice from the original WorkArena sampler, without an instance."""

import json


def fixed_l1_sample(sample, category_of, registered):
    """Keep the first original seed in each of the first three distinct categories."""
    chosen = []
    categories = set()
    for task, seed in sample:
        task_id = task.get_task_id()
        category = category_of(task_id)
        if category is None or category in categories:
            continue
        env_id = "browsergym/" + task_id
        if env_id not in registered or type(seed) is not int or not 0 <= seed < 1000:
            raise ValueError("original sampled task or seed not registered as expected")
        categories.add(category)
        chosen.append(
            {"task_id": task_id, "gym_id": env_id, "seed": seed, "category": category}
        )
        if len(chosen) == 3:
            return chosen
    raise ValueError("original L1 sample did not contain three registered categories")


def bind_l1_environment(benchmark, selected, candidate_count):
    """Resolve fixed seeds to the original BrowserGym evaluation configuration."""
    if (
        benchmark.name != "workarena_l1"
        or benchmark.backends != ["workarena"]
        or benchmark.is_multi_tab
        or len(benchmark.env_args_list) != candidate_count
    ):
        raise ValueError("original WorkArena L1 benchmark configuration changed")
    action = benchmark.high_level_action_set_args
    expected_actions = {
        "subsets": ("workarena",),
        "multiaction": False,
        "strict": False,
        "retry_with_force": True,
        "demo_mode": "off",
    }
    if any(getattr(action, key) != value for key, value in expected_actions.items()):
        raise ValueError("original WorkArena L1 action profile changed")
    bound = []
    for row in selected:
        matches = [
            args
            for args in benchmark.env_args_list
            if (args.task_name, args.task_seed) == (row["task_id"], row["seed"])
        ]
        if (
            len(matches) != 1
            or matches[0].max_steps != 15
            or matches[0].task_kwargs is not None
        ):
            raise ValueError(
                "missing, duplicated or changed original WorkArena environment"
            )
        bound.append(matches[0])
    return bound, expected_actions


def main():
    import browsergym.workarena as workarena
    from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS
    import gymnasium

    if workarena.__version__ != "0.5.3":
        raise ValueError("requires the pinned original WorkArena 0.5.3 package")
    sampled = workarena.get_all_tasks_agents(filter="l1", meta_seed=42, n_seed_l1=10)
    selected = fixed_l1_sample(
        sampled,
        lambda task_id: workarena.get_task_category(task_id)[1],
        gymnasium.registry,
    )
    benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()
    bound, actions = bind_l1_environment(benchmark, selected, len(sampled))
    print(
        json.dumps(
            {
                "status": "FIXED_ORIGINAL_L1_SAMPLE_PLANNED_NO_INSTANCE_OR_AGENT",
                "source_id": "WORKARENA",
                "workarena_version": workarena.__version__,
                "original_sampler": {"filter": "l1", "meta_seed": 42, "n_seed_l1": 10},
                "original_sampler_candidates": len(sampled),
                "selection_rule": "first original seed in each of the first three distinct publisher categories",
                "selected": selected,
                "browsergym_benchmark": benchmark.name,
                "original_env_args_bound": len(bound),
                "original_max_steps": [args.max_steps for args in bound],
                "original_action_profile": actions,
                "instance_accessed": False,
                "agent_or_scoring_executed": False,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
