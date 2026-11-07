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


def main():
    import browsergym.workarena as workarena
    import gymnasium

    if workarena.__version__ != "0.5.3":
        raise ValueError("requires the pinned original WorkArena 0.5.3 package")
    sampled = workarena.get_all_tasks_agents(filter="l1", meta_seed=42, n_seed_l1=10)
    selected = fixed_l1_sample(
        sampled,
        lambda task_id: workarena.get_task_category(task_id)[1],
        gymnasium.registry,
    )
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
                "instance_accessed": False,
                "agent_or_scoring_executed": False,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
