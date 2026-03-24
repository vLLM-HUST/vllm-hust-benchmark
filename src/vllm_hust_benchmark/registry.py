from __future__ import annotations

import json
from importlib import resources

from vllm_hust_benchmark.models import ScenarioDefinition


def load_official_scenarios() -> list[ScenarioDefinition]:
    with resources.files("vllm_hust_benchmark.data").joinpath(
        "official_scenarios.json"
    ).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scenarios: list[ScenarioDefinition] = []
    for item in payload["scenarios"]:
        scenarios.append(
            ScenarioDefinition(
                name=item["name"],
                title=item["title"],
                benchmark_type=item["benchmark_type"],
                description=item["description"],
                tags=tuple(item.get("tags", [])),
                defaults=dict(item.get("defaults", {})),
                leaderboard=dict(item.get("leaderboard", {})),
            )
        )
    return scenarios


def get_scenario(name: str) -> ScenarioDefinition:
    for scenario in load_official_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown scenario: {name}")


def filter_scenarios(
    *, benchmark_type: str | None = None, tag: str | None = None
) -> list[ScenarioDefinition]:
    scenarios = load_official_scenarios()
    if benchmark_type:
        scenarios = [item for item in scenarios if item.benchmark_type == benchmark_type]
    if tag:
        scenarios = [item for item in scenarios if tag in item.tags]
    return scenarios