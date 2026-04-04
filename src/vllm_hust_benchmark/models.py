from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SERVE_PARAMETER_ALIASES = {
    "num_iters_warmup": "num_warmups",
}

SERVE_DATASET_PARAMETER_ALIASES = {
    "random": {
        "batch_size": "random_batch_size",
    },
}


def _apply_parameter_aliases(
    parameters: dict[str, Any], aliases: dict[str, str]
) -> dict[str, Any]:
    normalized = dict(parameters)
    for source_key, target_key in aliases.items():
        if source_key in normalized and target_key not in normalized:
            normalized[target_key] = normalized.pop(source_key)
    return normalized


def normalize_scenario_parameters(
    benchmark_type: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(parameters)
    if benchmark_type != "serve":
        return normalized

    normalized = _apply_parameter_aliases(normalized, SERVE_PARAMETER_ALIASES)

    dataset_name = normalized.get("dataset_name")
    if isinstance(dataset_name, str):
        dataset_aliases = SERVE_DATASET_PARAMETER_ALIASES.get(dataset_name)
        if dataset_aliases:
            normalized = _apply_parameter_aliases(normalized, dataset_aliases)

    return normalized



def render_parameter_flags(parameters: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key, value in parameters.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                flags.append(flag)
            continue
        flags.extend([flag, str(value)])
    return flags


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    title: str
    benchmark_type: str
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    defaults: dict[str, Any] = field(default_factory=dict)
    leaderboard: dict[str, Any] = field(default_factory=dict)

    def merge_parameters(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = dict(self.defaults)
        if overrides:
            merged.update({key: value for key, value in overrides.items() if value is not None})
        return normalize_scenario_parameters(self.benchmark_type, merged)

    def render_command(
        self,
        *,
        model: str,
        overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        merged = self.merge_parameters(overrides)
        command = ["vllm", "bench", self.benchmark_type, "--model", model]
        command.extend(render_parameter_flags(merged))
        return command
