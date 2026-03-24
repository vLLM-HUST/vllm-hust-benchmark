from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    title: str
    benchmark_type: str
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    defaults: dict[str, Any] = field(default_factory=dict)
    leaderboard: dict[str, Any] = field(default_factory=dict)

    def render_command(
        self,
        *,
        model: str,
        overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        merged = dict(self.defaults)
        if overrides:
            merged.update({key: value for key, value in overrides.items() if value is not None})

        command = ["vllm", "bench", self.benchmark_type]
        command.extend(["--model", model])

        for key, value in merged.items():
            if value is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(flag)
                continue
            command.extend([flag, str(value)])

        return command