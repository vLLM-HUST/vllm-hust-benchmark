"""Source-native workload envelope; not a benchmark loop or model request renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any

WORKLOAD_SCHEMA_VERSION = "source-workload/v1"


@dataclass(frozen=True, repr=False)
class WorkloadRecord:
    source: dict[str, Any]
    task: dict[str, Any]
    runtime: dict[str, Any]
    replay: dict[str, Any]
    outcome: dict[str, Any] | None = field(default=None, repr=False)

    def __repr__(self):
        return f"WorkloadRecord(source_id={self.source['source_id']!r}, payload=<not displayed>)"

    def input_record(self) -> dict:
        """Return inputs only. Never merge scorer references or historical outcomes."""
        return deepcopy(
            {
                "schema_version": WORKLOAD_SCHEMA_VERSION,
                "source": self.source,
                "task": self.task,
                "runtime": self.runtime,
                "replay": self.replay,
            }
        )

    def outcome_record(self) -> dict | None:
        return deepcopy(self.outcome)

    def request_body(self) -> dict:
        if self.runtime["kind"] != "request_replay":
            raise ValueError(
                "not a ready model request; use the source-native task/trace/environment contract"
            )
        return deepcopy(self.task["input"]["request_body"])
