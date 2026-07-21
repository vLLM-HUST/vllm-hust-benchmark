"""Fail-closed exclusions for public leaderboard submissions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "vllm-hust-leaderboard-exclusions/v1"


@dataclass(frozen=True)
class LeaderboardExclusion:
    exclusion_id: str
    plugin_commit: str
    reason: str


def load_leaderboard_exclusions(path: Path) -> tuple[LeaderboardExclusion, ...]:
    if not path.is_file():
        return ()

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"invalid leaderboard exclusion file: {path}")

    raw_exclusions = payload.get("exclusions")
    if not isinstance(raw_exclusions, list):
        raise ValueError(f"leaderboard exclusions must be a list: {path}")

    exclusions: list[LeaderboardExclusion] = []
    for raw in raw_exclusions:
        if not isinstance(raw, Mapping) or raw.get("status") != "excluded":
            continue
        match = raw.get("match")
        if not isinstance(match, Mapping):
            raise ValueError(f"leaderboard exclusion is missing match criteria: {path}")
        plugin_commit = str(match.get("runtime_provenance.plugin.commit") or "").strip()
        exclusion_id = str(raw.get("id") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if len(plugin_commit) != 40 or not exclusion_id or not reason:
            raise ValueError(f"invalid leaderboard exclusion entry: {path}")
        exclusions.append(
            LeaderboardExclusion(
                exclusion_id=exclusion_id,
                plugin_commit=plugin_commit.lower(),
                reason=reason,
            )
        )
    return tuple(exclusions)


def match_leaderboard_exclusion(
    artifact: Mapping[str, Any],
    exclusions: tuple[LeaderboardExclusion, ...],
) -> LeaderboardExclusion | None:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    provenance = metadata.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        return None
    plugin = provenance.get("plugin")
    if not isinstance(plugin, Mapping):
        return None
    plugin_commit = str(plugin.get("commit") or "").strip().lower()
    return next(
        (item for item in exclusions if item.plugin_commit == plugin_commit),
        None,
    )
