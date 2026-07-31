"""Fail-closed exclusions for public leaderboard submissions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vllm-hust-leaderboard-exclusions/v1"


@dataclass(frozen=True)
class LeaderboardExclusion:
    exclusion_id: str
    plugin_commit: str  # 当 match_type=target_misalignment 时可为空字符串
    reason: str
    match_type: str = "plugin_commit"  # "plugin_commit" / "target_misalignment"


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
        match_type = str(match.get("match_type") or "plugin_commit").strip()
        if match_type not in ("plugin_commit", "target_misalignment"):
            raise ValueError(f"invalid leaderboard exclusion entry: {path}")
        plugin_commit = str(match.get("runtime_provenance.plugin.commit") or "").strip()
        exclusion_id = str(raw.get("id") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if match_type == "plugin_commit":
            if len(plugin_commit) != 40 or not exclusion_id or not reason:
                raise ValueError(f"invalid leaderboard exclusion entry: {path}")
        else:  # target_misalignment
            if not exclusion_id or not reason:
                raise ValueError(f"invalid leaderboard exclusion entry: {path}")
        exclusions.append(
            LeaderboardExclusion(
                exclusion_id=exclusion_id,
                plugin_commit=plugin_commit.lower(),
                reason=reason,
                match_type=match_type,
            )
        )
    return tuple(exclusions)


def match_leaderboard_exclusion(
    artifact: Mapping[str, Any],
    exclusions: tuple[LeaderboardExclusion, ...],
    *,
    misaligned_entry_ids: set[str] | None = None,
) -> LeaderboardExclusion | None:
    """匹配 exclusion。

    - match_type=plugin_commit: 通过 runtime_provenance.plugin.commit 匹配（原行为）
    - match_type=target_misalignment: 通过 entry_id 是否在 misaligned_entry_ids 中匹配
      （需要调用方传入 misaligned_entry_ids；未传入时此类 exclusion 不匹配任何 entry）
    """
    # 先尝试 target_misalignment 匹配
    if misaligned_entry_ids:
        entry_id = str(artifact.get("entry_id") or "")
        for item in exclusions:
            if (
                item.match_type == "target_misalignment"
                and entry_id in misaligned_entry_ids
            ):
                return item

    # 再走 plugin_commit 匹配（原行为）
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
        (
            item
            for item in exclusions
            if item.match_type == "plugin_commit"
            and item.plugin_commit == plugin_commit
        ),
        None,
    )
