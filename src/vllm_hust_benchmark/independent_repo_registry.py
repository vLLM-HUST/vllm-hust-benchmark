"""Independent optimization repo result card registry.

Issue #89: independent optimization repos (vllm-hust-bidkv,
vllm-ascend-hust-diffspec, vllm-ascend-quant-hust,
vllm-ascend-hust-LatchMoE, adaptive-selector-plugin) must have their own
canonical result entry points so they do not disappear from the results
page merely because they do not feed the 14B single-card main line.

This module validates ``independent-repo-result-card/v1`` artifacts
against the JSON Schema and applies semantic checks:

- Each repo must have at least one result card.
- ``status=blocked`` requires a non-empty ``blocker``; other statuses
  forbid a ``blocker`` (fail-closed: a blocked card without a reason is
  unactionable, and a non-blocked card with a blocker is contradictory).
- ``status=formal-presentable`` or ``experimental-presentable`` requires
  a non-null ``metrics`` block with at least one finite metric value
  (a presentable card with zero metrics is not presentable).
- ``repetitions`` (if present) must be >= 1.
- ``repo_commit`` and ``base_commit`` (if present) must be 40-char hex.
- Repo names must be unique within the registry (no duplicate entries).
- ``card_id`` values must be unique within the registry.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

SCHEMA_VERSION = "independent-repo-result-card/v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "independent_repo_result_card_v1.schema.json"
)

# Issue #89 §6: the five independent optimization repos that must have
# result entry points.  A registry missing any of these fails the
# coverage check.
REQUIRED_REPOS = frozenset(
    {
        "vllm-hust-bidkv",
        "vllm-ascend-hust-diffspec",
        "vllm-ascend-quant-hust",
        "vllm-ascend-hust-LatchMoE",
        "adaptive-selector-plugin",
    }
)

_STATUS_WITH_BLOCKER = frozenset({"blocked"})
_STATUS_WITH_METRICS = frozenset({"formal-presentable", "experimental-presentable"})


def load_schema() -> dict[str, Any]:
    """Load the independent_repo_result_card_v1 JSON Schema document."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def schema_validator() -> Draft7Validator:
    """Return a Draft7Validator for the independent-repo schema."""
    return Draft7Validator(load_schema())


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _has_finite_metric(metrics: Mapping[str, Any]) -> bool:
    for key in ("ttft_ms", "tbt_ms", "throughput_tps", "peak_mem_mb", "error_rate"):
        if key in metrics and _is_finite_number(metrics[key]):
            return True
    return False


def validate_registry_semantics(
    registry: Mapping[str, Any], *, context: str = "registry"
) -> None:
    """Apply semantic checks beyond the JSON Schema.

    Raises ``ValueError`` on the first semantic violation (fail-closed).
    """
    # Schema-level validation first.
    errors = sorted(schema_validator().iter_errors(registry), key=str)
    if errors:
        detail = errors[0].message
        raise ValueError(f"{context}: schema validation failed: {detail}")

    repos = registry.get("repos", [])
    if not isinstance(repos, list):
        raise ValueError(f"{context}: repos must be an array")

    seen_repos: set[str] = set()
    seen_cards: set[str] = set()

    for idx, repo in enumerate(repos):
        if not isinstance(repo, Mapping):
            raise ValueError(f"{context}: repos[{idx}] is not an object")
        repo_name = str(repo.get("repo_name") or "")
        if not repo_name:
            raise ValueError(f"{context}: repos[{idx}].repo_name is empty")
        if repo_name in seen_repos:
            raise ValueError(f"{context}: duplicate repo_name {repo_name!r}")
        seen_repos.add(repo_name)

        cards = repo.get("result_cards", [])
        if not isinstance(cards, list) or not cards:
            raise ValueError(f"{context}: repo {repo_name!r} must have >=1 result card")

        for card_idx, card in enumerate(cards):
            if not isinstance(card, Mapping):
                raise ValueError(
                    f"{context}: repo {repo_name!r} card[{card_idx}] is not an object"
                )
            card_id = str(card.get("card_id") or "")
            if not card_id:
                raise ValueError(
                    f"{context}: repo {repo_name!r} card[{card_idx}].card_id is empty"
                )
            if card_id in seen_cards:
                raise ValueError(f"{context}: duplicate card_id {card_id!r}")
            seen_cards.add(card_id)

            status = str(card.get("status") or "")
            blocker = card.get("blocker")
            metrics = card.get("metrics")

            if status in _STATUS_WITH_BLOCKER:
                if not blocker or not str(blocker).strip():
                    raise ValueError(
                        f"{context}: card {card_id!r} status=blocked requires "
                        f"a non-empty blocker reason"
                    )
            else:
                if blocker is not None and str(blocker).strip():
                    raise ValueError(
                        f"{context}: card {card_id!r} status={status!r} "
                        f"must not carry a blocker (got {blocker!r})"
                    )

            if status in _STATUS_WITH_METRICS:
                if not isinstance(metrics, Mapping) or not _has_finite_metric(metrics):
                    raise ValueError(
                        f"{context}: card {card_id!r} status={status!r} requires "
                        f"a non-null metrics block with >=1 finite value"
                    )


def check_required_repo_coverage(
    registry: Mapping[str, Any], *, context: str = "registry"
) -> None:
    """Verify all REQUIRED_REPOS have at least one result card.

    Issue #89 acceptance: "独立优化仓库均有结果入口，不能因不属于主线而从成果页消失。"
    """
    present: set[str] = set()
    for repo in registry.get("repos", []):
        if isinstance(repo, Mapping):
            name = str(repo.get("repo_name") or "")
            if (
                name
                and isinstance(repo.get("result_cards"), list)
                and repo["result_cards"]
            ):
                present.add(name)
    missing = REQUIRED_REPOS - present
    if missing:
        raise ValueError(
            f"{context}: missing required independent repos with result "
            f"cards: {sorted(missing)}"
        )


def load_registry(path: Path) -> dict[str, Any]:
    """Load and fully validate a result card registry file."""
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_registry_semantics(registry, context=str(path))
    check_required_repo_coverage(registry, context=str(path))
    return registry
