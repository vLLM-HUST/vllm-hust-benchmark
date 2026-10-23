# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension API for selecting requests to preempt under KV pressure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

PreemptionSchedulingPolicy = Literal["fcfs", "priority"]
PREEMPTION_POLICY_API_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class PreemptionCandidate:
    """Immutable request data available to a preemption policy."""

    request_id: str
    priority: int
    arrival_time: float
    num_prompt_tokens: int
    num_output_tokens: int
    num_computed_tokens: int
    num_preemptions: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class PreemptionContext:
    """Immutable snapshot of one scheduler preemption decision.

    ``requesting_request_id`` identifies the running request whose KV
    allocation just failed. Selecting it resets its computed KV state, so a
    policy that cannot establish a material benefit should return ``None``.
    ``builtin_victim_id`` lets policies compare against vLLM's stable fallback
    without copying FCFS/priority ordering rules. It is optional only for
    source compatibility with callers that constructed API 1.0 contexts.
    """

    candidates: tuple[PreemptionCandidate, ...]
    scheduling_policy: PreemptionSchedulingPolicy
    requesting_request_id: str
    kv_cache_usage: float
    now: float
    builtin_victim_id: str | None = None


@runtime_checkable
class PreemptionPolicy(Protocol):
    """Policy contract for request preemption.

    Returning ``None`` delegates the decision to vLLM's built-in policy.
    Implementations must return an ID from ``context.candidates`` and must not
    retain references to scheduler-owned mutable state.
    """

    def select_victim(self, context: PreemptionContext) -> str | None:
        """Return the request ID to preempt, or ``None`` to abstain."""
        ...


@dataclass(slots=True)
class PreemptionPolicyStats:
    """Cumulative policy invocation and failover counters."""

    policy_name: str = "builtin"
    enabled: bool = False
    calls: int = 0
    selections: int = 0
    abstentions: int = 0
    failures: int = 0
    invalid_selections: int = 0


def _builtin_victim_id(context: PreemptionContext) -> str:
    if not context.candidates:
        raise ValueError("cannot select a preemption victim from an empty set")
    candidate_ids = {candidate.request_id for candidate in context.candidates}
    if context.builtin_victim_id in candidate_ids:
        return context.builtin_victim_id
    if context.scheduling_policy == "priority":
        return max(
            context.candidates,
            key=lambda candidate: (candidate.priority, candidate.arrival_time),
        ).request_id
    return context.candidates[-1].request_id


def _policy_name(policy: PreemptionPolicy) -> str:
    policy_type = type(policy)
    return f"{policy_type.__module__}.{policy_type.__qualname__}"


def _load_policy(vllm_config: VllmConfig) -> PreemptionPolicy | None:
    configured: Any = vllm_config.scheduler_config.preemption_policy
    if configured is None:
        return None
    implementation = (
        resolve_obj_by_qualname(configured)
        if isinstance(configured, str)
        else configured
    )
    factory = getattr(implementation, "from_vllm_config", None)
    if callable(factory):
        policy = factory(vllm_config)
    elif isinstance(implementation, type):
        policy = implementation()
    else:
        policy = implementation
    if not isinstance(policy, PreemptionPolicy):
        raise TypeError(
            "preemption_policy must resolve to an object implementing PreemptionPolicy"
        )
    return policy


class PreemptionPolicyController:
    """Validate an external policy and fail over permanently on a fault."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        policy = _load_policy(vllm_config)
        self._policy = policy
        self.stats = PreemptionPolicyStats(
            policy_name=_policy_name(policy) if policy is not None else "builtin",
            enabled=policy is not None,
        )
        logger.info(
            "Preemption policy initialized: policy=%s enabled=%s",
            self.stats.policy_name,
            self.stats.enabled,
        )

    def select_victim(self, context: PreemptionContext) -> str:
        policy = self._policy
        if policy is None:
            return _builtin_victim_id(context)

        self.stats.calls += 1
        try:
            selected_id = policy.select_victim(context)
        except Exception:
            self._disable_after_failure("raised an exception", exc_info=True)
            return _builtin_victim_id(context)

        if selected_id is None:
            self.stats.abstentions += 1
            return _builtin_victim_id(context)

        candidate_ids = {candidate.request_id for candidate in context.candidates}
        if selected_id not in candidate_ids:
            self.stats.invalid_selections += 1
            self._disable_after_failure(f"returned unknown request ID {selected_id!r}")
            return _builtin_victim_id(context)

        self.stats.selections += 1
        return selected_id

    def export_stats(self) -> dict[str, str | int | bool]:
        """Return a serialization-safe cumulative stats snapshot."""
        return asdict(self.stats)

    def _disable_after_failure(self, reason: str, *, exc_info: bool = False) -> None:
        self.stats.failures += 1
        self.stats.enabled = False
        self._policy = None
        logger.error(
            "Preemption policy %s failed (%s); disabling it and restoring "
            "the built-in policy for this engine process",
            self.stats.policy_name,
            reason,
            exc_info=exc_info,
        )


__all__ = [
    "PREEMPTION_POLICY_API_VERSION",
    "PreemptionCandidate",
    "PreemptionContext",
    "PreemptionPolicy",
    "PreemptionPolicyController",
    "PreemptionPolicyStats",
]
