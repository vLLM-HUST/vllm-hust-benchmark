# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension API for admitting request groups to concurrent batch queues."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

BATCH_ADMISSION_POLICY_API_VERSION = "1.1"
BatchRequestState = Literal["waiting", "running"]


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """Immutable request data available to a batch-admission policy."""

    request_id: str
    state: BatchRequestState
    priority: int
    arrival_time: float
    num_prompt_tokens: int
    num_output_tokens: int
    num_computed_tokens: int
    num_in_flight_tokens: int
    max_tokens: int

    @property
    def context_length(self) -> int:
        """Current prompt and generated-token length."""
        return self.num_prompt_tokens + self.num_output_tokens


@dataclass(frozen=True, slots=True)
class BatchAdmissionContext:
    """Immutable snapshot for one concurrent-batch admission decision."""

    requests: tuple[BatchRequest, ...]
    in_flight_batch_ids: frozenset[str]
    max_concurrent_batches: int
    pipeline_parallel_size: int
    now: float


@dataclass(frozen=True, slots=True)
class BatchAdmission:
    """One policy-selected logical batch and its eligible requests."""

    batch_id: str
    request_ids: tuple[str, ...]


@runtime_checkable
class BatchAdmissionPolicy(Protocol):
    """Policy contract for grouping work in a concurrent batch queue.

    Returning ``None`` means that no logical batch is currently eligible. A
    policy must not retain references to scheduler-owned mutable state.
    """

    def admit_batch(self, context: BatchAdmissionContext) -> BatchAdmission | None:
        """Return the next eligible logical batch, or ``None`` to wait."""
        ...


class BatchAdmissionPolicyFactory(Protocol):
    """Optional factory contract for policies requiring JSON configuration."""

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], vllm_config: VllmConfig
    ) -> BatchAdmissionPolicy:
        """Build a policy from its isolated configuration and engine config."""
        ...


@dataclass(slots=True)
class BatchAdmissionPolicyStats:
    """Cumulative policy invocation and failover counters."""

    policy_name: str = "builtin"
    configured: bool = False
    enabled: bool = False
    calls: int = 0
    admissions: int = 0
    abstentions: int = 0
    completions: int = 0
    aborts: int = 0
    failures: int = 0
    invalid_admissions: int = 0
    builtin_fallbacks: int = 0


def _policy_name(policy: BatchAdmissionPolicy) -> str:
    policy_type = type(policy)
    return f"{policy_type.__module__}.{policy_type.__qualname__}"


def _load_policy(vllm_config: VllmConfig) -> BatchAdmissionPolicy | None:
    configured: Any = vllm_config.scheduler_config.batch_admission_policy
    if configured is None:
        return None
    implementation = (
        resolve_obj_by_qualname(configured)
        if isinstance(configured, str)
        else configured
    )
    factory = getattr(implementation, "from_config", None)
    if callable(factory):
        raw_config = vllm_config.scheduler_config.batch_admission_policy_config
        policy = factory(raw_config or {}, vllm_config)
    elif isinstance(implementation, type):
        policy = implementation()
    else:
        policy = implementation
    if not isinstance(policy, BatchAdmissionPolicy):
        raise TypeError(
            "batch_admission_policy must resolve to an object implementing "
            "BatchAdmissionPolicy"
        )
    return policy


class BatchAdmissionPolicyController:
    """Validate an external admission policy and fail over on a fault."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        policy = _load_policy(vllm_config)
        self._policy = policy
        self.stats = BatchAdmissionPolicyStats(
            policy_name=_policy_name(policy) if policy is not None else "builtin",
            configured=policy is not None,
            enabled=policy is not None,
        )
        logger.info(
            "Batch admission policy initialized: policy=%s configured=%s enabled=%s",
            self.stats.policy_name,
            self.stats.configured,
            self.stats.enabled,
        )

    @property
    def enabled(self) -> bool:
        return self._policy is not None

    def admit_batch(self, context: BatchAdmissionContext) -> BatchAdmission | None:
        policy = self._policy
        if policy is None:
            self.stats.builtin_fallbacks += int(self.stats.configured)
            return None

        self.stats.calls += 1
        try:
            admission = policy.admit_batch(context)
        except Exception:
            self._disable_after_failure("raised an exception", exc_info=True)
            return None

        if admission is None:
            self.stats.abstentions += 1
            if not context.in_flight_batch_ids and context.requests:
                self._disable_after_failure(
                    "abstained while work existed and no batch was in flight"
                )
            return None

        error = self._validate_admission(admission, context)
        if error is not None:
            self.stats.invalid_admissions += 1
            self._disable_after_failure(error)
            return None

        self.stats.admissions += 1
        return admission

    def on_batch_complete(self, batch_id: str) -> None:
        policy = self._policy
        if policy is None:
            return
        callback = getattr(policy, "on_batch_complete", None)
        if callback is not None:
            try:
                callback(batch_id)
            except Exception:
                self._disable_after_failure(
                    "completion callback raised an exception", exc_info=True
                )
                return
        self.stats.completions += 1

    def on_batch_abort(self, batch_id: str) -> None:
        policy = self._policy
        if policy is None:
            return
        callback = getattr(policy, "on_batch_abort", None)
        if callback is not None:
            try:
                callback(batch_id)
            except Exception:
                self._disable_after_failure(
                    "abort callback raised an exception", exc_info=True
                )
                return
        self.stats.aborts += 1

    def export_stats(self) -> dict[str, str | int | bool]:
        """Return a serialization-safe cumulative stats snapshot."""
        return asdict(self.stats)

    def _validate_admission(
        self, admission: BatchAdmission, context: BatchAdmissionContext
    ) -> str | None:
        if not admission.batch_id:
            return "returned an empty batch ID"
        if admission.batch_id in context.in_flight_batch_ids:
            return f"returned in-flight batch ID {admission.batch_id!r}"
        if not admission.request_ids:
            return "returned an empty request list"
        if len(set(admission.request_ids)) != len(admission.request_ids):
            return "returned duplicate request IDs"
        candidate_ids = {request.request_id for request in context.requests}
        unknown_ids = set(admission.request_ids) - candidate_ids
        if unknown_ids:
            return f"returned unknown request IDs {sorted(unknown_ids)!r}"
        return None

    def _disable_after_failure(self, reason: str, *, exc_info: bool = False) -> None:
        self.stats.failures += 1
        self.stats.enabled = False
        self._policy = None
        logger.error(
            "Batch admission policy %s failed (%s); disabling it and restoring "
            "the built-in scheduler for this engine process",
            self.stats.policy_name,
            reason,
            exc_info=exc_info,
        )


__all__ = [
    "BATCH_ADMISSION_POLICY_API_VERSION",
    "BatchAdmission",
    "BatchAdmissionContext",
    "BatchAdmissionPolicy",
    "BatchAdmissionPolicyFactory",
    "BatchAdmissionPolicyController",
    "BatchAdmissionPolicyStats",
    "BatchRequest",
    "BatchRequestState",
]
