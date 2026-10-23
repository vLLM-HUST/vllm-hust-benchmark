# SPDX-License-Identifier: Apache-2.0
"""Batch-admission policy for the current vLLM-HUST host contract."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from .planner import (
    MicroBatchCostModel,
    MicroBatchSnapshot,
    RequestWork,
    worst_rank_cost,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.batch_admission import BatchAdmissionContext

logger = logging.getLogger(__name__)

PolicyMode = Literal["balanced", "calibrated"]


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Structural implementation of Core's ``BatchAdmission`` result."""

    batch_id: str
    request_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineMicrobatchConfig:
    """Validated runtime configuration extracted from ``additional_config``."""

    mode: PolicyMode
    model_ids: tuple[str, ...]
    pipeline_parallel_size: int
    tensor_parallel_size: int
    microbatch_count: int
    cost_models: tuple[MicroBatchCostModel, ...] = ()

    @classmethod
    def from_config(
        cls, raw: Mapping[str, Any], vllm_config: VllmConfig
    ) -> PipelineMicrobatchConfig:
        parallel = vllm_config.parallel_config
        pp_size = parallel.pipeline_parallel_size
        tp_size = parallel.tensor_parallel_size
        if pp_size <= 1:
            raise ValueError("pipeline microbatch requires pipeline_parallel_size > 1")

        expected_pp = _positive_int(raw, "pipeline_parallel_size")
        expected_tp = _positive_int(raw, "tensor_parallel_size")
        if (pp_size, tp_size) != (expected_pp, expected_tp):
            raise ValueError(
                "pipeline microbatch topology mismatch: "
                f"runtime PP{pp_size}xTP{tp_size}, "
                f"profile PP{expected_pp}xTP{expected_tp}"
            )

        model_ids_raw = raw.get("model_ids")
        if (
            not isinstance(model_ids_raw, list)
            or not model_ids_raw
            or not all(
                isinstance(model_id, str) and model_id for model_id in model_ids_raw
            )
        ):
            raise ValueError("pipeline_microbatch.model_ids must be non-empty strings")
        model_ids = tuple(model_ids_raw)
        runtime_model = str(vllm_config.model_config.model)
        if not _model_matches(runtime_model, model_ids):
            raise ValueError(
                f"pipeline microbatch profile does not cover model {runtime_model!r}"
            )

        mode = raw.get("mode")
        if mode not in ("balanced", "calibrated"):
            raise ValueError("pipeline_microbatch.mode must be balanced or calibrated")

        microbatch_count = int(
            raw.get("microbatch_count", vllm_config.max_concurrent_batches)
        )
        if not 1 < microbatch_count <= vllm_config.max_concurrent_batches:
            raise ValueError(
                "microbatch_count must be between 2 and max_concurrent_batches"
            )

        cost_models = _parse_cost_models(raw.get("cost_models", []))
        if mode == "calibrated":
            covered_ranks = {model.pp_rank for model in cost_models}
            if covered_ranks != set(range(pp_size)):
                raise ValueError(
                    "calibrated mode requires exactly one or more models covering "
                    f"every PP rank; got {sorted(covered_ranks)!r}"
                )
        elif cost_models:
            raise ValueError("balanced mode must not include calibrated cost models")

        return cls(
            mode=mode,
            model_ids=model_ids,
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            microbatch_count=microbatch_count,
            cost_models=cost_models,
        )


@dataclass(slots=True)
class PipelineMicrobatchStats:
    calls: int = 0
    admissions: int = 0
    abstentions: int = 0
    assignments: int = 0
    completions: int = 0
    aborts: int = 0
    cancellations_reclaimed: int = 0


class PipelineMicrobatchPolicy:
    """Stable request grouping over Core's immutable admission snapshots."""

    def __init__(self, config: PipelineMicrobatchConfig) -> None:
        self.config = config
        self.stats = PipelineMicrobatchStats()
        self._assignments: dict[str, str] = {}
        self._batch_ids = tuple(
            f"pipeline-microbatch-{index}" for index in range(config.microbatch_count)
        )
        self._next_batch_index = 0
        logger.info(
            "Pipeline microbatch policy enabled: mode=%s models=%s PP=%d TP=%d "
            "microbatches=%d",
            config.mode,
            ",".join(config.model_ids),
            config.pipeline_parallel_size,
            config.tensor_parallel_size,
            config.microbatch_count,
        )

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], vllm_config: VllmConfig
    ) -> PipelineMicrobatchPolicy:
        return cls(PipelineMicrobatchConfig.from_config(config, vllm_config))

    def admit_batch(self, context: BatchAdmissionContext) -> AdmissionDecision | None:
        self.stats.calls += 1
        request_by_id = {request.request_id: request for request in context.requests}
        active_ids = set(request_by_id)
        stale_ids = set(self._assignments) - active_ids
        for request_id in stale_ids:
            del self._assignments[request_id]
        self.stats.cancellations_reclaimed += len(stale_ids)

        for request in context.requests:
            if request.request_id not in self._assignments:
                batch_id = self._select_assignment(request, request_by_id)
                self._assignments[request.request_id] = batch_id
                self.stats.assignments += 1

        for offset in range(len(self._batch_ids)):
            index = (self._next_batch_index + offset) % len(self._batch_ids)
            batch_id = self._batch_ids[index]
            if batch_id in context.in_flight_batch_ids:
                continue
            request_ids = tuple(
                request.request_id
                for request in context.requests
                if self._assignments.get(request.request_id) == batch_id
            )
            if not request_ids:
                continue
            self._next_batch_index = (index + 1) % len(self._batch_ids)
            self.stats.admissions += 1
            logger.debug(
                "Pipeline microbatch admitted: batch_id=%s request_count=%d "
                "in_flight=%s",
                batch_id,
                len(request_ids),
                sorted(context.in_flight_batch_ids),
            )
            return AdmissionDecision(batch_id, request_ids)

        self.stats.abstentions += 1
        return None

    def on_batch_complete(self, batch_id: str) -> None:
        if batch_id not in self._batch_ids:
            raise ValueError(f"unknown completed batch ID {batch_id!r}")
        self.stats.completions += 1

    def on_batch_abort(self, batch_id: str) -> None:
        if batch_id not in self._batch_ids:
            raise ValueError(f"unknown aborted batch ID {batch_id!r}")
        self.stats.aborts += 1

    def export_stats(self) -> dict[str, int]:
        return asdict(self.stats)

    def _select_assignment(self, request: Any, request_by_id: dict[str, Any]) -> str:
        snapshots = self._snapshots(request_by_id)
        if self.config.mode == "balanced":
            target = min(
                snapshots,
                key=lambda batch: (
                    batch.request_num,
                    batch.aggregated_ctx_length,
                    batch.microbatch_id,
                ),
            )
            return self._batch_ids[target.microbatch_id]

        request_work = RequestWork(request.request_id, request.context_length)
        target = min(
            snapshots,
            key=lambda batch: (
                worst_rank_cost(
                    MicroBatchSnapshot(
                        batch.microbatch_id, (*batch.requests, request_work)
                    ),
                    self.config.cost_models,
                ),
                batch.microbatch_id,
            ),
        )
        return self._batch_ids[target.microbatch_id]

    def _snapshots(
        self, request_by_id: dict[str, Any]
    ) -> tuple[MicroBatchSnapshot, ...]:
        work_by_batch: dict[str, list[RequestWork]] = {
            batch_id: [] for batch_id in self._batch_ids
        }
        for request_id, batch_id in self._assignments.items():
            request = request_by_id.get(request_id)
            if request is not None:
                work_by_batch[batch_id].append(
                    RequestWork(request_id, request.context_length)
                )
        return tuple(
            MicroBatchSnapshot(index, tuple(work_by_batch[batch_id]))
            for index, batch_id in enumerate(self._batch_ids)
        )


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"pipeline_microbatch.{key} must be a positive integer")
    return value


def _model_matches(runtime_model: str, model_ids: tuple[str, ...]) -> bool:
    runtime_name = runtime_model.rstrip("/").rsplit("/", 1)[-1]
    return any(
        runtime_model == model_id
        or runtime_name == model_id.rstrip("/").rsplit("/", 1)[-1]
        for model_id in model_ids
    )


def _parse_cost_models(raw_models: Any) -> tuple[MicroBatchCostModel, ...]:
    if not isinstance(raw_models, list):
        raise ValueError("pipeline_microbatch.cost_models must be a list")
    models: list[MicroBatchCostModel] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ValueError("each cost model must be a mapping")
        try:
            models.append(
                MicroBatchCostModel(
                    p0=float(raw["p0"]),
                    p1=float(raw["p1"]),
                    p2=float(raw["p2"]),
                    p3=float(raw["p3"]),
                    p4=float(raw["p4"]),
                    p5=float(raw["p5"]),
                    pp_rank=int(raw["pp_rank"]),
                    layer_num=int(raw["layer_num"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid cost model: {raw!r}") from exc
    return tuple(models)


__all__ = [
    "AdmissionDecision",
    "PipelineMicrobatchConfig",
    "PipelineMicrobatchPolicy",
    "PipelineMicrobatchStats",
]
