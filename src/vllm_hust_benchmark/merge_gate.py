"""PR 合并前强制可比 real-online 性能证据的 merge gate 判定逻辑（issue #95 Layer 1）。

复用 #104 fixed-target registry 和 #105 admission gate 的字段校验，新增 fail-closed
CI 状态判定、data_source real-online 校验、paired base/head 一致性校验。

判定流程见 issue_95_需求分析.md §5.1。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.fixed_target_registry import (
    find_matching_profile,
    load_fixed_target_registry,
)


# 受认可的 data_source 前缀（issue §4.2）。必须以 real-online 开头。
_REAL_ONLINE_PREFIXES: tuple[str, ...] = ("real-online",)

# docs-only / test-only / website-only 受控 label（issue §5.2.4）。
_SKIP_LABELS: tuple[str, ...] = (
    "perf-skip:docs-only",
    "perf-skip:test-only",
    "perf-skip:website-only",
)


class ArtifactStatus(str, Enum):
    """paired benchmark artifact 的 CI 状态。非 accepted 一律 fail closed。"""

    ACCEPTED = "accepted"
    MISSING = "missing"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RESOURCE_BUSY = "resource_busy"


@dataclass(frozen=True)
class ArtifactRef:
    """对 base 或 head 的一次 artifact 引用。

    path 指向 run_leaderboard.json；ci_status 描述 CI 跑完后 artifact 是否真正产出。
    fail-closed：ci_status != ACCEPTED 时直接 fail，不尝试读 path。
    """

    path: Path | None
    ci_status: ArtifactStatus = ArtifactStatus.MISSING


@dataclass(frozen=True)
class PRContext:
    """PR 上下文，用于 docs-only 例外、target 声明和 specialty 声明。"""

    repo: str
    number: int
    head_sha: str
    base_sha: str
    labels: tuple[str, ...] = ()
    declared_target_id: str | None = None
    declared_target_version: str | None = None
    declared_profile_id: str | None = None
    specialty_spec: str | None = None
    specialty_reason: str | None = None


@dataclass(frozen=True)
class MergeGateDecision:
    """merge gate 判定结果。disposition=pass/fail/skip。"""

    disposition: str
    reason: str
    repo: str
    pr_number: int
    head_sha: str
    base_sha: str
    target_id: str | None = None
    target_version: str | None = None
    profile_id: str | None = None
    registry_hash_matched: bool | None = None
    base_status: str = "missing"
    head_status: str = "missing"
    data_source: str | None = None
    model: str | None = None
    model_parameters: str | None = None
    hardware_chip_model: str | None = None
    chip_count: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    base_spec_id: str | None = None
    head_spec_id: str | None = None
    spec_id_match: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _numeric_equal(left: Any, right: Any) -> bool:
    """数值比较，无法转 float 时退化为直接相等（复用 #105 _fixed_target_numeric_equal）。"""
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _is_real_online(data_source: str) -> bool:
    """data_source 必须以 real-online 或受认可 CI real-online source 开头（issue §4.2）。"""
    ds = (data_source or "").strip().lower()
    return any(ds.startswith(prefix) for prefix in _REAL_ONLINE_PREFIXES)


def _load_artifact(path: Path | None) -> dict | None:
    """读取 run_leaderboard.json。失败返回 None（由调用方 fail closed）。"""
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _extract_spec_fields(payload: dict) -> tuple[str | None, str | None, dict]:
    """从 run_leaderboard.json 提取 spec_id / resolved_spec_hash / resolved_server_parameters。"""
    same_spec = payload.get("same_spec")
    if not isinstance(same_spec, dict):
        return None, None, {}
    spec_id = str(same_spec.get("spec_id") or "").strip() or None
    spec_hash = str(same_spec.get("resolved_spec_hash") or "").strip() or None
    server = same_spec.get("resolved_server_parameters")
    server = server if isinstance(server, dict) else {}
    return spec_id, spec_hash, server


def _extract_meta(payload: dict) -> dict:
    """提取 model / hardware / workload / data_source 等展示字段。"""
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    hardware = (
        payload.get("hardware") if isinstance(payload.get("hardware"), dict) else {}
    )
    metadata = (
        payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    )
    workload = (
        payload.get("workload") if isinstance(payload.get("workload"), dict) else {}
    )
    return {
        "model_repo_id": model.get("repo_id") or model.get("name"),
        "model_parameters": model.get("parameters"),
        "model_precision": model.get("precision"),
        "chip_model": hardware.get("chip_model"),
        "chip_count": hardware.get("chip_count"),
        "workload_name": workload.get("name"),
        "data_source": metadata.get("data_source"),
    }


def evaluate_merge_gate(
    base: ArtifactRef,
    head: ArtifactRef,
    pr_context: PRContext,
    registry_path: Path | None = None,
) -> MergeGateDecision:
    """判定 PR 是否有合规 paired real-online 性能证据。

    判定流程（issue §5.1）：
      1. docs-only / test-only / website-only 受控 label → skip
      2. fail-closed：base/head ci_status != accepted → fail
      3. 加载 artifact，读失败 → fail（missing evidence）
      4. data_source 不以 real-online 开头 → fail
      5. registry 匹配（find_matching_profile），无匹配 → fail（3B perfgate != 14B）
      6. PR 声明的 target_id 不在 registry → fail（registry hash mismatch）
      7. active profile 字段校验（gpu_memory_utilization / max_model_len）→ fail（config_drift / missing field）
      8. paired base/head spec_id 不一致 → fail
      9. specialty 必须携带 spec + reason → 否则 fail
    """
    common = dict(
        repo=pr_context.repo,
        pr_number=pr_context.number,
        head_sha=pr_context.head_sha,
        base_sha=pr_context.base_sha,
    )

    # 1. docs-only / test-only / website-only 受控 label（issue §5.2.4）
    for label in pr_context.labels:
        if label in _SKIP_LABELS:
            return MergeGateDecision(
                disposition="skip",
                reason=f"PR skipped via controlled label: {label}",
                **common,
                details={"skip_label": label},
            )

    # 2. fail-closed CI 状态检查（issue §4.3）
    if base.ci_status != ArtifactStatus.ACCEPTED:
        return MergeGateDecision(
            disposition="fail",
            reason=f"base artifact not accepted (ci_status={base.ci_status.value}): "
            f"paired evidence incomplete, fail closed",
            **common,
            base_status=base.ci_status.value,
            head_status=head.ci_status.value,
        )
    if head.ci_status != ArtifactStatus.ACCEPTED:
        return MergeGateDecision(
            disposition="fail",
            reason=f"head artifact not accepted (ci_status={head.ci_status.value}): "
            f"paired evidence incomplete, fail closed",
            **common,
            base_status=base.ci_status.value,
            head_status=head.ci_status.value,
        )

    # 3. 加载 artifact
    base_payload = _load_artifact(base.path)
    head_payload = _load_artifact(head.path)
    if base_payload is None or head_payload is None:
        return MergeGateDecision(
            disposition="fail",
            reason="missing evidence: base/head run_leaderboard.json unreadable or missing",
            **common,
            base_status="accepted",
            head_status="accepted",
        )

    base_meta = _extract_meta(base_payload)
    head_meta = _extract_meta(head_payload)
    data_source = base_meta.get("data_source")

    # 4. data_source real-online 校验（issue §4.2）
    for side, ds in (
        ("base", base_meta.get("data_source")),
        ("head", head_meta.get("data_source")),
    ):
        if not _is_real_online(ds or ""):
            return MergeGateDecision(
                disposition="fail",
                reason=f"data_source not real-online on {side}: {ds!r} "
                f"(must start with 'real-online')",
                **common,
                base_status="accepted",
                head_status="accepted",
                data_source=data_source,
                model=base_meta.get("model_repo_id"),
                model_parameters=base_meta.get("model_parameters"),
                hardware_chip_model=base_meta.get("chip_model"),
                chip_count=base_meta.get("chip_count"),
            )

    # 5. registry 匹配（复用 #104 find_matching_profile）
    registry = load_fixed_target_registry(registry_path)
    base_profile = find_matching_profile(base_payload, registry)
    head_profile = find_matching_profile(head_payload, registry)
    if base_profile is None or head_profile is None:
        return MergeGateDecision(
            disposition="fail",
            reason="no matching fixed-target profile: 3B perfgate does not equal "
            "14B public-target evidence (issue §4.4)",
            **common,
            base_status="accepted",
            head_status="accepted",
            data_source=data_source,
            model=base_meta.get("model_repo_id"),
            model_parameters=base_meta.get("model_parameters"),
            hardware_chip_model=base_meta.get("chip_model"),
            chip_count=base_meta.get("chip_count"),
        )

    profile = base_profile
    registry_hash_matched = base_profile.target_id == head_profile.target_id

    # 6. PR 声明的 target_id 必须在 registry（issue §4.4 registry hash 匹配）
    if pr_context.declared_target_id is not None:
        declared_ids = {p.target_id for p in registry}
        if pr_context.declared_target_id not in declared_ids:
            return MergeGateDecision(
                disposition="fail",
                reason=f"declared target_id not in registry: "
                f"{pr_context.declared_target_id!r} (registry hash mismatch)",
                **common,
                target_id=profile.target_id,
                target_version=profile.target_version,
                profile_id=profile.profile_name,
                registry_hash_matched=False,
                base_status="accepted",
                head_status="accepted",
                data_source=data_source,
                model=base_meta.get("model_repo_id"),
                model_parameters=base_meta.get("model_parameters"),
                hardware_chip_model=base_meta.get("chip_model"),
                chip_count=base_meta.get("chip_count"),
            )

    # 7. active profile 字段校验（复用 #105 config_drift / missing field 逻辑）
    base_spec_id, _, base_server = _extract_spec_fields(base_payload)
    head_spec_id, _, head_server = _extract_spec_fields(head_payload)
    base_gmu = base_server.get("gpu_memory_utilization")
    base_mml = base_server.get("max_model_len")

    if profile.status == "active":
        for field_name, required_value in (
            ("gpu_memory_utilization", profile.gpu_memory_utilization),
            ("max_model_len", profile.max_model_len),
        ):
            # missing field
            if field_name not in base_server:
                return MergeGateDecision(
                    disposition="fail",
                    reason=f"missing {field_name} in base same_spec.resolved_server_parameters",
                    **common,
                    target_id=profile.target_id,
                    target_version=profile.target_version,
                    profile_id=profile.profile_name,
                    registry_hash_matched=registry_hash_matched,
                    base_status="accepted",
                    head_status="accepted",
                    data_source=data_source,
                    model=base_meta.get("model_repo_id"),
                    model_parameters=base_meta.get("model_parameters"),
                    hardware_chip_model=base_meta.get("chip_model"),
                    chip_count=base_meta.get("chip_count"),
                    base_spec_id=base_spec_id,
                    head_spec_id=head_spec_id,
                )
            actual = base_server[field_name]
            if not _numeric_equal(actual, required_value):
                return MergeGateDecision(
                    disposition="fail",
                    reason=f"config_drift: {field_name}={actual!r} required={required_value!r} "
                    f"(profile={profile.profile_name})",
                    **common,
                    target_id=profile.target_id,
                    target_version=profile.target_version,
                    profile_id=profile.profile_name,
                    registry_hash_matched=registry_hash_matched,
                    base_status="accepted",
                    head_status="accepted",
                    data_source=data_source,
                    model=base_meta.get("model_repo_id"),
                    model_parameters=base_meta.get("model_parameters"),
                    hardware_chip_model=base_meta.get("chip_model"),
                    chip_count=base_meta.get("chip_count"),
                    gpu_memory_utilization=base_gmu,
                    max_model_len=base_mml,
                    base_spec_id=base_spec_id,
                    head_spec_id=head_spec_id,
                )

    # 8. paired base/head spec_id 一致性（复用 perfgate _validate_same_spec 语义）
    spec_id_match = base_spec_id == head_spec_id
    if not spec_id_match:
        return MergeGateDecision(
            disposition="fail",
            reason=f"paired spec_id mismatch: base={base_spec_id!r} head={head_spec_id!r}",
            **common,
            target_id=profile.target_id,
            target_version=profile.target_version,
            profile_id=profile.profile_name,
            registry_hash_matched=registry_hash_matched,
            base_status="accepted",
            head_status="accepted",
            data_source=data_source,
            model=base_meta.get("model_repo_id"),
            model_parameters=base_meta.get("model_parameters"),
            hardware_chip_model=base_meta.get("chip_model"),
            chip_count=base_meta.get("chip_count"),
            gpu_memory_utilization=base_gmu,
            max_model_len=base_mml,
            base_spec_id=base_spec_id,
            head_spec_id=head_spec_id,
            spec_id_match=False,
        )

    # 9. specialty 必须携带 spec + reason（issue §4.4）
    is_specialty = profile.status == "specialty"
    if is_specialty:
        if not pr_context.specialty_spec:
            return MergeGateDecision(
                disposition="fail",
                reason="specialty target declared but specialty_spec is empty "
                "(issue §4.4: specialty must carry independent spec)",
                **common,
                target_id=profile.target_id,
                target_version=profile.target_version,
                profile_id=profile.profile_name,
                registry_hash_matched=registry_hash_matched,
                base_status="accepted",
                head_status="accepted",
                data_source=data_source,
                model=base_meta.get("model_repo_id"),
                model_parameters=base_meta.get("model_parameters"),
                hardware_chip_model=base_meta.get("chip_model"),
                chip_count=base_meta.get("chip_count"),
                base_spec_id=base_spec_id,
                head_spec_id=head_spec_id,
                spec_id_match=True,
                details={"specialty": True},
            )
        if not pr_context.specialty_reason:
            return MergeGateDecision(
                disposition="fail",
                reason="specialty target declared but specialty_reason is empty "
                "(issue §4.4: specialty must carry reason)",
                **common,
                target_id=profile.target_id,
                target_version=profile.target_version,
                profile_id=profile.profile_name,
                registry_hash_matched=registry_hash_matched,
                base_status="accepted",
                head_status="accepted",
                data_source=data_source,
                model=base_meta.get("model_repo_id"),
                model_parameters=base_meta.get("model_parameters"),
                hardware_chip_model=base_meta.get("chip_model"),
                chip_count=base_meta.get("chip_count"),
                base_spec_id=base_spec_id,
                head_spec_id=head_spec_id,
                spec_id_match=True,
                details={"specialty": True},
            )

    # 10. pass
    return MergeGateDecision(
        disposition="pass",
        reason="paired real-online evidence accepted, all checks passed",
        **common,
        target_id=profile.target_id,
        target_version=profile.target_version,
        profile_id=profile.profile_name,
        registry_hash_matched=registry_hash_matched,
        base_status="accepted",
        head_status="accepted",
        data_source=data_source,
        model=base_meta.get("model_repo_id"),
        model_parameters=base_meta.get("model_parameters"),
        hardware_chip_model=base_meta.get("chip_model"),
        chip_count=base_meta.get("chip_count"),
        gpu_memory_utilization=base_gmu,
        max_model_len=base_mml,
        base_spec_id=base_spec_id,
        head_spec_id=head_spec_id,
        spec_id_match=True,
        details={"specialty": is_specialty} if is_specialty else {},
    )


def format_decision_log(decision: MergeGateDecision, pr_context: PRContext) -> str:
    """输出 issue §7.3 要求的结构化判定日志。"""
    lines = [
        "[merge-gate-check]",
        f"  PR: {decision.repo}#{decision.pr_number} "
        f"head={decision.head_sha} base={decision.base_sha}",
        f"  target_id: {decision.target_id}",
        f"  target_version: {decision.target_version}",
        f"  profile_id: {decision.profile_id}",
        f"  registry_hash_matched: {decision.registry_hash_matched}",
        f"  base_artifact: status={decision.base_status}",
        f"  head_artifact: status={decision.head_status}",
        f"  data_source: {decision.data_source}",
        f"  model: {decision.model} parameters={decision.model_parameters}",
        f"  hardware: {decision.hardware_chip_model} chip_count={decision.chip_count}",
        f"  gpu_memory_utilization: {decision.gpu_memory_utilization}",
        f"  max_model_len: {decision.max_model_len}",
        f"  same_spec.spec_id: base={decision.base_spec_id} "
        f"head={decision.head_spec_id} match={decision.spec_id_match}",
        f"  disposition: {decision.disposition}",
        f"  reason: {decision.reason}",
    ]
    return "\n".join(lines)


def write_decision_json(decision: MergeGateDecision, path: Path) -> None:
    """持久化判定结果为 merge-gate-decision.json（增强建议 §14.1 #4）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(decision), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
