"""merge gate 判定逻辑的 TDD 测试（issue #95 Layer 1）。

覆盖 issue_95_需求分析.md §7.1 的全部测试矩阵。
先写失败测试，再实现 merge_gate.py 让其通过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.merge_gate import (
    ArtifactRef,
    ArtifactStatus,
    PRContext,
    evaluate_merge_gate,
    format_decision_log,
    write_decision_json,
)

# ---------------------------------------------------------------------------
# Fixture 构造辅助
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "vllm_hust_benchmark"
    / "data"
    / "fixed_target_registry.json"
)

VALID_SPEC_ID = "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
VALID_SPEC_HASH = "f8cc8fc26b4b9bb06d50f079174894a95d2bc0f49799374a652e6e04b75c8feb"  # pragma: allowlist secret


def _make_run_leaderboard(
    *,
    data_source: str = "real-online-pr95-test",
    model_repo_id: str = "Qwen/Qwen2.5-14B-Instruct",
    model_parameters: str = "14B",
    model_precision: str = "FP16",
    chip_model: str = "910B2",
    chip_count: int = 1,
    workload_name: str = "random-online",
    gpu_memory_utilization: float = 0.6,
    max_model_len: int = 32768,
    spec_id: str = VALID_SPEC_ID,
    spec_hash: str = VALID_SPEC_HASH,
    same_spec_present: bool = True,
) -> dict:
    """构造一个合规的 14B 单卡 run_leaderboard.json payload。"""
    payload: dict = {
        "entry_id": "test-entry-001",
        "engine": "vllm-hust",
        "engine_version": "v0.23.1rc0-test",
        "config_type": "single_gpu",
        "hardware": {
            "vendor": "Huawei",
            "chip_model": chip_model,
            "chip_count": chip_count,
        },
        "model": {
            "repo_id": model_repo_id,
            "name": model_repo_id,
            "parameters": model_parameters,
            "precision": model_precision,
        },
        "workload": {"name": workload_name},
        "metrics": {"ttft_ms": 100.0, "tbt_ms": 30.0, "throughput_tps": 200.0},
        "metadata": {
            "submitted_at": "2026-07-29T00:00:00Z",
            "data_source": data_source,
        },
    }
    if same_spec_present:
        payload["same_spec"] = {
            "spec_id": spec_id,
            "resolved_spec_hash": spec_hash,
            "resolved_server_parameters": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            },
        }
    return payload


def _write_artifact(
    tmp_path: Path,
    name: str,
    payload: dict | None = None,
) -> Path:
    """写一个 run_leaderboard.json 到 tmp_path/name/ 下，返回其路径。"""
    art_dir = tmp_path / name
    art_dir.mkdir(parents=True, exist_ok=True)
    rl_path = art_dir / "run_leaderboard.json"
    rl_path.write_text(
        json.dumps(payload or _make_run_leaderboard(), indent=2),
        encoding="utf-8",
    )
    return rl_path


def _accepted(path: Path) -> ArtifactRef:
    return ArtifactRef(path=path, ci_status=ArtifactStatus.ACCEPTED)


def _default_pr_context(**overrides) -> PRContext:
    defaults = {
        "repo": "vllm-hust",
        "number": 193,
        "head_sha": "abc1234",
        "base_sha": "def5678",
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": "v0.18.0",
        "declared_profile_id": "core-text-14b",
    }
    defaults.update(overrides)
    return PRContext(**defaults)


# ---------------------------------------------------------------------------
# §7.1 判定函数单元测试
# ---------------------------------------------------------------------------


class TestFailClosedArtifacts:
    """fail-closed：artifact 缺失/CI 异常一律挡。"""

    def test_no_artifact_blocked(self, tmp_path):
        """PR 没有任何性能 artifact → fail（missing evidence）。"""
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert (
            "missing" in decision.reason.lower()
            or "evidence" in decision.reason.lower()
        )

    def test_only_base_artifact_blocked(self, tmp_path):
        """只有 base 没有 head → fail（unpaired）。"""
        base_path = _write_artifact(tmp_path, "base")
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert decision.head_status == "missing"

    def test_only_head_artifact_blocked(self, tmp_path):
        """只有 head 没有 base → fail（unpaired）。"""
        head_path = _write_artifact(tmp_path, "head")
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert decision.base_status == "missing"

    def test_baseline_missing_fail_closed(self, tmp_path):
        """central baseline 不存在 → fail closed。"""
        head_path = _write_artifact(tmp_path, "head")
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"

    def test_job_cancelled_fail_closed(self, tmp_path):
        """job 被 cancel → fail closed。"""
        base_path = _write_artifact(tmp_path, "base")
        head_path = _write_artifact(tmp_path, "head")
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=ArtifactRef(path=head_path, ci_status=ArtifactStatus.CANCELLED),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert decision.head_status == "cancelled"

    def test_job_skipped_fail_closed(self, tmp_path):
        """job 被 skip → fail closed。"""
        base_path = _write_artifact(tmp_path, "base")
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.SKIPPED),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert decision.head_status == "skipped"

    def test_resource_busy_fail_closed(self, tmp_path):
        """NPU 残留占用 → resource_busy → fail closed。"""
        head_path = _write_artifact(tmp_path, "head")
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.RESOURCE_BUSY),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert decision.base_status == "resource_busy"


class TestDataSourceRejection:
    """data_source 不以 real-online 开头 → fail。"""

    @pytest.mark.parametrize(
        "data_source",
        ["smoke-test", "replay-20260729", "derived-from-snapshot", "local-screenshot"],
    )
    def test_non_real_online_artifact_rejected(self, tmp_path, data_source):
        """smoke / replay / derived / screenshot 一律拒绝。"""
        payload = _make_run_leaderboard(data_source=data_source)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert (
            "data_source" in decision.reason.lower()
            or "real-online" in decision.reason.lower()
        )


class TestConfigDrift:
    """配置漂移（0.9/0.92/30720 混入 14B 文本）→ fail。"""

    def test_config_drift_0_9_blocked(self, tmp_path):
        """gpu_memory_utilization=0.9 → fail（config_drift）。"""
        payload = _make_run_leaderboard(gpu_memory_utilization=0.9)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert (
            "config_drift" in decision.reason
            or "gpu_memory_utilization" in decision.reason
        )

    def test_config_drift_0_92_blocked(self, tmp_path):
        """gpu_memory_utilization=0.92 → fail（config_drift）。"""
        payload = _make_run_leaderboard(gpu_memory_utilization=0.92)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"

    def test_wrong_max_model_len_blocked(self, tmp_path):
        """14B 文本用了 30720（vision 默认值）→ fail。"""
        payload = _make_run_leaderboard(max_model_len=30720)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "max_model_len" in decision.reason or "config_drift" in decision.reason

    def test_missing_gpu_memory_utilization_blocked(self, tmp_path):
        """缺 gpu_memory_utilization 字段 → fail。"""
        payload = _make_run_leaderboard()
        del payload["same_spec"]["resolved_server_parameters"]["gpu_memory_utilization"]
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "gpu_memory_utilization" in decision.reason

    def test_missing_max_model_len_blocked(self, tmp_path):
        """缺 max_model_len 字段 → fail。"""
        payload = _make_run_leaderboard()
        del payload["same_spec"]["resolved_server_parameters"]["max_model_len"]
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "max_model_len" in decision.reason


class TestRegistryAndTarget:
    """registry 匹配、3B perfgate 不等于 14B、registry hash 不匹配。"""

    def test_3b_perfgate_not_enough(self, tmp_path):
        """只有 3B perfgate，缺 14B → fail（不等于 14B evidence）。"""
        payload = _make_run_leaderboard(
            model_repo_id="Qwen/Qwen2.5-3B-Instruct",
            model_parameters="3B",
        )
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        # 3B 不匹配 registry 的 14B profile
        assert (
            decision.profile_id is None
            or "3b" in decision.reason.lower()
            or "14b" in decision.reason.lower()
        )

    def test_registry_hash_mismatch_blocked(self, tmp_path):
        """PR 声明的 target_id 不在 registry active 列表 → fail。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(declared_target_id="nonexistent-target"),
        )
        assert decision.disposition == "fail"
        assert "target" in decision.reason.lower()


class TestSpecialty:
    """specialty target 必须携带独立 spec 和理由。"""

    def test_specialty_without_spec_blocked(self, tmp_path):
        """声称 specialty 但没带 spec → fail。"""
        # 2chip 是 specialty profile
        payload = _make_run_leaderboard(chip_count=2)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(
                declared_profile_id="multi-chip-2chip-text-14b",
                specialty_reason="multi-chip test",
            ),
            # specialty_spec 为 None
        )
        assert decision.disposition == "fail"
        assert "specialty" in decision.reason.lower()

    def test_specialty_without_reason_blocked(self, tmp_path):
        """声称 specialty 但没理由 → fail。"""
        payload = _make_run_leaderboard(chip_count=2)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(
                declared_profile_id="multi-chip-2chip-text-14b",
                specialty_spec="multi-chip-spec",
            ),
            # specialty_reason 为 None
        )
        assert decision.disposition == "fail"
        assert (
            "specialty" in decision.reason.lower()
            or "reason" in decision.reason.lower()
        )


class TestPairedSpecIdMismatch:
    """paired base/head 的 same_spec.spec_id 不一致 → fail。"""

    def test_paired_spec_id_mismatch_blocked(self, tmp_path):
        base_payload = _make_run_leaderboard(
            spec_id="official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
        )
        head_payload = _make_run_leaderboard(
            spec_id="official-ascend-jan-2026-v0.18.0-sharegpt-online-qwen25-14b-910b2",
        )
        base_path = _write_artifact(tmp_path, "base", base_payload)
        head_path = _write_artifact(tmp_path, "head", head_payload)
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "spec_id" in decision.reason.lower() or "spec" in decision.reason.lower()
        assert decision.spec_id_match is False


class TestValidPasses:
    """合规证据 → pass。"""

    def test_valid_14b_paired_passes(self, tmp_path):
        """完整合规的 14B paired evidence → pass。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "pass"
        assert decision.target_id == "official-ascend-jan-2026-v0.18.0"
        assert decision.profile_id == "core-text-14b"
        assert decision.gpu_memory_utilization == 0.6
        assert decision.max_model_len == 32768
        assert decision.spec_id_match is True

    def test_valid_specialty_passes(self, tmp_path):
        """完整合规的 specialty evidence + spec + reason → pass。"""
        payload = _make_run_leaderboard(chip_count=2)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(
                declared_profile_id="multi-chip-2chip-text-14b",
                specialty_spec="multi-chip-2chip-text-14b",
                specialty_reason="验证 2-chip TP 并行扩展收益",
            ),
        )
        assert decision.disposition == "pass"
        assert decision.details.get("specialty") is True

    def test_vision_profile_passes(self, tmp_path):
        """vision profile（max_model_len=30720）合规 → pass。"""
        payload = _make_run_leaderboard(
            model_repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            model_parameters="7B",
            workload_name="visionarena-online",
            max_model_len=30720,
            spec_id="official-ascend-jan-2026-v0.18.0-visionarena-online-qwen25-vl-7b-910b2",
        )
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(declared_profile_id="vision-7b"),
        )
        assert decision.disposition == "pass"
        assert decision.max_model_len == 30720


class TestDocsOnlyException:
    """docs-only / test-only / website-only 受控 label。"""

    def test_docs_only_label_approved(self, tmp_path):
        """docs-only label + 审批人 → pass（skip）。"""
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(
                labels=("perf-skip:docs-only",),
                skip_approver="reviewer-alice",
            ),
        )
        assert decision.disposition == "skip"
        assert (
            "docs-only" in decision.reason.lower() or "skip" in decision.reason.lower()
        )

    def test_test_only_label_approved(self, tmp_path):
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(
                labels=("perf-skip:test-only",),
                skip_approver="reviewer-bob",
            ),
        )
        assert decision.disposition == "skip"

    def test_skip_label_without_approver_blocked(self, tmp_path):
        """M5 fix: skip label 但无审批人 → fail（fail closed）。"""
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(labels=("perf-skip:docs-only",)),
        )
        assert decision.disposition == "fail"
        assert "approver" in decision.reason.lower()

    def test_docs_only_without_label_blocked(self, tmp_path):
        """docs-only 但没 label → fail（fail closed）。"""
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"


class TestDecisionLogAndJson:
    """§7.3 结构化日志 + decision.json 持久化。"""

    def test_decision_log_contains_required_fields(self, tmp_path):
        """日志含 PR/target/artifact/data_source/model/hardware/server params/disposition。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        log = format_decision_log(decision, _default_pr_context())
        assert "merge-gate-check" in log
        assert "vllm-hust#193" in log
        assert "target_id" in log
        assert "data_source" in log
        assert "gpu_memory_utilization" in log
        assert "max_model_len" in log
        assert "disposition" in log
        assert "pass" in log

    def test_write_decision_json(self, tmp_path):
        """decision.json 持久化含全部字段。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        out = tmp_path / "merge-gate-decision.json"
        write_decision_json(decision, out)
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["disposition"] == "pass"
        assert saved["target_id"] == "official-ascend-jan-2026-v0.18.0"
        assert saved["gpu_memory_utilization"] == 0.6
        assert "base_spec_id" in saved
        assert "head_spec_id" in saved

    def test_decision_log_for_fail(self, tmp_path):
        """fail 场景的日志也打印判定详情。"""
        decision = evaluate_merge_gate(
            base=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            head=ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING),
            pr_context=_default_pr_context(),
        )
        log = format_decision_log(decision, _default_pr_context())
        assert "fail" in log
        assert "missing" in log.lower() or "evidence" in log.lower()


# ---------------------------------------------------------------------------
# Review fix 回归测试：C1-C4 + M1-M3/M5
# ---------------------------------------------------------------------------


class TestHeadOnlyConfigDrift:
    """C1 fix: head artifact 的配置漂移也必须被挡（PR 自身 commit 引入漂移）。"""

    def test_head_only_gpu_memory_drift_blocked(self, tmp_path):
        """base 合规、head gmu=0.9 → fail（C1 回归）。"""
        base_payload = _make_run_leaderboard()
        head_payload = _make_run_leaderboard(gpu_memory_utilization=0.9)
        base_path = _write_artifact(tmp_path, "base", base_payload)
        head_path = _write_artifact(tmp_path, "head", head_payload)
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "config_drift" in decision.reason
        assert "head" in decision.reason

    def test_head_only_max_model_len_drift_blocked(self, tmp_path):
        """base 合规、head mml=30720 → fail（C1 回归）。"""
        base_payload = _make_run_leaderboard()
        head_payload = _make_run_leaderboard(max_model_len=30720)
        base_path = _write_artifact(tmp_path, "base", base_payload)
        head_path = _write_artifact(tmp_path, "head", head_payload)
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "config_drift" in decision.reason
        assert "head" in decision.reason


class TestMissingTargetDeclaration:
    """C3 fix: target_id/target_version/profile_id 强制声明。"""

    def test_missing_target_id_blocked(self, tmp_path):
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(declared_target_id=None),
        )
        assert decision.disposition == "fail"
        assert "target_id" in decision.reason

    def test_missing_target_version_blocked(self, tmp_path):
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(declared_target_version=None),
        )
        assert decision.disposition == "fail"
        assert "target_version" in decision.reason

    def test_missing_profile_id_blocked(self, tmp_path):
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(declared_profile_id=None),
        )
        assert decision.disposition == "fail"
        assert "profile_id" in decision.reason


class TestDeclaredTargetMismatch:
    """M6 fix: 声明的 target_id/profile_id 必须与 artifact 实际匹配的 profile 一致。

    防止 PR 声明一个有效但与实际 artifact 不匹配的 target 来 bypass。
    注意：target_version 是版本号与 registry 描述性名称不同概念，不做一致性比较。
    """

    def test_declared_target_id_matches_profile_passes(self, tmp_path):
        """默认声明的 target_id 与 profile 一致 → pass（回归保护）。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
            registry_path=DEFAULT_REGISTRY_PATH,
        )
        assert decision.disposition == "pass"

    def test_declared_profile_id_mismatch_blocked(self, tmp_path):
        """declared_profile_id 与 profile.profile_name 不一致 → fail。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(declared_profile_id="wrong-profile"),
        )
        assert decision.disposition == "fail"
        assert (
            "profile_id" in decision.reason.lower()
            or "profile_name" in decision.reason.lower()
        )


class TestSpecIdNoneMatch:
    """C4 fix: spec_id 为 None/空时 fail（不允许 None==None 通过）。"""

    def test_both_spec_id_missing_blocked(self, tmp_path):
        """base/head 都没有 same_spec 块 → fail（不是 None==None 通过）。"""
        payload = _make_run_leaderboard(same_spec_present=False)
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "spec_id" in decision.reason

    def test_empty_spec_id_blocked(self, tmp_path):
        """spec_id 为空字符串 → fail。"""
        payload = _make_run_leaderboard(spec_id="")
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "spec_id" in decision.reason


class TestSpecHashMismatch:
    """M1 fix: resolved_spec_hash 不一致 → fail。"""

    def test_spec_hash_mismatch_blocked(self, tmp_path):
        base_payload = _make_run_leaderboard(
            spec_hash="a" * 64,
        )
        head_payload = _make_run_leaderboard(
            spec_hash="b" * 64,
        )
        base_path = _write_artifact(tmp_path, "base", base_payload)
        head_path = _write_artifact(tmp_path, "head", head_payload)
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "spec_hash" in decision.reason
        assert decision.spec_hash_match is False


class TestProfileMismatch:
    """M2 fix: base/head 匹配不同 profile → fail。"""

    def test_base_head_different_profile_blocked(self, tmp_path):
        """base 匹配 core-text-14b，head 匹配 coder-14b → fail。"""
        base_payload = _make_run_leaderboard()
        head_payload = _make_run_leaderboard(workload_name="instructcoder-online")
        head_path = _write_artifact(tmp_path, "head", head_payload)
        base_path = _write_artifact(tmp_path, "base", base_payload)
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
        )
        assert decision.disposition == "fail"
        assert "profile" in decision.reason.lower()


class TestRegistryLoadFailure:
    """M3 fix: registry 加载失败 → fail closed（返回 decision 而非 raise）。"""

    def test_registry_missing_fail_closed(self, tmp_path):
        """registry 文件不存在 → fail（不 raise）。"""
        payload = _make_run_leaderboard()
        base_path = _write_artifact(tmp_path, "base", payload)
        head_path = _write_artifact(tmp_path, "head", payload.copy())
        decision = evaluate_merge_gate(
            base=_accepted(base_path),
            head=_accepted(head_path),
            pr_context=_default_pr_context(),
            registry_path=tmp_path / "nonexistent_registry.json",
        )
        assert decision.disposition == "fail"
        assert "registry" in decision.reason.lower()


# ---------------------------------------------------------------------------
# --dry-run CLI 层测试（issue #95 增强建议 9）
# 用于 #105 phase 4 PR 补跑预检：判定 fail 时不阻塞，只输出结果。
# ---------------------------------------------------------------------------


class TestDryRunMode:
    """--dry-run 模式：fail 也返回 0，用于 PR 补跑预检。"""

    def test_dry_run_fail_returns_zero(self, tmp_path, capsys):
        """dry-run + fail → 退出码 0（不阻塞预检流程）。"""
        from vllm_hust_benchmark.cli import main

        exit_code = main(
            [
                "merge-gate-check",
                "--repo",
                "vllm-hust",
                "--pr-number",
                "999",
                "--head-sha",
                "aaa111",
                "--base-sha",
                "bbb222",
                "--base-status",
                "missing",
                "--head-status",
                "missing",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "fail" in captured.out.lower()
        assert "dry-run" in captured.out.lower() or "dry_run" in captured.out.lower()

    def test_dry_run_pass_returns_zero(self, tmp_path, capsys):
        """dry-run + pass → 退出码 0。"""
        from vllm_hust_benchmark.cli import main

        base_path = _write_artifact(tmp_path, "base")
        head_path = _write_artifact(tmp_path, "head", _make_run_leaderboard())
        exit_code = main(
            [
                "merge-gate-check",
                "--repo",
                "vllm-hust",
                "--pr-number",
                "193",
                "--head-sha",
                "abc1234",
                "--base-sha",
                "def5678",
                "--base-artifact",
                str(base_path),
                "--head-artifact",
                str(head_path),
                "--declared-target-id",
                "official-ascend-jan-2026-v0.18.0",
                "--declared-target-version",
                "v0.18.0",
                "--declared-profile-id",
                "core-text-14b",
                "--dry-run",
            ]
        )
        assert exit_code == 0

    def test_no_dry_run_fail_returns_one(self, tmp_path, capsys):
        """无 --dry-run + fail → 退出码 1（原有行为不变）。"""
        from vllm_hust_benchmark.cli import main

        exit_code = main(
            [
                "merge-gate-check",
                "--repo",
                "vllm-hust",
                "--pr-number",
                "999",
                "--head-sha",
                "aaa111",
                "--base-sha",
                "bbb222",
                "--base-status",
                "missing",
                "--head-status",
                "missing",
            ]
        )
        assert exit_code == 1

    def test_dry_run_writes_decision_json(self, tmp_path, capsys):
        """dry-run 模式仍然写 decision.json（用于预检留档）。"""
        from vllm_hust_benchmark.cli import main

        decision_path = tmp_path / "merge-gate-decision.json"
        exit_code = main(
            [
                "merge-gate-check",
                "--repo",
                "vllm-hust",
                "--pr-number",
                "999",
                "--head-sha",
                "aaa111",
                "--base-sha",
                "bbb222",
                "--base-status",
                "missing",
                "--head-status",
                "missing",
                "--dry-run",
                "--decision-output",
                str(decision_path),
            ]
        )
        assert exit_code == 0
        assert decision_path.exists()
        saved = json.loads(decision_path.read_text(encoding="utf-8"))
        assert saved["disposition"] == "fail"
