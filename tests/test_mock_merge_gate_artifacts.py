"""Mock merge gate artifact 生成器的 TDD 测试（issue #95 Layer 2 mock 模式）。

验证生成器产出的 mock artifact 能被 merge-gate-check 正确判定为期望的 disposition。
这样 workflow 接线正确性可以脱离真实 NPU 验证。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_hust_benchmark.merge_gate import (
    ArtifactRef,
    ArtifactStatus,
    PRContext,
    evaluate_merge_gate,
)

GENERATOR_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "generate_mock_merge_gate_artifacts.py"
)

# Must match the constants in generate_mock_merge_gate_artifacts.py
VALID_ENGINE_COMMIT = (
    "e4ce33646f2ef1781289e6dc651fad0d00177c55"  # pragma: allowlist secret
)
REGISTRY_TARGET_VERSION = "Official Ascend Jan 2026"


def _run_generator(tmp_path: Path, scenario: str) -> Path:
    """运行 mock 生成器，返回 output_dir。"""
    output_dir = tmp_path / "mock" / scenario
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR_SCRIPT),
            "--scenario",
            scenario,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, f"generator failed: {result.stderr}"
    return output_dir


def _load_artifact(output_dir: Path, side: str) -> Path:
    """加载 base 或 head 的 run_leaderboard.json 路径。"""
    path = output_dir / side / "run_leaderboard.json"
    assert path.is_file(), f"missing {side} artifact: {path}"
    return path


def _accepted(path: Path) -> ArtifactRef:
    return ArtifactRef(path=path, ci_status=ArtifactStatus.ACCEPTED)


def _missing() -> ArtifactRef:
    return ArtifactRef(path=None, ci_status=ArtifactStatus.MISSING)


def _default_pr_context(**overrides) -> PRContext:
    defaults = {
        "repo": "vllm-hust",
        "number": 193,
        "head_sha": VALID_ENGINE_COMMIT,
        "base_sha": VALID_ENGINE_COMMIT,
        "declared_target_id": "official-ascend-jan-2026-v0.18.0",
        "declared_target_version": REGISTRY_TARGET_VERSION,
        "declared_profile_id": "core-text-14b",
    }
    defaults.update(overrides)
    return PRContext(**defaults)


def _pr_context_from_manifest(output_dir: Path, **overrides) -> PRContext:
    """从 scenario_manifest.json 构建 PRContext（读取声明的 target/profile/skip）。"""
    manifest = json.loads((output_dir / "scenario_manifest.json").read_text())
    kwargs = {
        "repo": "vllm-hust",
        "number": 193,
        "head_sha": manifest.get("pr_head_sha", VALID_ENGINE_COMMIT),
        "base_sha": manifest.get("pr_base_sha", VALID_ENGINE_COMMIT),
        "declared_target_id": manifest.get("declared_target_id"),
        "declared_target_version": manifest.get("declared_target_version"),
        "declared_profile_id": manifest.get("declared_profile_id"),
    }
    if "pr_labels" in manifest:
        kwargs["labels"] = tuple(manifest["pr_labels"])
    if manifest.get("specialty_spec"):
        kwargs["specialty_spec"] = manifest["specialty_spec"]
    if manifest.get("specialty_reason"):
        kwargs["specialty_reason"] = manifest["specialty_reason"]
    if manifest.get("skip_approver"):
        kwargs["skip_approver"] = manifest["skip_approver"]
    kwargs.update(overrides)
    return PRContext(**kwargs)


# ---------------------------------------------------------------------------
# 场景：mock 生成器产出 + merge-gate 判定配对验证
# ---------------------------------------------------------------------------


class TestMockPassScenario:
    """pass 场景：合规 14B paired evidence。"""

    def test_generates_valid_paired_artifacts(self, tmp_path):
        out = _run_generator(tmp_path, "pass")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "pass"
        assert decision.target_id == "official-ascend-jan-2026-v0.18.0"
        assert decision.gpu_memory_utilization == 0.6
        assert decision.max_model_len == 32768

    def test_generator_writes_manifest(self, tmp_path):
        out = _run_generator(tmp_path, "pass")
        manifest = json.loads((out / "scenario_manifest.json").read_text())
        assert manifest["scenario"] == "pass"
        assert manifest["expected_disposition"] == "pass"


class TestMockConfigDriftScenario:
    """fail 场景：gpu_memory_utilization=0.9 配置漂移。"""

    def test_config_drift_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_config_drift")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert (
            "config_drift" in decision.reason
            or "gpu_memory_utilization" in decision.reason
        )


class TestMockDataSourceRejectScenario:
    """fail 场景：data_source 不是 real-online。"""

    def test_data_source_rejected(self, tmp_path):
        out = _run_generator(tmp_path, "fail_data_source")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert "data_source" in decision.reason.lower()


class TestMockUnpairedSpecScenario:
    """fail 场景：base/head spec_id 不一致。"""

    def test_unpaired_spec_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_unpaired_spec")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert "spec_id" in decision.reason.lower() or "spec" in decision.reason.lower()


class TestMock3BNot14BScenario:
    """fail 场景：3B perfgate 不等于 14B evidence。"""

    def test_3b_not_enough(self, tmp_path):
        out = _run_generator(tmp_path, "fail_3b_not_14b")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"


class TestMockMissingArtifactScenario:
    """fail 场景：只生成 base，head 缺失（CI 状态 missing）。"""

    def test_missing_artifact_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_missing_artifact")
        base = _accepted(_load_artifact(out, "base"))
        # head 目录不存在
        head = _missing()
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert decision.head_status == "missing"

    def test_manifest_declares_expected_head_missing(self, tmp_path):
        out = _run_generator(tmp_path, "fail_missing_artifact")
        manifest = json.loads((out / "scenario_manifest.json").read_text())
        assert manifest["expected_head_status"] == "missing"


class TestMockSkipDocsOnlyScenario:
    """skip 场景：docs-only label。"""

    def test_docs_only_skip(self, tmp_path):
        out = _run_generator(tmp_path, "skip_docs_only")
        decision = evaluate_merge_gate(
            base=_missing(),
            head=_missing(),
            pr_context=_pr_context_from_manifest(out),
        )
        assert decision.disposition == "skip"


class TestMockSpecialtyValidScenario:
    """pass 场景：合规 specialty（2-chip + spec + reason）。"""

    def test_specialty_valid_passes(self, tmp_path):
        out = _run_generator(tmp_path, "specialty_valid")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base,
            head=head,
            pr_context=_pr_context_from_manifest(out),
        )
        assert decision.disposition == "pass"
        assert decision.details.get("specialty") is True


class TestMockSpecialtyNoReasonScenario:
    """fail 场景：specialty 缺 reason（manifest 不含 specialty_reason）。"""

    def test_specialty_no_reason_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "specialty_no_reason")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base,
            head=head,
            pr_context=_pr_context_from_manifest(out),
        )
        assert decision.disposition == "fail"
        assert (
            "specialty" in decision.reason.lower()
            or "reason" in decision.reason.lower()
        )


class TestMockRegistryHashMismatchScenario:
    """fail 场景：声明的 target_id 不在 registry（manifest 已声明 nonexistent-target）。"""

    def test_registry_hash_mismatch_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_registry_hash_mismatch")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base,
            head=head,
            pr_context=_pr_context_from_manifest(out),
        )
        assert decision.disposition == "fail"


class TestMockConfigDriftHeadOnlyScenario:
    """fail 场景：仅 head 配置漂移（C1 fix 回归）。"""

    def test_head_only_drift_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_config_drift_head_only")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert "head" in decision.reason
        assert (
            "config_drift" in decision.reason
            or "gpu_memory_utilization" in decision.reason
        )


class TestMockMissingTargetDeclarationScenario:
    """fail 场景：未声明 target_id/target_version/profile_id（C3 fix）。"""

    def test_missing_target_declaration_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_missing_target_declaration")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert (
            "target" in decision.reason.lower() or "declared" in decision.reason.lower()
        )


class TestMockSkipDocsOnlyNoApproverScenario:
    """fail 场景：docs-only label 但无 skip_approver（M5 fix）。"""

    def test_skip_without_approver_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "skip_docs_only_no_approver")
        decision = evaluate_merge_gate(
            base=_missing(),
            head=_missing(),
            pr_context=_pr_context_from_manifest(out),
        )
        assert decision.disposition == "fail"
        assert (
            "skip" in decision.reason.lower() or "approver" in decision.reason.lower()
        )


class TestMockHeadCommitMismatchScenario:
    """fail 场景：head artifact 的 engine_commit 与 PR head_sha 不匹配。"""

    def test_head_commit_mismatch_blocked(self, tmp_path):
        out = _run_generator(tmp_path, "fail_head_commit_mismatch")
        base = _accepted(_load_artifact(out, "base"))
        head = _accepted(_load_artifact(out, "head"))
        decision = evaluate_merge_gate(
            base=base, head=head, pr_context=_pr_context_from_manifest(out)
        )
        assert decision.disposition == "fail"
        assert "head_sha" in decision.reason or "head artifact" in decision.reason


class TestMockScenarioList:
    """生成器必须支持全部场景。"""

    @pytest.mark.parametrize(
        "scenario",
        [
            "pass",
            "fail_config_drift",
            "fail_config_drift_head_only",
            "fail_data_source",
            "fail_unpaired_spec",
            "fail_3b_not_14b",
            "fail_missing_artifact",
            "fail_missing_target_declaration",
            "skip_docs_only",
            "skip_docs_only_no_approver",
            "specialty_valid",
            "specialty_no_reason",
            "fail_registry_hash_mismatch",
            "fail_head_commit_mismatch",
        ],
    )
    def test_scenario_supported(self, tmp_path, scenario):
        out = _run_generator(tmp_path, scenario)
        manifest = json.loads((out / "scenario_manifest.json").read_text())
        assert manifest["scenario"] == scenario
        assert "expected_disposition" in manifest
