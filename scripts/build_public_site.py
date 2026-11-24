"""Create a public website projection while retaining immutable research sources.

Run only on an assembled, validated website checkout. The output is a new
directory; this tool never deletes or rewrites the input research evidence.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import zipfile
from urllib.parse import unquote, urlsplit
from pathlib import Path

PUBLIC_ENGINE_HISTORY_PATH = "data/engine-release-history-v1.json"
PUBLIC_FIXED_B0_PATH = "data/fixed-b0-original018-v1.json"
PUBLIC_FIXED_B0_SHA256 = "8b5bbddb6ae681513f8476e02f27ec5333a486a51a45848ac01092b310f27d4d"  # pragma: allowlist secret -- fixed public B0 observations
PUBLIC_ENGINE_HISTORY_SHA256 = "c1263959ca8f1044a3c003746b3939cc0a57a59bcbed7fce6220fa73401a2c04"  # pragma: allowlist secret -- fixed public release catalog


PROJECT = re.compile(
    r"betterscale|traceloom|gennova|latchmoe|tricard|"
    r"prefix-cache-routing-reliability|prefix[ _-]routing|pr173-dev4|"
    r"ascend-adaptive-quantized|adaptive[ _-]+quantized[ _-]+kv|"
    r"vllm-hust-perf-analyzer|vllm-hust-cuda-(?:qk-rope|hybrid-apc|async-tp|fullgraph-matrix)",
    re.I,
)
IDENTITY_KEYS = {
    "id",
    "name",
    "name_en",
    "name_zh",
    "label",
    "title",
    "title_en",
    "title_zh",
    "engine",
    "engine_name",
    "solution",
    "component_id",
    "plugin_id",
    "repository",
    "repository_url",
    "component_ids",
    "mods",
    "mod",
    "series_id",
    "series_label",
    "implementation",
    "runtime_name",
    "worker_class",
    "worker_cls",
    "family",
}
DROP = object()

# The owner explicitly restored this historical C16 comparison on 2026-10-31.
C16_EVIDENCE_PATH = "data/leaderboard_frontier_swe_evidence.json"
PUBLIC_C16_RUN_IDS = frozenset(
    {
        "dab8ed7c4ee74f2d958dab5528514294",  # pragma: allowlist secret - public B0 run ID
        "fa0e01a32eef456bab4bebe51e4aa7dd",  # pragma: allowlist secret - public B1 run ID
    }
)

# Allowlisted MOD card with the historical service comparison and R10 component aggregate.
# The separate project page, raw artifacts, and other BetterScale records stay filtered.
BETTERSCALE_ID = "betterscale"
ACCEPTED_STATE_FUSION_ID = "accepted-state-fusion"
ACCEPTED_STATE_FUSION_HOST = "vLLM-HUST Engine R10 · BetterScale"
BETTERSCALE_CARD = {
    "id": BETTERSCALE_ID,
    "name": "BetterScale Execution and State Scheduling",
    "name_en": "BetterScale Execution and State Scheduling",
    "name_zh": "BetterScale 执行与状态调度",
    "artifact_type": "runtime_component",
    "system_role": "graph_execution_policy",
    "integration_contracts": [],
    "execution_planes": ["worker", "native", "device"],
    "deployment_topology": "in_process",
    "delivery_model": "python_distribution",
    "ownership": "hust_owned_subsystem",
    "repository_relationship": "organization_native",
    "repository_visibility": "private",
    "maturity": "experimental",
    "maintainers": ["CubeLander"],
    "maintainer_profiles": [{"login": "CubeLander", "name": "田景远"}],
    "summary_en": "Multi-device inference execution optimization for Ascend. The frozen B1 r6 profile uses its asynchronous execution and state path at E24/R36. Frozen R10 uses its resident-state path. The R10 fusion experiment is recorded under Accepted Token State Fusion.",
    "summary_zh": "面向昇腾的多卡推理执行优化。B1 r6 冻结组合使用其异步执行与状态管理路径，配置为 E24/R36。 冻结 R10 使用其状态管理路径。 已接受 Token 状态写回融合的 R10 试验记录见同名独立 MOD 条目。",
    "public_effect_status": "measured",
    "public_effect_en": "Historical DeepSeek-V4-Flash HTTP test: output throughput rose 39.63% at DP8/C16 and 35.17% at TP8/C4 against their respective baselines. R10 single-card component test of 16-row resident-slot tiling. All six decode and mixed shapes passed bitwise checks on graph replay with changed inputs. At 54 live rows, decode fell from 21.45 to 8.74 μs and mixed from 33.90 to 15.16 μs. The 1/2/4/8-warp settings generated identical device binaries within each branch. Model-level validation is pending. The same-host, same-card MMLU-Pro ten-item screen used a historical R10 control. Output throughput +1.96%, P95 TTFT +2.29%, P95 TPOT −1.07%, batch wall time +2.16%, correct answers 10 to 9. Larger model-level validation is pending for R11.",
    "public_effect_zh": "DeepSeek-V4-Flash 历史 HTTP 实测中，DP8／并发 16 和 TP8／并发 4 的输出吞吐分别较各自基线提高 39.63% 和 35.17%。 冻结 R10 单卡组件对照中，驻留状态槽改为 16 行分块。六种解码和混合形状均通过动态输入图回放逐位核对。54 活跃行时，解码 21.45→8.74 μs，混合 33.90→15.16 μs。1／2／4／8 warp 在同一分支生成相同设备二进制。分块仅取得组件证据，整模待验证。 同机同卡分时 MMLU-Pro 十题初筛中，输出吞吐 +1.96%，P95 首字 +2.29%，P95 每 token 耗时 −1.07%，整批耗时 +2.16%，答对 10→9。这组数据用于整模初筛，R11 仍待扩大复核。",
    "public_effect_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e2481edfeb341e856ac11705336f8e872f423a23/website-publication/data/r10-parallel-components-20261124/summary.json#L5",
    "public_effect_detail_en": [
        "2026-09-14 · DeepSeek-V4-Flash-0731 W8A8 on 8 × Ascend 910B2 (HCCS), vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1, K5.",
        "DP8, concurrency 16, 8 GiB KV/rank: 268.89 → 375.45 output tok/s (+39.63%), runs 158 → 155. The candidate included the DP startup preparation carried into 0.3.1.",
        "TP8, concurrency 4, 12 GiB KV/rank: 155.13 → 209.68 output tok/s (+35.17%), runs 160 → 161. The baseline included the LCM startup compatibility fix.",
        "Each comparison divides total output tokens by total timed duration across all nine synthetic HTTP rounds (three workloads, three repeats). Startup, the separately logged prelude and quality scoring were excluded. Each arm passed 32/32 retrieval questions. TP decode P99 chunk-gap worsened. These measurements do not establish maximum concurrency, SLO capacity, or performance of a newer wheel.",
    ],
    "public_effect_detail_zh": [
        "2026-09-14 · DeepSeek-V4-Flash-0731 W8A8，8 张昇腾 910B2（HCCS），vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1，K5。",
        "DP8，并发 16，每 rank 8 GiB KV。原生基线 268.89 → 优化后 375.45 输出 token/s（+39.63%，run 158 → 155）。候选包含后续收录于 0.3.1 的 DP 启动预热。",
        "TP8，并发 4，每 rank 12 GiB KV。含 LCM 启动兼容修复的基线 155.13 → 优化后 209.68 输出 token/s（+35.17%，run 160 → 161）。",
        "各自按九组固定合成 HTTP 负载（三种负载，各三次）的输出 token 总数除以计时总长。启动、单独记录的预备阶段和质量评分不计时。各组检索 32/32，TP 解码 P99 输出块间隔变差。结果仅适用于上述实测工况，不能用于推断最大并发、SLO 容量或新版软件包性能。",
    ],
    "evidence_level": "hardware_verified",
}
BETTERSCALE_CATALOG_ENTRIES = {
    "data/mod-taxonomy.json": {
        "kind": "runtime_mod",
        "capability": "execution_optimization",
        "phases": ["prefill", "decode"],
        "lifecycle": "implemented",
        "evidence": "measured_beneficial",
    },
    "data/plugin-workload-navigation.json": ["distributed_pipeline"],
}
PUBLIC_ADVISOR_FIELDS = ("name_zh", "name_en", "relationship")


def public_advisors(component):
    return [
        {key: advisor[key] for key in PUBLIC_ADVISOR_FIELDS}
        for advisor in component.get("advisors", [])
    ]


def add_betterscale_card(path, projected, original):
    if path == "data/ecosystem.json":
        source = next(
            (item for item in original["components"] if item["id"] == BETTERSCALE_ID),
            None,
        )
        if source is not None:
            card = copy.deepcopy(BETTERSCALE_CARD)
            card["advisors"] = public_advisors(source)
            projected["components"].append(card)
        fusion = next(
            (
                item
                for item in projected["components"]
                if item["id"] == ACCEPTED_STATE_FUSION_ID
            ),
            None,
        )
        if fusion is not None:
            fusion["compatibility"]["host"] = ACCEPTED_STATE_FUSION_HOST
    elif path in BETTERSCALE_CATALOG_ENTRIES:
        key = (
            "plugins"
            if path == "data/plugin-workload-navigation.json"
            else "components"
        )
        if BETTERSCALE_ID in original[key]:
            projected[key][BETTERSCALE_ID] = copy.deepcopy(
                BETTERSCALE_CATALOG_ENTRIES[path]
            )


def validate_betterscale_card(path, data):
    if path == "data/ecosystem.json":
        cards = [item for item in data["components"] if item["id"] == BETTERSCALE_ID]
        assert len(cards) <= 1, path
        if cards:
            card = cards[0]
            data["components"].remove(card)
            advisors = card.pop("advisors")
            assert isinstance(advisors, list), path
            assert all(
                isinstance(advisor, dict)
                and set(advisor) == set(PUBLIC_ADVISOR_FIELDS)
                and all(isinstance(value, str) and value for value in advisor.values())
                for advisor in advisors
            ), path
            assert not matches(advisors), path
            assert card == BETTERSCALE_CARD, path
        for item in data["components"]:
            if item["id"] == ACCEPTED_STATE_FUSION_ID:
                assert (
                    item["compatibility"].pop("host") == ACCEPTED_STATE_FUSION_HOST
                ), path
    elif path in BETTERSCALE_CATALOG_ENTRIES:
        key = (
            "plugins"
            if path == "data/plugin-workload-navigation.json"
            else "components"
        )
        if BETTERSCALE_ID in data[key]:
            assert data[key].pop(BETTERSCALE_ID) == BETTERSCALE_CATALOG_ENTRIES[path], (
                path
            )


# Explicit catalog-only scope for these MODs. Withdrawn pages and results retain their filters.
PUBLIC_MOD_PREVIEWS = {
    "traceloom": {
        "card": {
            "id": "traceloom",
            "name": "TraceLoom Inference Timeline Analysis",
            "name_en": "TraceLoom Inference Timeline Analysis",
            "name_zh": "TraceLoom 推理时间线分析",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Inference timeline analysis with scheduler-step attribution. Its analysis functions have been demonstrated on Ascend.",
            "summary_zh": "推理时间线分析工具，可把 profiler 事件关联到调度步骤。已有昇腾上的分析功能验证。",
            "system_role": "profiling_analysis",
            "execution_planes": ["scheduler", "worker", "offline"],
            "deployment_topology": "mixed",
            "delivery_model": "python_distribution",
            "ownership": "hust_owned_subsystem",
            "repository_relationship": "organization_native",
            "evidence_level": "hardware_verified",
        },
        "taxonomy": {
            "kind": "tool_mod",
            "capability": "observability_evaluation",
            "phases": ["observability"],
            "lifecycle": "implemented",
            "evidence": "no_performance_claim",
        },
    },
    "latchmoe": {
        "card": {
            "id": "latchmoe",
            "name": "LatchMoE Expert-Weight Offload",
            "name_en": "LatchMoE Expert-Weight Offload",
            "name_zh": "LatchMoE 专家权重卸载",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "MoE expert weight offloading. Functional validation covers one Ascend NPU in TP1 PIECEWISE mode. Historical TP4 throughput regressed.",
            "summary_zh": "MoE 专家权重卸载。功能验证限于昇腾单卡 TP1 PIECEWISE，历史 TP4 吞吐对照有回退。",
            "system_role": "operator_runtime",
            "execution_planes": ["worker", "native", "device"],
            "deployment_topology": "in_process",
            "delivery_model": "plugin_bundle",
            "ownership": "hust_owned_subsystem",
            "repository_relationship": "organization_native",
            "evidence_level": "hardware_verified",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "operators_attention",
            "phases": ["prefill", "decode"],
            "lifecycle": "implemented_restricted",
            "evidence": "qualified_restricted",
        },
    },
    "tricard-clm-lifecycle": {
        "card": {
            "id": "tricard-clm-lifecycle",
            "name": "Tricard Lifecycle Observation",
            "name_en": "Tricard Lifecycle Observation",
            "name_zh": "Tricard 生命周期观测",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Lifecycle event observer for long conversations. Validation covers Bundle packaging and enablement intent.",
            "summary_zh": "长会话生命周期事件观察组件。目前验证范围为 Bundle 打包和启用意图。",
            "system_role": "lifecycle_control_plane",
            "execution_planes": ["scheduler", "control_plane"],
            "deployment_topology": "mixed",
            "delivery_model": "plugin_bundle",
            "ownership": "original_contributor_maintained",
            "repository_relationship": "organization_native",
            "evidence_level": "cpu_smoke",
        },
        "taxonomy": {
            "kind": "control_plane",
            "capability": "lifecycle_control",
            "phases": ["service_lifecycle", "observability"],
            "lifecycle": "implemented_restricted",
            "evidence": "bridge_component",
        },
    },
    "ascend-adaptive-quantized-kv": {
        "card": {
            "id": "ascend-adaptive-quantized-kv",
            "name": "Ascend Adaptive KV Quantization",
            "name_en": "Ascend Adaptive KV Quantization",
            "name_zh": "昇腾自适应 KV 量化",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Adaptive KV-cache quantization research on Ascend. Installation, discovery, and inspection work. Runtime activation awaits host-interface qualification.",
            "summary_zh": "昇腾 KV 缓存自适应量化研究。目前可安装、发现和检查，运行时启用待宿主接口验证。",
            "compatibility": {"status": "inspect_only"},
            "system_role": "kv_quantization_research",
            "execution_planes": ["worker", "device"],
            "deployment_topology": "in_process",
            "delivery_model": "python_distribution",
            "ownership": "hust_owned_subsystem",
            "repository_relationship": "organization_native",
            "evidence_level": "cpu_smoke",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "kv_management",
            "phases": ["prefill", "decode", "state_lifecycle"],
            "lifecycle": "pending",
            "evidence": "activation_pending",
        },
    },
    "dla": {
        "card": {
            "id": "dla",
            "name": "DLA Output-Length-Aware Admission",
            "name_en": "DLA Output-Length-Aware Admission",
            "name_zh": "DLA 输出长度感知准入",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Length-aware admission and preemption research. A known-output-budget experiment exercised admission checks and recorded zero deferrals or preemptions.",
            "summary_zh": "输出长度感知的请求准入与抢占研究。已测版本使用已知输出预算，观测到准入检查，延后和抢占均未触发。",
            "system_role": "scheduler_policy",
            "execution_planes": ["scheduler"],
            "deployment_topology": "in_process",
            "delivery_model": "plugin_bundle",
            "ownership": "hust_owned_subsystem",
            "repository_relationship": "organization_native",
            "evidence_level": "performance_verified",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "scheduling_routing",
            "phases": ["request_admission"],
            "lifecycle": "implemented_restricted",
            "evidence": "measured_inconclusive",
        },
    },
    "stateaxis-automatic-prefix-cache": {
        "card": {
            "id": "stateaxis-automatic-prefix-cache",
            "name": "StateAxis Automatic Prefix Cache",
            "name_en": "StateAxis Automatic Prefix Cache",
            "name_zh": "StateAxis 自动前缀缓存",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Historical testing verified activation and rollback, with zero prefix-cache hits in the measured window.",
            "summary_zh": "历史测试验证了启用和回退，测试窗口中前缀缓存命中为零。",
            "repository_relationship": "external_reference",
            "system_role": "prefix_cache_policy",
            "execution_planes": ["scheduler", "worker"],
            "deployment_topology": "in_process",
            "delivery_model": "python_distribution",
            "ownership": "original_contributor_maintained",
            "evidence_level": "runtime_activation_verified_performance_unqualified",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "kv_management",
            "phases": ["prefill", "state_lifecycle"],
            "lifecycle": "implemented_restricted",
            "evidence": "runtime_effective_no_benefit_demonstrated",
        },
    },
    "stateaxis-async-kv-transfer": {
        "card": {
            "id": "stateaxis-async-kv-transfer",
            "name": "StateAxis Async KV Transfer",
            "name_en": "StateAxis Async KV Transfer",
            "name_zh": "StateAxis 异步 KV 传输",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Historical component tests observed host-to-device KV restoration overlapping with compute. Online gains of the current artifact await validation.",
            "summary_zh": "历史组件测试验证了主机到设备的 KV 恢复与计算重叠，当前制品的在线收益待验证。",
            "repository_relationship": "external_reference",
            "system_role": "kv_transfer_overlap",
            "execution_planes": ["worker", "device"],
            "deployment_topology": "in_process",
            "delivery_model": "python_distribution",
            "ownership": "original_contributor_maintained",
            "evidence_level": "runtime_activation_verified_historical_performance_only",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "kv_management",
            "phases": ["state_lifecycle"],
            "lifecycle": "implemented_restricted",
            "evidence": "historical_scoped_positive_current_unqualified",
        },
    },
    "stateaxis-pipelined-weight-loading": {
        "card": {
            "id": "stateaxis-pipelined-weight-loading",
            "name": "StateAxis Pipelined Weight Loading",
            "name_en": "StateAxis Pipelined Weight Loading",
            "name_zh": "StateAxis 流水化权重加载",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Historical Ascend TP2 diagnostics found improvements in startup timing. Formal activation and steady-state performance remain pending. R10 admission review found that the current manifest requires a StateAxis host and Qwen2. Frozen R10 uses Qwen3.5 MoE. No candidate was launched.",
            "summary_zh": "历史昇腾 TP2 诊断验证了部分启动阶段耗时改善，正式启用和稳态性能仍待验证。 R10 准入复核中，当前扩展清单限定 StateAxis 宿主和 Qwen2，冻结 R10 使用 Qwen3.5 MoE，未运行候选。",
            "repository_relationship": "external_reference",
            "system_role": "model_weight_loading",
            "execution_planes": ["worker", "native"],
            "deployment_topology": "in_process",
            "delivery_model": "python_distribution",
            "ownership": "original_contributor_maintained",
            "evidence_level": "hardware_startup_diagnostic",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "startup_deployment",
            "phases": ["service_startup"],
            "lifecycle": "implemented_restricted",
            "evidence": "startup_diagnostic_formal_ecpa_pending",
        },
    },
    "cuda-mtp": {
        "card": {
            "id": "cuda-mtp",
            "name": "CUDA Multi-Token Prediction",
            "name_en": "CUDA Multi-Token Prediction",
            "name_zh": "CUDA 多词元预测（MTP）",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "public",
            "summary_en": "CUDA activation carrier for Qwen3.5 native multi-token prediction. Bounded launch and functional checks ran on A6000 PP4/TP1.",
            "summary_zh": "Qwen3.5 原生多词元预测的 CUDA 启用组件。已有 A6000 PP4/TP1 的有限启动与功能验证。",
            "canonical_repository": "https://github.com/vLLM-HUST/vllm-hust-cuda-mtp",
            "repository_relationship": "organization_native",
            "system_role": "speculative_decoding",
            "execution_planes": ["worker", "device"],
            "delivery_model": "python_distribution",
            "ownership": "original_contributor_maintained",
            "evidence_level": "hardware_verified",
            "public_effect_en": "Bounded startup and functional checks cover A6000 PP4/TP1. The tested setup reduced available KV capacity and produced one answer mismatch; it does not establish a general speed or quality gain.",
            "public_effect_zh": "已有 A6000 PP4/TP1 的有限启动与功能检查。该配置可用 KV 容量下降，且出现一次回答不一致；尚不能据此确认普遍的速度或质量收益。",
            "public_effect_status": "preview",
            "public_effect_url": "https://github.com/vLLM-HUST/vllm-hust-cuda-mtp#readme",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "decoding",
            "phases": ["decode"],
            "lifecycle": "qualification_pending",
            "evidence": "hardware_validation_pending",
        },
    },
    "memory-tracker": {
        "card": {
            "id": "memory-tracker",
            "name": "Ascend Graph Memory Tracker",
            "name_en": "Ascend Graph Memory Tracker",
            "name_zh": "昇腾计算图显存跟踪器",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "private",
            "summary_en": "Captures Ascend graph-memory allocations and exports Perfetto traces. Hardware validation covers one 910B2 and bounded capture sizes.",
            "summary_zh": "采集昇腾计算图捕获的显存分配并导出 Perfetto 轨迹。实测覆盖单张 910B2 和有限捕获规格。",
            "maintainers": ["CubeLander"],
            "maintainer_profiles": [{"login": "CubeLander", "name": "田景远"}],
            "system_role": "profiling_analysis",
            "execution_planes": ["worker", "device"],
            "delivery_model": "python_distribution",
            "ownership": "original_contributor_maintained",
            "repository_relationship": "organization_native",
            "evidence_level": "hardware_verified",
        },
        "taxonomy": {
            "kind": "tool_mod",
            "capability": "observability_evaluation",
            "phases": ["observability"],
            "lifecycle": "implemented_restricted",
            "evidence": "no_performance_claim",
        },
    },
    "ascend-mapped-kv-offload": {
        "card": {
            "id": "ascend-mapped-kv-offload",
            "name": "Ascend Mapped-host KV Offload",
            "name_en": "Ascend Mapped-host KV Offload",
            "name_zh": "昇腾映射主机内存 KV 卸载",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "public",
            "summary_en": "Mapped-host memory KV offload adaptation for Ascend. The package supports installation and inspection while host integration and device validation are pending.",
            "summary_zh": "昇腾映射主机内存的 KV 卸载适配。目前可安装和检查，等待宿主接入和设备验证。",
            "canonical_repository": "https://github.com/vLLM-HUST/vllm-ascend-mapped-kv-offload-hust",
            "repository_relationship": "organization_native",
            "compatibility": {"status": "inspect_only"},
            "maintainers": ["CubeLander"],
            "maintainer_profiles": [{"login": "CubeLander", "name": "田景远"}],
            "system_role": "kv_tiering",
            "execution_planes": ["worker", "device"],
            "delivery_model": "plugin_bundle",
            "ownership": "original_contributor_maintained",
            "evidence_level": "cpu_smoke",
        },
        "taxonomy": {
            "kind": "runtime_mod",
            "capability": "kv_management",
            "phases": ["state_lifecycle"],
            "lifecycle": "pending",
            "evidence": "activation_pending",
        },
    },
    "scheduler-policy-lab": {
        "card": {
            "id": "scheduler-policy-lab",
            "name": "Scheduling Policy Research",
            "name_en": "Scheduling Policy Research",
            "name_zh": "调度策略研究与测试",
            "artifact_type": "runtime_component",
            "integration_contracts": [],
            "maturity": "experimental",
            "public_surface": True,
            "repository_visibility": "public",
            "summary_en": "Research workspace for scheduler policies, baseline ports and observability. Host-independent policy code has tests. vLLM runtime integration is pending.",
            "summary_zh": "调度策略、基线迁移和观测研究工作区。宿主无关的策略代码已有测试，vLLM 运行接入待完成。",
            "canonical_repository": "https://github.com/vLLM-HUST/vllm-hust-scheduler-policy-lab",
            "repository_relationship": "organization_native",
            "compatibility": {"status": "inspect_only"},
            "maintainers": ["Tkhkrnx"],
            "maintainer_profiles": [{"login": "Tkhkrnx", "name": "彭浩然"}],
            "system_role": "scheduler_policy",
            "execution_planes": ["scheduler"],
            "delivery_model": "source_toolkit",
            "ownership": "original_contributor_maintained",
            "evidence_level": "cpu_smoke",
        },
        "taxonomy": {
            "kind": "tool_mod",
            "capability": "scheduling_routing",
            "phases": ["request_admission"],
            "lifecycle": "research",
            "evidence": "host_integration_pending",
        },
    },
}


PUBLIC_PREVIEW_WORKLOADS = {
    "traceloom": ["profiling_analysis"],
    "latchmoe": ["moe", "memory_kv_pressure"],
    "tricard-clm-lifecycle": ["lifecycle_control", "repeated_prefix_agent"],
    "ascend-adaptive-quantized-kv": ["memory_kv_pressure"],
    "dla": ["latency_slo"],
    "stateaxis-automatic-prefix-cache": ["repeated_prefix_agent"],
    "stateaxis-async-kv-transfer": ["memory_kv_pressure"],
    "stateaxis-pipelined-weight-loading": ["service_cold_start"],
    "cuda-mtp": ["decode_heavy"],
    "memory-tracker": ["profiling_analysis"],
    "ascend-mapped-kv-offload": ["memory_kv_pressure"],
    "scheduler-policy-lab": ["latency_slo"],
}


def publish_mod_previews(path, projected, original):
    if path == "data/ecosystem.json":
        sources = {item["id"]: item for item in original["components"]}
        cards = []
        for mod_id, entry in PUBLIC_MOD_PREVIEWS.items():
            card = copy.deepcopy(entry["card"])
            source = sources.get(mod_id, {})
            for key in ("maintainers", "maintainer_profiles"):
                if key in source:
                    card[key] = copy.deepcopy(source[key])
            if "advisors" in source:
                card["advisors"] = public_advisors(source)
            cards.append(card)
        projected["components"] = [
            item
            for item in projected["components"]
            if item["id"] not in PUBLIC_MOD_PREVIEWS
        ] + cards
    elif path == "data/mod-taxonomy.json":
        projected["components"].update(
            {mod_id: entry["taxonomy"] for mod_id, entry in PUBLIC_MOD_PREVIEWS.items()}
        )
    elif path == "data/plugin-workload-navigation.json":
        projected["plugins"].update(PUBLIC_PREVIEW_WORKLOADS)


def validate_mod_previews(path, data):
    # Validate the exact public copy and role of each scoped entry before the
    # remaining payload is checked against the withdrawal filters.
    if path == "data/ecosystem.json":
        previews = [
            item for item in data["components"] if item["id"] in PUBLIC_MOD_PREVIEWS
        ]
        assert len(previews) == len(PUBLIC_MOD_PREVIEWS), path
        assert {item["id"] for item in previews} == set(PUBLIC_MOD_PREVIEWS), path
        for item in previews:
            card = PUBLIC_MOD_PREVIEWS[item["id"]]["card"]
            people_fields = {"maintainers", "maintainer_profiles", "advisors"}
            assert {k: v for k, v in item.items() if k not in people_fields} == {
                k: v for k, v in card.items() if k not in people_fields
            }, item["id"]
            people = {key: item[key] for key in people_fields if key in item}
            assert not matches(people), item["id"]
            assert all(
                set(person) == {"login", "name"}
                for person in people.get("maintainer_profiles", [])
            ), item["id"]
            assert all(
                set(advisor) == set(PUBLIC_ADVISOR_FIELDS)
                for advisor in people.get("advisors", [])
            ), item["id"]
            data["components"].remove(item)
    elif path == "data/mod-taxonomy.json":
        for mod_id, entry in PUBLIC_MOD_PREVIEWS.items():
            assert data["components"].pop(mod_id) == entry["taxonomy"], mod_id
    elif path == "data/plugin-workload-navigation.json":
        for mod_id, traits in PUBLIC_PREVIEW_WORKLOADS.items():
            assert data["plugins"].pop(mod_id) == traits, mod_id


def matches(value):
    return bool(PROJECT.search(json.dumps(value, ensure_ascii=False)))


def target_record(value):
    if not isinstance(value, dict):
        return False
    # Withhold the whole measured comparison if its evidence belongs to a
    # withdrawn work. Removing just the evidence URL corrupts its provenance.
    if ("metrics" in value or "configuration" in value) and matches(value):
        return True
    if any(matches(v) for k, v in value.items() if k in IDENTITY_KEYS):
        return True
    for key in ("configuration", "runtime", "provenance", "source_provenance"):
        nested = value.get(key)
        if isinstance(nested, dict) and target_record(nested):
            return True
    return False


def project_json(value, removed_ids, record=False, retained_runs=()):
    if isinstance(value, dict):
        run_id = value.get("run_id")
        if isinstance(run_id, str) and run_id in retained_runs:
            return copy.deepcopy(value)
        if record and (
            target_record(value)
            or (isinstance(value.get("id"), str) and value["id"] in removed_ids)
        ):
            if isinstance(value.get("id"), str):
                removed_ids.add(value["id"])
            return DROP
        result = {}
        for key, item in value.items():
            if PROJECT.search(key) or key in removed_ids:
                continue
            child = project_json(
                item,
                removed_ids,
                record=isinstance(item, dict),
                retained_runs=retained_runs,
            )
            if child is not DROP:
                result[key] = child
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            child = project_json(
                item, removed_ids, record=True, retained_runs=retained_runs
            )
            if child is not DROP:
                result.append(child)
        return result
    if isinstance(value, str) and value in removed_ids:
        return DROP
    if isinstance(value, str) and PROJECT.search(value):
        if value.startswith(("https://", "http://", "./", "data/", "docs/")):
            return DROP
        sentences = re.split(r"(?<=[.!?。！？;；])\s*", value)
        retained = " ".join(
            sentence for sentence in sentences if not PROJECT.search(sentence)
        ).strip()
        return retained if retained else DROP
    return value


def prune_frontier(data, removed_ids):
    """Keep declared plot selections consistent with the remaining evidence."""
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != "leaderboard-frontier/v1"
    ):
        return data
    for _ in range(len(data["cohorts"]) + 1):
        previous = len(data["cohorts"])
        point_ids = {p["id"] for p in data["points"]}
        cohorts = []
        for cohort in data["cohorts"]:
            points = [p for p in data["points"] if p["cohort_id"] == cohort["id"]]
            contract = cohort["workload"]["contract"]
            retain = bool(points)
            if "display_series_ids" in contract:
                series = {p.get("load", {}).get("concurrency_series") for p in points}
                contract["display_series_ids"] = [
                    s for s in contract["display_series_ids"] if s in series
                ]
                retain = retain and bool(contract["display_series_ids"])
            if "comparison_point_ids" in contract:
                contract["comparison_point_ids"] = [
                    p for p in contract["comparison_point_ids"] if p in point_ids
                ]
                retain = retain and bool(contract["comparison_point_ids"])
            if contract.get("default_groups") == []:
                contract.pop("default_groups")
            if retain:
                cohorts.append(cohort)
            else:
                removed_ids.add(cohort["id"])
        data["cohorts"] = cohorts
        cohort_ids = {c["id"] for c in cohorts}
        for point in data["points"]:
            if point["cohort_id"] not in cohort_ids:
                removed_ids.add(point["id"])
        data["points"] = [p for p in data["points"] if p["cohort_id"] in cohort_ids]
        if len(cohorts) == previous:
            break
    return data


def clean_html(text):
    # Remove links/images whose destination belongs to a withdrawn project.
    text = re.sub(
        r"<a\b[^>]*>.*?</a>", lambda m: "" if matches(m[0]) else m[0], text, flags=re.S
    )
    text = re.sub(r"<img\b[^>]*>", lambda m: "" if matches(m[0]) else m[0], text)
    # This shared card also describes PegaFlow; preserve that unrelated content.
    text = re.sub(r"LatchMoE is not applicable.*?(?=PegaFlow is)", "", text, flags=re.S)
    text = re.sub(r"LatchMoE 不适用于.*?(?=PegaFlow 是)", "", text, flags=re.S)
    text = re.sub(r"'atlas-latchmoe-role': '[^']*',\s*", "", text)
    # Remove obsolete project-specific metadata from the shared route table.
    return text


def clean_javascript(path, text):
    if path == "assets/plugins-page.js":
        for start, end in [
            ("    traceloom: {", '    "clm-lifecycle": {'),
            ('    "tricard-clm-lifecycle": {', "    betterscale: {"),
            ("    betterscale: {", "    bidkv: {"),
            ("    latchmoe: {", '    "pegaflow-vllm-connectors": {'),
            ('    "ascend-adaptive-quantized-kv": {', '    "kvcompress-ascend": {'),
        ]:
            if start in text:
                a = text.index(start)
                b = text.index(end, a)
                text = text[:a] + text[b:]
    if path == "assets/dataset-matrix-v1.js":
        text = text.replace(
            "(r.solution === 'betterscale' ? '2026-10-15' : r.dataset === 'mmlu-pro' ? '2026-10-12' : r.dataset === 'frontierscience' ? '2026-10-13' : '2026-10-14/15')",
            "(r.dataset === 'mmlu-pro' ? '2026-10-12' : r.dataset === 'frontierscience' ? '2026-10-13' : '2026-10-14/15')",
        )
        text = text.replace(
            "BetterScale 的未决判分分别为 20 / 27 条，保留在总分母中。", ""
        )
        text = text.replace(
            "BetterScale has 20 / 27 unresolved judges, retained in the overall denominators.",
            "",
        )
        text = text.replace(
            " · <a href=\"${data.betterscale_publication.url}\">${t('BetterScale 公开实测报告', 'BetterScale public report')}</a>",
            "",
        )
    if path == "assets/leaderboard-frontier.js":
        text = text.replace(
            "Only Pareto-frontier points are shown, connected separately for BetterScale and Native. BetterScale spans workload-tuned configurations, not a fixed-capacity sweep.",
            "Only currently published Pareto-frontier points are shown; each configuration retains its measurement scope.",
        )
        text = text.replace(
            "BetterScale lines connect best trade-offs across workload-tuned configurations; other lines follow measured concurrency series",
            "Lines follow the published measured concurrency series",
        )
        text = text.replace(
            "BetterScale 连线连接不同 workload 调优配置的最佳权衡点；其他连线按实测并发序列连接",
            "连线按当前公开的实测并发序列连接",
        )
        start = "            ${point.configuration.mods.includes('betterscale')?"
        end = "            <p class=\"frontier-popup-load\">${t('warmup')}"
        if start in text:
            a = text.index(start)
            text = text[:a] + text[text.index(end, a) :]
        text = re.sub(
            r"        // BetterScale checkbox.*?(?=        const mods=)",
            "",
            text,
            flags=re.S,
        )
        text = text.replace(
            "仅展示并分别连接 BetterScale 与 Native 的 Pareto 前沿点；BetterScale 跨 workload 调优配置，不代表固定容量扫描。",
            "展示当前公开方案的 Pareto 前沿点；不同配置不代表固定容量扫描。",
        )
        text = text.replace(
            "Only show and connect BetterScale and Native Pareto frontier points; BetterScale is tuned across workloads, not a fixed-capacity sweep.",
            "Show the Pareto frontier of currently published configurations; different settings are not a fixed-capacity sweep.",
        )
    if path == "assets/site.js":
        start, end = "        'betterscale.html': {", "        'index.html': {"
        if start in text:
            a = text.index(start)
            text = text[:a] + text[text.index(end, a) :]
    if path == "assets/midterm-report.js":
        text = (
            "\n".join(
                line
                for line in text.splitlines()
                if "website-publication/data/betterscale-" not in line
            )
            + "\n"
        )
    return text


def archive_matches(path):
    if path.suffix.lower() not in (".pptx", ".docx", ".xlsx", ".zip"):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if PROJECT.search(member.filename):
                    return True
                if (
                    member.filename.endswith((".xml", ".json", ".md", ".txt"))
                    and member.file_size < 5_000_000
                ):
                    if PROJECT.search(
                        archive.read(member).decode("utf-8", errors="replace")
                    ):
                        return True
    except zipfile.BadZipFile:
        pass
    return False


def prune_removed_artifact_links(value, removed):
    """Optional curve downloads must refer to files in the public artifact."""
    if isinstance(value, dict):
        url = value.get("concurrency_curves_url")
        if isinstance(url, str):
            parsed = urlsplit(url)
            local = unquote(parsed.path).removeprefix("./").lstrip("/")
            if not parsed.scheme and not parsed.netloc and local in removed:
                del value["concurrency_curves_url"]
        for child in value.values():
            prune_removed_artifact_links(child, removed)
    elif isinstance(value, list):
        for child in value:
            prune_removed_artifact_links(child, removed)


def build(source, destination, policy_revision):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Output must be a new directory outside the input checkout")
    destination.mkdir(parents=True)
    removed, changed, documents, removed_ids = [], [], {}, set()
    excluded = {
        ".git",
        ".github",
        ".agents",
        "tests",
        "scripts",
        "hooks",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "screenshots",
        "output",
        "deploy",
    }
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(source)
        if any(part in excluded for part in rel.parts) or any(
            part.startswith(".") for part in rel.parts
        ):
            continue
        name = rel.as_posix()
        if PROJECT.search(name):
            removed.append(name)
            continue
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        if archive_matches(path):
            removed.append(name)
            continue
        if suffix in (".svg", ".txt", ".log", ".csv", ".jsonl"):
            if PROJECT.search(raw.decode("utf-8", errors="replace")):
                removed.append(name)
                continue
        if suffix == ".md" and PROJECT.search(raw.decode("utf-8", errors="replace")):
            text = raw.decode("utf-8")
            first_heading = next(
                (line for line in text.splitlines() if line.startswith("# ")), ""
            )
            if PROJECT.search(first_heading) or re.search(
                r"FRONTIER-(?:DSV4|SMALL-FISH|REPEAT-SELECTION|QWEN35-(?:CONCURRENCY|SWE-PREFIX))",
                name,
            ):
                removed.append(name)
                continue
            raw = "\n".join(
                line for line in text.splitlines() if not PROJECT.search(line)
            ).encode("utf-8")
        if suffix == ".png" and re.search(
            r"sharegpt_story|multi_workload|current.research|当前研究成果|agent_research_online_candidate",
            name,
            re.I,
        ):
            removed.append(name)
            continue
        if suffix == ".json" and name != "data/benchmark-source-manifest.json":
            data = json.loads(raw)
            if name in (PUBLIC_ENGINE_HISTORY_PATH, PUBLIC_FIXED_B0_PATH):
                if hashlib.sha256(raw).hexdigest() != (
                    PUBLIC_ENGINE_HISTORY_SHA256
                    if name == PUBLIC_ENGINE_HISTORY_PATH
                    else PUBLIC_FIXED_B0_SHA256
                ):
                    raise ValueError("Unreviewed engine release history bytes")
                documents[name] = (data, copy.deepcopy(data), raw)
                continue
            projected = project_json(
                data,
                removed_ids,
                retained_runs=PUBLIC_C16_RUN_IDS if name == C16_EVIDENCE_PATH else (),
            )
            prune_frontier(projected, removed_ids)
            documents[name] = (data, projected, raw)
            continue
        if suffix == ".html":
            raw = clean_html(raw.decode("utf-8")).encode("utf-8")
        elif suffix == ".js":
            raw = clean_javascript(name, raw.decode("utf-8")).encode("utf-8")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if raw != path.read_bytes():
            changed.append(name)
    # Repeat with the full removed-id set so cross-file references cannot expose orphan records.
    for name, (original, _, raw) in documents.items():
        projected = (
            copy.deepcopy(original)
            if name in (PUBLIC_ENGINE_HISTORY_PATH, PUBLIC_FIXED_B0_PATH)
            else project_json(
                original,
                removed_ids,
                retained_runs=PUBLIC_C16_RUN_IDS if name == C16_EVIDENCE_PATH else (),
            )
        )
        prune_frontier(projected, removed_ids)
        prune_removed_artifact_links(projected, set(removed))
        if projected is DROP:
            removed.append(name)
            continue
        add_betterscale_card(name, projected, original)
        publish_mod_previews(name, projected, original)
        out = (
            raw
            if projected == original
            else (json.dumps(projected, ensure_ascii=False, indent=2) + "\n").encode()
        )
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out)
        if out != raw:
            changed.append(name)
    manifest_path = destination / "data/benchmark-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = []
    for entry in manifest["files"]:
        target = destination / entry["website_path"]
        if not target.is_file():
            continue
        raw = target.read_bytes()
        item = copy.deepcopy(entry)
        item["unprojected_sha256"] = item["sha256"]
        item["sha256"] = hashlib.sha256(raw).hexdigest()
        item["bytes"] = len(raw)
        entries.append(item)
    manifest["files"] = entries
    manifest["projection"] = {
        "policy": "selected-research-withdrawal-20261030",
        "public_run_ids": sorted(PUBLIC_C16_RUN_IDS),
        "revision": policy_revision,
        "scope": "Public website projection; original research records are not rewritten.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Remove rendered links to files omitted from this public artifact.
    omitted = set(removed)
    for path in destination.glob("*.html"):
        text = path.read_text()

        def link(match):
            href = re.search(r'href=["\']([^"\']+)', match[0])
            if href and unquote(urlsplit(href[1]).path).removeprefix("./") in omitted:
                return ""
            return match[0]

        path.write_text(re.sub(r"<a\b[^>]*>.*?</a>", link, text, flags=re.S))
    audit = {
        "removed_files": sorted(set(removed)),
        "projected_files": sorted(set(changed)),
        "removed_record_ids": sorted(removed_ids),
        "policy_revision": policy_revision,
    }
    return audit


def validate(destination):
    manifest = json.loads(
        (destination / "data/benchmark-source-manifest.json").read_text()
    )
    for entry in manifest["files"]:
        raw = (destination / entry["website_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["website_path"]
        assert len(raw) == entry["bytes"], entry["website_path"]
    for path in (destination / "data").rglob("*.json"):
        if path.name == "benchmark-source-manifest.json":
            continue
        data = json.loads(path.read_text())
        name = path.relative_to(destination).as_posix()
        if name in (PUBLIC_ENGINE_HISTORY_PATH, PUBLIC_FIXED_B0_PATH):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                PUBLIC_ENGINE_HISTORY_SHA256
                if name == PUBLIC_ENGINE_HISTORY_PATH
                else PUBLIC_FIXED_B0_SHA256
            ), path
            continue
        validate_betterscale_card(name, data)
        validate_mod_previews(name, data)
        if name == C16_EVIDENCE_PATH:
            for run in data.get("runs", []):
                if run.get("run_id") in PUBLIC_C16_RUN_IDS:
                    run.clear()
        assert not matches(data), path
    for name in [
        "betterscale.html",
        "traceloom.html",
        "gennova.html",
        "assets/betterscale.js",
        "assets/gennova.js",
    ]:
        assert not (destination / name).exists(), name
    for name in [
        "index.html",
        "plugins.html",
        "dataset-introductions.html",
        "leaderboard-runs.html",
        "assets/site.js",
    ]:
        assert (destination / name).is_file(), name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--website", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    audit = build(args.website, args.output, args.policy_revision)
    validate(args.output)
    # Exercise the same schema gate used by the rendered chart, including
    # references between configurations, cohorts and comparison series.
    subprocess.run(
        [
            "node",
            "-e",
            "const fs=require('fs');const M=require('./assets/leaderboard-frontier-model.js');M.validate(JSON.parse(fs.readFileSync('data/leaderboard_frontier.json')));",
        ],
        cwd=args.output,
        check=True,
    )
    for javascript in (args.output / "assets").glob("*.js"):
        subprocess.run(["node", "--check", str(javascript.resolve())], check=True)
    if args.audit:
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "removed_files": len(audit["removed_files"]),
                "projected_files": len(audit["projected_files"]),
                "removed_record_ids": len(audit["removed_record_ids"]),
                "validated": True,
            }
        )
    )
