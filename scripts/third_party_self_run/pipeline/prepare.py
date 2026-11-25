"""Resolve execution inputs, preserve unavailable requirements, freeze a diagnostic."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
KIT = Path(os.environ.get("VLLM_HUST_KIT", ROOT.parent)).resolve()


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    if os.environ.get("VLLM_HUST_ALLOW_LEGACY_DIAGNOSTIC") != "1":
        raise RuntimeError(
            "Legacy ten-request diagnostic is disabled; use one_click.py for fixed three-dataset execution"
        )
    if (ROOT / "execution-plan.json").exists():
        raise RuntimeError("Already frozen; use a new campaign directory")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    for name in [
        "run_native.py",
        "export_receipt.py",
        "run_vllm_hust_tests.py",
        "inference_test_report.py",
        "requirements-report.txt",
    ]:
        shutil.copyfile(KIT / name, ROOT / name)
    inventory = {}
    for cmd, name in [
        (["npu-smi", "info"], "npu-before.txt"),
        (["npu-smi", "info", "-t", "board", "-i", "5"], "board-5.txt"),
        (["npu-smi", "info", "-t", "board", "-i", "6"], "board-6.txt"),
    ]:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        (ROOT / name).write_text(result.stdout + result.stderr)
        inventory[name] = {"returncode": result.returncode}
    inventory.update(
        host=socket.gethostname(),
        observed_at=now,
        versions={
            m: importlib.metadata.version(m)
            for m in ["vllm", "vllm-ascend", "torch", "torch-npu"]
        },
        actual_site="当前容器宿主机；未取得浦江国家实验室场地确认",
        ssh_peer={"host_alias": "hust", "result": "Connection refused on port 32038"},
        available_physical_devices=[3, 4, 5, 6],
        selected_diagnostic_devices=[5, 6],
    )
    for name, repo in [
        ("core", "/vllm-workspace/vllm"),
        ("plugin", "/vllm-workspace/vllm-ascend"),
    ]:
        inventory[name + "_commit"] = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
        inventory[name + "_status"] = subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True
        )
    save("environment.json", inventory)
    assets = {
        "MMLU-Pro": "/data/shared_upload/lumi-dataset-validation-20261015/assets/mmlu/assets/test-00000-of-00001.parquet",
        "HLE-Verified": "/data/shared_upload/lumi-dataset-validation-20261015/assets/hle/revision/test.jsonl",
        "FrontierScience-olympiad": "/data/shared_upload/lumi-dataset-validation-20261015/assets/frontier/olympiad/test.jsonl",
        "FrontierScience-research": "/data/shared_upload/lumi-dataset-validation-20261015/assets/frontier/research/test.jsonl",
        "SWE-bench Pro V1": "/data/shared_upload/hust-fjj-256GB--12364c7a-42ec-4b16-97c1-16089dd6d478/swe-pro-v1-200-c8-20261103-run01/task-manifest-200.jsonl",
        "Terminal-Bench 2.1 records": "/data/shared_upload/terminalbench-migration-9450-20261105/final-containers.jsonl",
    }
    save(
        "local-assets.json",
        {
            k: {
                "path": str(p),
                "exists": p.is_file(),
                "sha256": digest(p) if p.is_file() else None,
                "role": "本地原件/历史任务资料；版本、赛道和本次既定任务范围仍须由实际客户端核对",
            }
            for k, value in assets.items()
            for p in [Path(value)]
        },
    )
    model = Path("/models/Qwen3.5-35B-A3B")
    weights = []
    for p in sorted(model.glob("*")):
        if p.is_file() and (p.suffix in [".json", ".safetensors"]):
            print("Hashing", p.name, flush=True)
            weights.append(
                {"file": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
            )
    save(
        "model-manifest.json",
        {
            "path": str(model),
            "files": weights,
            "identity_scope": "本次本地逐文件SHA256；未宣称已核对历史HF/MS revision",
        },
    )
    source = Path("/root/test7-gdn-cache-20261018/inputs-private-r1/questions.json")
    questions = json.loads(source.read_text())
    request_list = []
    for q in questions:
        request_list.append(
            {
                "id": q["case_id"],
                "group": "mmlu-diagnostic",
                "prompt_token_ids": q["prompt_token_ids"],
                "input_tokens": len(q["prompt_token_ids"]),
                "max_tokens": 1024,
                "ignore_eos": False,
                "expected_answer": q["answer"],
            }
        )
    for length in [4096, 16384]:
        tokens = (
            questions[0]["prompt_token_ids"]
            * (length // len(questions[0]["prompt_token_ids"]) + 1)
        )[:length]
        for i in range(4):
            request_list.append(
                {
                    "id": f"length-{length}-{i}",
                    "group": f"context-{length}",
                    "prompt_token_ids": tokens,
                    "input_tokens": len(tokens),
                    "max_tokens": 64,
                    "ignore_eos": True,
                }
            )
    save(
        "diagnostic-workload.json",
        {
            "source": str(source),
            "source_sha256": digest(source),
            "scope": "2个历史物化MMLU-Pro题目，非五数据集正式任务清单；长输入为明确标识的重复题文长度探测，非准确率样本",
            "selection_rule": "源文件全部2题保持顺序；随后4K与16K各4次，顺序预冻结，不择优",
            "warmup_tokens": [1, 2, 3, 4],
            "requests": request_list,
        },
    )
    cards = [f"{socket.gethostname()}/physical-{d}" for d in [5, 6]]
    save(
        "diagnostic-config.json",
        {
            "schema": "vllm-hust-local-diagnostic/v1",
            "cards": cards,
            "runtime_identity": inventory,
            "workload_file": str(ROOT / "diagnostic-workload.json"),
            "source_weights": digest(ROOT / "model-manifest.json"),
            "kwargs": {
                "model": str(model),
                "tensor_parallel_size": 2,
                "dtype": "bfloat16",
                "max_model_len": 32768,
                "max_num_seqs": 2,
                "max_num_batched_tokens": 2048,
                "kv_cache_memory_bytes": 2147483648,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "enforce_eager": False,
                "async_scheduling": False,
                "limit_mm_per_prompt": {"image": 0, "video": 0},
                "additional_config": {"enable_cpu_binding": False},
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 2],
                },
                "seed": 0,
                "disable_log_stats": False,
            },
        },
    )
    profile = os.environ.get("VLLM_HUST_PROFILE", "stock")
    config = json.loads((ROOT / "diagnostic-config.json").read_text())
    executable = "python3"
    if profile == "baseline":
        executable = str(KIT / "runtime-b0/bin/python")
        config["arm"] = "B0"
        config["role"] = (
            "vLLM/vLLM-Ascend 0.18.0 real paired diagnostic; formal quality/load coverage is assessed separately"
        )
        config["kwargs"]["compilation_config"]["cudagraph_mode"] = "FULL_AND_PIECEWISE"
        config["compatibility_role"] = (
            "0.18 with recorded CANN header and Qwen3.5 text-graph compatibility patches; not unmodified upstream"
        )
        for name in [
            "baseline-python-requirements.lock",
            "baseline-cann91-patch.json",
            "baseline-qwen35-patch.json",
        ]:
            shutil.copyfile(KIT / name, ROOT / name)
    elif profile == "hust":
        # Version admission happens before this preparation and before any NPU use.
        import sys

        sys.path.insert(0, str(KIT))
        from version_gate import check

        identity = check(profiles=["B1"])
        if not identity["passed"]:
            raise ValueError(
                "Unqualified B1 version/source; refusing the old August runtime"
            )
        deployment = json.loads((KIT / "runtime-profile.json").read_text())["B1"]
        executable = deployment["python"]
        config["arm"] = "B1"
        config["role"] = (
            "Test-version-qualified school 0.28/0.25 runtime; diagnostic workload only, not full formal acceptance"
        )
        config["extra_pythonpath"] = (
            deployment.get("pythonpath", "").split(":")
            if deployment.get("pythonpath")
            else []
        )
        config["version_receipt"] = identity
        if identity["arms"]["B1"]["identity_mode"] == "reconstructed":
            config["role"] = (
                "Reconstructed 0.28/0.25 test-family native diagnostic; not the historical optimized B1 or a best-configuration claim"
            )
        save("hust-source-receipt.json", identity)
    config["profile"] = profile
    if profile == "hust":
        config["general_plugins"] = [
            "ascend",
            "ascend_model",
            "ascend_model_loader",
            "ascend_kv_connector",
        ]
    config["python_executable"] = executable
    save("diagnostic-config.json", config)
    save(
        "diagnostic-contract.json",
        {
            "role": "supplementary_real_inference_only",
            "formal_acceptance": False,
            "requests_planned": len(request_list),
            "repeat_rule": "all recorded, no best-run selection",
            "cache_policy": "新服务、新编译缓存；APC关闭；单次无任务题文预热；允许只读权重页缓存",
            "measurement": "逐请求generate调用墙钟，原生engine metrics另存；无HTTP，非同轮F/T算力测量",
            "quality": "原输出/题目答案保留；1024预算下的截断和无法解析单列，不证明完整回答质量",
            "failure": "首次模型或请求失败即停止，保留未执行分母；不切eager，不换模型/版本",
        },
    )
    save(
        "native-workflow.json",
        {
            "schema": "vllm-hust-native-workflow/v1",
            "steps": [
                {
                    "name": "real_tp2_graph_inference",
                    "argv": [executable, str(ROOT / "native_probe.py")],
                    "timeout_s": 1500,
                },
                {
                    "name": "export_receipt",
                    "argv": [
                        "python3",
                        "export_receipt.py",
                        "--data",
                        "{output}/normalized.json",
                    ],
                    "timeout_s": 15,
                },
            ],
            "cleanup": [],
        },
    )
    save(
        "authorization.json",
        {
            "time": now,
            "source": "本聊天用户明确要求补齐清单并执行真实推理测试",
            "scope": "本地执行及出图",
            "formal_test_owner_approval": False,
            "note": "用户执行授权不能伪装成主责单位的场地、费用、任务与可比性确认",
        },
    )
    reasons = {
        "compute": "缺本次同轮有效F、双卡同步T、独立计算/访存/通信标定、整链参考适用性证据。历史59.7573%不回填本次；监控利用率不替代有效算力。",
        "throughput": "本轮是冻结的轻负载串行诊断；尚未完成正式高并发多轮/工具场景、独立质量及SLO验证。诊断吞吐比不能直接签发本目标达标。",
        "latency": "本轮记录原生TTFT/TPOT；串行闭环请求不是同一冻结开放到达序列压测，正式最小样本和混合波次未确认，不能直接签发达标。",
        "stability": "本地MMLU-Pro数据已定位；本次先冻结2个物化题目诊断，未接通完整任务清单及原生评分流程，不能宣称全范围稳定性达标。",
        "context": "缺主责单位确认的4K/16K保持率、尾时延比、连续窗口和样本门槛；仅做各4次长度探测，无正式连续窗口或五数据集覆盖。",
        "cost": "缺本机每卡月综合费用与费用范围依据、冻结利用率改善幅度和统计规则；轻负载诊断不替代高负载验证。历史类似云机报价不能冒充本机账单。",
    }
    save("blockers.json", reasons)
    plan = json.loads(
        (KIT / "execution-plan.example.json").read_text(encoding="utf-8-sig")
    )
    plan.update(
        version="20261111-local-v3",
        approval_record="authorization.json",
        site=inventory["actual_site"],
        readiness="blocked_for_formal_acceptance",
        prepared_at=now,
    )
    for arm in ["B0", "B1"]:
        save(
            f"deployment-{arm}.json",
            {
                "engine": "vLLM" if arm == "B0" else "vLLM-HUST",
                "plugin": "vLLM-Ascend" if arm == "B0" else "vllm-ascend-hust",
                "engine_version": "0.18.0" if arm == "B0" else None,
                "plugin_version": "0.18.0" if arm == "B0" else None,
                "version_role": "required, not installed/effective",
                "source_weights": digest(ROOT / "model-manifest.json"),
                "runtime_identity": None,
                "precision": "BF16 (proposed, matching local weights)",
                "parallelism": {"tensor_parallel_size": 2, "physical_cards": cards},
                "availability": "not delivered; local actual 0.23 recorded separately",
            },
        )
    for case in plan["cases"]:
        kind = case["kind"]
        case["sequence"] = [[*case["arms"]]]
        case["readiness"] = "blocked"
        case["blocker"] = reasons[kind]
        if kind == "stability" and case["dataset"] != "MMLU-Pro":
            case["blocker"] = (
                f"本地已有{case['dataset']}相关数据/历史资料（见local-assets.json）；尚未接通该数据集本次冻结任务与原生客户端/评分环境，缺冻结B1部署。"
            )
        case["contract"] = f"contract-{case['id']}.json"
        save(
            case["contract"],
            {
                "workload": {
                    "dataset": case.get("dataset"),
                    "status": "not_delivered",
                    "details": case["blocker"],
                },
                "request_policy": "待交付原生请求语义、到达序列和采样输出约束；不以诊断替代",
                "cache_policy": "拟每轮独立空服务/APC状态，权重只读可复用；需正式确认",
                "measurement": "遵循原scope第3.4节；缺项见blocker",
                "quality_slo": {"status": "not_delivered"},
                "failure_policy": "停止正式执行并保留全部失败/缺失；不降级换口径",
                "repeat_rule": "all_pairs",
                "repeat_count_status": "one attempted readiness pass, formal repetitions not confirmed",
            },
        )
        for arm, spec in case["arms"].items():
            spec.update(
                config=f"deployment-{'B1' if arm in ['4K', '16K'] else arm}.json",
                argv=["python3", "run_native.py", "formal-workflow-unavailable.json"],
                timeout_s=1530,
            )
        if kind == "context":
            case["context_engine"] = "B1"
        if kind in ["throughput", "latency", "context"]:
            case["rules"]["throughput_basis"] = "completed_in_window"
    # Unknown business thresholds, task lists, fees and runtime identities stay null.
    # Filling them with made-up values would change the requested acceptance criteria.
    plan["files"] = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in ROOT.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name
        not in ["execution-plan.json", "execution-plan.sha256", "bundle-checksums.json"]
    }
    save("execution-plan.json", plan)
    (ROOT / "execution-plan.sha256").write_text(
        digest(ROOT / "execution-plan.json") + "\n"
    )
    print("PREPARED", ROOT / "execution-plan.json", flush=True)


if __name__ == "__main__":
    main()
