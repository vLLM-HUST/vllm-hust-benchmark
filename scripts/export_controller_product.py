"""Import completed real-engine pairs; export metrics without task or answer text.

Usage: python scripts/export_controller_product.py --multi-turn <sealed-run> --profiles <sealed-run>
This is a CPU-only import. It never starts model requests or modifies source evidence.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def export(args):
    repo = Path(__file__).resolve().parents[1]
    report = repo / "reports/controller-r8-20261118"
    report.mkdir(exist_ok=True)
    publication = repo / "website-publication/data"
    sources = {}

    def read(root, name):
        raw = (root / name).read_bytes()
        sources[f"{root.name}/{name}"] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    result = read(args.multi_turn, "paired-result.json")
    diagnostics = read(args.multi_turn, "paired-diagnostics.json")
    profiles = read(args.profiles, "paired-profiles.json")
    assert result["completed_pair"] and profiles["all_four_pairs_complete"]
    assert (
        result["direct"]["input_tokens"]
        == result["request_throttling_controller"]["input_tokens"]
    )
    assert (
        result["direct"]["output_tokens"]
        == result["request_throttling_controller"]["output_tokens"]
    )
    assert all(v["complete_matched_pair"] for v in profiles["profiles"].values())
    labels = {
        "mmlu": "MMLU-Pro",
        "hle": "Humanity’s Last Exam",
        "frontier": "FrontierScience",
        "swe": "SWE-bench Pro public trace",
        "terminal": "Terminal-Bench public trace",
    }
    rows = []
    for key in ("mmlu", "hle", "frontier", "swe", "terminal"):
        if key == "hle":
            rows.append(
                {
                    "id": key,
                    "label": labels[key],
                    "status": "deferred",
                    "reason_zh": "按用户要求暂缓",
                    "reason_en": "Deferred by request",
                }
            )
            continue
        source = profiles["profiles"][key]
        rows.append(
            dict(
                id=key,
                label=labels[key],
                status="complete",
                direct=source["arms"]["direct"],
                controller=source["arms"]["controller"],
                **{
                    k: source[k]
                    for k in (
                        "same_client_request_bodies",
                        "same_per_case_engine_prompt_tokens",
                        "time_reduction_percent",
                        "output_token_change_percent",
                        "output_throughput_change_percent",
                    )
                },
            )
        )
    quality = []
    a, b = (
        profiles["profiles"]["mmlu"]["arms"][arm] for arm in ("direct", "controller")
    )
    quality.append(
        {
            "label_zh": "MMLU-Pro正确数 / 200",
            "label_en": "MMLU-Pro correct / 200",
            "direct": f"{a['quality']['correct']} / 200",
            "controller": f"{b['quality']['correct']} / 200",
        }
    )
    for track, label in (("olympiad", "Olympiad"), ("research", "Research")):
        a, b = (
            profiles["profiles"]["frontier"]["arms"][arm]["quality"]["tracks"][track]
            for arm in ("direct", "controller")
        )
        for field, zh, en in [
            ("successes", "达标数 / 总数", "successes / total"),
            ("judge_invalid", "评分未解析", "unparsed grades"),
            ("judge_transport_failures", "评分传输失败", "judge transport failures"),
        ]:
            quality.append(
                {
                    "label_zh": f"FrontierScience {label} {zh}",
                    "label_en": f"FrontierScience {label} {en}",
                    "direct": f"{a[field]} / {a['total']}"
                    if field == "successes"
                    else a[field],
                    "controller": f"{b[field]} / {b['total']}"
                    if field == "successes"
                    else b[field],
                }
            )
    overrides = [
        {
            "zh": "引擎基于第八版；执行上限沿官方36。两组共同将任务状态驻留数54改为80，80不是vLLM默认值。",
            "en": "Based on engine r8 with its execution limit of 36. Both arms change resident-state capacity from 54 to 80; 80 is not a vLLM default.",
        },
        {
            "zh": "两组共同覆盖逐请求强制重算，保留稳定program身份和跨轮状态复用；未宣称完全未改动的官方全部优化。",
            "en": "Both arms override forced per-request recomputation to retain stable program identity and cross-turn state reuse; this is not an unmodified official stack.",
        },
        {
            "zh": "Controller由RIDE local独立管理，使用原版ThunderAgent；业务请求、转发与恢复不设timeout。",
            "en": "RIDE local manages a separate controller using original ThunderAgent. Business requests, forwarding and restoration have no timeout.",
        },
    ]
    entry = {
        "id": "r8",
        "label": "2026-11-18 · Engine r8 + Request Throttling Controller",
        "engine_release": "full-answer-b1-20261117-r8",
        "max_num_seqs": 36,
        "state_resident_seats": 80,
        "repetitions_per_arm": 1,
        "report_path": "reports/controller-r8-20261118/README.md",
        "result": result,
        "profiles": rows,
        "quality_rows": quality,
        "overrides": overrides,
        "scope_zh": "56份SWE-smith长度/工具等待模板各生成两个独立token-ID实例，共112个三轮程序、336次请求；新输入为合成token-ID，续轮继承模型真实输出。不是原始题目文本或完整Agent任务解决率。",
        "scope_en": "56 SWE-smith length/tool-wait templates, each with two independent token-ID instances: 112 three-turn programs and 336 requests. New inputs are synthetic token IDs; continuations inherit real model outputs. This does not measure original task text or full Agent solve rate.",
        "limitations_zh": "仅一次固定顺序配对；未证明最优或稳定收益。104个程序变快、8个变慢。缓存命中按输入token加权；引擎prefill计算减少，不代表所有状态驱逐问题已根治。",
        "limitations_en": "One pair in fixed order; optimality and repeatability are unproven. 104 programs became faster and 8 slower. Cache hits are input-token weighted. Reduced prefill does not establish that all state-eviction issues are fixed.",
        "quality_note_zh": "MMLU-Pro使用原提取器和200题分母；FrontierScience使用官网同款Qwen替代judge，未解析保留原分母，不等同官方judge评分。公开轨迹没有执行完整工具任务。HTTP时延不含等待客户端并发槽位；整批耗时包含该排队、release及收尾，评分另计。",
        "quality_note_en": "MMLU-Pro uses the original extractor and 200-case denominator. FrontierScience uses the website’s Qwen substitute judge, retaining invalid grades in the planned denominator; these are not official-judge scores. Public traces do not execute complete tool tasks. HTTP latency excludes waiting for client worker slots; batch time includes this queue, release and teardown. Grading is timed separately.",
        "changelog": [
            {
                "date": "2026-11-18",
                "zh": "保留第六版30.06%历史结果；第八版另立运行，不重标旧成绩。",
                "en": "Retain the historical r6 result of 30.06%. R8 is a separate run, not a relabelled measurement.",
            },
            {
                "date": "2026-11-18",
                "zh": "第八版共同应用驻留80、跨轮复用、无业务timeout；模型与控制面分别冷启动和管理。",
                "en": "Both r8 arms use resident capacity 80, cross-turn reuse and no business timeout. Model and controller have independent lifecycles.",
            },
            {
                "date": "2026-11-18",
                "zh": "四项官网固定负载完成匹配对照和原评分，保留负收益、输出量变化与未解析评分；HLE暂缓。",
                "en": "Four fixed website profiles completed matched comparisons and original scoring. Retain regressions, output-count differences and unparsed grades; HLE is deferred.",
            },
        ],
    }
    old = json.loads(
        (publication / "controller-experiment-progress-v1.json").read_text()
    )
    legacy = copy.deepcopy(entry)
    legacy.update(
        id="r6",
        label="2026-11-18 · Engine r6 historical pair",
        engine_release=old["engine_release"],
        max_num_seqs=24,
        report_path=old["report_path"],
        result=old["result"],
        quality_rows=[],
        quality_note_zh="本历史配对未测任务解决率或这四项固定数据集。",
        quality_note_en="This historical pair did not evaluate task solve rate or these four fixed profiles.",
        limitations_zh="第六版独立历史配对，不与第八版构成单参数消融；未独立重复。",
        limitations_en="A separate historical r6 pair, not a single-variable ablation against r8; no independent repetitions.",
        changelog=old["changelog"],
    )
    legacy["profiles"] = [
        {
            "id": key,
            "label": labels[key],
            "status": "not_run",
            "reason_zh": "本历史配对未执行",
            "reason_en": "Not run in this historical pair",
        }
        for key in labels
    ]
    legacy["overrides"][0] = {
        "zh": "第六版历史配对：执行上限24、任务状态驻留80，不是vLLM默认值。",
        "en": "Historical r6 pair: execution limit 24 and resident-state capacity 80; these are not vLLM defaults.",
    }
    product = {
        "schema": "controller-product-results/v1",
        "updated": "2026-11-18",
        "default_release": "r8",
        "releases": [entry, legacy],
    }

    def write(path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    write(publication / "controller-product-results-v1.json", product)
    for name, value in [
        ("multi-turn-paired-result.json", result),
        ("multi-turn-diagnostics.json", diagnostics),
        ("fixed-profile-pairs.json", profiles),
    ]:
        write(report / name, value)
    for name in (
        "matched-engine.json",
        "website-profiles/input-manifest.json",
        "controller-archive-receipt.json",
        "cleanup.json",
    ):
        write(report / Path(name).name, read(args.profiles, name))
    for name in (
        "execution-and-analysis-source-final-sha256.json",
        "workload-instance-scope.json",
    ):
        write(report / name, read(args.multi_turn, name))
    # Keep public per-request evidence, without prompts, output text or device URLs.
    for arm in ("direct", "controller"):
        rows = []
        for dataset in ("terminal", "swe", "mmlu", "frontier"):
            for path in sorted(
                (
                    args.profiles
                    / "client-raw/website-profiles"
                    / arm
                    / dataset
                    / "requests"
                ).glob("*.json")
            ):
                record = json.loads(path.read_text())
                safe = {
                    k: record.get(k)
                    for k in (
                        "case_id",
                        "status",
                        "http_status",
                        "elapsed_s",
                        "usage",
                        "finish_reason",
                        "stream_sha256",
                    )
                }
                safe.update(
                    dataset=dataset,
                    original_record_sha256=hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    request_sha256=hashlib.sha256(
                        json.dumps(
                            record["request"], sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                )
                rows.append(safe)
        assert len(rows) == 472
        write(report / (arm + "-requests.json"), rows)
    write(report / "import-source-sha256.json", sources)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-turn", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    export(parser.parse_args())
