"""Separate V1.5 stability evidence from answer quality, using sealed request rows."""

import hashlib
import json
import math
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def request_metrics(rows):
    ttft = [
        1000 * (r["first_token"]["monotonic"] - r["start"]["monotonic"]) for r in rows
    ]
    tpot = [
        1000
        * (r["end"]["monotonic"] - r["first_token"]["monotonic"])
        / (r["usage"]["completion_tokens"] - 1)
        for r in rows
        if r["usage"]["completion_tokens"] > 1
    ]
    return {
        "requests": len(rows),
        "tpot_requests": len(tpot),
        **{
            f"{name}_p{p}_ms": quantile(values, p / 100)
            for name, values in (("ttft", ttft), ("tpot", tpot))
            for p in (95, 99)
        },
    }


def context_coverage(rows):
    return {
        "input_at_least16384": sum(r["usage"]["prompt_tokens"] >= 16384 for r in rows),
        "input_plus_output_at_least16384": sum(
            r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] >= 16384
            for r in rows
        ),
    }


def read_rows(root, native=False):
    folder = root / ("service" if native else "service/evaluation/requests")
    return [
        r
        for p in sorted(folder.glob("*.json"))
        if isinstance(r := json.loads(p.read_bytes()), dict)
        and all(k in r for k in ("start", "end", "usage", "first_token"))
    ]


def log_evidence(root):
    receipt = json.loads((root / "receipt.json").read_bytes())
    cleanup = json.loads((root / "cleanup.json").read_bytes())
    lines = (root / "service.log").read_text(errors="replace").splitlines()
    shutdowns = [
        i + 1
        for i, line in enumerate(lines)
        if (
            "[shutdown]" in line
            and (
                "trigger received signal=SIGTERM" in line
                or "shutdown triggered" in line
            )
        )
        or "Shutdown initiated" in line
    ]
    errors = [i + 1 for i, line in enumerate(lines) if re.search(r"\) ERROR ", line)]
    shutdown = min(shutdowns) if shutdowns else None
    return {
        "service_log_sha256": digest(root / "service.log"),
        "receipt_sha256": digest(root / "receipt.json"),
        "cleanup_sha256": digest(root / "cleanup.json"),
        "completed": receipt["status"] == "completed",
        "quiescent": bool(receipt.get("quiescent") and cleanup.get("quiescent")),
        "cleanup_signals": sorted({a["signal"] for a in cleanup.get("actions", [])}),
        "shutdown_marker_line": shutdown,
        "error_lines_before_shutdown": [
            n for n in errors if shutdown is None or n < shutdown
        ],
        "error_lines_after_shutdown": [
            n for n in errors if shutdown is not None and n >= shutdown
        ],
        "scope": "vLLM severity ERROR lines and explicit shutdown markers; post-shutdown errors remain recorded, not counted as serving failures. INFO/WARNING text about possible OOM is not an observed OOM.",
    }


def continuous_arm(root):
    rows = read_rows(root, native=True)
    receipt = json.loads((root / "receipt.json").read_bytes())
    phases = json.loads((root / "service/continuous.json").read_bytes())
    if not (
        receipt["status"] == "completed"
        and receipt["quiescent"]
        and phases["status"] == "completed"
        and all(r.get("success") for r in rows)
    ):
        raise ValueError("Continuous evidence must be complete; do not filter failures")
    groups = {}
    for phase in phases["phases"]:
        cohort = [r for r in rows if r["phase"] == phase["name"]]
        if len(cohort) != phase["requests"] or phase["failed"]:
            raise ValueError("Continuous phase denominator changed")
        if not phase["name"].endswith("formal"):
            continue
        start = phase["start"]["monotonic"]
        end = phase["end"]["monotonic"]
        if phase["seconds"] != 300 or any(
            not (start <= r["start"]["monotonic"] < start + 300) for r in cohort
        ):
            raise ValueError("Expected original 300 s submission phase")
        groups[phase["name"]] = {
            **request_metrics(cohort),
            **context_coverage(cohort),
            "duration_including_drain_seconds": end - start,
            "output_tokens_per_second": sum(
                r["usage"]["completion_tokens"] for r in cohort
            )
            / (end - start),
            "submission_cohorts": [
                {
                    "start_seconds": offset,
                    "end_seconds": offset + 60,
                    **request_metrics(
                        [
                            r
                            for r in cohort
                            if offset <= r["start"]["monotonic"] - start < offset + 60
                        ]
                    ),
                }
                for offset in range(0, 300, 60)
            ],
        }
    if set(groups) != {"4096-formal", "16384-formal"}:
        raise ValueError("Both original contexts are required")
    short, long = groups["4096-formal"], groups["16384-formal"]
    keys = ["output_tokens_per_second"] + [
        f"{name}_p{p}_ms" for name in ("ttft", "tpot") for p in (95, 99)
    ]
    return {
        "source_seal_sha256": digest(root / "checksums.json"),
        "continuous_phase_sha256": digest(root / "service/continuous.json"),
        "all_recorded_requests": len(rows),
        "failed_requests": 0,
        "groups": groups,
        "ratio_16k_to_4k": {k: long[k] / short[k] for k in keys},
        "scope": "Original synthetic closed-loop C8; not a five-dataset task window",
    }


def assess(stage_roots, datasets):
    """Caller must verify campaign and child seals before invoking this analysis."""
    profiles = {}
    for dataset in datasets:
        name = dataset["id"]
        profiles[name] = {}
        for role in ("B0", "B1"):
            root = (
                stage_roots["mmlu-pair"] / role
                if name == "mmlu"
                else stage_roots[
                    name
                    + ("-trace-" if name in ("swe", "terminal") else "-solver-")
                    + role
                ]
            )
            rows = read_rows(root, native=name == "mmlu")
            arm = dataset["arms"][role]
            if len(rows) != arm["planned"] or arm["failed"] or arm["unreturned"]:
                raise ValueError("Do not discard unsuccessful or unreturned requests")
            long = [
                r
                for r in rows
                if r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
                >= 16384
            ]
            lengths = [r for r in rows if r["finish_reason"] == "length"]
            budget = 8192 if name == "mmlu" else None
            mismatch = sum(
                r["usage"]["completion_tokens"]
                != (budget if budget is not None else r["request"]["max_tokens"])
                for r in lengths
            )
            profiles[name][role] = {
                "recorded": len(rows),
                "failed": arm["failed"],
                "unreturned": arm["unreturned"],
                **context_coverage(rows),
                "length_stops": len(lengths),
                "length_stop_budget_mismatches": mismatch,
                "terminal_context_16k_cohort": request_metrics(long),
                "source_seal_sha256": digest(root / "checksums.json"),
            }
    return {
        "schema": "v15-stability-evidence-audit/v1",
        "criterion": "3.4.3",
        "execution_complete": True,
        "profiles": profiles,
        "stage_logs": {
            name: log_evidence(root)
            for name, root in {
                **{k: v for k, v in stage_roots.items() if k != "mmlu-pair"},
                **{
                    "mmlu-" + role: stage_roots["mmlu-pair"] / role
                    for role in ("B0", "B1")
                },
            }.items()
        },
        "continuous": {
            role: continuous_arm(stage_roots["continuous-" + role])
            for role in ("B0", "B1")
        },
        "quality_is_a_separate_assessment": True,
        "total_context_includes_output": True,
        "each_dataset_input_16k_is_not_required": True,
        "cohort_method": "60 s bins by request submission; include complete request latency, including drain. All five bins retained. Percentiles interpolate at (n-1)q. These are diagnostic bins, not new acceptance thresholds.",
        "limitations": [
            "Task counts use terminal input+output length. They do not establish how long live context remained above16K; final usage alone cannot timestamp the crossing.",
            "C8 keeps all original 60 s warmups and 300 s formal phases; it is separate synthetic evidence, not a replacement for the task-based continuous-window conclusion in V1.5.",
            "No new numerical definition of abnormal steep rise is imposed. Observed 4K/16K ratios and consecutive cohorts are disclosed for review, not automatically labelled pass or engine failure.",
            "Budget length termination is explicit, not silent truncation. Raw stream checks and successful transport do not prove semantic answer completeness.",
        ],
        "status": "execution_complete_long_context_assessment_open",
    }


def milestone_target(evidence):
    if not evidence["execution_complete"]:
        raise ValueError("Completed evidence is required")
    profiles = evidence["profiles"]
    coverage = "、".join(
        f"{name}"
        + "/".join(
            str(profiles[name][r]["input_plus_output_at_least16384"])
            for r in ("B0", "B1")
        )
        for name in ("hle", "frontier", "swe", "terminal", "mmlu")
    )
    groups = evidence["continuous"]["B1"]["groups"]
    short, long = groups["4096-formal"], groups["16384-formal"]
    tails = "、".join(
        f"{w['ttft_p99_ms'] / 1000:.3f}" for w in long["submission_cohorts"]
    )
    return {
        "name": "五数据集与16K稳定性",
        "accepted": False,
        "status": "固定五项执行完成；16K稳定性结论待复核",
        "display_value": "Executed; 16K assessment open",
        "reason": (
            "固定五项两臂均已完成，无失败或未返回请求。按输入+输出计实际16K覆盖："
            + coverage
            + "（B0/B1）。裁判未决与准确率另列，不作为本项稳定性失败。"
            "独立C8两臂4K/16K正式窗口均完成：B1吞吐"
            f"{short['output_tokens_per_second']:.2f}→{long['output_tokens_per_second']:.2f} token/s，"
            f"TTFT P99为{short['ttft_p99_ms'] / 1000:.3f}→{long['ttft_p99_ms'] / 1000:.3f}秒；"
            f"16K五个连续提交区间的TTFT P99为{tails}秒。"
            "尚需完成大纲所要求的真实任务实际16K区间、可比负载和尾时延的联合判读；"
            "有限终态长度计数或合成C8不能单独替代该结论。"
            "未将其判成已证实的引擎稳定性失败。"
        ),
        "evidence_path": "stability-audit-v1.json",
        "quality_is_a_separate_assessment": True,
    }
