"""Project verified same-workload measurements into the Dataset Matrix."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from microflow_candidate import RUN as MICROFLOW_RUN, load as load_microflow


REPORT = Path(__file__).resolve().parent
DEST = REPORT.parents[1] / "leaderboard-data/dataset-validation"
FILE = "dataset_validation_qwen35_mmlu_school028_inf_out256.json"
SCENARIO = "qwen35-35b-a3b-bf16-tp2-school028-mmlu-inf-out256"
METRICS = [
    (
        "output_token_throughput",
        "Output token throughput",
        "token/s",
        "higher",
        lambda m: m["output_tps"],
    ),
    ("ttft_mean", "Mean TTFT", "ms", "lower", lambda m: m["ttft_s"]["mean"] * 1000),
    ("tpot_mean", "Mean TPOT", "ms", "lower", lambda m: m["tpot_s"]["mean"] * 1000),
]


def build(evidence_commit):
    assert len(evidence_commit) == 40 and all(
        c in "0123456789abcdef" for c in evidence_commit
    )
    pair = json.loads((REPORT / "aligned-mmlu-results.json").read_text())
    report_url = f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{evidence_commit}/reports/{REPORT.name}/中期测试报告.md"
    sources = ["aligned-mmlu-results.json"]
    arms = [
        (
            "e16-r36",
            "vLLM-HUST EP off / 64K / APC on / MTP2 / FULL / E16/R36",
            pair["b1"],
            "mmlu-b1-r2",
            "not-recorded",
        )
    ]
    ended = []
    for kind in ("e32", "e24", "kvmat", "kvmat-e24"):
        name = f"b1-{kind}-screen.json"
        result = json.loads((REPORT / name).read_text())
        sources.append(name)
        ended.append(result["receipt"]["ended"])
        archive = REPORT / result["archive"]["file"]
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == result["archive"]["sha256"]
        )
        if result["status"] == "measured":
            assert (
                result["measured"]["input_tokens"]
                == pair["b0"]["input_tokens"]
                == 282265
            )
            assert result["measured"]["output_tokens"] == 51200
            arms.append(
                (
                    kind,
                    result["candidate"],
                    result["measured"],
                    result["run_id"],
                    "exercised" if kind.startswith("kvmat") else "not-recorded",
                )
            )

    microflow, measured = load_microflow()
    sources.append("microflow-school028/summary.json")
    ended.extend(row["ended"] for row in microflow["runs"])
    arms.append(
        (
            "microflow-fa-median",
            "vLLM-HUST E24/R36 + KV recompute + FA Microflow (median of 3 starts)",
            measured,
            MICROFLOW_RUN,
            "exercised",
        )
    )

    def provenance(run):
        return {
            "repository": "vLLM-HUST/vllm-hust-benchmark",
            "repository_commit": evidence_commit,
            "report_url": report_url,
            "run_id": run,
        }

    cells = []
    for metric, label, unit, direction, read in METRICS:
        baseline = read(pair["b0"])
        candidates = [
            {
                "candidate_id": ident,
                "series_id": run,
                "label": arm_label,
                "value": read(measured),
                "delta_pct": 100 * (read(measured) / baseline - 1),
                "runtime_effectiveness": effectiveness,
                "provenance": provenance(run),
            }
            for ident, arm_label, measured, run, effectiveness in arms
        ]
        selected = (max if direction == "higher" else min)(
            candidates, key=lambda row: row["value"]
        )
        if selected["candidate_id"] == "microflow-fa-median":
            key = {
                "output_token_throughput": "output_throughput",
                "ttft_mean": "mean_ttft_ms",
                "tpot_mean": "mean_tpot_ms",
            }[metric]
            assert microflow["improves_current_best_per_metric"][key]
        improved = (
            selected["value"] > baseline
            if direction == "higher"
            else selected["value"] < baseline
        )
        cells.append(
            {
                "dataset_id": "mmlu-pro",
                "metric_id": metric,
                "status": "passed",
                "baseline_value": baseline,
                "value": selected["value"],
                "delta_pct": selected["delta_pct"],
                "updated_at": "2026-10-19",
                "selected_candidate_id": selected["candidate_id"],
                "candidate_values": candidates,
                "comparison": {
                    "direction": f"{direction}_is_better",
                    "trend": "improved" if improved else "regressed",
                    "selection": "Best measured service-performance value per metric; FA Microflow uses the median of 3 independent starts; not a joint deployment result",
                },
                "provenance": selected["provenance"],
                "baseline_provenance": provenance("mmlu-b0-r2"),
                "note": "Sequential TP2 services: 200 requests per run, out256, infinite arrival. FA Microflow uses the median of 3 independent ON starts and matched OFF controls with the same initial compiler-cache snapshot; small differences are not claims of statistical significance. Task accuracy was not rescored. Each metric selects its own configuration. KV recompute only; partial reuse and GDN Microflow were not exercised.",
                "note_zh": "同机单TP2服务，每轮200请求、out256、无限到达率。FA Microflow取3次独立ON运行的中位数，配对OFF使用相同初始编译缓存快照；小幅差异不代表统计显著性。未重评任务正确率，各指标分别选择配置。KV仅逐请求重算，分段复用和GDN Microflow未执行。",
            }
        )
    return {
        "contract_version": "dataset-validation-v1",
        "generated_at": datetime.fromtimestamp(max(ended), timezone.utc).isoformat(),
        "status": "complete_comparison",
        "source": {
            "service": "vLLM-HUST MMLU-Pro same-workload measurements",
            "commit": evidence_commit,
            "report_url": report_url,
            "files": {
                name: hashlib.sha256((REPORT / name).read_bytes()).hexdigest()
                for name in sources
            },
        },
        "baseline": {
            "id": "mmlu-b0-r2",
            "label": "Native EP on / 32K / APC off / MTP off / FULL_DECODE_ONLY",
            "generated_at": "2026-10-19",
        },
        "candidate_policy": {
            "id": "best-per-cell",
            "label": "当前已实测最优的vLLM-HUST优化方案",
            "single_version_required": False,
            "rule": "Same model revision, BF16, TP2, source-pinned runtime, prepared inputs, arrival and output contract; optimization configuration may differ and is retained per candidate. Service performance only; task quality not rescored.",
        },
        "scenario": {
            "id": SCENARIO,
            "label": "Qwen3.5-35B-A3B · BF16 · TP2 · school 0.28 · MMLU-Pro · inf · out256",
            "model": "Qwen3.5-35B-A3B",
            "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",  # pragma: allowlist secret (model commit)
            "hardware": "2× Ascend 910B2 (4 reserved)",
            "precision": "BF16",
            "prepared_workload_sha256": "fb79b6fa98f232282a948e4a17f404f80c03dc2bae05af57b95b9cc0651b3506",  # pragma: allowlist secret (input hash)
            "accuracy_evaluated": False,
            "request_rate": "inf",
            "num_prompts": 200,
            "output_length": 256,
        },
        "datasets": [
            {
                "id": "mmlu-pro",
                "label": "MMLU-Pro",
                "group": "MMLU-Pro",
                "description": "200-request service-performance burst; not task accuracy",
                "applicable_metric_ids": [m[0] for m in METRICS],
            }
        ],
        "metrics": [
            {"id": ident, "label": label, "unit": unit, "direction": direction}
            for ident, label, unit, direction, _ in METRICS
        ],
        "results": cells,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-commit", required=True)
    args = parser.parse_args()
    data = build(args.evidence_commit)
    (DEST / FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    index_path = DEST / "dataset_validation_index_v1.json"
    index = json.loads(index_path.read_text())
    if not any(row["id"] == SCENARIO for row in index["scenarios"]):
        index["scenarios"].insert(
            4,
            {
                "id": SCENARIO,
                "label": "Paired B0/B1 · " + data["scenario"]["label"],
                "model": "Qwen3.5-35B-A3B",
                "hardware": "2× Ascend 910B2",
                "precision": "BF16",
                "data_file": FILE,
            },
        )
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
