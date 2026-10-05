#!/usr/bin/env python3
"""Build explicit Dataset Matrix coverage states from the canonical artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "leaderboard-data" / "dataset-validation"
QWEN25_FILE = "dataset_validation_v1.b0.json"
QWEN35_FILE = "dataset_validation_qwen35_tp2_matrix.json"
BIDKV_FILE = "dataset_validation_qwen35_bidkv.json"
INDEX_FILE = "dataset_validation_index_v1.json"
ONLINE_REASON = (
    "No matching measurement has been admitted for this dataset, model, and "
    "serving configuration."
)
AGENT_DATASET_REASON = (
    "This agent evaluation uses resolve rate rather than online serving metrics."
)
AGENT_METRIC_REASON = (
    "Agent resolve rate applies only to the SZYN OpenCode issue-resolution dataset."
)


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name: str, payload: dict) -> None:
    (ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_qwen25() -> dict:
    payload = load(QWEN25_FILE)
    metric_ids = [metric["id"] for metric in payload["metrics"]]
    result_by_cell = {
        (result["dataset_id"], result["metric_id"]): result
        for result in payload["results"]
    }

    existing_unadmitted = {
        candidate["id"]: candidate
        for candidate in payload.get("candidate_search", {}).get(
            "unadmitted_candidates", []
        )
    }
    candidate_id = "gsm8k-vspec-b16-eagle-adaptive-unmatched-b0"
    if candidate_id in existing_unadmitted:
        gsm8k = copy.deepcopy(existing_unadmitted[candidate_id]["results"])
    else:
        gsm8k = [
            copy.deepcopy(result)
            for result in payload["results"]
            if result["dataset_id"] == "gsm8k"
        ]
    payload["candidate_search"]["unadmitted_candidates"] = [
        {
            "id": candidate_id,
            "dataset_id": "gsm8k",
            "reason": (
                "The measured candidate has no same-spec B0 observation; retain its "
                "raw provenance without publishing a B1 comparison."
            ),
            "results": gsm8k,
        }
    ]
    payload["candidate_search"]["eligible_cells"] = 6
    payload["coverage_contract"] = {
        "mode": "explicit-dataset-metric-applicability",
        "complete_cell_states_required": True,
        "non_measured_reason_required": True,
        "matched_b0_required_for_b1": True,
    }

    results = []
    for dataset in payload["datasets"]:
        dataset_id = dataset["id"]
        applicable = (
            [] if dataset_id == "szyn-opencode-swebench-verified-500" else metric_ids
        )
        dataset["applicable_metric_ids"] = applicable
        for metric_id in metric_ids:
            cell = result_by_cell.get((dataset_id, metric_id))
            if dataset_id == "gsm8k":
                cell = {
                    "dataset_id": dataset_id,
                    "metric_id": metric_id,
                    "status": "not_tested",
                    "baseline_value": None,
                    "value": None,
                    "reason": (
                        "A candidate measurement exists but lacks a matching B0; "
                        "see candidate_search.unadmitted_candidates."
                    ),
                }
            elif metric_id not in applicable:
                cell = {
                    "dataset_id": dataset_id,
                    "metric_id": metric_id,
                    "status": "not_applicable",
                    "baseline_value": None,
                    "value": None,
                    "reason": AGENT_DATASET_REASON,
                }
            elif cell is None:
                cell = {
                    "dataset_id": dataset_id,
                    "metric_id": metric_id,
                    "status": "not_tested",
                    "baseline_value": None,
                    "value": None,
                    "reason": ONLINE_REASON,
                }
            results.append(cell)
    payload["results"] = results
    payload["status"] = "explicit_partial_comparison"
    return payload


def build_qwen35(qwen25: dict) -> dict:
    existing = load(QWEN35_FILE)
    measured = {
        (result["dataset_id"], result["metric_id"]): result
        for result in existing["results"]
        if result["status"] in {"baseline_only", "passed", "failed"}
    }
    online_metrics = copy.deepcopy(qwen25["metrics"])
    agent_metric = {
        "id": "agent_resolve_rate",
        "label": "Agent resolve rate",
        "unit": "%",
        "direction": "higher",
    }
    metric_ids = [metric["id"] for metric in online_metrics]
    datasets = copy.deepcopy(qwen25["datasets"])
    results = []
    for dataset in datasets:
        is_agent = dataset["id"] == "szyn-opencode-swebench-verified-500"
        dataset["applicable_metric_ids"] = (
            [agent_metric["id"]] if is_agent else metric_ids
        )
        for metric_id in [*metric_ids, agent_metric["id"]]:
            applicable = metric_id in dataset["applicable_metric_ids"]
            if is_agent and applicable:
                results.append(
                    {
                        "dataset_id": dataset["id"],
                        "metric_id": metric_id,
                        "status": "queued",
                        "baseline_value": None,
                        "value": None,
                        "reason": (
                            "The 1/500 collector qualification is not an aggregate. "
                            "The remaining 499 tasks are tracked by dev-hub issue #87."
                        ),
                        "tracking_url": (
                            "https://github.com/vLLM-HUST/vllm-hust-dev-hub/issues/87"
                        ),
                    }
                )
            elif applicable:
                results.append(
                    {
                        "dataset_id": dataset["id"],
                        "metric_id": metric_id,
                        "status": "not_tested",
                        "baseline_value": None,
                        "value": None,
                        "reason": ONLINE_REASON,
                    }
                )
            else:
                results.append(
                    {
                        "dataset_id": dataset["id"],
                        "metric_id": metric_id,
                        "status": "not_applicable",
                        "baseline_value": None,
                        "value": None,
                        "reason": (
                            AGENT_DATASET_REASON if is_agent else AGENT_METRIC_REASON
                        ),
                    }
                )

    results = [
        copy.deepcopy(measured.get((cell["dataset_id"], cell["metric_id"]), cell))
        for cell in results
    ]
    publication = {
        "contract_version": "dataset-validation-v1",
        "generated_at": "2026-10-03T08:00:00Z",
        "status": "campaign_defined_no_aggregate_measurements",
        "source": {
            "service": "vLLM-HUST Dataset Matrix campaign",
            "run_id": "qwen35-tp2-dataset-matrix-v1",
            "repository": "vLLM-HUST/vllm-hust-benchmark",
            "tracking_url": (
                "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/231"
            ),
        },
        "baseline": {
            "id": "native-qwen35-tp2-bf16-dataset-matrix",
            "label": "Native Qwen3.5 TP2 BF16",
            "commit": None,
            "generated_at": None,
        },
        "candidate_policy": {
            "id": "best-per-cell",
            "label": "Best compliant measured result per cell",
            "single_version_required": False,
            "rule": (
                "Admit B1 only with a same-spec B0 and exact per-cell runtime, "
                "workload, artifact, and revision provenance."
            ),
        },
        "coverage_contract": {
            "mode": "explicit-dataset-metric-applicability",
            "complete_cell_states_required": True,
            "non_measured_reason_required": True,
            "matched_b0_required_for_b1": True,
        },
        "scenario": {
            "id": "qwen35-35b-a3b-bf16-tp2-dataset-matrix-v1",
            "label": "Qwen3.5-35B-A3B · BF16 · TP2 · Dataset Matrix",
            "model": "Qwen3.5-35B-A3B",
            "model_revision": "712cf74392b05026a6db2bf213d343747d1f6d45",  # pragma: allowlist secret
            "hardware": "2× Ascend 910B2",
            "precision": "BF16",
            "tensor_parallel_size": 2,
        },
        "datasets": datasets,
        "metrics": [*online_metrics, agent_metric],
        "results": results,
        "limitations": [
            "This campaign is separate from the TP4/C12 BidKV specialty cell.",
            "The existing SZYN 1/500 record remains a qualification, not an aggregate.",
            "No empty or queued cell is a performance or coverage claim.",
        ],
    }
    if measured:
        for field in ("generated_at", "status", "baseline", "limitations"):
            publication[field] = copy.deepcopy(existing[field])
    return publication


def update_index() -> dict:
    index = load(INDEX_FILE)
    scenario = {
        "id": "qwen35-35b-a3b-bf16-tp2-dataset-matrix-v1",
        "label": "Qwen3.5-35B-A3B · BF16 · TP2 · Dataset Matrix",
        "model": "Qwen3.5-35B-A3B",
        "hardware": "2× Ascend 910B2",
        "precision": "BF16",
        "data_file": QWEN35_FILE,
    }
    index["scenarios"] = [
        item for item in index["scenarios"] if item["id"] != scenario["id"]
    ]
    index["scenarios"].insert(1, scenario)
    return index


def update_bidkv() -> dict:
    payload = load(BIDKV_FILE)
    payload["datasets"][0]["applicable_metric_ids"] = [payload["metrics"][0]["id"]]
    payload["coverage_contract"] = {
        "mode": "explicit-dataset-metric-applicability",
        "complete_cell_states_required": True,
        "non_measured_reason_required": True,
        "matched_b0_required_for_b1": True,
    }
    return payload


def update_checksums() -> None:
    names = sorted(path.name for path in ROOT.glob("*.json"))
    lines = [
        f"{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    qwen25 = build_qwen25()
    write(QWEN25_FILE, qwen25)
    write(QWEN35_FILE, build_qwen35(qwen25))
    write(BIDKV_FILE, update_bidkv())
    write(INDEX_FILE, update_index())
    update_checksums()


if __name__ == "__main__":
    main()
