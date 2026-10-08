#!/usr/bin/env python3
"""Build explicit Dataset Matrix coverage states from the canonical artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "leaderboard-data" / "dataset-validation"
QWEN25_FILE = "dataset_validation_v1.b0.json"
QWEN35_FILE = "dataset_validation_qwen35_tp2_matrix.json"
BIDKV_FILE = "dataset_validation_qwen35_bidkv.json"
QWEN35_WORKBOOK_FILE = "dataset_validation_qwen35_tp2_ep_ctx32k_apcoff_inf_out256.json"
INDEX_FILE = "dataset_validation_index_v1.json"
REPORT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "qwen35-b0-xlsx-cross-validation-20261008"
)
EVIDENCE_COMMIT = "254ab4a71c1b82a9470b876c74a708502faebbe5"  # pragma: allowlist secret
EVIDENCE_BASE_URL = (
    "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
    f"{EVIDENCE_COMMIT}/reports/qwen35-b0-xlsx-cross-validation-20261008"
)
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


def load_report(name: str) -> dict:
    return json.loads((REPORT_ROOT / name).read_text(encoding="utf-8"))


def parse_workbook_bindings() -> dict[str, dict]:
    """Parse the checksummed per-dataset binding table from the evidence manifest."""
    manifest = (REPORT_ROOT / "MANIFEST-engineer-config-e1de9a4d.md").read_text(
        encoding="utf-8"
    )
    bindings: dict[str, dict] = {}
    in_table = False
    sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
    for line in manifest.splitlines():
        if line.startswith("| 数据集 | 准备操作.md SHA-256"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        parts = [part.strip().strip("`") for part in line.strip("|").split("|")]
        if len(parts) != 6:
            raise ValueError(f"invalid workbook binding row: {line}")
        label, preparation_sha, materialized_sha, result_sha, run_time, counts = parts
        if result_sha == "MISSING":
            continue
        if not all(
            sha256_pattern.fullmatch(value)
            for value in (preparation_sha, materialized_sha, result_sha)
        ):
            raise ValueError(f"invalid workbook binding hash: {label}")
        completed, failed, planned = (int(value) for value in counts.split("/"))
        bindings[label] = {
            "preparation_sha256": preparation_sha,
            "materialized_sha256": materialized_sha,
            "result_json_sha256": result_sha,
            "run_time": run_time,
            "completed_requests": completed,
            "failed_requests": failed,
            "planned_requests": planned,
        }
    if len(bindings) != 21:
        raise ValueError(f"expected 21 workbook bindings, found {len(bindings)}")
    return bindings


def build_qwen35_workbook() -> dict:
    """Publish the corrected 35B B0 workbook as its own measured configuration."""
    assessment = load_report("assessment.json")
    comparison = load_report("workbook-structural-comparison.json")
    bindings = parse_workbook_bindings()
    registry = load(QWEN35_FILE)
    registry_datasets = {item["id"]: item for item in registry["datasets"]}
    registry_metrics = {item["id"]: item for item in registry["metrics"]}
    metric_ids = [
        "request_throughput",
        "output_token_throughput",
        "total_token_throughput",
        "ttft_mean",
        "tpot_mean",
        "request_success_rate",
    ]
    row_metrics = {
        "2": "request_throughput",
        "4": "output_token_throughput",
        "5": "total_token_throughput",
        "6": "ttft_mean",
        "7": "tpot_mean",
    }
    corrected = {
        (item["dataset"], row_metrics[item["cell"][1:]]): item["corrected"]
        for item in comparison["differences"]
    }

    datasets = []
    results = []
    for observed in assessment["datasets"]:
        dataset_id = observed["dataset_id"]
        label = observed["dataset_label"]
        if label not in bindings:
            raise ValueError(f"missing workbook binding for {label}")
        dataset = copy.deepcopy(registry_datasets[dataset_id])
        dataset["applicable_metric_ids"] = metric_ids
        datasets.append(dataset)
        binding = bindings[label]
        provenance = {
            "repository": "vLLM-HUST/vllm-hust-benchmark",
            "evidence_commit": EVIDENCE_COMMIT,
            "report_url": f"{EVIDENCE_BASE_URL}/HISTORICAL-B0-EVIDENCE-SUPPLEMENT.md",
            "corrected_workbook_url": (
                f"{EVIDENCE_BASE_URL}/35B-B0-baseline-corrected-ddb9d8da.xlsx"
            ),
            "corrected_workbook_sha256": (
                "ddb9d8dab05dbccb2adb094b0598c812f2c2e69976128337ce23343fb8c16506"  # pragma: allowlist secret
            ),
            "manifest_url": (
                f"{EVIDENCE_BASE_URL}/MANIFEST-engineer-config-e1de9a4d.md"
            ),
            "manifest_sha256": (
                "e1de9a4d70e7d4800781df6d18619d1fb411bad421f7a00e6cd70d5fd3ba933c"  # pragma: allowlist secret
            ),
            **binding,
            "file_observation_grade": "R",
            "run_binding_grade": "D",
        }
        for metric_id in metric_ids:
            if metric_id == "request_success_rate":
                baseline_value = (
                    100.0 * binding["completed_requests"] / binding["planned_requests"]
                )
                value_source = "engineer manifest completed/planned counts"
            else:
                baseline_value = observed["metrics"][metric_id]["xlsx_value"]
                baseline_value = corrected.get((label, metric_id), baseline_value)
                value_source = "corrected workbook"
            results.append(
                {
                    "dataset_id": dataset_id,
                    "metric_id": metric_id,
                    "status": "baseline_only",
                    "baseline_value": baseline_value,
                    "value": None,
                    "updated_at": "2026-10-08",
                    "value_source": value_source,
                    "provenance": copy.deepcopy(provenance),
                }
            )

    scenario_id = (
        "qwen35-35b-a3b-bf16-tp2-ep-on-ctx32k-apc-off-full-decode-only-inf-out256"
    )
    return {
        "contract_version": "dataset-validation-v1",
        "generated_at": "2026-10-08T00:00:00Z",
        "status": "baseline_measurements",
        "source": {
            "service": "Qwen3.5-35B B0 workbook import",
            "run_id": scenario_id,
            "repository": "vLLM-HUST/vllm-hust-benchmark",
            "evidence_commit": EVIDENCE_COMMIT,
            "report_url": f"{EVIDENCE_BASE_URL}/README.md",
        },
        "baseline": {
            "id": "native-qwen35-tp2-ep-on-ctx32k-apc-off-inf-out256",
            "label": "Native Qwen3.5 TP2/EP on · 32K · APC off · inf · out256",
            "generated_at": "2026-10-04/05",
            "measurement_count": len(results),
        },
        "candidate_policy": {
            "id": "best-per-cell",
            "label": "Best same-configuration measured result per cell",
            "single_version_required": False,
            "rule": (
                "A B1 comparison requires the same model, topology, runtime settings, "
                "input materialization, and load contract as this B0."
            ),
        },
        "coverage_contract": {
            "mode": "measured-datasets-only",
            "complete_cell_states_required": True,
            "non_measured_reason_required": True,
            "matched_b0_required_for_b1": True,
        },
        "scenario": {
            "id": scenario_id,
            "label": (
                "Qwen3.5-35B-A3B · BF16 · TP2/EP on · 32K · "
                "APC off · FULL_DECODE_ONLY · inf · out256"
            ),
            "model": "Qwen3.5-35B-A3B",
            "model_revision": "712cf74392b05026a6db2bf213d343747d1f6d45",  # pragma: allowlist secret
            "model_revision_evidence": (
                "project model identity supplied separately; not embedded in workbook"
            ),
            "hardware": "2× Ascend 910B2",
            "precision": "BF16",
            "kv_cache_dtype": "auto",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "expert_parallel": True,
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "block_size": 128,
            "gpu_memory_utilization": 0.85,
            "prefix_caching": False,
            "chunked_prefill": True,
            "enforce_eager": False,
            "graph_mode": "FULL_DECODE_ONLY",
            "capture_sizes": [1, 2, 4, 8, 16],
            "speculative_decoding": "unset",
            "request_rate": "inf",
            "num_prompts": 200,
            "num_prompts_humaneval": 164,
            "output_length": 256,
            "temperature": 0,
            "seed": 0,
        },
        "evidence_quality": {
            "workbook_and_cell_verification": "R",
            "documented_run_configuration": "D",
            "same_run_cryptographic_binding": False,
            "policy": (
                "Publish measured B0 values with their evidence grade; do not treat "
                "a metadata gap as absence of a measurement."
            ),
        },
        "datasets": datasets,
        "metrics": [copy.deepcopy(registry_metrics[item]) for item in metric_ids],
        "results": results,
        "limitations": [
            "This is a distinct valid configuration, not the unified Frontier contract.",
            "The workbook values and ten corrections are locally reproduced; server-to-result binding is documentary.",
            "Runtime and model metadata repaired from the checksummed engineer manifest and project model identity retain explicit evidence grades.",
            "Raw result JSON hashes are retained per dataset; the raw JSON bytes are not copied into this publication bundle.",
        ],
    }


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
    workbook_scenario = {
        "id": (
            "qwen35-35b-a3b-bf16-tp2-ep-on-ctx32k-apc-off-full-decode-only-inf-out256"
        ),
        "label": (
            "Qwen3.5-35B-A3B · BF16 · TP2/EP on · 32K · "
            "APC off · FULL_DECODE_ONLY · inf · out256"
        ),
        "model": "Qwen3.5-35B-A3B",
        "hardware": "2× Ascend 910B2",
        "precision": "BF16",
        "data_file": QWEN35_WORKBOOK_FILE,
    }
    index["scenarios"] = [
        item
        for item in index["scenarios"]
        if item["id"] not in {scenario["id"], workbook_scenario["id"]}
    ]
    index["scenarios"].insert(1, scenario)
    index["scenarios"].insert(2, workbook_scenario)
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
    write(QWEN35_WORKBOOK_FILE, build_qwen35_workbook())
    write(BIDKV_FILE, update_bidkv())
    write(INDEX_FILE, update_index())
    update_checksums()


if __name__ == "__main__":
    main()
