#!/usr/bin/env python3
"""Build the Qwen3.5 unified Frontier B0/B1 Dataset Validation artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "leaderboard-data" / "dataset-validation"
OUTPUT = PUBLICATION / "dataset_validation_qwen35_frontier_unified_900s.json"
INDEX = PUBLICATION / "dataset_validation_index_v1.json"
CHECKSUMS = PUBLICATION / "SHA256SUMS"
CONCURRENCIES = [1, 2, 4, 8, 16]
METRICS = {
    "output_token_throughput": {
        "label": "Output token throughput",
        "unit": "token/s",
        "direction": "higher",
        "source_key": "output_tps",
    },
    "decode_p90_tps_per_user": {
        "label": "P90 decode throughput per user",
        "unit": "token/s/user",
        "direction": "higher",
        "source_key": "decode_p90_tps",
    },
}
REPORT_BASE = "https://github.com/vLLM-HUST/vllm-hust-website/blob/6ac7463355ae35efc029ca4d0fbf90c9c26cd019"

NATIVE = {
    "series_id": "swe-unified-native-20260927",
    "output_tps": [91.54555555555555, 152.7211111111111, 217.47555555555556, 291.1066666666667, 359.75],
    "decode_p90_tps": [107.36076237098999, 97.81036014052084, 78.15060902600783, 59.758209174967206, 33.54398219355827],
    "run_ids": [
        "562a9f3349c04f69b9069f81d518a198", "edda8b45980944edb315abc2172910c9",
        "2357194fcf6143ca9c78d65d8f7eca09", "9f9dc7fa5e40427b8ef9fe99a95c737e",
        "f7e5da90a1a243deb3e24b8c86f68303",
    ],
}

CANDIDATES = [
    {
        "id": "bidkv", "label": "BidKV", "series_id": "swe-unified-bidkv-20260927",
        "repository": "vLLM-HUST/vllm-hust-bidkv", "commit": "a0cba97d9abdc99908e46616db622f0e0099127f",
        "report": f"{REPORT_BASE}/docs/FRONTIER-QWEN35-UNIFIED-MODS.md",
        "output_tps": [92.13333333333334, 153.43, 216.70222222222222, 293.44555555555553, 358.70666666666665],
        "decode_p90_tps": [107.68752556522423, 98.83310399422402, 78.1774927481624, 59.13227474995871, 33.39448751175928],
        "run_ids": ["9cd43584a1ab4a7085da2dec73ffb5ad", "7d1281c1e2474bcdbf4e5fde3b8a6b1f", "b041218b96c3473bb50230ce679483a2", "eeb2c659cf2c434baa1daab4e3393728", "8b09ecc3847f4578a0b044e359400f04"],
    },
    {
        "id": "dla", "label": "DLA", "series_id": "swe-unified-dla-20260927",
        "repository": "vLLM-HUST/vllm-hust-dla", "commit": "dc20d0f8ea8d09106f77571e1947b9a2f8702545",
        "report": f"{REPORT_BASE}/docs/FRONTIER-QWEN35-UNIFIED-MODS.md",
        "output_tps": [91.84666666666666, 152.60666666666665, 215.56222222222223, 288.1488888888889, 358.1988888888889],
        "decode_p90_tps": [106.95998116424124, 98.15888496241027, 77.57745642141361, 57.033470748952965, 34.83995993020349],
        "run_ids": ["e4c11fa8ea304eb181c12192248a4070", "ef7efb2e134d480580de65f3102ce866", "dfdb8d3893cb4292ad2a3bb7be291a00", "2ab00cdf0a0a44159154c1a38b131f14", "5bf7b8ce60ba4b68864debbbf91bacb7"],
    },
    {
        "id": "kv-materialization-arrival-control", "label": "KV materialization arrival control", "series_id": "swe-unified-kv-materialization-arrival-control-20260928",
        "repository": "vLLM-HUST/vllm-hust-kv-materialization-arrival-control", "commit": "53a6b1130cbff58925782eb62d7e88cd37e4e20a",
        "report": f"{REPORT_BASE}/docs/FRONTIER-QWEN35-KV-MATERIALIZATION.md",
        "output_tps": [90.5911111111111, 154.7288888888889, 237.52444444444444, 325.75111111111113, 414.2922222222222],
        "decode_p90_tps": [107.36876391861011, 98.92568495208843, 84.76666140724171, 59.8154350121774, 36.79618457742906],
        "run_ids": ["ad9925f9329b4f87ad3da5285ebab36b", "aa7b91b673fb4b1caf4adbc2134ed65b", "5a723cba6cf146379aee490dc398fc8b", "5c9cd4af1452439f90d746ee86f4f843", "1f9c022d2026433682cffd3d25211cbc"],
        "runtime_effectiveness": "exercised",
    },
    {
        "id": "kv-tiering-migration", "label": "KV tiering migration", "series_id": "swe-unified-kv-tiering-migration-20260927",
        "repository": "vLLM-HUST/vllm-hust-kv-tiering", "commit": "7ba646a780c3bd0a8906309ea59719f5ccf6187e",
        "report": f"{REPORT_BASE}/docs/FRONTIER-QWEN35-UNIFIED-MODS.md",
        "output_tps": [91.03222222222222, 150.45222222222222, 215.9322222222222, 286.12, 341.6333333333333],
        "decode_p90_tps": [106.13549438369412, 97.13844486718608, 77.40249120147915, 56.49823410867081, 33.25649016831222],
        "run_ids": ["4f4aa33f2ab046ada0de34df16fbc6c3", "99370030aacb43918f0a3f3ad924ad5c", "9ed47b70f85549c09efdb9e115df2b3c", "d650f3a508814a5c8525631111a0238b", "2cd49521050b4aa8a756e6edb1327885"],
    },
    {
        "id": "kvcompress-ascend", "label": "KVCompress Ascend", "series_id": "swe-unified-kvcompress-ascend-20260928",
        "repository": "vLLM-HUST/vllm-ascend-kvcompress-hust", "commit": "2ca0f9335399a698342285870df1514e9c519bd4",
        "report": "https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/a020c164ec68cf3a7c53a40c0129b37a14ed1a05/handoff/2026-09-28-qwen35-frontier-formal/README.md",
        "output_tps": [94.57222222222222, 136.70555555555555, 206.5988888888889, 286.1011111111111, 347.75333333333333],
        "decode_p90_tps": [110.35405668201226, 88.41768829712161, 77.34354504472113, 61.13461820415613, 34.93487505412306],
        "run_ids": ["b31b00379e034d7e813ae774a6ca41a2", "7fe75902ced340b9b7b24abf1058794c", "2042394037de4949926f5ceea42b04ea", "bda15187dacf4d4b994fb34aaf7bb310", "436f78f03dd24c9ea2dd462803db3d4b"],
        "runtime_effectiveness": "exercised",
    },
    {
        "id": "pegaflow-vllm-connectors", "label": "PegaFlow vLLM connectors", "series_id": "swe-unified-pegaflow-vllm-connectors-20260929",
        "repository": "vLLM-HUST/pegaflow-hust", "commit": "cd64ecc283ff856a44437a9a25659929ef3a0653",
        "report": "https://github.com/vLLM-HUST/pegaflow-hust/blob/64ef9d0b5c4f11589d296f8500dc1664ffc346dc/handoff/2026-09-29-qwen35-frontier-formal/README.md",
        "output_tps": [65.52333333333333, 167.14333333333335, 237.96777777777777, 361.4411111111111, 459.71555555555557],
        "decode_p90_tps": [113.9060417414391, 109.48681228877716, 87.30566240630844, 73.03849278224763, 46.37622765934938],
        "run_ids": ["c01435f8e7a34e19aec12f2c24031d5c", "9d697fe6af154fe785a626a10864c7f5", "d09d02d878be40d883d746c4ec660956", "618a071154cb4b8e810c0d4f4efd5125", "eed86dbce9f74fc583ef30f3073cfdec"],
        "runtime_effectiveness": "exercised",
    },
]


def pct(value: float, baseline: float) -> float:
    return (value / baseline - 1) * 100


def candidate_cell(candidate: dict, metric_key: str, index: int, baseline: float) -> dict:
    value = candidate[metric_key][index]
    return {
        "candidate_id": candidate["id"],
        "label": candidate["label"],
        "series_id": candidate["series_id"],
        "value": value,
        "delta_pct": pct(value, baseline),
        "runtime_effectiveness": candidate.get("runtime_effectiveness", "not-recorded"),
        "provenance": {
            "repository": candidate["repository"],
            "repository_commit": candidate["commit"],
            "report_url": candidate["report"],
            "run_id": candidate["run_ids"][index],
        },
    }


def build() -> dict:
    results = []
    for index, concurrency in enumerate(CONCURRENCIES):
        for metric_id, metric in METRICS.items():
            source_key = metric["source_key"]
            baseline = NATIVE[source_key][index]
            candidates = [candidate_cell(c, source_key, index, baseline) for c in CANDIDATES]
            selected = max(candidates, key=lambda item: item["value"])
            results.append({
                "dataset_id": f"swe-prefix-reuse-c{concurrency}",
                "metric_id": metric_id,
                "status": "passed",
                "baseline_value": baseline,
                "value": selected["value"],
                "delta_pct": selected["delta_pct"],
                "updated_at": "2026-09-29",
                "selected_candidate_id": selected["candidate_id"],
                "candidate_values": candidates,
                "comparison": {"direction": "higher_is_better", "trend": "improved", "selection": "best compliant measured candidate for this cell"},
                "provenance": selected["provenance"],
                "baseline_provenance": {
                    "repository": "vLLM-HUST/vllm-hust-website",
                    "repository_commit": "6ac7463355ae35efc029ca4d0fbf90c9c26cd019",
                    "report_url": f"{REPORT_BASE}/docs/FRONTIER-QWEN35-UNIFIED-MODS.md",
                    "series_id": NATIVE["series_id"],
                    "run_id": NATIVE["run_ids"][index],
                },
            })
    return {
        "contract_version": "dataset-validation-v1",
        "generated_at": "2026-10-08T00:00:00Z",
        "status": "complete_comparison",
        "source": {
            "service": "Qwen3.5 unified Frontier paired measurements",
            "repository": "vLLM-HUST/vllm-hust-website",
            "commit": "6ac7463355ae35efc029ca4d0fbf90c9c26cd019",
            "frontier_data_sha256": "5913253b83455ce375d0252ed803beed76fb6a5a98348f60c103e6f44f246a7d",
            "comparison_contract_sha256": "d174bd33ab9b4e1b9b02a836f3f24b0be8fbfd156ae33e19539d8f5984e1e4c2",
            "tracking_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/245",
        },
        "baseline": {"id": NATIVE["series_id"], "label": "Native unified 900s", "generated_at": "2026-09-27"},
        "candidate_policy": {
            "id": "best-per-cell-with-full-candidate-set",
            "label": "Best compliant measured result per cell; retain every admitted candidate",
            "single_version_required": False,
            "rule": "Select the highest value for the cell from the six series admitted by the published comparison contract; retain every candidate and its provenance in candidate_values.",
        },
        "scenario": {
            "id": "qwen35-35b-a3b-bf16-tp2-pp1-dp1-ep-off-ctx262k-apc-on-mtp2-full-piecewise-sweprefix-900s",
            "label": "Qwen3.5-35B-A3B · BF16 · TP2/PP1/DP1 · EP off · 262K · APC on · MTP2 · FULL_AND_PIECEWISE · SWE Prefix Reuse · 900s",
            "model": "Qwen3.5-35B-A3B", "model_revision": "712cf74392b05026a6db2bf213d343747d1f6d45",
            "hardware": "2× Ascend 910B2", "precision": "BF16", "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1, "data_parallel_size": 1, "expert_parallel": False,
            "max_model_len": 262144, "max_num_seqs": 16, "max_num_batched_tokens": 4096,
            "prefix_caching": True, "async_scheduling": True, "mamba_cache_mode": "align",
            "mtp_draft_tokens": 2, "thinking": True, "temperature": 0,
            "graph_mode": "FULL_AND_PIECEWISE", "kv_cache_memory_bytes_per_chip": 26038239232,
            "workload": "swe-prefix-reuse/v1", "prepared_workload_sha256": "8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85",
            "tokenizer_fingerprint": "3f9ca78537850303ee04bfa6640c020be89723c62f37121c0f27a4c0babc53e0",
            "measurement_seconds": 900,
        },
        "datasets": [{
            "id": f"swe-prefix-reuse-c{c}", "label": f"SWE Prefix Reuse · C{c}", "group": "SWE Prefix Reuse",
            "description": f"swe-prefix-reuse/v1 at closed-loop client concurrency {c}",
            "applicable_metric_ids": list(METRICS),
        } for c in CONCURRENCIES],
        "metrics": [{"id": key, "label": value["label"], "unit": value["unit"], "direction": value["direction"]} for key, value in METRICS.items()],
        "results": results,
        "limitations": [
            "These cells cover only SWE Prefix Reuse at five closed-loop concurrency levels; they are not measurements for the other Dataset Matrix datasets.",
            "The source Frontier projection labels its legacy campaign profile as smoke, but every admitted cell here is a full 900-second formal window.",
            "runtime_effectiveness=not-recorded means this projection does not independently claim a control action; it does not rewrite the producing repository's evidence.",
        ],
        "coverage_contract": {"mode": "explicit-dataset-metric-applicability", "complete_cell_states_required": True, "non_measured_reason_required": True, "matched_b0_required_for_b1": True},
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    write_json(OUTPUT, build())
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    scenario = build()["scenario"]
    entry = {key: scenario[key] for key in ("id", "label", "model", "hardware", "precision")}
    entry["data_file"] = OUTPUT.name
    index["scenarios"] = [item for item in index["scenarios"] if item["id"] != entry["id"]] + [entry]
    write_json(INDEX, index)
    names = sorted([INDEX.name, *(item["data_file"] for item in index["scenarios"])])
    CHECKSUMS.write_text("".join(f"{hashlib.sha256((PUBLICATION / name).read_bytes()).hexdigest()}  {name}\n" for name in names), encoding="utf-8")


if __name__ == "__main__":
    main()
