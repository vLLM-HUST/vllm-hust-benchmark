"""Audit published per-cell selection without pooling different workloads."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    documents = []
    cells = []
    for path in sorted(root.glob("dataset_validation_*.json")):
        data = json.loads(path.read_text())
        documents.append(
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        for cell in data.get("results", []):
            candidates = cell.get("candidate_values", [])
            if not candidates:
                continue
            direction = cell.get("comparison", {}).get("direction", "higher_is_better")
            if direction not in {"higher_is_better", "lower_is_better"}:
                raise ValueError(f"Unknown metric direction: {direction}")
            best = (max if direction == "higher_is_better" else min)(
                c["value"] for c in candidates
            )
            selected = next(
                c
                for c in candidates
                if c["candidate_id"] == cell["selected_candidate_id"]
            )
            cells.append(
                {
                    "file": path.name,
                    "dataset_id": cell["dataset_id"],
                    "metric_id": cell["metric_id"],
                    "direction": direction,
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_value": selected["value"],
                    "best_value": best,
                    "selection_is_best": selected["value"] == best,
                    "candidates": [
                        {
                            k: c.get(k)
                            for k in (
                                "candidate_id",
                                "value",
                                "runtime_effectiveness",
                                "provenance",
                            )
                        }
                        for c in candidates
                    ],
                }
            )
    return {
        "schema": "b1-selection-audit/v1",
        "scope": "Published candidate lists only; not a complete organization source audit",
        "definition_zh": "当前已实测最优的vLLM-HUST优化方案",
        "source_files": documents,
        "cells": cells,
        "selected_cell_count": len(cells),
        "non_best_selections": sum(not c["selection_is_best"] for c in cells),
    }


def audit_website(root):
    paths = [
        root / "data/plugin-performance.json",
        root / "data/request-throttling-dataset-matrix-20261018/mmlu-pro.json",
        root / "data/leaderboard_betterscale_width_evidence.json",
    ]
    registry, throttle, width = [json.loads(p.read_text()) for p in paths]
    return {
        "source_files": [
            {
                "file": str(p.relative_to(root)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        ],
        "published_mod_inventory": [
            {
                "id": entry["id"],
                "url": entry.get("url"),
                "observations": [
                    {
                        "id": o["id"],
                        "kind": o["kind"],
                        "comparisons": o.get("comparisons", []),
                    }
                    for o in entry.get("observations", [])
                ],
            }
            for entry in registry["entries"]
        ],
        "request_throttling_mmlu": {
            "configuration": throttle["configuration"],
            "input_sha256": throttle["performance"]["input_sha256"],
            "aggregate": throttle["performance"]["aggregate"],
            "admission": "Same input; runtime variation is allowed, but this is not a contemporaneous paired comparison",
            "mechanism_boundary": throttle["performance"]["note"],
        },
        "width_candidate_basis": {
            "scope": width["scope"],
            "observations": [
                {
                    k: o.get(k)
                    for k in (
                        "arm",
                        "execution",
                        "resident",
                        "concurrency",
                        "valid",
                        "output_tok_s_chip",
                        "failed_requests",
                        "workload_sha256",
                    )
                }
                for o in width["observations"]
            ],
            "admission": "SWE evidence motivates width screening; not MMLU evidence",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--website-root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.website_root:
        result["website_audit"] = audit_website(args.website_root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
