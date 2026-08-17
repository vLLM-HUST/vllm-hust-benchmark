#!/usr/bin/env python3
"""Generate PPT audit attachments for the 910B2 stable-checkpoint matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "leaderboard-data/snapshots/leaderboard_historical.json"
OUTPUT_DIR = ROOT / "reports/ppt-stable-trend-audit-20260817"

CHECKPOINTS = {
    "C1": {
        "rank": 1,
        "core": "0657f3f2a6867c0ad33de6227b137ddc49a0c638",  # pragma: allowlist secret
        "plugin": "03a12f9bddd944952bd029c6b62e23d68fa3a28e",  # pragma: allowlist secret
        "label": "2026-07-13 core health checkpoint",
    },
    "C2": {
        "rank": 2,
        "core": "73187bc8ba89b8f83652cbc24042433fb7032add",  # pragma: allowlist secret
        "plugin": "03a12f9bddd944952bd029c6b62e23d68fa3a28e",  # pragma: allowlist secret
        "label": "2026-07-16 core health checkpoint",
    },
    "C3": {
        "rank": 3,
        "core": "1aa7cd10b7b16e82fdb29fcc47d3a3cd93bd01dc",  # pragma: allowlist secret
        "plugin": "03ae1d03db8049cd2a5c3f824039814459542e25",  # pragma: allowlist secret
        "label": "2026-07-19 core health checkpoint",
    },
}

WORKLOADS = (
    "agent-research-online",
    "instructcoder-online",
    "prefix-repetition-online",
    "random-latency",
    "random-online",
    "sharegpt-online",
    "sharegpt-throughput",
    "sonnet-throughput",
    "visionarena-online",
)

# Declared before looking at per-line direction: interactive/online workloads use
# response-start latency, explicit throughput workloads use token throughput, and
# the offline latency workload uses its mean measured latency.
PRIMARY_METRICS = {
    "agent-research-online": ("ttft_ms", "lower"),
    "instructcoder-online": ("ttft_ms", "lower"),
    "prefix-repetition-online": ("ttft_ms", "lower"),
    "random-latency": ("ttft_ms", "lower"),
    "random-online": ("ttft_ms", "lower"),
    "sharegpt-online": ("ttft_ms", "lower"),
    "sharegpt-throughput": ("throughput_tps", "higher"),
    "sonnet-throughput": ("throughput_tps", "higher"),
    "visionarena-online": ("ttft_ms", "lower"),
}

METRIC_FIELDS = ("throughput_tps", "ttft_ms", "tbt_ms", "error_rate")

CAPABILITY_COVERAGE = [
    {
        "point": "C1",
        "core_commit": CHECKPOINTS["C1"]["core"],
        "plugin_commit": CHECKPOINTS["C1"]["plugin"],
        "point_type": "historical version health checkpoint",
        "core_pr_42": True,
        "core_pr_124": False,
        "core_pr_173": False,
        "core_pr_220": False,
        "core_pr_236": False,
        "ascend_pr_151": True,
        "ascend_pr_153": False,
        "ascend_pr_216": False,
        "matrix_coverage": "9/9 workloads, mostly single invocation",
    },
    {
        "point": "C2",
        "core_commit": CHECKPOINTS["C2"]["core"],
        "plugin_commit": CHECKPOINTS["C2"]["plugin"],
        "point_type": "historical version health checkpoint",
        "core_pr_42": True,
        "core_pr_124": False,
        "core_pr_173": False,
        "core_pr_220": False,
        "core_pr_236": False,
        "ascend_pr_151": True,
        "ascend_pr_153": False,
        "ascend_pr_216": False,
        "matrix_coverage": "9/9 workloads, mostly single invocation",
    },
    {
        "point": "C3",
        "core_commit": CHECKPOINTS["C3"]["core"],
        "plugin_commit": CHECKPOINTS["C3"]["plugin"],
        "point_type": "historical version health checkpoint",
        "core_pr_42": True,
        "core_pr_124": True,
        "core_pr_173": False,
        "core_pr_220": False,
        "core_pr_236": False,
        "ascend_pr_151": False,
        "ascend_pr_153": False,
        "ascend_pr_216": False,
        "matrix_coverage": "9/9 workloads, mostly single invocation",
    },
    {
        "point": "H1-partial",
        "core_commit": "e4ce33646f2ef1781289e6dc651fad0d00177c55",  # pragma: allowlist secret
        "plugin_commit": "0f38988f47b55e2e896551bc6125fda27fae5392",  # pragma: allowlist secret
        "point_type": "post-Core-PR-173 partial capability checkpoint",
        "core_pr_42": True,
        "core_pr_124": True,
        "core_pr_173": True,
        "core_pr_220": False,
        "core_pr_236": False,
        "ascend_pr_151": True,
        "ascend_pr_153": False,
        "ascend_pr_216": False,
        "matrix_coverage": "5/9 workloads in snapshot; not a replacement axis",
    },
    {
        "point": "H-current-candidate",
        "core_commit": "43341b177dbaa8c7f04662f71e885ee7dfe22704",  # pragma: allowlist secret
        "plugin_commit": "0a46364814eedd3314f04eff3490c3ab422438bd",  # pragma: allowlist secret
        "point_type": "hardened current candidate; not yet benchmarked",
        "core_pr_42": True,
        "core_pr_124": True,
        "core_pr_173": True,
        "core_pr_220": True,
        "core_pr_236": True,
        "ascend_pr_151": True,
        "ascend_pr_153": True,
        "ascend_pr_216": True,
        "matrix_coverage": "0/9 at this exact pair",
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_pair(entry: dict[str, Any]) -> tuple[str, str]:
    provenance = entry.get("metadata", {}).get("runtime_provenance", {})
    return (
        provenance.get("engine", {}).get("commit", ""),
        provenance.get("plugin", {}).get("commit", ""),
    )


def checkpoint_for(entry: dict[str, Any]) -> str | None:
    pair = runtime_pair(entry)
    for checkpoint, definition in CHECKPOINTS.items():
        if pair == (definition["core"], definition["plugin"]):
            return checkpoint
    return None


def source_dir(entry: dict[str, Any]) -> Path:
    source = entry["historical_recovery"]["source_path"]
    return ROOT / source_dir_string(source)


def source_dir_string(source: str) -> str:
    return str(Path(source).parent)


def request_accounting(
    workload: str, raw: dict[str, Any], client: dict[str, Any]
) -> tuple[int | None, str]:
    if workload == "random-latency":
        return len(raw.get("latencies", [])) or client.get("num_iters"), (
            f"measured iterations; batch_size={client.get('batch_size', 'unknown')}"
        )
    if workload.endswith("-throughput"):
        return raw.get("num_requests") or client.get("num_prompts"), "requests"
    return raw.get("num_prompts") or raw.get("completed") or client.get(
        "num_prompts"
    ), ("requests")


def warmup_accounting(workload: str, client: dict[str, Any]) -> tuple[int | None, str]:
    if workload == "random-latency":
        value = client.get("num_iters_warmup")
        return (
            value,
            f"{value} warmup iterations" if value is not None else "not recorded",
        )
    if workload.endswith("-throughput"):
        value = client.get("num_warmups")
        return (
            value,
            f"{value} benchmark warmup requests"
            if value is not None
            else "not recorded",
        )
    return None, "no benchmark warmup phase declared in the serve invocation"


def execution_mode(workload: str) -> str:
    return (
        "offline"
        if workload == "random-latency" or workload.endswith("-throughput")
        else "online"
    )


def evidence_detail(
    mode: str, raw: dict[str, Any], directory: Path, repeat_count: int
) -> str:
    if mode == "online":
        return (
            f"raw serve result: completed={raw.get('completed')}, failed={raw.get('failed')}; "
            f"server_log={str((directory / 'server.stdout.log').exists()).lower()}; "
            f"independent_invocations={repeat_count}"
        )
    return (
        "not real-online: offline raw benchmark result; "
        f"offline_graph_proof={str((directory / 'offline_graph_proof.json').exists()).lower()}; "
        f"independent_invocations={repeat_count}"
    )


def evidence_grade(directory: Path, independent_median: bool) -> str:
    if independent_median:
        return "A: raw artifacts plus independent 3-repeat median"
    if (directory / "checksums.sha256").exists() and (
        directory / "bench.stdout.log"
    ).exists():
        return "B: single invocation with raw result, logs, environment/checksums"
    return "C: single invocation raw result/manifest only"


def raw_change_pct(first: float | None, last: float | None) -> float | None:
    if first in (None, 0) or last is None:
        return None
    return (last - first) / first * 100.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    entries = load_json(SNAPSHOT)
    selected: list[tuple[str, dict[str, Any]]] = []
    for entry in entries:
        checkpoint = checkpoint_for(entry)
        if checkpoint and entry.get("workload", {}).get("name") in WORKLOADS:
            selected.append((checkpoint, entry))

    counts: dict[tuple[str, str], int] = {}
    for checkpoint, entry in selected:
        key = (checkpoint, entry["workload"]["name"])
        counts[key] = counts.get(key, 0) + 1
    expected = {
        (checkpoint, workload) for checkpoint in CHECKPOINTS for workload in WORKLOADS
    }
    assert set(counts) == expected, set(counts) ^ expected
    assert all(count == 1 for count in counts.values())

    evidence_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    json_cells: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}

    for checkpoint, entry in sorted(
        selected,
        key=lambda item: (
            CHECKPOINTS[item[0]]["rank"],
            WORKLOADS.index(item[1]["workload"]["name"]),
        ),
    ):
        workload = entry["workload"]["name"]
        source_path = entry["historical_recovery"]["source_path"]
        directory = ROOT / source_dir_string(source_path)
        raw_path = directory / "raw_benchmark_result.json"
        raw = load_json(raw_path)
        resolved_path = directory / "resolved_same_spec.json"
        client = (
            load_json(resolved_path).get("resolved_client_parameters", {})
            if resolved_path.exists()
            else entry.get("same_spec", {}).get("resolved_client_parameters", {})
        )
        suite_path = directory / "repeat_suite.json"
        suite = load_json(suite_path) if suite_path.exists() else {}
        repeat_dirs = suite.get("repeat_result_dirs", [])
        repeat_count = len(repeat_dirs) if repeat_dirs else 1
        independent_median = bool(
            repeat_count >= 3
            and suite.get("selection", {}).get("median_value") is not None
            and suite.get("selection", {}).get("selected_result_dir")
        )
        request_count, request_unit = request_accounting(workload, raw, client)
        warmup_count, warmup_detail = warmup_accounting(workload, client)
        mode = execution_mode(workload)
        core, plugin = runtime_pair(entry)
        inferred = entry.get("historical_recovery", {}).get("inferred_fields", [])
        metrics = entry["metrics"]
        row = {
            "checkpoint": checkpoint,
            "checkpoint_label": CHECKPOINTS[checkpoint]["label"],
            "workload": workload,
            "core_commit": core,
            "plugin_commit": plugin,
            "source_path": source_path,
            "raw_result_path": str(raw_path.relative_to(ROOT)),
            "execution_mode": mode,
            "real_online_evidence": evidence_detail(mode, raw, directory, repeat_count),
            "benchmark_invocation_repeat_count": repeat_count,
            "request_count_per_invocation": request_count,
            "request_count_unit": request_unit,
            "warmup_count": warmup_count,
            "warmup_detail": warmup_detail,
            "independent_3_repeat_median": independent_median,
            "public_display_label": "representative measured result",
            "internal_evidence_status": (
                "formally hardened: independent 3-invocation median"
                if independent_median
                else "temporary evidence: single invocation; formal statistical hardening pending"
            ),
            "repeat_suite_path": str(suite_path.relative_to(ROOT))
            if suite_path.exists()
            else "",
            "repeat_group_present": bool(entry.get("repeat_group")),
            "canonical_aggregate_present": bool(entry.get("canonical_aggregate")),
            "inferred_fields": ";".join(inferred),
            "exact_spec_hash": entry["same_spec"]["resolved_spec_hash"],
            "snapshot_data_source_label": entry.get("metadata", {}).get(
                "data_source", ""
            ),
            "evidence_grade": evidence_grade(directory, independent_median),
            **{field: metrics.get(field) for field in METRIC_FIELDS},
        }
        evidence_rows.append(row)
        indexed[(checkpoint, workload)] = row
        json_cells.append({**row, "inferred_fields": inferred})

    for workload in WORKLOADS:
        primary_metric, direction = PRIMARY_METRICS[workload]
        first = indexed[("C1", workload)][primary_metric]
        last = indexed[("C3", workload)][primary_metric]
        raw_delta = raw_change_pct(first, last)
        benefit_delta = (
            None
            if raw_delta is None
            else (raw_delta if direction == "higher" else -raw_delta)
        )
        for checkpoint in CHECKPOINTS:
            cell = indexed[(checkpoint, workload)]
            raw_rows.append(
                {
                    "workload": workload,
                    "checkpoint": checkpoint,
                    "core_commit": cell["core_commit"],
                    "plugin_commit": cell["plugin_commit"],
                    "primary_metric": primary_metric,
                    "primary_direction": direction,
                    "throughput_tps": cell["throughput_tps"],
                    "ttft_ms": cell["ttft_ms"],
                    "tbt_ms": cell["tbt_ms"],
                    "error_rate": cell["error_rate"],
                    "m1_to_m3_primary_raw_change_pct": raw_delta
                    if checkpoint == "C3"
                    else "",
                    "m1_to_m3_primary_benefit_pct": benefit_delta
                    if checkpoint == "C3"
                    else "",
                    "exact_spec_hash": cell["exact_spec_hash"],
                }
            )

    independent_cells = sum(row["independent_3_repeat_median"] for row in evidence_rows)
    online_cells = sum(row["execution_mode"] == "online" for row in evidence_rows)
    offline_cells = len(evidence_rows) - online_cells
    grades: dict[str, int] = {}
    for row in evidence_rows:
        grade = row["evidence_grade"].split(":", 1)[0]
        grades[grade] = grades.get(grade, 0) + 1

    audit = {
        "schema_version": "stable-trend-ppt-audit/v1",
        "generated_from_commit": "f67475ec32692c9f65dbabe80a37c50905bf6443",  # pragma: allowlist secret
        "scope": "Ascend 910B2, one chip, nine workloads, three version health checkpoints",
        "checkpoint_semantics": "historical version health checkpoints; not capability milestones and not current latest",
        "checkpoint_pairs": CHECKPOINTS,
        "primary_metric_policy": {
            workload: {"metric": metric, "direction": direction}
            for workload, (metric, direction) in PRIMARY_METRICS.items()
        },
        "summary": {
            "cells": len(evidence_rows),
            "online_cells": online_cells,
            "offline_cells": offline_cells,
            "independent_3_repeat_median_cells": independent_cells,
            "single_invocation_cells": len(evidence_rows) - independent_cells,
            "internally_hardened_cells": independent_cells,
            "internally_pending_hardening_cells": len(evidence_rows)
            - independent_cells,
            "evidence_grade_counts": grades,
        },
        "capability_coverage": CAPABILITY_COVERAGE,
        "cells": json_cells,
    }

    baseline_rows: list[dict[str, Any]] = []
    for workload in WORKLOADS:
        candidates = [
            entry
            for entry in entries
            if entry.get("engine") == "vllm"
            and entry.get("workload", {}).get("name") == workload
            and entry.get("hardware", {}).get("chip_model") == "910B2"
            and entry.get("hardware", {}).get("chip_count") == 1
            and entry.get("same_spec", {})
            .get("spec_id", "")
            .startswith("official-ascend-jan-2026-v0.18.0-")
        ]
        baseline = candidates[0] if candidates else None
        source_path = (
            baseline.get("historical_recovery", {}).get("source_path", "")
            if baseline
            else ""
        )
        source_directory = (
            ROOT / source_dir_string(source_path) if source_path else None
        )
        repeat_suite_path = (
            source_directory / "repeat_suite.json" if source_directory else None
        )
        strict_three_repeat = bool(repeat_suite_path and repeat_suite_path.exists())
        if workload == "random-latency":
            strict_path = (
                ROOT / "reports/historical-recovery-evidence/reruns/"
                "bcf2be9612-e18643f8a4-random-latency/repeat_suite.json"
            )
            strict_three_repeat = strict_path.exists()
            if strict_three_repeat:
                source_path = str(strict_path.relative_to(ROOT))
        if strict_three_repeat:
            status = "strict 3-repeat baseline available"
        elif baseline:
            status = "aggregate snapshot only; not strict repeat evidence"
        else:
            status = "no admitted exact-spec baseline result"
        baseline_rows.append(
            {
                "workload": workload,
                "baseline_core_commit": "bcf2be96120005e9aea171927f85055a6a5c0cf6",  # pragma: allowlist secret
                "baseline_plugin_commit": "e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
                "baseline_status": status,
                "existing_source_path": source_path,
                "exact_spec_hash": baseline.get("same_spec", {}).get(
                    "resolved_spec_hash", ""
                )
                if baseline
                else "",
                "required_independent_invocations": 0 if strict_three_repeat else 3,
            }
        )
    audit["baseline_gap"] = baseline_rows
    audit["summary"]["strict_baseline_workloads"] = sum(
        row["required_independent_invocations"] == 0 for row in baseline_rows
    )
    audit["summary"]["baseline_invocations_required"] = sum(
        row["required_independent_invocations"] for row in baseline_rows
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "stable_trend_evidence_matrix.csv",
        list(evidence_rows[0]),
        evidence_rows,
    )
    write_csv(OUTPUT_DIR / "stable_trend_raw_metrics.csv", list(raw_rows[0]), raw_rows)
    write_csv(
        OUTPUT_DIR / "capability_coverage.csv",
        list(CAPABILITY_COVERAGE[0]),
        CAPABILITY_COVERAGE,
    )
    write_csv(OUTPUT_DIR / "baseline_gap.csv", list(baseline_rows[0]), baseline_rows)
    (OUTPUT_DIR / "stable_trend_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    attachment_names = (
        "README.md",
        "baseline_gap.csv",
        "capability_coverage.csv",
        "stable_trend_audit.json",
        "stable_trend_evidence_matrix.csv",
        "stable_trend_raw_metrics.csv",
    )
    checksum_lines = []
    for name in attachment_names:
        path = OUTPUT_DIR / name
        if path.exists():
            checksum_lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}"
            )
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "cells": len(evidence_rows),
                "online_cells": online_cells,
                "offline_cells": offline_cells,
                "independent_3_repeat_median_cells": independent_cells,
                "output_dir": str(OUTPUT_DIR),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
