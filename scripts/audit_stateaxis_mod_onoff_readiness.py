#!/usr/bin/env python3
"""Fail-closed admission audit for StateAxis MOD ON/OFF campaigns.

This is deliberately a preflight, not a benchmark simulator.  A descriptor
whose implementation is import-only cannot produce an ON arm, so the runner
records ``ON_NOT_RUNNABLE`` and leaves every performance field null.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "vllm-hust-benchmark/stateaxis-mod-on-off-preflight/v1"
REQUIRED_METRICS = [
    "request_throughput",
    "output_token_throughput",
    "ttft_p50_p95_p99",
    "tpot_p50_p95_p99",
    "e2e_p50_p95_p99",
    "peak_hbm",
    "error_rate",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or None


def _find_manifest(repo: Path) -> Path:
    matches = sorted(repo.glob("src/**/vllm-hust-extension-v0.3.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Manifest 0.3 descriptor, found {len(matches)}"
        )
    return matches[0]


def inspect_repo(entry: dict[str, Any], repos_root: Path) -> dict[str, Any]:
    slug = entry["repository"].split("/", 1)[1]
    repo = repos_root / slug
    blockers: list[dict[str, str]] = []

    if not repo.is_dir():
        return {
            "mod_id": entry["mod_id"],
            "repository": entry["repository"],
            "pinned_commit": entry["commit"],
            "on_status": "ON_NOT_RUNNABLE",
            "off_status": "NOT_RUN_NO_PAIRED_ON_ARM",
            "performance_result": None,
            "blockers": [{"code": "REPOSITORY_MISSING", "detail": str(repo)}],
        }

    manifest_path = _find_manifest(repo)
    manifest = _load(manifest_path)
    provenance_path = repo / "PROVENANCE.json"
    metadata_path = repo / "MOD_METADATA.json"
    provenance = _load(provenance_path)
    metadata = _load(metadata_path)

    statuses = [row.get("status") for row in manifest.get("implementation", [])]
    if not statuses or any(status != "active" for status in statuses):
        blockers.append(
            {
                "code": "IMPLEMENTATION_NOT_ACTIVE",
                "detail": f"implementation statuses are {statuses or ['missing']}",
            }
        )
    entry_points = manifest.get("activation", {}).get("entry_points", [])
    if not entry_points:
        blockers.append(
            {
                "code": "NO_ACTIVATION_ENTRY_POINT",
                "detail": "Manifest activation.entry_points is empty",
            }
        )
    if provenance.get("implementation_extracted") is not True:
        blockers.append(
            {
                "code": "IMPLEMENTATION_NOT_EXTRACTED",
                "detail": "PROVENANCE.json implementation_extracted is not true",
            }
        )
    activation_contract = metadata.get("lifecycle", {}).get("activation_contract", "")
    if "import-only" in activation_contract.lower():
        blockers.append(
            {
                "code": "ACTIVATION_CONTRACT_REFUSES_ON",
                "detail": activation_contract,
            }
        )

    runnable = not blockers
    return {
        "mod_id": entry["mod_id"],
        "repository": entry["repository"],
        "pinned_commit": entry["commit"],
        "local_head": _git_head(repo),
        "manifest": {
            "path": str(manifest_path.relative_to(repo)),
            "sha256": _sha256(manifest_path),
            "schema_version": manifest.get("schema_version"),
            "implementation_statuses": statuses,
        },
        "provenance_sha256": _sha256(provenance_path),
        "metadata_sha256": _sha256(metadata_path),
        "on_status": "READY_FOR_MATCHED_ON_OFF" if runnable else "ON_NOT_RUNNABLE",
        "off_status": "PENDING_PAIRED_EXECUTION"
        if runnable
        else "NOT_RUN_NO_PAIRED_ON_ARM",
        "performance_result": None,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--repos-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npu-snapshot", type=Path)
    parser.add_argument("--write-repo-evidence", action="store_true")
    args = parser.parse_args()

    catalog = _load(args.catalog)
    entries = catalog["repositories"]
    rows = [inspect_repo(entry, args.repos_root) for entry in entries]
    args.output_dir.mkdir(parents=True, exist_ok=False)

    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    runnable = sum(row["on_status"] == "READY_FOR_MATCHED_ON_OFF" for row in rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "generated_at": generated_at,
        "benchmark_repository": "vLLM-HUST/vllm-hust-benchmark",
        "benchmark_commit": _git_head(Path(__file__).resolve().parents[1]),
        "catalog": {
            "path": str(args.catalog),
            "sha256": _sha256(args.catalog),
            "source_commit": catalog.get("source_commit"),
        },
        "protocol": {
            "phase": "activation_admission",
            "fresh_service_repetitions_per_arm": 3,
            "pair_order": ["OFF/ON", "ON/OFF", "OFF/ON"],
            "same_spec_required": True,
            "correctness_required": True,
            "activation_telemetry_required": True,
            "resource_release_required": True,
            "failure_recovery_required": True,
            "required_metrics": REQUIRED_METRICS,
            "rule": "No OFF-only or inert-import run may be reported as an ON/OFF performance result.",
        },
        "hardware_snapshot": None,
        "summary": {
            "total": len(rows),
            "ready_for_matched_on_off": runnable,
            "on_not_runnable": len(rows) - runnable,
            "performance_pairs_completed": 0,
        },
        "results": rows,
    }
    if args.npu_snapshot:
        report["hardware_snapshot"] = {
            "path": str(args.npu_snapshot),
            "sha256": _sha256(args.npu_snapshot),
        }

    report_path = args.output_dir / "preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table_rows = []
    for row in rows:
        codes = ", ".join(blocker["code"] for blocker in row["blockers"]) or "none"
        table_rows.append((f"`{row['mod_id']}`", f"`{row['on_status']}`", codes))
    table_headers = ("MOD", "ON status", "Blockers")
    table_widths = tuple(
        max(len(table_headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(table_headers))
    )

    def table_line(values: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(table_widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    lines = [
        "# StateAxis MOD ON/OFF benchmark admission",
        "",
        f"- Generated: `{generated_at}`",
        f"- MODs audited: **{len(rows)}**",
        f"- Ready for real matched ON/OFF: **{runnable}**",
        f"- ON not runnable: **{len(rows) - runnable}**",
        "- Performance pairs completed: **0**",
        "",
        "No performance delta is emitted when the ON arm cannot be activated. This is a",
        "fail-closed benchmark result, not a zero-percent performance result.",
        "",
        table_line(table_headers),
        table_line(tuple("-" * width for width in table_widths)),
    ]
    lines.extend(table_line(row) for row in table_rows)
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    if args.write_repo_evidence:
        for row in rows:
            slug = row["repository"].split("/", 1)[1]
            evidence_dir = (
                args.repos_root / slug / "evidence" / "on-off-preflight-20261010"
            )
            evidence_dir.mkdir(parents=True, exist_ok=False)
            payload = {
                "schema_version": SCHEMA,
                "generated_at": generated_at,
                "benchmark_repository": report["benchmark_repository"],
                "benchmark_commit": report["benchmark_commit"],
                "protocol": report["protocol"],
                "result": row,
            }
            (evidence_dir / "RESULT.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (evidence_dir / "README.md").write_text(
                "# ON/OFF benchmark preflight\n\n"
                f"Status: **{row['on_status']}**.\n\n"
                "No performance number was produced because the ON arm is not an active, "
                "Manager-admissible implementation. The OFF arm was intentionally not run "
                "alone because it cannot form a matched pair. See `RESULT.json` for the exact "
                "blockers and frozen identities.\n",
                encoding="utf-8",
            )

    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if runnable == len(rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
