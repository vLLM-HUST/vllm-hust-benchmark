#!/usr/bin/env python3
"""Issue #105 data cleanup: prune non-active-spec entries from public leaderboard snapshots.

Active fixed-target profiles (per docs/official-baselines/ + issue #105):
  - core-text-14b: random-online, Qwen/Qwen2.5-14B-Instruct, FP16, 0.6/32768
  - coder-14b: instructcoder-online, Qwen/Qwen2.5-Coder-14B-Instruct, FP16, 0.6/32768
  - vision-7b: visionarena-online, Qwen/Qwen2.5-VL-7B-Instruct, FP16, 0.6/32768

Outputs:
  - leaderboard-data/snapshots/leaderboard_single.json (cleaned)
  - leaderboard-data/snapshots/leaderboard_multi.json (cleaned)
  - leaderboard-data/snapshots/admission_report.json
  - leaderboard-data/snapshots/pre_cleanup_freeze.json (copy of pre-cleanup single)
  - leaderboard-data/snapshots/rejected_superseded_report.json

Quarantine policy (issue #105, fail closed):
  - keep: entry whose (workload, model, precision, gpu_mem, max_model_len) matches one of 3 active
    specs AND passes the admission gate (metadata.verified is True, metrics.peak_mem_mb > 0,
    metrics.error_rate < 1.0, same_spec.resolved_spec_hash is 64-char hex, and both
    metadata.runtime_provenance.{engine,plugin}.commit are 40-char hex SHA)
  - quarantine: everything else (specialty/out-of-scope/config drift/missing fields/admission
    gate failure). Missing gpu_mem or max_len is a fail-closed rejection, not a skipped check.
  - The "current_main" entries (engine_version containing e4ce33646f or matching active spec) are
    kept only when they also pass the admission gate, so that paired evidence (baseline vs
    current_main) is consumable from the public snapshot without reintroducing unverifiable data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "leaderboard-data" / "snapshots"

# Issue #105 fail-closed admission helpers. A leaderboard entry is only
# eligible for the public snapshot when it carries a verified flag, real
# peak memory, a sub-100% error rate, a resolved spec hash, and 40-char
# hex SHA provenance for both engine and plugin commits. Missing or
# invalid values => quarantine (never keep).
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Active fixed-target profiles: (workload_name, model_repo_id, precision, gpu_mem_util, max_model_len)
ACTIVE_PROFILES = [
    {
        "profile": "core-text-14b",
        "workload": "random-online",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "precision": "FP16",
        "gpu_memory_utilization": 0.6,
        "max_model_len": 32768,
    },
    {
        "profile": "coder-14b",
        "workload": "instructcoder-online",
        "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "precision": "FP16",
        "gpu_memory_utilization": 0.6,
        "max_model_len": 32768,
    },
    {
        "profile": "vision-7b",
        "workload": "visionarena-online",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "precision": "FP16",
        "gpu_memory_utilization": 0.6,
        "max_model_len": 32768,
    },
]


def _get(entry: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = entry
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _entry_workload(entry: dict) -> str:
    w = entry.get("workload")
    if isinstance(w, dict):
        return w.get("name", "") or ""
    if isinstance(w, str):
        return w
    return ""


def _entry_model(entry: dict) -> str:
    m = entry.get("model")
    if isinstance(m, dict):
        return m.get("name", "") or m.get("repo_id", "") or ""
    if isinstance(m, str):
        return m
    return ""


def _entry_precision(entry: dict) -> str:
    m = entry.get("model")
    if isinstance(m, dict):
        return m.get("precision", "") or ""
    return ""


def _entry_gpu_mem(entry: dict) -> Any:
    # Try several known locations across schema versions
    for path in (
        ("environment", "gpu_memory_utilization"),
        ("environment", "server_parameters", "gpu_memory_utilization"),
        ("cluster", "gpu_memory_utilization"),
        ("constraints", "gpu_memory_utilization"),
        ("metadata", "gpu_memory_utilization"),
    ):
        v = _get(entry, *path)
        if v not in (None, "", 0):
            return v
    # Try top-level
    return entry.get("gpu_memory_utilization")


def _entry_max_model_len(entry: dict) -> Any:
    for path in (
        ("environment", "max_model_len"),
        ("environment", "server_parameters", "max_model_len"),
        ("cluster", "max_model_len"),
        ("constraints", "max_model_len"),
        ("metadata", "max_model_len"),
    ):
        v = _get(entry, *path)
        if v not in (None, "", 0):
            return v
    return entry.get("max_model_len")


def _entry_target_id(entry: dict) -> str:
    ss = entry.get("same_spec") or {}
    if isinstance(ss, dict):
        spec_id = ss.get("spec_id", "") or ""
        if spec_id:
            return spec_id
    return entry.get("target_id", "") or ""


def _entry_engine_version(entry: dict) -> str:
    return entry.get("engine_version", "") or ""


def _match_active_profile(entry: dict) -> dict | None:
    """Return matching profile dict if entry aligns with one of the 3 active specs, else None.

    Alignment is strict (issue #105 "fail closed"): workload + model + precision + gpu_mem +
    max_model_len must all be present AND match. Missing gpu_mem or max_len is a fail-closed
    rejection (returns None) rather than a skipped comparison, so an entry with absent
    config fields can never be admitted purely on workload/model shape.
    """
    workload = _entry_workload(entry)
    model = _entry_model(entry)
    precision = _entry_precision(entry)
    gpu_mem = _entry_gpu_mem(entry)
    max_len = _entry_max_model_len(entry)

    # Fail closed: gpu_mem and max_len are required config fields. Missing
    # values must not fall through to a profile match.
    if gpu_mem is None or max_len is None:
        return None

    for prof in ACTIVE_PROFILES:
        if workload != prof["workload"]:
            continue
        if model != prof["model"]:
            continue
        if precision and prof["precision"] and precision != prof["precision"]:
            continue
        try:
            if abs(float(gpu_mem) - float(prof["gpu_memory_utilization"])) > 1e-6:
                continue
        except (TypeError, ValueError):
            # Invalid (non-numeric) => fail closed
            return None
        try:
            if int(max_len) != int(prof["max_model_len"]):
                continue
        except (TypeError, ValueError):
            return None
        return prof
    return None


def _passes_admission_gate(entry: dict) -> tuple[bool, list[str]]:
    """Issue #105 fail-closed admission gate for a single leaderboard entry.

    Returns (passes, failure_fields). A passing entry must carry:
    - ``metadata.verified`` is exactly ``True`` (not None/False)
    - ``metrics.peak_mem_mb`` is a positive number (0 indicates missing/invalid)
    - ``metrics.error_rate`` is present and below 1.0 (100% errors = invalid)
    - ``same_spec.resolved_spec_hash`` is a 64-char hex string
    - ``metadata.runtime_provenance.engine.commit`` is a 40-char hex SHA
    - ``metadata.runtime_provenance.plugin.commit`` is a 40-char hex SHA

    Any missing or invalid field => the entry is rejected (fail closed),
    per the project hard constraint "Missing registry or required fields in
    aggregator must result in fail closed".
    """
    failures: list[str] = []

    metadata = entry.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    verified = metadata.get("verified")
    if verified is not True:
        failures.append("metadata.verified")

    metrics = entry.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    peak_mem = metrics.get("peak_mem_mb")
    if not (isinstance(peak_mem, (int, float)) and peak_mem > 0):
        failures.append("metrics.peak_mem_mb")
    error_rate = metrics.get("error_rate")
    if not (isinstance(error_rate, (int, float)) and 0 <= error_rate < 1):
        failures.append("metrics.error_rate")

    same_spec = entry.get("same_spec")
    same_spec = same_spec if isinstance(same_spec, dict) else {}
    resolved_hash = same_spec.get("resolved_spec_hash")
    if not (isinstance(resolved_hash, str) and _HEX64_RE.match(resolved_hash)):
        failures.append("same_spec.resolved_spec_hash")

    provenance = metadata.get("runtime_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    engine_prov = provenance.get("engine")
    engine_prov = engine_prov if isinstance(engine_prov, dict) else {}
    engine_commit = engine_prov.get("commit")
    if not (isinstance(engine_commit, str) and _HEX40_RE.match(engine_commit)):
        failures.append("metadata.runtime_provenance.engine.commit")
    plugin_prov = provenance.get("plugin")
    plugin_prov = plugin_prov if isinstance(plugin_prov, dict) else {}
    plugin_commit = plugin_prov.get("commit")
    if not (isinstance(plugin_commit, str) and _HEX40_RE.match(plugin_commit)):
        failures.append("metadata.runtime_provenance.plugin.commit")

    return (len(failures) == 0, failures)


def _build_admission_entry(
    entry: dict,
    disposition: str,
    profile: dict | None,
    reason: str,
    drift_fields: list[str],
    missing_fields: list[str],
) -> dict:
    actual_config = {
        "model": _entry_model(entry) or None,
        "hardware_chip_model": _get(entry, "hardware", "chip_model"),
        "chip_count": _get(entry, "hardware", "chip_count"),
        "model_precision": _entry_precision(entry) or None,
        "gpu_memory_utilization": _entry_gpu_mem(entry),
        "max_model_len": _entry_max_model_len(entry),
        "workload_name": _entry_workload(entry) or None,
        "engine_version": _entry_engine_version(entry) or None,
        "same_spec_id": _entry_target_id(entry) or None,
    }
    required_config = None
    if profile:
        required_config = {
            "model": profile["model"],
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "model_precision": profile["precision"],
            "gpu_memory_utilization": profile["gpu_memory_utilization"],
            "max_model_len": profile["max_model_len"],
            "workload_name": profile["workload"],
        }
    return {
        "entry_id": entry.get("entry_id"),
        "scenario": _entry_workload(entry) or None,
        "actual_config": actual_config,
        "artifact_path": None,
        "profile_name": profile["profile"] if profile else None,
        "required_config": required_config,
        "missing_fields": missing_fields,
        "drift_fields": drift_fields,
        "disposition": disposition,
        "reason": reason,
    }


def _classify_entry(entry: dict) -> tuple[dict | None, str, str, list[str], list[str]]:
    """Return (profile, disposition, reason, drift_fields, missing_fields)."""
    missing = []
    drift = []
    if not _entry_workload(entry):
        missing.append("workload.name")
    if not _entry_model(entry):
        missing.append("model.name")
    if not _entry_precision(entry):
        missing.append("model.precision")
    if _entry_gpu_mem(entry) is None:
        missing.append("gpu_memory_utilization")
    if _entry_max_model_len(entry) is None:
        missing.append("max_model_len")

    prof = _match_active_profile(entry)
    if prof is not None:
        # Issue #105 fail-closed: profile shape match alone is not enough.
        # The entry must also pass the admission gate (verified flag, real
        # peak memory, sub-100% error rate, resolved spec hash, 40-char hex
        # engine+plugin commits). Any missing field => quarantine.
        admission_ok, admission_failures = _passes_admission_gate(entry)
        if admission_ok:
            return prof, "keep", None, [], []
        return (
            prof,
            "quarantine",
            f"admission_gate_failure: {','.join(admission_failures)}",
            [],
            admission_failures,
        )

    # Build drift fields if a profile could be inferred from workload alone
    inferred_prof = None
    wl = _entry_workload(entry)
    for p in ACTIVE_PROFILES:
        if p["workload"] == wl:
            inferred_prof = p
            break
    if inferred_prof:
        if _entry_model(entry) and _entry_model(entry) != inferred_prof["model"]:
            drift.append("model")
        if (
            _entry_precision(entry)
            and _entry_precision(entry) != inferred_prof["precision"]
        ):
            drift.append("model.precision")
        try:
            if (
                _entry_gpu_mem(entry) is not None
                and abs(
                    float(_entry_gpu_mem(entry))
                    - float(inferred_prof["gpu_memory_utilization"])
                )
                > 1e-6
            ):
                drift.append("gpu_memory_utilization")
        except (TypeError, ValueError):
            pass
        try:
            if _entry_max_model_len(entry) is not None and int(
                _entry_max_model_len(entry)
            ) != int(inferred_prof["max_model_len"]):
                drift.append("max_model_len")
        except (TypeError, ValueError):
            pass
        if drift:
            return (
                inferred_prof,
                "quarantine",
                f"config drift: {','.join(drift)}",
                drift,
                missing,
            )
        if missing:
            return (
                inferred_prof,
                "quarantine",
                f"missing fields: {','.join(missing)}",
                [],
                missing,
            )
        return (
            inferred_prof,
            "quarantine",
            "out-of-scope (specialty workload)",
            [],
            missing,
        )

    # No workload match: specialty or unknown
    if missing:
        return None, "quarantine", f"missing fields: {','.join(missing)}", [], missing
    return None, "quarantine", "out-of-scope (specialty workload)", [], missing


def cleanup_snapshot(
    snapshot_path: Path,
    output_path: Path,
    admission_entries: list[dict],
    rejected_entries: list[dict],
) -> tuple[int, int]:
    """Filter snapshot to keep only active-spec entries. Returns (kept, removed)."""
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{snapshot_path}: expected JSON list, got {type(data).__name__}"
        )

    kept: list[dict] = []
    for entry in data:
        prof, disposition, reason, drift, missing = _classify_entry(entry)
        admission_entries.append(
            _build_admission_entry(entry, disposition, prof, reason, drift, missing)
        )
        if disposition == "keep":
            kept.append(entry)
        else:
            rejected_entries.append(
                {
                    "entry_id": entry.get("entry_id"),
                    "scenario": _entry_workload(entry) or None,
                    "model": _entry_model(entry) or None,
                    "engine_version": _entry_engine_version(entry) or None,
                    "same_spec_id": _entry_target_id(entry) or None,
                    "disposition": disposition,
                    "reason": reason,
                    "drift_fields": drift,
                    "missing_fields": missing,
                }
            )

    output_path.write_text(
        json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(kept), len(data) - len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SNAPSHOT_DIR,
        help="Directory containing leaderboard_single.json and leaderboard_multi.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass idempotency guard (allow re-running on already-cleaned input). "
        "Use only when intentionally regenerating from a restored original input.",
    )
    args = parser.parse_args()

    snapshot_dir: Path = args.snapshot_dir
    single_path = snapshot_dir / "leaderboard_single.json"
    multi_path = snapshot_dir / "leaderboard_multi.json"

    if not single_path.is_file():
        print(f"ERROR: {single_path} not found", file=sys.stderr)
        return 2

    # Idempotency guard (issue #105 reviewer round 3+4): if the input has
    # already been cleaned (0 entries) but pre_cleanup_freeze.json shows a
    # prior run with > 0 entries, refuse to proceed unless --force is given.
    # This prevents a second run from overwriting the real quarantine with
    # an empty one.  The guard is fail-closed: a corrupted, unreadable, or
    # schema-violating freeze file is treated as evidence that a prior run
    # existed but cannot be verified — the script refuses to continue rather
    # than silently treating it as a first run.  --force additionally
    # requires the current input to be non-empty (i.e. restored from a real
    # original snapshot) so that the guard cannot be trivially bypassed on
    # an empty input.
    input_data = json.loads(single_path.read_text(encoding="utf-8"))
    input_is_empty = isinstance(input_data, list) and len(input_data) == 0
    freeze_path = snapshot_dir / "pre_cleanup_freeze.json"
    freeze_exists = freeze_path.is_file()

    if args.force:
        # --force is an explicit recovery flow: require the current input to
        # be non-empty so the guard cannot be bypassed on an already-cleaned
        # empty input.  An empty input with --force is only allowed when no
        # freeze file exists (genuine first run on empty data).
        if input_is_empty and freeze_exists:
            print(
                f"ERROR: --force on an empty {single_path.name} with an existing "
                f"pre_cleanup_freeze.json is refused. Restore the original "
                f"non-empty leaderboard before using --force to regenerate.",
                file=sys.stderr,
            )
            return 4
    else:
        if input_is_empty and freeze_exists:
            # Fail-closed validation of the freeze file: any deviation from
            # the expected schema is grounds for refusal.
            try:
                freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"ERROR: Idempotency guard — {single_path.name} has 0 entries "
                    f"and pre_cleanup_freeze.json exists but cannot be parsed "
                    f"({exc}). Refusing to proceed: a prior cleanup run likely "
                    f"occurred but its freeze record is corrupted. Restore the "
                    f"original leaderboard or use --force with a restored "
                    f"non-empty input to override.",
                    file=sys.stderr,
                )
                return 3
            if not isinstance(freeze, dict):
                print(
                    f"ERROR: Idempotency guard — pre_cleanup_freeze.json is not a "
                    f"JSON object (got {type(freeze).__name__}). Refusing to "
                    f"proceed: freeze record is malformed.",
                    file=sys.stderr,
                )
                return 3
            if "entry_count" not in freeze:
                print(
                    "ERROR: Idempotency guard — pre_cleanup_freeze.json is "
                    "missing the required 'entry_count' field. Refusing to "
                    "proceed: cannot verify prior run history.",
                    file=sys.stderr,
                )
                return 3
            prior_count = freeze["entry_count"]
            if not isinstance(prior_count, int) or isinstance(prior_count, bool):
                print(
                    f"ERROR: Idempotency guard — pre_cleanup_freeze.json "
                    f"'entry_count' has wrong type {type(prior_count).__name__} "
                    f"(expected int). Refusing to proceed: freeze record is "
                    f"malformed.",
                    file=sys.stderr,
                )
                return 3
            if prior_count > 0:
                print(
                    f"ERROR: Idempotency guard — {single_path.name} has 0 "
                    f"entries but pre_cleanup_freeze.json records "
                    f"{prior_count} entries from a prior run. Refusing to "
                    f"overwrite quarantine with empty result. Restore the "
                    f"original leaderboard or use --force with a restored "
                    f"non-empty input to override.",
                    file=sys.stderr,
                )
                return 3

    # Pre-cleanup freeze (single snapshot only, per issue #105 step 1)
    pre_cleanup = json.loads(single_path.read_text(encoding="utf-8"))
    pre_cleanup_summary = {
        "schema_version": "pre-cleanup-freeze/v1",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(single_path.relative_to(REPO_ROOT))
        if single_path.is_relative_to(REPO_ROOT)
        else str(single_path),
        "entry_count": len(pre_cleanup),
        "entry_ids": [e.get("entry_id") for e in pre_cleanup],
        "frozen_entries": pre_cleanup,
    }

    admission_entries: list[dict] = []
    rejected_entries: list[dict] = []

    if args.dry_run:
        # In-memory classification only
        temp_data = json.loads(single_path.read_text(encoding="utf-8"))
        kept_count = 0
        for entry in temp_data:
            prof, disposition, reason, drift, missing = _classify_entry(entry)
            admission_entries.append(
                _build_admission_entry(entry, disposition, prof, reason, drift, missing)
            )
            if disposition == "keep":
                kept_count += 1
            else:
                rejected_entries.append(
                    {"entry_id": entry.get("entry_id"), "reason": reason}
                )
        print(
            f"[dry-run] leaderboard_single.json: {len(temp_data)} entries -> keep {kept_count}, quarantine {len(temp_data) - kept_count}"
        )
        if multi_path.is_file():
            multi_data = json.loads(multi_path.read_text(encoding="utf-8"))
            multi_kept = sum(1 for e in multi_data if _classify_entry(e)[1] == "keep")
            print(
                f"[dry-run] leaderboard_multi.json: {len(multi_data)} entries -> keep {multi_kept}"
            )
        return 0

    # Real run: write cleaned snapshots + reports
    single_kept, single_removed = cleanup_snapshot(
        single_path, single_path, admission_entries, rejected_entries
    )

    multi_kept = multi_removed = 0
    if multi_path.is_file():
        multi_kept, multi_removed = cleanup_snapshot(
            multi_path, multi_path, admission_entries, rejected_entries
        )

    # Write pre_cleanup_freeze.json (record of pre-cleanup state)
    (snapshot_dir / "pre_cleanup_freeze.json").write_text(
        json.dumps(pre_cleanup_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write quarantine_leaderboard_entries.json (full data of removed entries)
    # Issue #105: original submission artifacts are not physically deleted; they
    # remain in submissions/ dirs.  This file preserves the full leaderboard
    # entries that were removed from the public snapshot, for audit without
    # rescanning submissions/ directories.
    quarantine_entries = [e for e in pre_cleanup if _classify_entry(e)[1] != "keep"]
    quarantine_report = {
        "schema_version": "quarantine-leaderboard-entries/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(single_path.relative_to(REPO_ROOT))
        if single_path.is_relative_to(REPO_ROOT)
        else str(single_path),
        "quarantined_count": len(quarantine_entries),
        "policy": (
            "Issue #105: entries not matching active fixed-target specs are "
            "removed from public leaderboard snapshots but preserved here for "
            "audit. Original submission artifacts remain in submissions/ dirs."
        ),
        "quarantined_entries": quarantine_entries,
    }
    (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
        json.dumps(quarantine_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write admission_report.json
    admission_report = {
        "schema_version": "admission-report/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registry_version": "official-ascend-jan-2026-v0.18.0",
        "active_profiles": [p["profile"] for p in ACTIVE_PROFILES],
        "summary": {
            "leaderboard_single": {
                "total": single_kept + single_removed,
                "keep": single_kept,
                "quarantine": single_removed,
            },
            "leaderboard_multi": {
                "total": multi_kept + multi_removed,
                "keep": multi_kept,
                "quarantine": multi_removed,
            },
        },
        "entries": admission_entries,
    }
    (snapshot_dir / "admission_report.json").write_text(
        json.dumps(admission_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write rejected_superseded_report.json (conforms to schemas/rejected_superseded_report_v1.schema.json)
    rejected_submissions = [
        {
            "dir": str(r.get("entry_id") or ""),
            "reason": "exclusion_match",
            "detail": (
                f"{r.get('reason', 'unknown')}; "
                f"scenario={r.get('scenario', '')}, model={r.get('model', '')}, "
                f"engine_version={r.get('engine_version', '')}, "
                f"drift_fields={r.get('drift_fields', [])}, "
                f"missing_fields={r.get('missing_fields', [])}"
            ),
        }
        for r in rejected_entries
    ]
    rejected_report = {
        "schema_version": "rejected-superseded-report/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rejected_submissions": rejected_submissions,
        "superseded_entries": [],
        "excluded_plugin_commits": [],
    }
    (snapshot_dir / "rejected_superseded_report.json").write_text(
        json.dumps(rejected_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Cleanup complete:\n"
        f"  leaderboard_single.json: keep {single_kept}, quarantine {single_removed}\n"
        f"  leaderboard_multi.json:  keep {multi_kept}, quarantine {multi_removed}\n"
        f"  pre_cleanup_freeze.json:  {pre_cleanup_summary['entry_count']} entries frozen\n"
        f"  admission_report.json:   {len(admission_entries)} entries classified\n"
        f"  rejected_superseded_report.json: {len(rejected_entries)} rejected entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
