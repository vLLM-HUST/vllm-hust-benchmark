#!/usr/bin/env python3
"""Sample peak HBM usage for an explicitly scoped set of Ascend devices."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEVICE_ROW = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|")
HBM_CELL = re.compile(r"(\d+)\s*/\s*(\d+)")


def parse_hbm_usage(output: str) -> dict[int, tuple[int, int]]:
    """Return physical NPU id -> (used MiB, capacity MiB)."""
    lines = output.splitlines()
    parsed: dict[int, tuple[int, int]] = {}
    for index, line in enumerate(lines[:-1]):
        match = DEVICE_ROW.match(line)
        if not match:
            continue
        cells = lines[index + 1].split("|")
        if len(cells) < 3:
            continue
        hbm_matches = HBM_CELL.findall(cells[-2])
        if hbm_matches:
            used, capacity = hbm_matches[-1]
            parsed[int(match.group(1))] = (
                int(used),
                int(capacity),
            )
    return parsed


def _device_scope(value: str | None) -> list[int]:
    raw = value or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_VISIBLE_DEVICES"
    )
    if not raw:
        raise ValueError(
            "an explicit --devices, ASCEND_RT_VISIBLE_DEVICES, or "
            "ASCEND_VISIBLE_DEVICES scope is required"
        )
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"invalid Ascend device scope: {raw}")
    return devices


def _write_summary(
    output_path: Path,
    *,
    devices: list[int],
    per_device_peak: dict[int, int],
    capacities: dict[int, int],
    samples: int,
    failures: int,
) -> None:
    payload = {
        "schema_version": "ascend-peak-hbm-evidence/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "npu-smi info",
        "devices": devices,
        "sample_count": samples,
        "sample_failure_count": failures,
        "peak_hbm_mb": sum(per_device_peak.values()),
        "per_device_peak_hbm_mb": {
            str(device): per_device_peak.get(device, 0) for device in devices
        },
        "per_device_capacity_mb": {
            str(device): capacities.get(device) for device in devices
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--npu-smi", default="npu-smi")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    try:
        devices = _device_scope(args.devices)
    except ValueError as error:
        parser.error(str(error))

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    per_device_peak = {device: 0 for device in devices}
    capacities: dict[int, int] = {}
    samples = 0
    failures = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    while not stop:
        try:
            result = subprocess.run(
                [args.npu_smi, "info"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            usage = parse_hbm_usage(result.stdout)
            if any(device not in usage for device in devices):
                raise RuntimeError("npu-smi output omitted a scoped device")
            for device in devices:
                used, capacity = usage[device]
                per_device_peak[device] = max(per_device_peak[device], used)
                capacities[device] = capacity
            samples += 1
        except (OSError, subprocess.SubprocessError, RuntimeError):
            failures += 1
        _write_summary(
            args.output,
            devices=devices,
            per_device_peak=per_device_peak,
            capacities=capacities,
            samples=samples,
            failures=failures,
        )
        if not stop:
            time.sleep(args.interval_seconds)
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
