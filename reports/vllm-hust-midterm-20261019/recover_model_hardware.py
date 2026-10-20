"""Project explicitly selected per-device HBM databases onto request intervals."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from integrate_hbm_samples import check_export, has_complete_hbm_coverage, integrate


def request_phase(receipt, result):
    offsets = [
        receipt[f"bench_{edge}_wall"] - receipt[f"bench_{edge}_monotonic"]
        for edge in ("started", "ended")
    ]
    if abs(offsets[0] - offsets[1]) > 0.001:
        raise ValueError("Wall/monotonic clock offset changed by more than 1 ms")
    starts, latencies = result["start_times"], result["latencies"]
    if len(starts) != 200 or len(latencies) != 200:
        raise ValueError("Expected exactly 200 request timestamps")
    if result["completed"] != 200 or result["failed"] != 0:
        raise ValueError("Incomplete request cohort")
    begin = min(starts)
    end = max(s + latency for s, latency in zip(starts, latencies, strict=True))
    if (
        not receipt["bench_started_monotonic"]
        <= begin
        < end
        <= receipt["bench_ended_monotonic"]
    ):
        raise ValueError("Request interval falls outside client lifecycle")
    offset = sum(offsets) / 2
    return dict(
        name="first_request_to_last_response",
        start_wall=begin + offset,
        end_wall=end + offset,
        duration_seconds=end - begin,
        wall_monotonic_offset_difference_seconds=abs(offsets[0] - offsets[1]),
        method="Request perf_counter timestamps mapped to wall time by mean of controller start/end clock offsets.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--db", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.run / "receipt.json").read_text())
    result = json.loads((args.run / "mmlu-pro.json").read_text())
    if receipt["status"] != "measured" or not receipt["all_devices_released"]:
        raise ValueError("Workload or resource-release check failed")
    phase = request_phase(receipt, result)
    devices = []
    seen = set()
    for path in args.db:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            samples = connection.execute(
                "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
            ).fetchall()
            check = check_export(
                samples, connection.execute("SELECT * FROM HBMbwData").fetchall()
            )
        ids = {row[0] for row in samples}
        if len(ids) != 1 or seen & ids:
            raise ValueError("Select exactly one database per device")
        seen |= ids
        counts = integrate(samples, phase["start_wall"] * 1e9, phase["end_wall"] * 1e9)
        devices.append(
            dict(
                db=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decoder_crosscheck=check,
                **counts,
            )
        )
    if len(seen) != 2:
        raise ValueError("TP2 requires two distinct device samples")
    complete = all(has_complete_hbm_coverage(d) for d in devices)
    files = [args.run / name for name in ("receipt.json", "mmlu-pro.json")]
    output = dict(
        schema="midterm-model-hardware-hbm/v1",
        phase=phase,
        devices=devices,
        status="complete_hbm_coverage" if complete else "incomplete_hbm_coverage",
        decoder=dict(
            path=str(args.decoder_source),
            sha256=hashlib.sha256(args.decoder_source.read_bytes()).hexdigest(),
            unit="32 bytes/event; per-interval HBM counts",
        ),
        input_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        total_prorated_bytes=sum(d["prorated_bytes"] for d in devices)
        if complete
        else None,
        total_fully_contained_bytes=sum(d["fully_contained_bytes"] for d in devices)
        if complete
        else None,
        total_intersecting_interval_bytes=sum(
            d["intersecting_interval_bytes"] for d in devices
        )
        if complete
        else None,
        request_completed=result["completed"],
        request_failed=result["failed"],
        observed_profiled_output_throughput=result["output_throughput"],
        model_effective_flops=None,
        attainment_percent=None,
        scope="Hardware-sampled Microflow B1 request cohort; diagnostic only, no replacement of unprofiled B1 performance. Device-wide incidental traffic included.",
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "devices"}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
