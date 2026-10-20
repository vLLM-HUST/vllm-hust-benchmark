"""Integrate hardware HBM interval counts with explicit boundary uncertainty."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3


BYTES_PER_EVENT = 32  # CANN HbmModel: HBMC(256) / HBM_EVENT(8).


def has_complete_hbm_coverage(measured):
    expected = {
        (channel, direction) for channel in range(4) for direction in ("read", "write")
    }
    return (
        len(measured["channels"]) == 8
        and {(c["channel"], c["direction"]) for c in measured["channels"]} == expected
        and measured["minimum_channel_coverage"] == 1
    )


def check_export(samples, exported):
    series = defaultdict(list)
    for device, timestamp, count, channel, direction in samples:
        series[device, channel, direction].append((timestamp, count))
    lookup = {(d, c, e, t): b for d, t, b, c, e in exported}
    checked = 0
    for (device, channel, direction), values in series.items():
        values.sort()
        for (previous, _), (current, count) in zip(values, values[1:]):
            if current <= previous:
                raise ValueError("Non-increasing channel timestamps")
            expected = count * BYTES_PER_EVENT / 1024**2 * 1e9 / (current - previous)
            actual = lookup[device, channel, direction, current]
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Hardware decoder and exported bandwidth disagree")
            checked += 1
    if not checked:
        raise ValueError("No bandwidth samples to crosscheck")
    return {"samples_crosschecked": checked, "passed": True}


def integrate(samples, start_ns, end_ns):
    if end_ns <= start_ns:
        raise ValueError("Nonpositive measurement window")
    series = defaultdict(list)
    for device, timestamp, count, channel, direction in samples:
        if count < 0 or direction not in ("read", "write"):
            raise ValueError("Invalid HBM sample")
        series[device, channel, direction].append((timestamp, count))
    channels = []
    for (device, channel, direction), values in sorted(series.items()):
        values.sort()
        lower = upper = 0
        estimate = covered = 0.0
        largest_interval = 0
        for (previous, _), (current, count) in zip(values, values[1:]):
            if current <= previous:
                raise ValueError("Non-increasing channel timestamps")
            overlap = max(0, min(current, end_ns) - max(previous, start_ns))
            if not overlap:
                continue
            amount = count * BYTES_PER_EVENT
            upper += amount
            if previous >= start_ns and current <= end_ns:
                lower += amount
            estimate += amount * overlap / (current - previous)
            covered += overlap
            largest_interval = max(largest_interval, current - previous)
        channels.append(
            dict(
                device=device,
                channel=channel,
                direction=direction,
                fully_contained_bytes=lower,
                intersecting_interval_bytes=upper,
                prorated_bytes=estimate,
                coverage=covered / (end_ns - start_ns),
                largest_interval_ns=largest_interval,
            )
        )
    return dict(
        channels=channels,
        fully_contained_bytes=sum(c["fully_contained_bytes"] for c in channels),
        intersecting_interval_bytes=sum(
            c["intersecting_interval_bytes"] for c in channels
        ),
        prorated_bytes=sum(c["prorated_bytes"] for c in channels),
        minimum_channel_coverage=min((c["coverage"] for c in channels), default=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as connection:
        samples = connection.execute(
            "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
        ).fetchall()
        decoder_check = check_export(
            samples, connection.execute("SELECT * FROM HBMbwData").fetchall()
        )
    result = []
    for phase in json.loads(args.phases.read_text())["phases"]:
        measured = integrate(
            samples, phase["start_wall"] * 1e9, phase["end_wall"] * 1e9
        )
        measured["phase"] = phase
        if "logical_read_bytes" in phase and "logical_write_bytes" in phase:
            measured["logical_copy_bytes"] = (
                phase["logical_read_bytes"] + phase["logical_write_bytes"]
            )
            measured["physical_to_logical_ratio"] = (
                measured["prorated_bytes"] / measured["logical_copy_bytes"]
            )
        measured["prorated_bandwidth_bytes_per_second"] = measured["prorated_bytes"] / (
            phase["end_wall"] - phase["start_wall"]
        )
        result.append(measured)
    source = dict(
        path=str(args.decoder_source),
        sha256=hashlib.sha256(args.decoder_source.read_bytes()).hexdigest(),
        unit="32 bytes/event; raw counts are per sampling interval; bandwidth export uses MiB/s",
    )
    output = dict(
        schema="midterm-hbm-sampler-qualification/v1",
        decoder=source,
        decoder_crosscheck=decoder_check,
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.db, args.phases)
        },
        phases=result,
        method="First sample in each channel is excluded because its start boundary is unknown. Fully contained and all intersecting interval sums bound boundary uncertainty. Prorated estimate assumes uniform rate within each interval.",
        scope="Device-wide physical HBM read+write traffic during supplied intervals, including incidental traffic; not a complete model acceptance metric.",
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
