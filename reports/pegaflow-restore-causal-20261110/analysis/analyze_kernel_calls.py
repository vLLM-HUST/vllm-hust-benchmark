"""Pair the private driver kprobe capture; elapsed times do not measure CPU time."""

import argparse
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

FUNCTIONS = [
    "trs_hw_sq_send_task",
    "hal_kernel_trs_chan_send",
    "trs_chan_fill_sqe",
    "trs_chan_flush_sqe_cache",
]
HEADER = re.compile(
    r"^\s*(?P<comm>.+)-(?P<tid>\d+)\s+(?:\(\s*\d+\)\s+)?"
    r"\[(?P<cpu>\d+)\]\s+(?:\S+\s+)?(?P<time>\d+\.\d+):\s+"
    r"(?P<event>[\w:]+):\s*(?P<payload>.*)$"
)


def distribution(values):
    if not values:
        return {"count": 0}
    values = sorted(values)

    def percentile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return dict(
        count=len(values),
        min_ns=values[0],
        max_ns=values[-1],
        p50_ns=percentile(0.5),
        p95_ns=percentile(0.95),
        p99_ns=percentile(0.99),
        sum_ns=sum(values),
    )


def parse_capture(text, group):
    events, invalid, lost = [], [], []
    pattern = re.compile(re.escape(group) + r"_(\d+)_(enter|leave)$")
    for line_number, line in enumerate(text.splitlines(), 1):
        if "LOST" in line.upper() and "EVENT" in line.upper():
            lost.append(line_number)
            continue
        if group not in line:
            continue
        match = HEADER.match(line)
        event = pattern.search(match["event"]) if match else None
        if not event or int(event[1]) >= len(FUNCTIONS):
            invalid.append(line_number)
            continue
        arg = re.search(r"\barg0=(?:0x)?([0-9a-fA-F]+)\b", match["payload"])
        chan = re.search(r"\bchan_id=(\d+)\b", match["payload"])
        events.append(
            dict(
                line=line_number,
                tid=int(match["tid"]),
                cpu=int(match["cpu"]),
                ns=int(Decimal(match["time"]) * 1_000_000_000),
                function=int(event[1]),
                kind=event[2],
                channel=int(chan[1]) if chan else None,
                pointer=arg[1].lower().lstrip("0") or "0" if arg else None,
            )
        )
    # Mono clock permits migration; retain file order to break equal timestamps.
    inversions = sum(a["ns"] > b["ns"] for a, b in zip(events, events[1:]))
    events.sort(key=lambda row: (row["ns"], row["line"]))
    return events, dict(
        unparsed_target_lines=invalid,
        explicit_loss_lines=lost,
        parsed_event_counts=dict(
            collections.Counter(f"{group}_{r['function']}_{r['kind']}" for r in events)
        ),
        input_time_inversions=inversions,
    )


def pair_calls(text, group):
    events, quality = parse_capture(text, group)
    stacks = collections.defaultdict(list)
    thread_ids, objects, calls = {}, {}, []
    orphan_leaves = nesting_errors = dropped_open = suspect_pairs = 0
    damaged_lines = quality["unparsed_target_lines"] + quality["explicit_loss_lines"]
    for event in events:
        tid = event["tid"]
        thread_ids.setdefault(tid, "thread_" + str(len(thread_ids) + 1))
        stack = stacks[tid]
        if event["kind"] == "enter":
            identity = (event["function"], event["pointer"])
            objects.setdefault(identity, "object_" + str(len(objects) + 1))
            active = event["channel"]
            if active is None and stack:
                active = stack[-1]["active_channel"]
            inherited = {active} if active is not None else set()
            stack.append(
                dict(
                    **event,
                    child_ns=0,
                    object=objects[identity],
                    channels=inherited,
                    active_channel=active,
                )
            )
            continue
        if not stack:
            orphan_leaves += 1
            continue
        if stack[-1]["function"] != event["function"]:
            # A lost nested event can make a superficially matching outer pair unsafe.
            nesting_errors += 1
            dropped_open += len(stack)
            stack.clear()
            continue
        enter = stack.pop()
        if any(
            min(enter["line"], event["line"]) <= n <= max(enter["line"], event["line"])
            for n in damaged_lines
        ):
            suspect_pairs += 1
            dropped_open += len(stack)
            stack.clear()
            continue
        duration = event["ns"] - enter["ns"]
        if stack:
            stack[-1]["child_ns"] += duration
            stack[-1]["channels"].update(enter["channels"])
        calls.append(
            dict(
                function=FUNCTIONS[event["function"]],
                thread=thread_ids[tid],
                object=enter["object"],
                start_ns=enter["ns"],
                end_ns=event["ns"],
                elapsed_ns=duration,
                channels=sorted(enter["channels"]),
                outside_traced_children_ns=duration - enter["child_ns"],
                migrated=enter["cpu"] != event["cpu"],
            )
        )
    quality.update(
        orphan_leaves=orphan_leaves,
        nesting_errors=nesting_errors,
        discarded_open_calls=dropped_open,
        discarded_pairs_spanning_damage=suspect_pairs,
        open_calls_at_end=sum(map(len, stacks.values())),
    )
    # Any sign of missing events weakens aggregate pairing. Still expose paired
    # samples so an operator can inspect a partial window; never certify coverage.
    quality["pairing_complete"] = not any(
        [
            quality["unparsed_target_lines"],
            quality["explicit_loss_lines"],
            orphan_leaves,
            nesting_errors,
            quality["open_calls_at_end"],
        ]
    )
    return calls, quality, len(events)


def loss_metadata(directory, group):
    profile = directory / "probe-profile.txt"
    rows = []
    invalid = 0
    if profile.exists():
        for line in profile.read_text().splitlines():
            bits = line.split()
            if group not in line:
                continue
            try:
                rows.append(
                    dict(event=bits[0], hits=int(bits[-2]), missed=int(bits[-1]))
                )
            except (ValueError, IndexError):
                invalid += 1
    losses = []
    before, after = directory / "per-cpu-before", directory / "per-cpu-after"
    missing_stats = []
    unread = []
    if after.exists():
        for end in sorted(after.glob("cpu*.txt")):
            start = before / end.name
            if not start.exists():
                missing_stats.append(end.name)
                continue

            def counters(path):
                return {
                    key.strip(): int(value.strip())
                    for key, value in (
                        line.split(":", 1)
                        for line in path.read_text().splitlines()
                        if ":" in line
                    )
                    if key.strip() in ["overrun", "commit overrun", "dropped events"]
                }

            a, b = counters(start), counters(end)
            entry = re.search(r"^entries:\s*(\d+)\s*$", end.read_text(), re.MULTILINE)
            if entry is None:
                missing_stats.append(end.name + ":entries")
            elif int(entry[1]):
                unread.append(dict(cpu=end.stem, entries=int(entry[1])))
            if set(a) != set(b) or not all(
                k in b for k in ["overrun", "commit overrun", "dropped events"]
            ):
                missing_stats.append(end.name)
                continue
            for name in b:
                if b[name] != a[name]:
                    losses.append(
                        dict(cpu=end.stem, counter=name, delta=b[name] - a[name])
                    )
    covered_events = {r["event"].split("/")[-1].split(":")[-1] for r in rows}
    expected = {
        f"{group}_{index}_{kind}" for index in range(4) for kind in ["enter", "leave"]
    }
    same_cpus = {p.name for p in before.glob("cpu*.txt")} == {
        p.name for p in after.glob("cpu*.txt")
    }
    complete = (
        expected == covered_events
        and len(rows) == len(expected)
        and not invalid
        and not missing_stats
        and same_cpus
        and before.exists()
        and after.exists()
        and bool(list(after.glob("cpu*.txt")))
    )
    return dict(
        probe_profile=rows,
        invalid_profile_rows=invalid,
        per_cpu_losses=losses,
        missing_cpu_stats=missing_stats,
        unread_buffer_entries=unread,
        metadata_complete=complete,
        loss_free=complete
        and not losses
        and not unread
        and all(r["missed"] == 0 for r in rows),
    )


def summarize(directory, target_channel=None):
    environment = dict(
        line.split("=", 1)
        for line in (directory / "environment.txt").read_text().splitlines()
        if "=" in line
    )
    group = environment["group"]
    trace = directory / "kernel-calls.txt"
    calls, quality, event_count = pair_calls(trace.read_text(errors="replace"), group)
    loss = loss_metadata(directory, group)
    mismatches = []
    for row in loss["probe_profile"]:
        event = row["event"].split("/")[-1].split(":")[-1]
        observed = quality["parsed_event_counts"].get(event, 0)
        if observed != row["hits"]:
            mismatches.append(
                dict(event=event, probe_hits=row["hits"], parsed=observed)
            )
    loss["event_hit_count_mismatches"] = mismatches
    loss["loss_free"] = loss["loss_free"] and not mismatches
    all_call_count = len(calls)
    ambiguous_excluded = 0
    if target_channel is not None:
        ambiguous_excluded = sum(
            target_channel in row["channels"] and len(row["channels"]) > 1
            for row in calls
        )
        calls = [row for row in calls if row["channels"] == [target_channel]]
    channel_aliases = {}
    for row in calls:
        row["channels"] = [
            channel_aliases.setdefault(
                value, "channel_" + str(len(channel_aliases) + 1)
            )
            for value in row["channels"]
        ]
    grouped = collections.defaultdict(list)
    objects = collections.defaultdict(list)
    for call in calls:
        grouped[call["function"]].append(call)
        objects[(call["thread"], call["function"], call["object"])].append(call)
    gaps = []
    for (thread, function, obj), rows in objects.items():
        rows.sort(key=lambda row: row["start_ns"])
        # Gap includes other work on the thread; it is not a measurement of idle time.
        values = [
            b["start_ns"] - a["end_ns"]
            for a, b in zip(rows, rows[1:])
            if b["start_ns"] >= a["end_ns"]
        ]
        gaps.append(
            dict(
                thread=thread,
                function=function,
                object=obj,
                gap_between_same_context_calls=distribution(values),
            )
        )
    origin = min((r["start_ns"] for r in calls), default=0)
    largest = []
    for row in sorted(calls, key=lambda r: r["elapsed_ns"], reverse=True)[:20]:
        largest.append(
            {
                **{k: v for k, v in row.items() if k not in ["start_ns", "end_ns"]},
                "relative_start_ns": row["start_ns"] - origin,
            }
        )
    return dict(
        schema="driver-kernel-pairing/v1",
        source_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        parsed_event_count=event_count,
        scope="target_channel"
        if target_channel is not None
        else "all_observed_channels",
        all_channels_paired_call_count=all_call_count,
        ambiguous_multichannel_calls_excluded=ambiguous_excluded,
        paired_call_count=len(calls),
        pairing_quality=quality,
        loss_checks=loss,
        complete_loss_free_pairing=(
            quality["pairing_complete"] and loss["loss_free"] and bool(calls)
        ),
        timing_definition="Function-boundary elapsed time, including off-CPU time; "
        "outside_traced_children includes untraced work and is not CPU self time. "
        "Same-context gaps can include other contexts and are not idle time.",
        functions={
            name: dict(
                elapsed=distribution([r["elapsed_ns"] for r in rows]),
                outside_traced_children=distribution(
                    [r["outside_traced_children_ns"] for r in rows]
                ),
            )
            for name, rows in grouped.items()
        },
        per_thread_context_gaps=gaps,
        largest_elapsed_calls=largest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--channel", type=int, help="Experiment-owned numeric /proc/trs_chan channel ID"
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(summarize(args.capture_directory, args.channel), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
