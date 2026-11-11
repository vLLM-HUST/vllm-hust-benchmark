"""Recompute the healthy-host kernel queue observer validation."""

import argparse
import json
import re
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def channel(text):
    sq = re.search(r"sq: id\((\d+)\).*?head\((\d+)\),tail\((\d+)\)", text)
    tx = re.search(r"stat: tx\((\d+)\)", text)
    hw = re.search(r"hw info: sq status (\d+) sq head (\d+) sq tail (\d+)", text)
    if not (sq and tx and hw):
        return None
    return dict(
        sq=int(sq[1]),
        cached_head=int(sq[2]),
        cached_tail=int(sq[3]),
        forwarded_entries=int(tx[1]),
        hardware_status=int(hw[1]),
        hardware_head=int(hw[2]),
        hardware_tail=int(hw[3]),
    )


def summarize(root):
    host = root / "hust-zsx-81"
    result = {"native": {}, "standalone": {}}
    for directory in sorted(host.glob("native-kernel-view-*-device1")):
        ops = [
            row
            for row in jsonl(directory / "operations.jsonl")
            if row["event"] == "restore"
        ]
        frames = jsonl(directory / "kernel-queue.jsonl")
        result["native"][directory.name] = {
            "completed": json.loads((directory / "done.json").read_text()),
            "restores": [
                {
                    k: r[k]
                    for k in [
                        "phase",
                        "complete_s",
                        "verified_tag_count",
                        "descriptors",
                    ]
                }
                for r in ops
            ],
            "frames": [],
        }
        for frame in frames:
            channels = []
            for path, row in frame["files"].items():
                if path.startswith("/proc/trs_chan/"):
                    parsed = channel(row.get("text", ""))
                    if parsed is not None:
                        channels.append({"path": path, **parsed})
            result["native"][directory.name]["frames"].append(
                {
                    "when": frame["when"],
                    "wall_read_ms": (frame["end_ns"] - frame["begin_ns"]) / 1e6,
                    "direct_channel_count": len(channels),
                    "channels": channels,
                }
            )
    for directory in sorted(host.glob("kernel-queue-calibration*-device0")):
        output = jsonl(directory / "stdout.jsonl")
        totals = [row for row in output if "wall_ns" in row][-1]
        queries = [row for row in output if row.get("event") == "known_backlog"]
        mapping = [row for row in output if row.get("event") == "stream_mapping"][0]
        matches = []
        for frame in jsonl(directory / "kernel.jsonl"):
            for path, row in frame["files"].items():
                if not path.startswith("/proc/trs_chan/"):
                    continue
                parsed = channel(row.get("text", ""))
                if parsed is None or parsed["sq"] != mapping["value0"]:
                    continue
                item = {"begin_ns": row["begin_ns"], **parsed}
                if queries:
                    closest = min(
                        queries, key=lambda q: abs(q["monotonic_ns"] - row["begin_ns"])
                    )
                    delta = (closest["monotonic_ns"] - row["begin_ns"]) / 1e6
                    if abs(delta) <= 20:
                        item["nearest_hal_query"] = closest
                        item["query_delta_ms"] = delta
                matches.append(item)
        result["standalone"][directory.name] = {
            "completed": json.loads((directory / "done.json").read_text()),
            "copy_result": totals,
            "channel_frames": matches,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.capsule), indent=2) + "\n")


if __name__ == "__main__":
    main()
