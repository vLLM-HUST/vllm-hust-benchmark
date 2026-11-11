"""Recompute finite DMA-preparation experiments from immutable D/E/F capsules."""

import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path
import statistics

from analyze_resume_20261111 import analyze_native, cpu_stat, jsonl


def distribution(values):
    return dict(
        n=len(values),
        seconds=values,
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
    )


def native(directory):
    ops = [r for r in jsonl(directory / "operations.jsonl") if r["event"] == "restore"]
    done = json.loads((directory / "done.json").read_text())
    assert done["restores"] == len(ops)
    assert not (directory / "failure.json").exists()
    config = json.loads((directory / "config.json").read_text())
    rows = jsonl(directory / "raw/api.jsonl")
    timing_offsets = [
        r["epoch_ns"] - r["start_ns"]
        for r in rows
        if "epoch_ns" in r and "start_ns" in r
    ]
    offset = statistics.median(timing_offsets)
    lifecycle = [r for r in rows if r.get("event", "").startswith("shadow_")]
    assert all(r["error"] == 0 for r in lifecycle)
    for r in lifecycle:
        assert r["end_ns"] >= r["start_ns"]
    base = analyze_native(directory)
    tasks = [
        r
        for r in rows
        if r.get("event") == "hal_send"
        and r.get("kind") == "task"
        and r["error"] == 0
        and r["sq"] == base["sq"]
        and "sqe_type6" in r
    ]
    assert all(r.get("sqe_type6") == 17 and r.get("flags8") == 64 for r in tasks)
    group = collections.defaultdict(list)
    result = []
    allocated = False
    registration_history = False
    for op, short in zip(ops, base["restores"], strict=True):
        assert op["all_tags"] and op["verified_tag_count"] == op["descriptors"] == 15456
        assert op["bytes"] == 4051697664
        expected = [
            [(i + plane * 521 + 1000) % 30000 for i in range(161)]
            for plane in range(96)
        ]
        assert op["verified_tags"] == expected
        start = op["start_epoch_ns"] - offset
        end = start + op["complete_s"] * 1e9
        components = collections.defaultdict(float)
        for event in lifecycle:
            overlap = max(0, min(end, event["end_ns"]) - max(start, event["start_ns"]))
            if overlap:
                components[event["event"]] += overlap / 1e9
        lifecycle_s = sum(components.values())
        assert lifecycle_s <= op["complete_s"] + 0.002
        policy = op["policy"]
        if policy in (29, 30) and not allocated:
            condition = "shadow_initialization"
            allocated = True
        elif policy == 29:
            condition = "shadow_dma_registered"
        elif policy == 30:
            condition = (
                "shadow_unregistered_after"
                if registration_history
                else "shadow_unregistered_before"
            )
        elif components:
            condition = "original_with_shadow_cleanup"
            allocated = False
        else:
            condition = "original"
        registration_history |= policy == 29
        residual = max(0, op["complete_s"] - lifecycle_s)
        batches = sorted(
            [
                r
                for r in rows
                if r.get("event") == "batch" and start <= r["start_ns"] < end
            ],
            key=lambda r: r["start_ns"],
        )
        assert sum(r["copies"] for r in batches) == op["descriptors"]
        assert sum(r["bytes"] for r in batches) == op["bytes"]
        pointer_signature = [
            (r["src_first"], r["dst_first"], r["copies"], r["bytes"]) for r in batches
        ]
        pointer_digest = hashlib.sha256(
            json.dumps(pointer_signature).encode()
        ).hexdigest()
        syncs = [
            r for r in rows if r.get("event") == "sync" and start <= r["start_ns"] < end
        ]
        threads_before = {r["tid"]: r for r in op["native_threads_before"]}
        thread_deltas = []
        for thread in op["native_threads_after"]:
            if thread["tid"] in threads_before:
                before = threads_before[thread["tid"]]
                delta = [
                    int(a) - int(b)
                    for a, b in zip(
                        thread["schedstat"].split(),
                        before["schedstat"].split(),
                        strict=True,
                    )
                ]
                thread_deltas.append(
                    dict(
                        name=thread["name"],
                        runtime_s=delta[0] / 1e9,
                        runqueue_wait_s=delta[1] / 1e9,
                    )
                )
        group[condition].append(residual)
        result.append(
            dict(
                **short,
                policy=policy,
                condition=condition,
                lifecycle_s=dict(components),
                lifecycle_total_s=lifecycle_s,
                residual_restore_s=residual,
                batch_pointer_signature_sha256=pointer_digest,
                batch_wall_s=sum(r["end_ns"] - r["start_ns"] for r in batches) / 1e9,
                batch_thread_cpu_s=sum(r["thread_cpu_ns"] for r in batches) / 1e9,
                sync_wall_s=sum(r["end_ns"] - r["start_ns"] for r in syncs) / 1e9,
                native_thread_deltas=thread_deltas,
                payload_bytes=op["bytes"],
                all_descriptor_tags_checked=True,
            )
        )
    shadow_signatures = {
        r["batch_pointer_signature_sha256"] for r in result if r["policy"] in (29, 30)
    }
    assert len(shadow_signatures) <= 1, "shadow pointer or batch geometry changed"
    points = base["kernel_points"]
    return dict(
        host=config["host"],
        logical_device=config["device"],
        physical_device=json.loads((directory / "kernel-resources.json").read_text())[
            "physical_device"
        ],
        numa=config["numa"],
        completed=len(result),
        driver_accepted=base["driver_accepted"],
        verified_tags_per_restore=15456,
        driver_entries_with_validated_header=len(tasks),
        shadow_source_and_destination_first_pointers_stable=bool(shadow_signatures),
        restores=result,
        lifecycle_events=lifecycle,
        residual_distributions={k: distribution(v) for k, v in group.items()},
        clock_offset_spread_ns=max(timing_offsets) - min(timing_offsets),
        max_sampled_driver_wait_ms=max(
            (r["next_task_driver_wait_ms"] or 0 for r in points), default=0
        ),
        kernel_points=points,
    )


def c_control(directory, kind):
    rows = jsonl(directory / "results.jsonl")
    exit_record = json.loads((directory / "exit.json").read_text())
    if kind in ("counter-history", "allocation-geometry", "host-dma-cache"):
        cases = [r for r in rows if r.get("event") == "copy"]
        hal = sorted(
            [
                r
                for r in jsonl(directory / "api.jsonl")
                if r.get("event") == "hal_send"
                and r.get("kind") == "task"
                and r["error"] == 0
            ],
            key=lambda r: r["end_ns"],
        )
        ends = [r["end_ns"] for r in hal]
        for case in cases:
            calls = hal[
                bisect.bisect_left(ends, case["start_ns"]) : bisect.bisect_right(
                    ends, case["end_ns"]
                )
            ]
            entries = sum(r["count"] for r in calls)
            assert entries == case["descriptors"], (
                directory.name,
                case["trial"],
                entries,
            )
            case["driver_entries"] = entries
    else:
        cases = [r for r in rows if r.get("event") in ("copy", "prepare_destroy")]
    if cases:
        assert exit_record["exit_code"] == 0
        assert all(
            r.get("verified_tags", r.get("verified_unchanged_tags")) == 8 for r in cases
        )
    before = json.loads((directory / "wrapper-start.json").read_text())
    stat_before = cpu_stat(before["cpu_stat_before"])
    stat_after = cpu_stat(exit_record["cpu_stat_after"])
    return dict(
        kind=kind,
        exit=exit_record["exit_code"],
        cases=cases,
        lifecycle=[
            r
            for r in rows
            if r.get("event")
            in ("cache_prepare", "cache_destroy", "dma_register", "dma_unregister")
        ],
        errors=[r for r in rows if "error" in r],
        throttled_usec_delta=stat_after["throttled_usec"]
        - stat_before["throttled_usec"],
    )


def summarize(roots):
    result = dict(
        schema="restore-container-dma/v1",
        native={},
        controls={},
        environments={},
        randomized_pairs=[],
    )
    for root in roots:
        for directory in sorted(root.glob("hust-zsx-*/container1111-*-device*")):
            if (directory / "operations.jsonl").exists():
                key = root.name + "/" + directory.relative_to(root).as_posix()
                result["native"][key] = native(directory)
        for name in (
            "counter-history",
            "allocation-geometry",
            "dma-prepare",
            "dma-prepare-other-card",
            "dma-window",
            "dma-reuse",
            "host-dma-cache",
        ):
            for directory in sorted(root.glob("hust-zsx-*/container1111-" + name)):
                key = root.name + "/" + directory.relative_to(root).as_posix()
                result["controls"][key] = c_control(directory, name)
        for path in root.glob("hust-zsx-*/environment-container1111.json"):
            result["environments"][path.parent.name] = json.loads(path.read_text())
        for path in root.glob(
            "hust-zsx-*/container1111-vmalloc-inventory/iommu-focused-summary.json"
        ):
            result["environments"][path.parent.parent.name]["iommu"] = json.loads(
                path.read_text()
            )
        for path in sorted(
            root.glob("hust-zsx-*/container1111-dma-randomized-controller/plan.json")
        ):
            plan = json.loads(path.read_text())
            controller_done = json.loads((path.parent / "done.json").read_text())
            assert controller_done["devices"] == len(plan["plan"])
            for entry in plan["plan"]:
                status = json.loads(
                    (path.parent / f"device{entry['device']}-exit.json").read_text()
                )
                assert status["exit_code"] == 0
                key = (
                    root.name
                    + "/"
                    + path.parent.parent.name
                    + f"/container1111-native-shadow-randomized-device{entry['device']}"
                )
                run = result["native"][key]
                assert run["completed"] == entry["expected_restores"]
                for block in entry["blocks"]:
                    arms = {}
                    for index, mode in enumerate(block["order"]):
                        stages = [
                            f"stage-{block['start_stage'] + index * 2 + i}"
                            for i in range(2)
                        ]
                        chosen = [r for r in run["restores"] if r["phase"] in stages]
                        assert len(chosen) == 2 and all(
                            r["policy"] == mode for r in chosen
                        )
                        arms[str(mode)] = dict(
                            mean_complete_s=statistics.mean(
                                r["complete_s"] for r in chosen
                            ),
                            mean_residual_s=statistics.mean(
                                r["residual_restore_s"] for r in chosen
                            ),
                            second_complete_s=chosen[1]["complete_s"],
                            stages=stages,
                        )
                    ratio = (
                        arms["29"]["mean_complete_s"] / arms["30"]["mean_complete_s"]
                    )
                    result["randomized_pairs"].append(
                        dict(
                            run=key,
                            host=run["host"],
                            physical_device=run["physical_device"],
                            block=block["block"],
                            order=block["order"],
                            seed=entry["seed"],
                            arms=arms,
                            registered_over_unregistered=ratio,
                            registered_faster=ratio < 1,
                        )
                    )
    ops = [r for run in result["native"].values() for r in run["restores"]]
    payload = [
        r
        for run in result["controls"].values()
        for r in run["cases"]
        if r.get("event") == "copy"
    ]
    prep = [
        r
        for run in result["controls"].values()
        for r in run["cases"]
        if r.get("event") == "prepare_destroy"
    ]
    result["totals"] = dict(
        native_restores=len(ops),
        native_descriptors_checked=sum(r["descriptors"] for r in ops),
        native_throttled_usec_delta=sum(r["throttled_usec_delta"] for r in ops),
        c_payload_cases=len(payload),
        pure_dma_preparation_cases=len(prep),
        invalid_c_prepare_attempts=sum(
            r["exit"] != 0 for r in result["controls"].values()
        ),
    )
    result["resource_events"] = [
        dict(
            run=key,
            phase=r["phase"],
            complete_s=r["complete_s"],
            throttled_usec_delta=r["throttled_usec_delta"],
        )
        for key, run in result["native"].items()
        for r in run["restores"]
        if r["throttled_usec_delta"]
    ]
    result["randomized_summary"] = {}
    for host in ("hust-zsx-81", "hust-zsx-91"):
        pairs = [r for r in result["randomized_pairs"] if r["host"] == host]
        if pairs:
            result["randomized_summary"][host] = dict(
                pairs=len(pairs),
                registered_faster=sum(r["registered_faster"] for r in pairs),
                ratios=[r["registered_over_unregistered"] for r in pairs],
                median_ratio=statistics.median(
                    r["registered_over_unregistered"] for r in pairs
                ),
            )
    for host in ("hust-zsx-81", "hust-zsx-91"):
        selected = [
            r
            for key, run in result["native"].items()
            if run["host"] == host and "shadow" in key
            for r in run["restores"]
        ]
        result.setdefault("shadow_summary", {})[host] = {
            condition: dict(
                total=distribution(
                    [r["complete_s"] for r in selected if r["condition"] == condition]
                ),
                residual=distribution(
                    [
                        r["residual_restore_s"]
                        for r in selected
                        if r["condition"] == condition
                    ]
                ),
            )
            for condition in (
                "original",
                "shadow_dma_registered",
                "shadow_unregistered_before",
                "shadow_unregistered_after",
            )
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsules", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.capsules)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["totals"]))


if __name__ == "__main__":
    main()
