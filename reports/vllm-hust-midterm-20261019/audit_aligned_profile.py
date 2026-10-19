"""Inventory profiler coverage without treating hardware work as useful FLOPs."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def audit(root):
    files = []
    for path in sorted(root.rglob("kernel_details.csv")):
        types = Counter()
        missing_shapes = Counter()
        missing_counters = Counter()
        count = shapes = 0
        gm_read = gm_write = 0.0
        counter_rows = 0
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                count += 1
                types[row.get("Type", "")] += 1
                shape = row.get("Input Shapes", "").strip('" ')
                has_shape = bool(shape and shape not in ("N/A", "nan", "None"))
                shapes += has_shape
                if not has_shape:
                    missing_shapes[row.get("Type", "")] += 1
                fields = [
                    f"{core}_{direction}_main_memory_datas(KB)"
                    for core in ("aic", "aiv")
                    for direction in ("read", "write")
                ]
                try:
                    values = [float(row[key]) for key in fields]
                except (KeyError, ValueError):
                    missing_counters[row.get("Type", "")] += 1
                    continue
                counter_rows += 1
                gm_read += values[0] + values[2]
                gm_write += values[1] + values[3]
        files.append(
            dict(
                path=str(path.relative_to(root)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                kernel_rows=count,
                rows_with_input_shapes=shapes,
                rows_with_memory_counters=counter_rows,
                types=dict(types),
                missing_shapes_by_type=dict(missing_shapes),
                missing_memory_counters_by_type=dict(missing_counters),
                gm_read_kb=gm_read,
                gm_write_kb=gm_write,
            )
        )
    return dict(
        trace_root=str(root),
        files=files,
        whole_model_effective_flops=None,
        physical_hbm_bytes=None,
        communication_bytes=None,
        attainment_percent=None,
        scope="Coverage inventory only; GM counters are not physical HBM traffic, and shape presence does not prove effective-FLOP coverage",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(audit(args.trace_root), indent=2, allow_nan=False) + "\n"
    )
