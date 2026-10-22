# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarise the W2b e2e TPOT matrix from the saved ``serve_*.json`` files.

Every number printed here is read from a落盘 result file produced by
``run_e2e_matrix.sh``; the script does not re-measure anything.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CASES = ("api1", "joint")
MODES = ("off", "on")
METRICS = ("median_tpot_ms", "p95_tpot_ms", "p99_tpot_ms", "mean_tpot_ms")


def _load(case: str, mode: str) -> list[tuple[str, dict]]:
    """Saved results for one (case, mode): single run and repeat-tagged.

    ``results/superseded/`` holds the first api1 pair taken before the
    fallback-ordering fix (REPORT.md §4) and is deliberately NOT summarised.
    """
    pattern = f"serve_{case}_{mode}*.json"
    return [(p.name, json.loads(p.read_text())) for p in sorted(RESULTS.glob(pattern))]


def main() -> None:
    for case in CASES:
        print(f"\n### case {case}")
        rows: dict[str, list[float]] = {
            f"{m}_{mode}": [] for m in METRICS for mode in MODES
        }
        for mode in MODES:
            for name, data in _load(case, mode):
                for metric in METRICS:
                    rows[f"{metric}_{mode}"].append(data[metric])
                print(
                    f"  {name:<32} median={data['median_tpot_ms']:8.2f} "
                    f"mean={data['mean_tpot_ms']:8.2f} "
                    f"p95={data['p95_tpot_ms']:8.2f} p99={data['p99_tpot_ms']:8.2f} "
                    f"out_tput={data['output_throughput']:7.2f} "
                    f"ok={data['completed']} fail={data['failed']}"
                )
        print("  --- medians over repeats ---")
        for metric in METRICS:
            off = rows[f"{metric}_off"]
            on = rows[f"{metric}_on"]
            if not off or not on:
                print(f"  {metric}: missing data (off={len(off)}, on={len(on)})")
                continue
            off_med = statistics.median(off)
            on_med = statistics.median(on)
            delta = (on_med - off_med) / off_med * 100.0
            print(
                f"  {metric:<16} off={off_med:8.2f} on={on_med:8.2f} "
                f"delta={delta:+6.2f}%  (off={off}, on={on})"
            )


if __name__ == "__main__":
    main()
