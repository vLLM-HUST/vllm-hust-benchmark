"""Serial bounded fresh comparisons, immutable labels, fail-closed result gates."""

import ast
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
settings = json.loads((root / "q35-environments.json").read_text())
status = root / "evidence/gamma4-campaign-status.json"


def save(**record):
    record["time"] = time.time()
    tmp = status.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(status)


def counter(ms, key):
    values = [m["value"] for m in ms if m.get("name") == key]
    assert len(values) == 1, (key, values)
    return values[0]


def main():
    for repeat in range(1, 4):
        order = (
            ["q35-mtp4", "q35-adaptive"] if repeat % 2 else ["q35-adaptive", "q35-mtp4"]
        )
        for arm in order:
            label = f"gamma4-performance-{arm}-r{repeat}"
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("HUST_VSPEC_", "VSPEC_", "LUMI_"))
            }
            env.update(
                settings["q35-mtp2" if arm == "q35-mtp4" else arm],
                LUMI_TIMEOUT_SECONDS="1800",
            )
            save(state="running", label=label)
            rc = subprocess.run(
                [
                    str(root / "venv/bin/python"),
                    "launch_gamma4.py",
                    "0,1",
                    label,
                    "screen_gamma4.py",
                    arm,
                    "perf56",
                ],
                cwd=root,
                env=env,
                check=False,
            ).returncode
            assert rc == 0, (label, rc)
            result = json.loads(
                (root / "evidence" / f"{label}-result.json").read_text()
            )
            assert len(result["rows"]) == 56
            assert (
                counter(result["metrics_after"], "vllm:num_preemptions")
                - counter(result["metrics_before"], "vllm:num_preemptions")
                == 0
            )
            if arm == "q35-adaptive":
                summaries = [
                    ast.literal_eval(
                        line.split("vSpec Online Adaptive summary: ", 1)[1]
                    )
                    for line in (root / "evidence" / f"{label}.log")
                    .read_text()
                    .split("\n")
                    if "vSpec Online Adaptive summary: " in line
                ]
                assert summaries and all(
                    s["gamma_switches"] == 0 and set(s["selection_counts"]) <= {"4"}
                    for s in summaries
                ), "Candidate changed depth: not a matched-gamma comparison"
            save(state="arm_complete", label=label)
    save(state="complete")


try:
    main()
except BaseException as e:
    save(state="failed", error=repr(e))
    traceback.print_exc()
    raise
