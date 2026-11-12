"""Run the archived fixed two/four-slot CPU forecasts without requesting hardware."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUN = "r69-2p2d-original95-P-ingress-resources-2400-RR"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.workspace.resolve() / "simulation"
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    code = destination / "simulator2p2d"
    shutil.copytree(source / "simulator2p2d", code)
    for path in source.glob("shared-resources-*.json"):
        shutil.copyfile(path, destination / path.name)
    plan = json.loads((source / "forecast-plan.json").read_bytes())
    assert plan["seeds"] == list(range(32))
    assert plan["hardware_runs"] == 0
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    conditions = {"two128": 2, "four64": 4}
    for name, slots in conditions.items():
        command = [
            sys.executable,
            "-B",
            str(code / "run_component_forecast.py"),
            "--run",
            RUN,
            "--seeds",
            "32",
            "--output",
            str(destination / name),
            *["--" + option for option in plan["options"]],
            "--d-session-affinity",
            "--native-staging-slots",
            str(slots),
        ]
        with (destination / (name + ".log")).open("x", encoding="utf8") as log:
            subprocess.run(
                command,
                cwd=destination,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=240,
            )
        for seed in range(32):
            path = destination / name / RUN / f"seed-{seed:02d}.json"
            trajectory = json.loads(path.read_bytes())
            assert len(trajectory["requests"]) == 95
            model = trajectory["native_staging_completion_model"]
            assert model["slot_count"] == slots
            assert not model["target_timing_feedback_used"]
        print(json.dumps({"condition": name, "complete_trajectories": 32}))


if __name__ == "__main__":
    main()
