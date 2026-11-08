"""Export saved fixed native captures with unchanged upstream reference CLIs.

No generation, ASR or scoring. Default prints a plan. v1.5 explicitly selects
the same twelve pairs so the exporter cannot silently expand to all tasks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_full_duplex_original_capture import FAMILIES, MODULE_HASHES

V15_EXPORT_HASH = "c5cd488c530bbb060266e6811751bed10ac9d61353c8b365f6a61d59822f69b4"  # pragma: allowlist secret - public original CLI hash


def export_command(args):
    rows = json.loads((args.capture_root / "fixed-plan-public.json").read_text("utf-8"))
    expected = [
        (version, f"{family}/{case}")
        for version, families in FAMILIES.items()
        for family, ids in families.items()
        for case in ids
    ]
    if [(row["version"], row["sample_id"]) for row in rows] != expected:
        raise ValueError("the fixed original 27-case capture plan is required")
    selected = []
    for index, row in enumerate(rows):
        output = args.capture_root / f"case-{index:02d}"
        if Path(row["output"]).resolve() != output.resolve():
            raise ValueError("capture plan output is not bound to its original case")
        if row["version"] == args.version:
            selected.append((row, output))
    if args.version == "v1.0":
        module = "benchmarks.eval.benchmark_duplex_v10"
        command = [args.python, "-m", module, "reference-export"]
    else:
        module = "benchmarks.eval.benchmark_duplex_reference"
        command = [
            args.python,
            "-m",
            module,
            "export",
            "--engine",
            args.engine,
            "--trace-format",
            "realtime-pcm16-v1",
        ]
    command.extend(
        [
            "--dataset-root",
            str(args.dataset_root / args.version),
            "--out",
            str(args.output / "tree"),
        ]
    )
    for _, output in selected:
        command.extend(["--run", str(output)])
    if args.version == "v1.5":
        for row, _ in selected:
            command.extend(["--only", row["sample_id"]])
    return command, selected


def execute(args, command, selected):
    module = command[2]
    path = args.omni_root / (module.replace(".", "/") + ".py")
    expected = MODULE_HASHES["v1.0"] if args.version == "v1.0" else V15_EXPORT_HASH
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("pinned original exporter CLI differs")
    if args.output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("use an external fresh output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.omni_root) + os.pathsep + env.get("PYTHONPATH", "")
    with (args.output / "export.log").open("xb") as log:
        code = subprocess.run(
            command,
            env=env,
            cwd=args.omni_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    report = {
        "version": args.version,
        "command": command,
        "exit_code": code,
        "selected_original_cases": len(selected),
        "model_calls": 0,
        "original_observation_windows_modified": False,
        "source_checkbox_eligible": False,
    }
    (args.output / "qualification-public.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1.0", "v1.5"], required=True)
    for flag in ["omni-root", "capture-root", "dataset-root", "output"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--engine", default="minicpmo-npu-derived")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for key in ["omni_root", "capture_root", "dataset_root", "output"]:
        setattr(args, key, getattr(args, key).resolve())
    command, selected = export_command(args)
    if args.execute:
        return execute(args, command, selected)
    print(
        json.dumps(
            {
                "status": "PLAN_ONLY_NO_MODEL_CALLS",
                "command": command,
                "selected_original_cases": len(selected),
                "source_checkbox_eligible": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
