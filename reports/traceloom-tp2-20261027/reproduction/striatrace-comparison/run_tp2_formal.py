"""Run the frozen HOST/AIV/AIV/HOST capture order; stop on first failure."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tp2_pilot import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("run_tp2_formal.py", "tp2_pilot.py", "launch_tp2_pilot.py"):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    cells = [
        dict(name=f"c{i}-{arm.lower()}", backend=arm)
        for i, arm in enumerate(("HOST", "AIV", "AIV", "HOST"))
    ]
    protocol = dict(
        status="frozen_before_capture",
        cells=cells,
        runtime=str(a.runtime.resolve()),
        model=str(a.model.resolve()),
        independent_unit="engine process; ranks and replays are paired/nested observations",
        purpose="Archive a complete current-runtime two-repetition structural comparison; not historical FFTS+ reproduction",
        configuration="Same qualified TP2 pilot; two warmup and two profiled sequential requests per process",
        no_unprofiled_speedup_claim=True,
        scripts={p.name: digest(p) for p in scripts.iterdir()},
    )
    with (root / "protocol.json").open("x") as f:
        json.dump(protocol, f, indent=2)
    for cell in cells:
        command = [
            sys.executable,
            str(scripts / "launch_tp2_pilot.py"),
            "--runtime",
            str(a.runtime.resolve()),
            "--model",
            str(a.model.resolve()),
            "--output",
            str(root / cell["name"]),
            "--backend",
            cell["backend"],
        ]
        print("START", cell["name"], flush=True)
        subprocess.run(command, check=True)
        print("COMPLETE", cell["name"], flush=True)
    print("ALL_FOUR_CAPTURES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
