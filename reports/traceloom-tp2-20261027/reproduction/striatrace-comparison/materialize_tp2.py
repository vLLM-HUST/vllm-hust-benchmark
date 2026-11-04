"""Export immutable TP2 profile copies and materialize each native worker trace."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from tp2_pilot import digest


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cell", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analyzer", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(s) for s in (a.cell / "receipt.jsonl").read_text().splitlines()]
    if rows[-1]["kind"] != "complete":
        raise RuntimeError("Capture incomplete")
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = a.cell.resolve() / "profile"
    copy = output / "profile-copy"
    shutil.copytree(source, copy)
    files = {
        str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()
    }
    for name, value in files.items():
        if (
            digest(copy / name) != value
            or (copy / name).stat().st_ino == (source / name).stat().st_ino
        ):
            raise RuntimeError("Profile copy verification failed")
    write(
        output / "inputs.json",
        dict(
            source=str(source),
            files=files,
            analyzer_sha256=digest(a.analyzer),
            script_sha256=digest(__file__),
        ),
    )
    with (output / "export.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from torch_npu.profiler.profiler import analyse; "
                "from torch_npu.profiler import ExportType; "
                "analyse(sys.argv[1], max_process_number=1, export_type=ExportType.Db)",
                str(copy),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    profiles = sorted(copy.glob("*_ascend_pt/PROF_*"))
    if len(profiles) != 2:
        raise RuntimeError(("Expected two worker profiles", profiles))
    for i, profile in enumerate(profiles):
        with (output / f"parse-{i}.log").open("x") as log:
            subprocess.run(
                ["msprof", "--parse=on", "--output=" + str(profile)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=240,
            )
    databases = sorted(
        copy.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db")
    )
    if len(databases) != 2:
        raise RuntimeError(("Expected two exported worker databases", databases))
    results = []
    for i, raw in enumerate(databases):
        with sqlite3.connect(raw.as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Corrupt input")
            tables = {
                r[0]
                for r in db.execute("select name from sqlite_master where type='table'")
            }
            counts = {
                name: db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                for name in (
                    "TASK",
                    "CANN_API",
                    "COMPUTE_TASK_INFO",
                    "COMMUNICATION_TASK_INFO",
                )
                if name in tables
            }
        command = [
            str(a.analyzer.resolve()),
            str(raw),
            "--threads",
            "4",
            "--output",
            str(output / f"worker-{i}.db"),
            "--structural-order",
            "host-launch",
        ]
        start = time.perf_counter()
        with (output / f"materialization-{i}.log").open("x") as log:
            result = subprocess.run(
                [
                    "/usr/bin/time",
                    "-v",
                    "-o",
                    str(output / f"resources-{i}.txt"),
                    *command,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        results.append(
            dict(
                command=command,
                raw=str(raw),
                raw_sha256=digest(raw),
                raw_counts=counts,
                elapsed_seconds=time.perf_counter() - start,
                returncode=result.returncode,
            )
        )
        write(output / f"worker-{i}.json", results[-1])
    write(output / "materialization.json", results)
    if any(r["returncode"] for r in results):
        raise RuntimeError(
            "One or more worker analyses failed; inspect individual receipts"
        )
    print(output, flush=True)


if __name__ == "__main__":
    main()
