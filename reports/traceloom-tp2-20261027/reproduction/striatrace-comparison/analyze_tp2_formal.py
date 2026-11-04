"""Materialize and independently audit all eight workers of a complete ABBA run."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
from tp2_pilot import digest


def run(script, *args):
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name(script)), *map(str, args)],
        check=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analyzer", type=Path, required=True)
    a = p.parse_args()
    root, out = a.campaign.resolve(), a.output.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    records = {}
    for cell in protocol["cells"]:
        path = root / cell["name"] / "receipt.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows[-1]["kind"] != "complete":
            raise RuntimeError(("Incomplete cell", cell))
        if (
            json.loads((root / (cell["name"] + ".exit.json")).read_text())["returncode"]
            != 0
        ):
            raise RuntimeError(("Nonzero capture exit", cell))
        records[cell["name"]] = {r["kind"]: r for r in rows}
    reference = next(iter(records.values()))
    for name, rows in records.items():
        if rows["effective_configuration"]["graph_mode"] != "PIECEWISE":
            raise RuntimeError(("Wrong graph mode", name))
        if (
            not rows["profile_outputs"]["warmup_parity"]
            or rows["profile_outputs"]["outputs"]
            != reference["profile_outputs"]["outputs"]
        ):
            raise RuntimeError(("Output mismatch", name))
        for key in (
            "script_sha256",
            "runtime_manifest_sha256",
            "model_config_sha256",
            "prompt_token_ids",
            "output_tokens",
        ):
            if rows["configuration"][key] != reference["configuration"][key]:
                raise RuntimeError(("Input configuration differs", name, key))
    out.mkdir(parents=True, exist_ok=False)

    def analyze(cell):
        name = cell["name"]
        dest = out / name
        run(
            "materialize_tp2.py",
            root / name,
            "--output",
            dest,
            "--analyzer",
            a.analyzer,
        )
        run(
            "inspect_tp2.py",
            dest / "worker-0.db",
            dest / "worker-1.db",
            "--output",
            dest / "inspection.json",
        )
        for rank in range(2):
            raw = list(
                (dest / "profile-copy").glob(
                    f"rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db"
                )
            )
            if len(raw) != 1:
                raise RuntimeError(("Expected one exported rank DB", name, rank))
            stream = list(
                raw[0].parent.parent.glob("PROF_*/host/sqlite/stream_info.db")
            )
            if len(stream) != 1:
                raise RuntimeError(("Expected one stream metadata DB", name, rank))
            run(
                "raw_tp2_audit.py",
                "--raw",
                raw[0],
                "--streams",
                stream[0],
                "--analysis",
                dest / f"worker-{rank}.db",
                "--output",
                dest / f"raw-worker-{rank}.json",
            )
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for name in pool.map(analyze, protocol["cells"]):
            print("AUDITED", name, flush=True)
    comparisons = []
    for index, (left, right) in enumerate(
        (("c0-host", "c1-aiv"), ("c3-host", "c2-aiv"))
    ):
        output = out / f"comparison-{index}.json"
        run(
            "compare_tp2_structures.py",
            "--left",
            *[out / left / f"worker-{i}.db" for i in range(2)],
            "--right",
            *[out / right / f"worker-{i}.db" for i in range(2)],
            "--raw-left",
            *[out / left / f"raw-worker-{i}.json" for i in range(2)],
            "--raw-right",
            *[out / right / f"raw-worker-{i}.json" for i in range(2)],
            "--output",
            output,
        )
        comparison = json.loads(output.read_text())
        if not all(all(r.values()) for r in comparison["answer_agreement"]):
            raise RuntimeError(
                "Cost decomposition differs from independent raw workflow"
            )
        comparisons.append(
            dict(
                path=str(output),
                sha256=digest(output),
                answer_agreement=comparison["answer_agreement"],
                per_rank=[
                    dict(
                        members=r["members_per_unit"],
                        left_medians_ns=r["left_costs"]["medians_ns"],
                        right_medians_ns=r["right_costs"]["medians_ns"],
                    )
                    for r in comparison["ranks"]
                ],
            )
        )
    workers = []
    for cell in protocol["cells"]:
        for rank in range(2):
            path = out / cell["name"] / f"raw-worker-{rank}.json"
            raw = json.loads(path.read_text())
            if (
                raw["unresolved"]
                or raw["missing_from_joint"]
                or raw["absent_from_raw"]
                or raw["coordinate_audit"]["differences"]
            ):
                raise RuntimeError("Recovery mismatch")
            workers.append(
                dict(
                    cell=cell["name"],
                    backend=cell["backend"],
                    rank=rank,
                    launches=raw["native_launches"],
                    members=raw["raw_members"],
                    units=raw["repeated_units"],
                    matching_coordinates=raw["coordinate_audit"]["equal"],
                    receipt=str(path),
                    sha256=digest(path),
                )
            )
    result = dict(
        status="COMPLETE_ABBA_STRUCTURAL_CONTROL",
        protocol=protocol,
        independent_processes=4,
        profiles=8,
        outputs_identical=True,
        workers=workers,
        totals={
            k: sum(r[k] for r in workers)
            for k in ("launches", "members", "units", "matching_coordinates")
        },
        comparisons=comparisons,
        script_sha256=digest(__file__),
        analyzer_sha256=digest(a.analyzer),
        runtime_manifest_sha256=reference["configuration"]["runtime_manifest_sha256"],
        interpretation="Structural correctness and current-runtime repeatability; raw workflow returns the same answers. "
        "Not a StriaTrace run, historical FFTS+ reproduction, or unprofiled speedup result.",
    )
    with (out / "summary.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(dict(status=result["status"], totals=result["totals"])), flush=True
    )


if __name__ == "__main__":
    main()
