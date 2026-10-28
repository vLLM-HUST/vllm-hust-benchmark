"""Verify original StructEval render protocols and report runtime prerequisites.

No generation, rendering, model calls or generated-code execution. A discovered
container binary is only a prerequisite, never proof of a qualified sandbox.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from audit_structeval_saved_nonrender import ARCHIVE_SHA, DATA_SHA, verify_author


def protocol_inventory(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or type(row.get("rendering")) is not bool
            or not isinstance(row.get("input_type"), str)
            or not isinstance(row.get("output_type"), str)
        ):
            raise ValueError("original explicit rendering/type fields required")
        if row["rendering"]:
            groups[(row["input_type"], row["output_type"])].append(index)
    return [
        {
            "input_type": key[0],
            "output_type": key[1],
            "original_rows": len(indices),
            "fixed_first_three_original_indices": indices[:3],
        }
        for key, indices in sorted(groups.items())
    ]


def prerequisites(which=shutil.which):
    names = (
        "docker",
        "podman",
        "bwrap",
        "unshare",
        "pdflatex",
        "typst",
        "node",
        "chromium",
    )
    return {
        "executables": {name: which(name) for name in names},
        "sandbox_qualified": False,
        "policy": "Do not execute generated Python/JavaScript/TeX on the host. Rendering requires a separately qualified isolated worker with no network, credentials or NPU devices, read-only original inputs, bounded writable output and explicit CPU/RAM/PID/time limits. A binary path or user namespace alone is not qualification.",
        "judge_policy": "Use the original VQA evaluator and its complete questions; a separately declared local NPU VLM may serve the API. No external paid judge is implicitly authorized. Render success alone is not VQA evaluation or task completion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh metadata output required; preserve prior evidence")
    for path, expected in (
        (args.data_file, DATA_SHA),
        (args.official_archive, ARCHIVE_SHA),
    ):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("fixed original asset identity mismatch")
    modules = verify_author(args.official_archive, args.package_root)
    import pyarrow.parquet as pq

    rows = pq.read_table(args.data_file).to_pylist()
    protocols = protocol_inventory(rows)
    result = {
        "source_id": "STRUCTEVAL",
        "status": "ORIGINAL_RENDER_PROTOCOL_PLAN_ONLY_NOT_EXECUTION",
        "original_rows": len(rows),
        "original_rendering_rows": sum(item["original_rows"] for item in protocols),
        "original_nonrendering_rows": sum(not row["rendering"] for row in rows),
        "official_source_modules_verified": len(modules),
        "render_protocols": protocols,
        "runtime_prerequisites": prerequisites(),
        "nonrendering_results_rerun": 0,
        "generated_code_executed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "render_protocols"}
        )
    )


if __name__ == "__main__":
    main()
