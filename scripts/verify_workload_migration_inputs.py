"""A few real records per explicit source through the preparation pipeline.

No network, model, native task execution, or performance measurement. Raw input
stays in the original files; the SQLite output contains references, not exports.
Only aggregate state is printed. Keep all generated artifacts outside public Git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def verify(
    component: Path,
    bindings: Path,
    bindings_base: Path,
    output: Path,
    *,
    installed=False,
    samples_per_source=3,
) -> dict:
    component, bindings, bindings_base, output = (
        path.resolve() for path in (component, bindings, bindings_base, output)
    )
    if output.exists():
        raise FileExistsError("choose a fresh output directory")
    if not 1 <= samples_per_source <= 100:
        raise ValueError("samples per source must be between 1 and 100")
    package_root = component / "src"
    if not installed:
        sys.path.insert(0, str(package_root))
    package = importlib.import_module("vllm_hust_workload_data")
    imported_from_source = Path(package.__file__).resolve().is_relative_to(package_root)
    if not installed and not imported_from_source:
        raise ValueError("another editable installation shadowed the staged component")
    provider = importlib.import_module("vllm_hust_workload_data.provider")
    identity = provider.provider_identity()
    if installed and (imported_from_source or identity["installed_version"] != "0.1.0"):
        raise ValueError("wheel verification must not import the source checkout")
    pipeline = importlib.import_module("vllm_hust_workload_data.pipeline")
    consumer = importlib.import_module("vllm_hust_workload_data.consumer")
    original = json.loads(bindings.read_text(encoding="utf-8"))
    config = {"sources": []}
    roots = {}
    for entry in original["sources"]:
        sid = entry["source_id"]
        if sid in roots:
            raise ValueError("duplicate source binding")
        root = Path(entry["root"])
        root = (root if root.is_absolute() else bindings_base / root).resolve()
        roots[sid] = root
        config["sources"].append(
            {"source_id": sid, "data_root": str(root), "limit": samples_per_source}
        )
    # Do not import private QBI roots or evaluate any historical task reference.
    output.mkdir(parents=True, exist_ok=False)
    report = pipeline.run(config, output=output / "prepared", allow_partial=False)
    database = output / "prepared/workload.sqlite"
    rows = list(pipeline.prepared_rows(database))
    receipts = []
    for sid, root in roots.items():
        selected = [row for row in rows if row["source_id"] == sid]
        if len(selected) != samples_per_source:
            raise ValueError("source did not produce the requested number of records")
        kinds, units, request_bodies_checked = set(), set(), 0
        for sample in selected:
            record = consumer.resolve_input(
                database, sample["record_id"], data_root=root
            )
            if "outcome" in record.input_record():
                raise ValueError("outcome was merged into model input")
            kinds.add(record.runtime["kind"])
            units.add(sample["source_unit"])
            if record.runtime["kind"] == "request_replay":
                body = record.request_body()
                if not isinstance(body, dict) or not body:
                    raise ValueError("original request body is not usable")
                json.dumps(body, ensure_ascii=False, allow_nan=False)
                request_bodies_checked += 1
        receipts.append(
            {
                "source_id": sid,
                "sample_units": sorted(units),
                "samples": len(selected),
                "runtime_kinds": sorted(kinds),
                "read_prepare_resolve": "PASS",
                "original_request_bodies_checked": request_bodies_checked,
                "actual_model_execution": "NOT_RUN",
                "native_task_execution": "NOT_RUN",
                "scoring": "NOT_RUN",
                "full_dataset_validation": False,
            }
        )
        print(
            json.dumps({"source_id": sid, "read_prepare_resolve": "PASS"}), flush=True
        )
    result = {
        "schema_version": "workload-migration-read-smoke/v1",
        "status": "PASS_READ_PREPARE_RESOLVE_ONLY",
        "source_count": len(receipts),
        "enterprise_count": sum(
            row["source_id"].startswith("PAIO-") for row in receipts
        ),
        "samples_per_source": samples_per_source,
        "sample_selection": "FIRST_N_NATIVE_RECORDS_NOT_RANDOM_OR_REPRESENTATIVE",
        "total_records_verified": sum(row["samples"] for row in receipts),
        "model_calls": 0,
        "native_tasks_run": 0,
        "upstream_source_commit": identity["provenance"]["receiving_commit"],
        "provider_version": identity["version"],
        "provider_contract": identity["contract"],
        "verification_mode": "INSTALLED_WHEEL" if installed else "SOURCE_CHECKOUT",
        "bindings_sha256": hashlib.sha256(bindings.read_bytes()).hexdigest(),
        "raw_data_published": False,
        "sources": receipts,
        "pipeline_status": report.get("status"),
    }
    (output / "read-smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=3)
    parser.add_argument(
        "--installed",
        action="store_true",
        help="Verify installed wheel, without injecting source path",
    )
    args = parser.parse_args(argv)
    try:
        result = verify(
            args.component,
            args.bindings,
            args.bindings_base,
            args.output,
            installed=args.installed,
            samples_per_source=args.samples_per_source,
        )
    except Exception as error:
        # Source payloads and private paths are never included in CLI errors.
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "model_calls": 0,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "source_count",
                    "enterprise_count",
                    "model_calls",
                    "native_tasks_run",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
