"""Export the completed SWE Trace Heavy Load repeat for the controller view."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/controller-r8-20261122-repeat"
PUBLICATION = ROOT / "website-publication/data/controller-r8-heavy-load-repeat-v1.json"


def generate():
    pair_raw = (REPORT / "paired-result.json").read_bytes()
    pair = json.loads(pair_raw)
    identity = json.loads((REPORT / "matched-engine.json").read_bytes())
    if not (
        pair["completed_pair"]
        and identity["runtime_match"]
        and identity["engine_arguments_match"]
    ):
        raise ValueError("The submitted pair is incomplete or used different engines")
    direct, controller = (pair[k] for k in ("direct", "request_throttling_controller"))
    measures = {}
    for label, arm in (("direct", direct), ("controller", controller)):
        if (
            arm["completed_tasks"] != 112
            or arm["successful_requests"] != 336
            or arm["failed_requests"]
        ):
            raise ValueError("The SWE Trace Heavy Load request census differs")
        if arm["output_tokens"] != 43516 or arm["input_tokens"] != 5992408:
            raise ValueError("The completed input or output population differs")
        measures[label] = {
            "programs": arm["completed_tasks"],
            "requests": arm["successful_requests"],
            "input_tokens": arm["input_tokens"],
            "output_tokens": arm["output_tokens"],
            "elapsed_seconds": arm["elapsed_seconds"],
            "output_tokens_per_second": arm["output_tokens"] / arm["elapsed_seconds"],
            "cached_input_percent": arm["cached_input_percent"],
        }
    return (
        json.dumps(
            {
                "schema": "controller-r8-heavy-load-repeat/v1",
                "date": "2026-11-22",
                "engine_release": "full-answer-b1-20261117-r8",
                "program_state_residency": 80,
                "report_path": "reports/controller-r8-20261122-repeat/README.md",
                "report_pair_sha256": hashlib.sha256(pair_raw).hexdigest(),
                **measures,
            },
            indent=2,
        )
        + "\n"
    ).encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    actual = generate()
    if args.check:
        if PUBLICATION.read_bytes() != actual:
            raise SystemExit(
                "SWE Trace Heavy Load publication differs from the repeat evidence"
            )
    else:
        PUBLICATION.write_bytes(actual)
