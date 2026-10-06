#!/usr/bin/env python3
"""Verify a sealed 112 bundle and emit an external admission attestation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_hust_benchmark.evaluation_bundle_verifier import verify_bundle
from vllm_hust_benchmark.evaluation_job_adapter import AdapterError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--job-record", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()
    if args.attestation.exists():
        print("refusing to overwrite admission attestation", file=sys.stderr)
        return 2
    try:
        payload = verify_bundle(args.bundle, args.job_record, args.schedule)
        args.attestation.parent.mkdir(parents=True, exist_ok=True)
        args.attestation.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (AdapterError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"evaluation bundle rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
