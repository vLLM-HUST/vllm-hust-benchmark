"""Verify the delivered experiment offline and compare its committed analysis."""

import argparse
import json
import sys
from pathlib import Path

from analyze import analyze
from evidence import digest, load_capsule, read_json, require
from qualification import verify_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        raw = load_capsule(args.report)
        verified = verify_evidence(raw, args.private_root)
        analysis = analyze(raw, verified)
        committed = (args.report / "analysis.json").read_bytes()
        require(
            analysis == read_json(committed),
            "Analysis differs from recomputed originals",
        )
        public_verified = {
            **verified,
            "coverage": {
                **verified["coverage"],
                "private_payloads_verified": 0,
                "official_f1_recomputed": 0,
            },
        }
        require(
            public_verified
            == read_json((args.report / "public-verification.json").read_bytes()),
            "Public verification receipt differs from recomputation",
        )
        receipt = {
            "status": "verified",
            "coverage": verified["coverage"],
            "lifecycles": len(verified["lifecycles"]),
            "compression_transactions": analysis["compression_transactions"],
            "quality_gate_passed": analysis["quality"]["gate_passed"],
            "runtime_snapshot_gaps": verified["runtime_snapshot_gaps"],
            "analysis_sha256": digest(committed),
        }
        encoded = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(encoded)
        print(encoded, end="")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
