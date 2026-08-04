#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.simllm_official_attestation import (  # noqa: E402
    attest_simllm_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attest a completed official paired SimLLM campaign."
    )
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--result-spec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verified-by", required=True)
    args = parser.parse_args()
    attestation = attest_simllm_campaign(
        REPO_ROOT,
        args.result_spec_dir.resolve(),
        args.output_dir.resolve(),
        target_id=args.target_id,
        verified_by=args.verified_by,
    )
    print(args.output_dir.resolve())
    print(attestation["target_id"])
    print(attestation["successful_repeats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
