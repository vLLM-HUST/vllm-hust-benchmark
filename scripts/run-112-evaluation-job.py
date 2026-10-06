#!/usr/bin/env python3
"""Entrypoint for the dev-hub 112 worker runner_command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_hust_benchmark.evaluation_job_adapter import main

if __name__ == "__main__":
    raise SystemExit(main())
