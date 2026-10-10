#!/usr/bin/env python3
from __future__ import annotations

import json

from vllm_hust_benchmark.acceptance_v5_4 import (
    build_measurement_queue,
    load_and_validate,
)


def main() -> int:
    declaration = load_and_validate()
    queue = build_measurement_queue(declaration)
    print(
        json.dumps(
            {
                "test_plan_version": declaration["test_plan_version"],
                "source_fidelity": declaration["source_fidelity"]["status"],
                "mandatory_queue_entries": len(queue),
                "optional_queue_entries": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
