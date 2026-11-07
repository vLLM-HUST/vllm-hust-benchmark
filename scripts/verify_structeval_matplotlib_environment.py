"""Check observed renderer dependencies/imports, never execute task code."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys


def verify(lock):
    pins = [
        line.strip().split("==")
        for line in Path(lock).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if (
        len(pins) != 12
        or any(len(pair) != 2 for pair in pins)
        or len({name for name, _ in pins}) != 12
    ):
        raise ValueError("complete observed12-package closure required")
    if sys.version_info[:2] != (3, 11):
        raise ValueError("observed Python3.11 profile required")
    for name, expected in pins:
        if metadata.version(name) != expected:
            raise ValueError("observed renderer version mismatch: " + name)
    import matplotlib.pyplot
    from PIL import Image

    assert matplotlib.pyplot is not None and Image is not None
    return {
        "status": "OBSERVED_RENDERER_IMPORTS_PASSED_NOT_BENCHMARK_TASK",
        "observed_versions_verified": 12,
        "author_lock": False,
        "original_renderer_imported": False,
        "task_executed": False,
        "vqa_executed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().lock), indent=2))


if __name__ == "__main__":
    main()
