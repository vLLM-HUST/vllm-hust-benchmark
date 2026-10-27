"""Prepare pinned publisher MBPP splits externally, without executing code."""

from pathlib import Path

from prepare_xsum_original_assets import main

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mbpp-original-assets.json"
)


if __name__ == "__main__":
    main(pins_path=PINS, source_id="MBPP", parts=("data",))
