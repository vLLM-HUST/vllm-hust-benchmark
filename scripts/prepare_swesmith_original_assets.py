"""Reuse immutable asset preparer for original SWE-smith tasks, not trajectories."""

from pathlib import Path

from prepare_xsum_original_assets import main

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/swesmith-original-task-assets.json"
)


if __name__ == "__main__":
    main(pins_path=PINS, source_id="SWESMITH", parts=("data",))
