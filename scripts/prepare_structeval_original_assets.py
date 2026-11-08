"""Prepare fixed original StructEval data and author source outside the repository."""

from pathlib import Path

from prepare_xsum_original_assets import main

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/structeval-original-assets.json"
)


if __name__ == "__main__":
    main(pins_path=PINS, source_id="STRUCTEVAL", parts=("data", "scoring"))
