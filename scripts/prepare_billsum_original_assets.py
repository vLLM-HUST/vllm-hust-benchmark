"""Pinned publisher BillSum HF view and official task-specific Pegasus assets."""

import importlib.util
from pathlib import Path

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/billsum-author-pins.json"
)


def main():
    spec = importlib.util.spec_from_file_location(
        "original_asset_preparation",
        Path(__file__).with_name("prepare_xsum_original_assets.py"),
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.main(pins_path=PINS, source_id="BILLSUM")


if __name__ == "__main__":
    main()
