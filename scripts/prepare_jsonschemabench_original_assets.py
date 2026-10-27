"""Acquire/verify pinned original official HF configs on external server storage.

Reuses the existing immutable asset preparer. No schema rewriting, formatter,
benchmark execution, inference or research analysis. Dataset payload stays private.
"""

from pathlib import Path

from prepare_xsum_original_assets import main

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/jsonschemabench-official-assets.json"
)


if __name__ == "__main__":
    main(pins_path=PINS, source_id="JSONSCHEMABENCH", parts=("data",))
