"""Acquire the original default8B embedding model; never start a service."""

from pathlib import Path

from prepare_xsum_original_assets import main

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/longmemeval-v2-embedding-pins.json"
)


if __name__ == "__main__":
    main(
        pins_path=PINS,
        source_id="LONGMEMEVAL_V2_EMBEDDING",
        parts=("model",),
        hf_download_query=True,
    )
