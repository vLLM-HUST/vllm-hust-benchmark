"""Prepare the publisher-recommended Qwen3.5-9B reader; never start a service."""

from pathlib import Path

from prepare_xsum_original_assets import main


if __name__ == "__main__":
    main(
        pins_path=Path(__file__).resolve().parents[1]
        / "configs/workload-native/longmemeval-v2-reader-pins.json",
        source_id="LONGMEMEVAL_V2_READER",
        parts=("model",),
        hf_download_query=True,
    )
