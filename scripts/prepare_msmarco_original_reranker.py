"""Acquire the publisher's MS MARCO cross-encoder; never execute a neural model."""

from pathlib import Path

from prepare_xsum_original_assets import main


if __name__ == "__main__":
    main(
        pins_path=Path(__file__).resolve().parents[1]
        / "configs/workload-native/msmarco-reranker-pins.json",
        source_id="MSMARCO_RERANKER",
        parts=("model",),
        hf_download_query=True,
    )
