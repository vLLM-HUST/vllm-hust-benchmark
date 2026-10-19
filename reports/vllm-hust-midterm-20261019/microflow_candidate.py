"""Load the sealed repeated FA screen for report and website projections."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABEL = "E24/R36 + KV + FA Microflow（3次中位数）"
RUN = "mmlu-microflow-school028-3pair-median"


def load():
    directory = ROOT / "microflow-school028"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["independent_pairs"] == 3
    assert len(summary["runs"]) == 6
    for row in summary["runs"]:
        assert row["verification"]["requests"] == 200
        assert row["verification"]["failures"] == 0
        assert (
            hashlib.sha256((directory / row["archive"]).read_bytes()).hexdigest()
            == row["archive_sha256"]
        )
    values = summary["aggregates"]["on"]
    measured = {
        "requests": 600,
        "failures": 0,
        "repetitions": 3,
        "aggregation": "Median of per-run statistics; not pooled requests",
        "input_tokens_per_run": 282265,
        "output_tokens_per_run": 51200,
        "output_tps": values["output_throughput"]["median"],
        "ttft_s": {"mean": values["mean_ttft_ms"]["median"] / 1000},
        "tpot_s": {
            "mean": values["mean_tpot_ms"]["median"] / 1000,
            "p99": values["p99_tpot_nearest_ms"]["median"] / 1000,
        },
    }
    return summary, measured
