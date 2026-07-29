from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark import perfgate_measurement


def _per_run_entry(
    run_index: int,
    *,
    throughput: float = 100.0,
    ttft: float = 50.0,
    tbt: float = 10.0,
    error_rate: float = 0.0,
    peak_mem: float | None = None,
) -> dict:
    return {
        "run_index": run_index,
        "raw_result_sha256": hashlib.sha256(str(run_index).encode()).hexdigest(),
        "metrics": {
            "throughput_tps": throughput,
            "ttft_ms": ttft,
            "tbt_ms": tbt,
            "error_rate": error_rate,
            "peak_mem_mb": peak_mem,
        },
    }


def _three_runs() -> list[dict]:
    return [
        _per_run_entry(1, throughput=90.0, ttft=55.0, tbt=12.0),
        _per_run_entry(2, throughput=110.0, ttft=45.0, tbt=9.0),
        _per_run_entry(3, throughput=100.0, ttft=50.0, tbt=10.0),
    ]


def _warmup_runs(count: int = 1) -> list[dict]:
    return [
        {
            "run_index": index,
            "raw_result_sha256": hashlib.sha256(f"warmup-{index}".encode()).hexdigest(),
        }
        for index in range(1, count + 1)
    ]


def test_median_is_computed_per_metric_independently() -> None:
    aggregated, measurement = perfgate_measurement.aggregate_measured_runs(
        _three_runs(), warmup_runs=1, warmup=_warmup_runs()
    )
    assert aggregated == {
        "throughput_tps": 100.0,
        "ttft_ms": 50.0,
        "tbt_ms": 10.0,
        "error_rate": 0.0,
    }
    assert measurement["strategy"] == "warmup+median"
    assert measurement["warmup_runs"] == 1
    assert measurement["measured_runs"] == 3
    assert measurement["aggregation"] == "median"
    assert [entry["run_index"] for entry in measurement["warmup"]] == [1]
    assert [entry["run_index"] for entry in measurement["per_run"]] == [1, 2, 3]


def test_single_measured_run_median_is_identity() -> None:
    aggregated, measurement = perfgate_measurement.aggregate_measured_runs(
        [_per_run_entry(1)], warmup_runs=0
    )
    assert aggregated["throughput_tps"] == 100.0
    assert measurement["measured_runs"] == 1


def test_rejects_non_zero_error_rate() -> None:
    runs = _three_runs()
    runs[1]["metrics"]["error_rate"] = 0.25
    with pytest.raises(ValueError, match="non-zero error_rate"):
        perfgate_measurement.aggregate_measured_runs(
            runs, warmup_runs=1, warmup=_warmup_runs()
        )


def test_rejects_non_finite_metric() -> None:
    runs = _three_runs()
    runs[0]["metrics"]["ttft_ms"] = float("nan")
    with pytest.raises(ValueError, match="ttft_ms"):
        perfgate_measurement.aggregate_measured_runs(
            runs, warmup_runs=1, warmup=_warmup_runs()
        )


def test_peak_memory_is_nullable_but_validated_when_present() -> None:
    runs = _three_runs()
    runs[0]["metrics"]["peak_mem_mb"] = None
    runs[1]["metrics"]["peak_mem_mb"] = 2048
    perfgate_measurement.aggregate_measured_runs(
        runs, warmup_runs=1, warmup=_warmup_runs()
    )

    runs[1]["metrics"]["peak_mem_mb"] = float("nan")
    with pytest.raises(ValueError, match="peak_mem_mb"):
        perfgate_measurement.aggregate_measured_runs(
            runs, warmup_runs=1, warmup=_warmup_runs()
        )


def test_rejects_run_index_mismatch() -> None:
    runs = _three_runs()
    runs[2]["run_index"] = 5
    with pytest.raises(ValueError, match="run_index mismatch"):
        perfgate_measurement.aggregate_measured_runs(
            runs, warmup_runs=1, warmup=_warmup_runs()
        )


def test_rejects_missing_raw_result_sha256() -> None:
    runs = _three_runs()
    runs[0]["raw_result_sha256"] = ""
    with pytest.raises(ValueError, match="raw_result_sha256"):
        perfgate_measurement.aggregate_measured_runs(
            runs, warmup_runs=1, warmup=_warmup_runs()
        )


def test_rejects_malformed_raw_result_sha256() -> None:
    runs = _three_runs()
    runs[0]["raw_result_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        perfgate_measurement.aggregate_measured_runs(
            runs, warmup_runs=1, warmup=_warmup_runs()
        )


@pytest.mark.parametrize("warmup_runs", [True, "1", 1.5])
def test_rejects_non_integer_warmup_runs(warmup_runs: object) -> None:
    with pytest.raises(ValueError, match="warmup_runs must be an integer"):
        perfgate_measurement.aggregate_measured_runs(
            _three_runs(),
            warmup_runs=warmup_runs,  # type: ignore[arg-type]
        )


def test_rejects_empty_run_list_and_bad_aggregation() -> None:
    with pytest.raises(ValueError, match="at least one measured run"):
        perfgate_measurement.aggregate_measured_runs(
            [], warmup_runs=1, warmup=_warmup_runs()
        )
    with pytest.raises(ValueError, match="unsupported aggregation"):
        perfgate_measurement.aggregate_measured_runs(
            _three_runs(), warmup_runs=1, aggregation="mean"
        )
    with pytest.raises(ValueError, match="warmup_runs"):
        perfgate_measurement.aggregate_measured_runs(_three_runs(), warmup_runs=-1)


def test_rejects_missing_or_malformed_warmup_evidence() -> None:
    with pytest.raises(ValueError, match="warmup evidence"):
        perfgate_measurement.aggregate_measured_runs(_three_runs(), warmup_runs=1)

    warmup = _warmup_runs()
    warmup[0]["raw_result_sha256"] = "invalid"
    with pytest.raises(ValueError, match="warmup run 1.*raw_result_sha256"):
        perfgate_measurement.aggregate_measured_runs(
            _three_runs(), warmup_runs=1, warmup=warmup
        )


def test_validate_measurement_block_shape_and_cross_check() -> None:
    _, measurement = perfgate_measurement.aggregate_measured_runs(
        _three_runs(), warmup_runs=1, warmup=_warmup_runs()
    )
    perfgate_measurement.validate_measurement_block(
        measurement,
        artifact_metrics={
            "throughput_tps": 100.0,
            "ttft_ms": 50.0,
            "tbt_ms": 10.0,
            "error_rate": 0.0,
        },
    )

    with pytest.raises(ValueError, match="does not match median"):
        perfgate_measurement.validate_measurement_block(
            measurement,
            artifact_metrics={
                "throughput_tps": 101.0,
                "ttft_ms": 50.0,
                "tbt_ms": 10.0,
                "error_rate": 0.0,
            },
        )

    broken = dict(measurement, measured_runs=2)
    with pytest.raises(ValueError, match="per_run"):
        perfgate_measurement.validate_measurement_block(broken)

    with pytest.raises(ValueError, match="strategy"):
        perfgate_measurement.validate_measurement_block(
            dict(measurement, strategy="single")
        )

    with pytest.raises(ValueError, match="must be an object"):
        perfgate_measurement.validate_measurement_block("not-a-dict")


def test_publication_policy_rejects_single_cold_measurement() -> None:
    _, measurement = perfgate_measurement.aggregate_measured_runs(
        [_per_run_entry(1)], warmup_runs=0
    )
    with pytest.raises(ValueError, match="publication requires at least"):
        perfgate_measurement.validate_measurement_block(
            measurement,
            artifact_metrics=measurement["per_run"][0]["metrics"],
            require_publication_policy=True,
        )


def test_apply_median_metrics_preserves_template_except_median_metrics(
    tmp_path: Path,
) -> None:
    template = tmp_path / "run_leaderboard.json"
    template.write_text(
        json.dumps(
            {
                "metrics": {
                    "throughput_tps": 90.0,
                    "ttft_ms": 55.0,
                    "tbt_ms": 12.0,
                    "error_rate": 0.0,
                    "peak_mem_mb": 2048,
                },
                "same_spec": {"spec_id": "spec", "resolved_spec_hash": "abc"},
                "metadata": {"git_commit": "1" * 40},
            }
        ),
        encoding="utf-8",
    )
    aggregated, _ = perfgate_measurement.aggregate_measured_runs(
        _three_runs(), warmup_runs=1, warmup=_warmup_runs()
    )
    output = tmp_path / "aggregated.json"
    payload = perfgate_measurement.apply_median_metrics(template, aggregated, output)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == payload
    assert written["metrics"]["throughput_tps"] == 100.0
    assert written["metrics"]["ttft_ms"] == 50.0
    assert written["metrics"]["tbt_ms"] == 10.0
    assert written["metrics"]["error_rate"] == 0.0
    # Server-lifetime and identity fields keep the template values.
    assert written["metrics"]["peak_mem_mb"] == 2048
    assert written["same_spec"] == {"spec_id": "spec", "resolved_spec_hash": "abc"}
    assert written["metadata"] == {"git_commit": "1" * 40}


def test_run_metrics_from_raw_result_derives_and_validates(tmp_path: Path) -> None:
    raw = tmp_path / "raw_benchmark_result.json"
    raw.write_text(
        json.dumps(
            {
                "completed": 8,
                "failed": 0,
                "mean_ttft_ms": 51.5,
                "mean_tpot_ms": 10.5,
                "output_throughput": 99.5,
            }
        ),
        encoding="utf-8",
    )
    metrics = perfgate_measurement.run_metrics_from_raw_result(raw)
    assert metrics["throughput_tps"] == 99.5
    assert metrics["ttft_ms"] == 51.5
    assert metrics["tbt_ms"] == 10.5
    assert metrics["error_rate"] == 0.0

    failing = tmp_path / "failing.json"
    failing.write_text(
        json.dumps(
            {
                "completed": 7,
                "failed": 1,
                "mean_ttft_ms": 51.5,
                "mean_tpot_ms": 10.5,
                "output_throughput": 99.5,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-zero error_rate"):
        perfgate_measurement.run_metrics_from_raw_result(failing)


def test_aggregate_cli_records_warmup_checksum_for_one_measured_run(
    tmp_path: Path,
) -> None:
    raw_payload = {
        "completed": 8,
        "failed": 0,
        "mean_ttft_ms": 50.0,
        "mean_tpot_ms": 10.0,
        "output_throughput": 100.0,
    }
    warmup = tmp_path / "warmup.json"
    measured = tmp_path / "measured.json"
    for path in (warmup, measured):
        path.write_text(json.dumps(raw_payload), encoding="utf-8")
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "metrics": {
                    "throughput_tps": 100.0,
                    "ttft_ms": 50.0,
                    "tbt_ms": 10.0,
                    "error_rate": 0.0,
                    "peak_mem_mb": None,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run_leaderboard.json"
    measurement_out = tmp_path / "measurement.json"

    result = perfgate_measurement.run_aggregate_runs_cli(
        argparse.Namespace(
            warmup_raw_results=[str(warmup)],
            run_raw_results=[str(measured)],
            warmup_runs=1,
            aggregation="median",
            template=str(template),
            output=str(output),
            measurement_out=str(measurement_out),
        )
    )

    assert result == 0
    measurement = json.loads(measurement_out.read_text(encoding="utf-8"))
    assert measurement["warmup_runs"] == 1
    assert measurement["measured_runs"] == 1
    assert (
        measurement["warmup"][0]["raw_result_sha256"]
        == hashlib.sha256(warmup.read_bytes()).hexdigest()
    )
