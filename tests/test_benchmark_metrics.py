from vllm_hust_benchmark.benchmark_metrics import derive_metrics_from_benchmark_result


def test_offline_batch_completion_is_not_first_token_latency():
    metrics = derive_metrics_from_benchmark_result(
        {"avg_latency": 2.5}, peak_mem_mb=None, benchmark_type="latency"
    )
    assert metrics["batch_latency_ms"] == 2500
    assert metrics["ttft_ms"] is metrics["tbt_ms"] is metrics["throughput_tps"] is None
    assert metrics["peak_mem_mb"] is metrics["error_rate"] is None


def test_request_rate_is_not_a_token_rate():
    metrics = derive_metrics_from_benchmark_result(
        {"requests_per_second": 20}, peak_mem_mb=None
    )
    assert metrics["throughput_tps"] is None


def test_measured_zero_and_output_token_basis_are_preserved():
    metrics = derive_metrics_from_benchmark_result(
        {
            "completed": 3,
            "failed": 1,
            "output_throughput": 0,
            "tokens_per_second": 100,
            "mean_ttft_ms": 0,
            "mean_tpot_ms": 0,
        },
        peak_mem_mb=0,
    )
    assert (
        metrics["throughput_tps"]
        == metrics["ttft_ms"]
        == metrics["tbt_ms"]
        == metrics["peak_mem_mb"]
        == 0
    )
    assert metrics["error_rate"] == 0.25


def test_offline_total_token_rate_does_not_invent_streaming_latency():
    metrics = derive_metrics_from_benchmark_result(
        {"tokens_per_second": 500}, peak_mem_mb=None, benchmark_type="throughput"
    )
    assert metrics["throughput_tps"] == 500
    assert metrics["ttft_ms"] is None
