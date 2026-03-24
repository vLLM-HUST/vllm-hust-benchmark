from vllm_hust_benchmark.cli import main


def test_build_command_prints_upstream_equivalent(capsys) -> None:
    exit_code = main(
        [
            "build-command",
            "sharegpt-online",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "vllm bench serve" in captured.out
    assert "--dataset-name sharegpt" in captured.out


def test_run_without_execute_only_prints(capsys) -> None:
    exit_code = main(
        [
            "run",
            "random-latency",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--set",
            "input_len=2048",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "vllm bench latency" in captured.out
    assert "--input-len 2048" in captured.out


def test_export_leaderboard_artifact(tmp_path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(
                """
{
    "metrics": {
        "ttft_ms": 42.0,
        "throughput_tps": 321.0,
        "peak_mem_mb": 10240,
        "error_rate": 0.0
    },
    "constraints_metrics": {
        "single_chip_effective_utilization_pct": 92.0,
        "typical_throughput_ratio_vs_baseline": 2.2,
        "typical_ttft_reduction_pct_vs_baseline": 23.0,
        "typical_tpot_reduction_pct_vs_baseline": 25.0,
        "long_context_length": 32768,
        "long_context_throughput_stable": true,
        "long_context_ttft_p95_ms": 80.0,
        "long_context_ttft_p99_ms": 95.0,
        "long_context_tpot_p95_ms": 9.0,
        "long_context_tpot_p99_ms": 10.0,
        "long_context_ttft_p95_stable": true,
        "long_context_ttft_p99_stable": true,
        "long_context_tpot_p95_stable": true,
        "long_context_tpot_p99_stable": true,
        "unit_token_cost_reduction_pct": 35.0,
        "multi_tenant_high_utilization": true
    }
}
""".strip()
                + "\n",
                encoding="utf-8",
        )

        output_dir = tmp_path / "export"
        exit_code = main(
                [
                        "export-leaderboard-artifact",
                        "sharegpt-online",
                        "--metrics-file",
                        str(metrics_file),
                        "--output-dir",
                        str(output_dir),
                        "--run-id",
                        "smoke-run-1",
                        "--engine",
                        "vllm-hust",
                        "--engine-version",
                        "0.7.3",
                        "--model-name",
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "--hardware-chip-model",
                        "Ascend-910B",
                        "--submitter",
                        "ci",
                ]
        )

        assert exit_code == 0
        assert (output_dir / "run_leaderboard.json").is_file()
        assert (output_dir / "leaderboard_manifest.json").is_file()