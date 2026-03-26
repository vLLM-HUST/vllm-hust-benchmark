from pathlib import Path

from vllm_hust_benchmark.cli import main
from vllm_hust_benchmark.integration import RepoLayout


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


def test_show_repos_prints_resolved_layout(capsys, monkeypatch, tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["show-repos"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(layout.vllm_hust_repo) in captured.out
    assert str(layout.website_repo) in captured.out


def test_list_tests_prints_upstream_tests(capsys, monkeypatch, tmp_path: Path) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests_dir / "serving-tests.json").write_text(
        '[{"test_name":"serving_llama8B_tp1_sharegpt","qps_list":["inf"],"server_parameters":{"model":"foo/bar"},"client_parameters":{"model":"foo/bar"}}]\n',
        encoding="utf-8",
    )
    (tests_dir / "latency-tests.json").write_text("[]\n", encoding="utf-8")
    (tests_dir / "throughput-tests.json").write_text("[]\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["list-tests", "--benchmark-type", "serve"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "serving_llama8B_tp1_sharegpt" in captured.out


def test_show_test_prints_wrapped_commands(capsys, monkeypatch, tmp_path: Path) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests_dir / "serving-tests.json").write_text(
        '[{"test_name":"serving_llama8B_tp1_sharegpt","qps_list":["inf"],"server_parameters":{"model":"foo/bar","tensor_parallel_size":1},"client_parameters":{"model":"foo/bar","backend":"vllm"}}]\n',
        encoding="utf-8",
    )
    (tests_dir / "latency-tests.json").write_text("[]\n", encoding="utf-8")
    (tests_dir / "throughput-tests.json").write_text("[]\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["show-test", "serving_llama8B_tp1_sharegpt"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "server_command: vllm serve foo/bar --tensor-parallel-size 1" in captured.out
    assert "client_command: vllm bench serve --request-rate inf --model foo/bar --backend vllm" in captured.out


def test_run_test_without_execute_prints_suite_wrapper(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    vllm_repo = tmp_path / "vllm-hust"
    tests_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "tests"
    tests_dir.mkdir(parents=True)
    (vllm_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (vllm_repo / "benchmarks").mkdir()
    suite_dir = vllm_repo / ".buildkite" / "performance-benchmarks" / "scripts"
    suite_dir.mkdir(parents=True)
    (suite_dir / "run-performance-benchmarks.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests_dir / "serving-tests.json").write_text("[]\n", encoding="utf-8")
    (tests_dir / "latency-tests.json").write_text(
        '[{"test_name":"latency_llama8B_tp1","parameters":{"model":"foo/bar"}}]\n',
        encoding="utf-8",
    )
    (tests_dir / "throughput-tests.json").write_text("[]\n", encoding="utf-8")
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=vllm_repo,
        website_repo=website_repo,
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)

    exit_code = main(["run-test", "latency_llama8B_tp1", "--env", "DRY_RUN=1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "TEST_SELECTOR='^latency_llama8B_tp1$' DRY_RUN=1 bash" in captured.out


def test_bench_without_execute_prints_wrapped_command(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr("vllm_hust_benchmark.cli.validate_repo_layout", lambda _layout: None)

    exit_code = main(["bench", "--", "serve", "--model", "foo/bar"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "-m vllm.entrypoints.cli.main bench serve --model foo/bar" in captured.out


def test_publish_website_without_execute_prints_aggregate_command(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )
    monkeypatch.setattr("vllm_hust_benchmark.cli.resolve_repo_layout", lambda: layout)
    monkeypatch.setattr("vllm_hust_benchmark.cli.validate_repo_layout", lambda _layout: None)

    source_dir = tmp_path / "exports"
    exit_code = main(["publish-website", "--source-dir", str(source_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aggregate_results.py" in captured.out
    assert f"--source-dir {source_dir}" in captured.out


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


def test_export_leaderboard_artifact_from_raw_benchmark_result(tmp_path) -> None:
    benchmark_result = tmp_path / "serve_result.json"
    benchmark_result.write_text(
        """
{
    "completed": 10,
    "failed": 1,
    "request_throughput": 5.5,
    "output_throughput": 321.0,
    "mean_ttft_ms": 42.0,
    "errors": [null, null, null, null, null, null, null, null, null, null, "boom"]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    constraints_file = tmp_path / "constraints.json"
    constraints_file.write_text(
        """
{
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
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "export_raw"
    exit_code = main(
        [
            "export-leaderboard-artifact",
            "sharegpt-online",
            "--benchmark-result-file",
            str(benchmark_result),
            "--constraints-file",
            str(constraints_file),
            "--peak-mem-mb",
            "10240",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "smoke-run-2",
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