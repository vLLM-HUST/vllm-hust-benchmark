from pathlib import Path

import pytest

from vllm_hust_benchmark.integration import (
    RepoLayout,
    aggregate_to_website,
    build_benchmark_script_command,
    build_performance_suite_command,
    build_vllm_bench_command,
    resolve_repo_layout,
    validate_repo_layout,
)


def test_resolve_repo_layout_defaults_to_repo_sibling_workspace(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_HUST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("VLLM_HUST_REPO", raising=False)
    monkeypatch.delenv("VLLM_HUST_WEBSITE_REPO", raising=False)

    layout = resolve_repo_layout()

    assert layout.workspace_root == Path("/home/shuhao").resolve()
    assert layout.vllm_hust_repo == Path("/home/shuhao/vllm-hust").resolve()


def test_resolve_repo_layout_from_workspace_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_HUST_WORKSPACE_ROOT", str(tmp_path))

    layout = resolve_repo_layout()

    assert layout.vllm_hust_repo == (tmp_path / "vllm-hust").resolve()
    assert layout.website_repo == (tmp_path / "vllm-hust-website").resolve()


def test_validate_repo_layout_requires_expected_repos(tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )

    with pytest.raises(ValueError, match="vllm-hust repository not found"):
        validate_repo_layout(layout)


def test_build_benchmark_script_command(tmp_path: Path) -> None:
    repo = tmp_path / "vllm-hust"
    script = repo / "benchmarks" / "benchmark_serving.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=repo,
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_benchmark_script_command(layout, "benchmark_serving.py", ["--help"])

    assert command[1] == str(script)
    assert command[-1] == "--help"


def test_build_vllm_bench_command() -> None:
    command = build_vllm_bench_command(["serve", "--model", "foo/bar"])
    assert command[1:4] == ["-m", "vllm.entrypoints.cli.main", "bench"]
    assert command[-3:] == ["serve", "--model", "foo/bar"]


def test_build_performance_suite_command(tmp_path: Path) -> None:
    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=tmp_path / "vllm-hust-website",
    )

    command = build_performance_suite_command(layout)

    assert command[0] == "bash"
    assert command[1].endswith("run-performance-benchmarks.sh")


def test_aggregate_to_website_without_execute_prints_command(
    capsys, tmp_path: Path
) -> None:
    website_repo = tmp_path / "vllm-hust-website"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "vllm-hust-benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        website_repo=website_repo,
    )

    exit_code = aggregate_to_website(
        layout=layout,
        source_dir=tmp_path / "exports",
        execute=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aggregate_results.py" in captured.out
    assert "--output-dir" in captured.out