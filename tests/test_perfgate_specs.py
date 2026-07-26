from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_benchmark import perfgate_specs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_registry(path: Path, entries: list[dict[str, str]]) -> None:
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def test_resolve_random_online_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="random-online",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_sharegpt_online_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="sharegpt-online",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_prefix_repetition_online_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="prefix-repetition-online",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-prefix-repetition-online-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_random_latency_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="random-latency",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-random-latency-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_sharegpt_throughput_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="sharegpt-throughput",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-sharegpt-throughput-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_sonnet_throughput_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="sonnet-throughput",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-sonnet-throughput-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_instructcoder_online_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="instructcoder-online",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-instructcoder-online-qwen25-coder-3b-910b2.json"
        ).resolve()
    )


def test_resolve_agent_research_online_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="agent-research-online",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-agent-research-online-qwen25-3b-910b2.json"
        ).resolve()
    )


def test_resolve_visionarena_online_perfgate_spec() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="visionarena-online",
        hardware_chip_model="910B2",
        repo_root=REPO_ROOT,
    )

    assert (
        spec_path
        == (
            REPO_ROOT
            / "docs"
            / "official-baselines"
            / "perfgate-ascend-visionarena-online-qwen25-vl-3b-910b2.json"
        ).resolve()
    )


def test_resolve_without_repo_root_returns_repo_relative_path() -> None:
    spec_path = perfgate_specs.resolve_perfgate_spec_file(
        scenario="random-online",
        hardware_chip_model="910b2",
    )

    assert spec_path == Path(
        "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json"
    )


def test_resolve_rejects_unsupported_pair() -> None:
    try:
        perfgate_specs.resolve_perfgate_spec_file(
            scenario="visionarena-online-2chip",
            hardware_chip_model="910B2",
            repo_root=REPO_ROOT,
        )
    except ValueError as error:
        message = str(error)
        assert "No perfgate spec registered" in message
        assert "agent-research-online/910B2" in message
        assert "instructcoder-online/910B2" in message
        assert "prefix-repetition-online/910B2" in message
        assert "random-latency/910B2" in message
        assert "random-online/910B2" in message
        assert "sharegpt-online/910B2" in message
        assert "sharegpt-throughput/910B2" in message
        assert "sonnet-throughput/910B2" in message
        assert "visionarena-online/910B2" in message
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected unsupported pair to fail")


def test_registry_rejects_duplicate_entries(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    entry = {
        "scenario": "random-online",
        "hardware_chip_model": "910B2",
        "spec_file": "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json",
    }
    _write_registry(registry_file, [entry, entry])

    try:
        perfgate_specs.load_perfgate_spec_registry(registry_file)
    except ValueError as error:
        assert "duplicate perfgate spec registry entry" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected duplicate registry entry to fail")


def test_resolve_rejects_missing_spec_file(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    _write_registry(
        registry_file,
        [
            {
                "scenario": "random-online",
                "hardware_chip_model": "910B2",
                "spec_file": "docs/official-baselines/missing.json",
            }
        ],
    )

    try:
        perfgate_specs.resolve_perfgate_spec_file(
            scenario="random-online",
            hardware_chip_model="910B2",
            repo_root=REPO_ROOT,
            registry_file=registry_file,
        )
    except ValueError as error:
        assert "perfgate spec file not found" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected missing spec file to fail")


def test_resolve_rejects_spec_scenario_mismatch(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    registry_file = tmp_path / "registry.json"
    spec_file.write_text(
        json.dumps(
            {
                "id": "test-spec",
                "scenario": "sharegpt-online",
                "hardware_chip_model": "910B2",
            }
        ),
        encoding="utf-8",
    )
    _write_registry(
        registry_file,
        [
            {
                "scenario": "random-online",
                "hardware_chip_model": "910B2",
                "spec_file": spec_file.name,
            }
        ],
    )

    try:
        perfgate_specs.resolve_perfgate_spec_file(
            scenario="random-online",
            hardware_chip_model="910B2",
            repo_root=tmp_path,
            registry_file=registry_file,
        )
    except ValueError as error:
        assert "scenario mismatch" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected spec scenario mismatch to fail")


def test_resolve_rejects_spec_chip_mismatch(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    registry_file = tmp_path / "registry.json"
    spec_file.write_text(
        json.dumps(
            {
                "id": "test-spec",
                "scenario": "random-online",
                "hardware_chip_model": "910B3",
            }
        ),
        encoding="utf-8",
    )
    _write_registry(
        registry_file,
        [
            {
                "scenario": "random-online",
                "hardware_chip_model": "910B2",
                "spec_file": spec_file.name,
            }
        ],
    )

    try:
        perfgate_specs.resolve_perfgate_spec_file(
            scenario="random-online",
            hardware_chip_model="910B2",
            repo_root=tmp_path,
            registry_file=registry_file,
        )
    except ValueError as error:
        assert "hardware_chip_model mismatch" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected spec chip mismatch to fail")


def test_resolve_rejects_missing_spec_id(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    registry_file = tmp_path / "registry.json"
    spec_file.write_text(
        json.dumps(
            {
                "scenario": "random-online",
                "hardware_chip_model": "910B2",
            }
        ),
        encoding="utf-8",
    )
    _write_registry(
        registry_file,
        [
            {
                "scenario": "random-online",
                "hardware_chip_model": "910B2",
                "spec_file": spec_file.name,
            }
        ],
    )

    try:
        perfgate_specs.resolve_perfgate_spec_file(
            scenario="random-online",
            hardware_chip_model="910B2",
            repo_root=tmp_path,
            registry_file=registry_file,
        )
    except ValueError as error:
        assert "missing required field: id" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected missing spec id to fail")


def test_cli_resolve_prints_one_absolute_path(capsys) -> None:
    exit_code = perfgate_specs.main(
        [
            "resolve",
            "--scenario",
            "sharegpt-online",
            "--hardware-chip-model",
            "910B2",
            "--repo-root",
            str(REPO_ROOT),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.splitlines() == [
        str(
            (
                REPO_ROOT
                / "docs"
                / "official-baselines"
                / "perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json"
            ).resolve()
        )
    ]


def test_cli_resolve_failure_uses_stderr(capsys) -> None:
    exit_code = perfgate_specs.main(
        [
            "resolve",
            "--scenario",
            "visionarena-online-2chip",
            "--hardware-chip-model",
            "910B2",
            "--repo-root",
            str(REPO_ROOT),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "No perfgate spec registered" in captured.err
