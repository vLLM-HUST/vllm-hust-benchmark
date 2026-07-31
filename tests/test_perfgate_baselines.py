from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vllm_hust_benchmark import perfgate_baselines


TARGET_REPOSITORY = "vLLM-HUST/vllm-hust"
TARGET_SHA = "1" * 40
PLUGIN_SHA = "2" * 40
BENCHMARK_SHA = "3" * 40
SPEC_ID = "perfgate-ascend-qwen25-3b-910b2"
SAME_SPEC_PAYLOAD = {
    "schema_version": "benchmark-same-spec/v1",
    "spec_id": "perfgate-ascend-qwen25-3b-910b2",
    "scenario": "random-online",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "model_parameters": "3B",
    "model_precision": "BF16",
    "model_quantization": "",
    "hardware_vendor": "Ascend",
    "hardware_chip_model": "910B2",
    "chip_count": 1,
    "node_count": 1,
    "resolved_server_parameters": {
        "dtype": "bfloat16",
        "enforce_eager": "",
        "max_model_len": 4096,
    },
    "resolved_client_parameters": {
        "num_prompts": 8,
        "request_rate": "inf",
    },
}
SPEC_HASH = perfgate_baselines.compute_resolved_spec_hash(SAME_SPEC_PAYLOAD)


def _identity(**overrides: str) -> perfgate_baselines.BaselineIdentity:
    values = {
        "target_repository": TARGET_REPOSITORY,
        "target_sha": TARGET_SHA,
        "scenario": "random-online",
        "spec_id": SPEC_ID,
        "spec_hash": SPEC_HASH,
        **overrides,
    }
    return perfgate_baselines.BaselineIdentity(**values)


def _provenance(
    **overrides: str,
) -> perfgate_baselines.BaselineProvenance:
    values = {
        "vllm_hust_sha": TARGET_SHA,
        "vllm_ascend_hust_sha": PLUGIN_SHA,
        "benchmark_runner_sha": BENCHMARK_SHA,
        "hardware_chip_model": "910B2",
        "cann_version": "9.0.0",
        "torch_version": "2.10.0",
        "torch_npu_version": "2.10.0",
        "runtime_manager_sha": "d" * 40,
        **overrides,
    }
    return perfgate_baselines.BaselineProvenance(**values)


def _write_artifact(
    path: Path,
    *,
    identity: perfgate_baselines.BaselineIdentity | None = None,
    throughput: float = 100.0,
    error_rate: float | None = 0.0,
) -> None:
    identity = perfgate_baselines.normalize_identity(identity or _identity())
    engine_sha = (
        identity.target_sha
        if identity.target_repository == "vLLM-HUST/vllm-hust"
        else TARGET_SHA
    )
    plugin_sha = (
        identity.target_sha
        if identity.target_repository == "vLLM-HUST/vllm-ascend-hust"
        else PLUGIN_SHA
    )
    metrics = {
        "throughput_tps": throughput,
        "ttft_ms": 50.0,
        "tbt_ms": 10.0,
    }
    if error_rate is not None:
        metrics["error_rate"] = error_rate
    path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "same_spec": {
                    **SAME_SPEC_PAYLOAD,
                    "resolved_spec_hash": identity.spec_hash,
                },
                "metadata": {
                    "github_repository": identity.target_repository,
                    "git_commit": identity.target_sha,
                    "runtime_provenance": {
                        "engine": {
                            "repository": "vLLM-HUST/vllm-hust",
                            "commit": engine_sha,
                        },
                        "plugin": {
                            "repository": "vLLM-HUST/vllm-ascend-hust",
                            "commit": plugin_sha,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_target_main(repository: Path) -> str:
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    _git("config", "user.name", "Test User", cwd=repository)
    _git("config", "user.email", "test@example.com", cwd=repository)
    (repository / "README.md").write_text("target\n", encoding="utf-8")
    _git("add", "README.md", cwd=repository)
    _git("commit", "-m", "initial target", cwd=repository)
    return _git("rev-parse", "HEAD", cwd=repository)


def test_central_path_contains_complete_identity() -> None:
    relative = perfgate_baselines.baseline_relative_dir(_identity())

    assert relative.as_posix() == (
        "baselines/vLLM-HUST/vllm-hust/"
        f"{TARGET_SHA}/random-online/{SPEC_ID}/{SPEC_HASH}"
    )


def test_store_and_exact_fetch_validate_provenance(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    output = tmp_path / "fetched.json"
    _write_artifact(artifact)

    destination = perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(),
        update_latest_pointer=True,
    )
    fetched = perfgate_baselines.fetch_baseline(
        central,
        output,
        _identity(),
        expected_provenance=_provenance(),
    )

    assert destination.read_bytes() == artifact.read_bytes()
    assert fetched.read_bytes() == artifact.read_bytes()
    manifest = json.loads(
        (destination.parent / "baseline-metadata.json").read_text(encoding="utf-8")
    )
    assert manifest["provenance"]["runtime_manager_sha"] == "d" * 40
    pointer = central / perfgate_baselines.latest_pointer_relative_path(_identity())
    assert json.loads(pointer.read_text(encoding="utf-8"))["identity"] == {
        "scenario": "random-online",
        "spec_hash": SPEC_HASH,
        "spec_id": SPEC_ID,
        "target_repository": TARGET_REPOSITORY,
        "target_sha": TARGET_SHA,
    }


def test_runtime_manager_sha_is_persisted_and_compared(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    provenance = _provenance(runtime_manager_sha="d" * 40)

    destination = perfgate_baselines.store_baseline(
        central, artifact, _identity(), provenance
    )
    manifest = json.loads(
        (destination.parent / "baseline-metadata.json").read_text(encoding="utf-8")
    )
    assert manifest["provenance"]["runtime_manager_sha"] == "d" * 40
    perfgate_baselines.fetch_baseline(
        central,
        tmp_path / "fetched.json",
        _identity(),
        expected_provenance=provenance,
    )

    with pytest.raises(ValueError, match="exact central baseline provenance mismatch"):
        perfgate_baselines.fetch_baseline(
            central,
            tmp_path / "mismatched.json",
            _identity(),
            expected_provenance=_provenance(runtime_manager_sha="e" * 40),
        )


def test_runtime_versions_accept_local_version_identifiers(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    destination = perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(
            torch_version="2.10.0+cpu",
            torch_npu_version="2.10.0.post1+git.abc123",
        ),
    )

    assert destination.is_file()


def test_store_is_idempotent_but_refuses_overwrite(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    first = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance()
    )
    second = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance()
    )
    assert first == second

    _write_artifact(artifact, throughput=99.0)
    with pytest.raises(ValueError, match="different content"):
        perfgate_baselines.store_baseline(central, artifact, _identity(), _provenance())


def _measurement(
    *,
    throughputs: tuple[float, float, float] = (90.0, 100.0, 110.0),
    ttft: float = 50.0,
    tbt: float = 10.0,
) -> dict:
    ordered_run_indices = sorted(
        range(1, len(throughputs) + 1),
        key=lambda index: (throughputs[index - 1], index),
    )
    selected_position = len(ordered_run_indices) // 2 + 1
    selected_run_index = ordered_run_indices[selected_position - 1]
    return {
        "schema_version": "perfgate-measurement/v2",
        "strategy": "warmup+primary-median-run",
        "warmup_runs": 1,
        "measured_runs": 3,
        "aggregation": "primary-median-run",
        "selection": {
            "primary_metric": "throughput_tps",
            "sort_direction": "ascending",
            "secondary_sort_key": "run_index",
            "ordered_run_indices": ordered_run_indices,
            "selected_position": selected_position,
            "selected_run_index": selected_run_index,
            "selected_raw_result_sha256": f"{selected_run_index:064x}",
        },
        "warmup": [
            {
                "run_index": 1,
                "raw_result_sha256": "b" * 64,
            }
        ],
        "per_run": [
            {
                "run_index": index,
                "raw_result_sha256": f"{index:064x}",
                "metrics": {
                    "throughput_tps": value,
                    "ttft_ms": ttft,
                    "tbt_ms": tbt,
                    "error_rate": 0.0,
                    "peak_mem_mb": None,
                },
            }
            for index, value in enumerate(throughputs, start=1)
        ],
    }


def test_store_embeds_measurement_and_round_trips_manifest(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    # Artifact metrics match the complete middle run after throughput sorting.
    _write_artifact(artifact)

    destination = perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(),
        measurement=_measurement(),
    )
    manifest = json.loads(
        (destination.parent / "baseline-metadata.json").read_text(encoding="utf-8")
    )
    assert manifest["measurement"]["strategy"] == "warmup+primary-median-run"
    assert manifest["measurement"]["selection"]["selected_run_index"] == 2
    assert manifest["measurement"]["measured_runs"] == 3

    # load_manifest validates the measurement block against the artifact.
    loaded, _provenance_out = perfgate_baselines.load_manifest(central, _identity())
    assert loaded == destination


def test_store_rejects_selected_run_metric_mismatch(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    with pytest.raises(ValueError, match="does not match selected run"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(),
            _provenance(),
            measurement=_measurement(throughputs=(80.0, 90.0, 95.0)),
        )


def test_store_rejects_legacy_per_metric_median_measurement(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    measurement = _measurement()
    measurement.pop("schema_version")
    measurement.pop("selection")
    measurement["strategy"] = "warmup+median"
    measurement["aggregation"] = "median"

    with pytest.raises(ValueError, match="schema_version"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(),
            _provenance(),
            measurement=measurement,
        )


def test_store_rejects_measurement_below_publication_policy(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    measurement = _measurement()
    measurement["warmup_runs"] = 0
    measurement["warmup"] = []

    with pytest.raises(ValueError, match="publication requires at least"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(),
            _provenance(),
            measurement=measurement,
        )


def test_store_with_measurement_is_idempotent_but_refuses_changes(
    tmp_path: Path,
) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    first = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance(), measurement=_measurement()
    )
    second = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance(), measurement=_measurement()
    )
    assert first == second

    changed = _measurement()
    changed["warmup_runs"] = 2
    changed["warmup"].append(
        {
            "run_index": 2,
            "raw_result_sha256": "c" * 64,
        }
    )
    with pytest.raises(ValueError, match="different content"):
        perfgate_baselines.store_baseline(
            central, artifact, _identity(), _provenance(), measurement=changed
        )


def test_manifest_without_measurement_remains_valid(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    destination = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance()
    )
    manifest = json.loads(
        (destination.parent / "baseline-metadata.json").read_text(encoding="utf-8")
    )
    assert "measurement" not in manifest
    loaded, _provenance_out = perfgate_baselines.load_manifest(central, _identity())
    assert loaded == destination
    with pytest.raises(ValueError, match="measurement metadata is missing"):
        perfgate_baselines.load_manifest(central, _identity(), require_measurement=True)


def test_load_manifest_rejects_corrupt_measurement_block(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    destination = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance(), measurement=_measurement()
    )
    manifest_path = destination.parent / "baseline-metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["measurement"]["per_run"][0]["metrics"]["throughput_tps"] = 500.0
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="measurement"):
        perfgate_baselines.load_manifest(central, _identity())


@pytest.mark.parametrize("status", ["quarantined", "withdrawn"])
def test_revoked_baseline_is_retained_but_consumer_rejects_it(
    tmp_path: Path, status: str
) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    destination = perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(),
        measurement=_measurement(),
    )

    revocation = perfgate_baselines.record_revocation(
        central,
        _identity(),
        status=status,
        reason="post-publication quality failure",
        detected_at="2026-07-29T12:00:00Z",
        detection_run_url="https://github.example/runs/detection",
        actor="baseline-owner",
        workflow_url="https://github.example/runs/revocation",
        publication_commit="9" * 40,
    )
    payload = json.loads(revocation.read_text(encoding="utf-8"))
    assert payload["release_visibility_status"] == status
    assert payload["artifact_sha256"]
    assert payload["actor"] == "baseline-owner"
    assert destination.is_file()

    with pytest.raises(ValueError, match=f"exact central baseline is {status}"):
        perfgate_baselines.load_manifest(central, _identity(), require_measurement=True)
    with pytest.raises(ValueError, match="cannot store a revoked exact baseline"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(),
            _provenance(),
            measurement=_measurement(),
        )


def test_revocation_record_is_immutable_and_idempotent(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(),
        measurement=_measurement(),
    )
    arguments = {
        "status": "quarantined",
        "reason": "quality failure",
        "detected_at": "2026-07-29T12:00:00Z",
        "detection_run_url": "https://github.example/runs/detection",
        "actor": "baseline-owner",
        "workflow_url": "https://github.example/runs/revocation",
        "publication_commit": "9" * 40,
    }

    first = perfgate_baselines.record_revocation(central, _identity(), **arguments)
    second = perfgate_baselines.record_revocation(central, _identity(), **arguments)
    assert first == second

    with pytest.raises(ValueError, match="different content"):
        perfgate_baselines.record_revocation(
            central, _identity(), **dict(arguments, actor="different-actor")
        )

    withdrawn = perfgate_baselines.record_revocation(
        central,
        _identity(),
        **dict(
            arguments,
            status="withdrawn",
            reason="formal withdrawal completed",
            workflow_url="https://github.example/runs/withdrawal",
        ),
    )
    assert withdrawn != first
    with pytest.raises(ValueError, match="exact central baseline is withdrawn"):
        perfgate_baselines.load_manifest(central, _identity())
    with pytest.raises(ValueError, match="after it has been withdrawn"):
        perfgate_baselines.record_revocation(
            central,
            _identity(),
            **dict(
                arguments,
                status="quarantined",
                reason="late quarantine must be rejected",
            ),
        )


def test_revoke_cli_writes_a_consumer_blocking_record(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(),
        measurement=_measurement(),
    )

    result = perfgate_baselines.main(
        [
            "revoke",
            "--repository-root",
            str(central),
            "--status",
            "quarantined",
            "--reason",
            "quality failure",
            "--detected-at",
            "2026-07-29T12:00:00Z",
            "--detection-run-url",
            "https://github.example/runs/detection",
            "--actor",
            "baseline-owner",
            "--workflow-url",
            "https://github.example/runs/revocation",
            "--publication-commit",
            "9" * 40,
            "--target-repository",
            _identity().target_repository,
            "--target-sha",
            _identity().target_sha,
            "--scenario",
            _identity().scenario,
            "--spec-id",
            _identity().spec_id,
            "--spec-hash",
            _identity().spec_hash,
        ]
    )

    assert result == 0
    with pytest.raises(ValueError, match="exact central baseline is quarantined"):
        perfgate_baselines.load_manifest(central, _identity())


def test_revocation_verifies_publication_commit_in_git_checkout(
    tmp_path: Path,
) -> None:
    central = tmp_path / "central"
    central.mkdir()
    _git("init", "-b", "benchmark-baselines", cwd=central)
    _git("config", "user.name", "Test User", cwd=central)
    _git("config", "user.email", "test@example.com", cwd=central)
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    perfgate_baselines.store_baseline(
        central,
        artifact,
        _identity(),
        _provenance(),
        measurement=_measurement(),
    )
    _git("add", "baselines", cwd=central)
    _git("commit", "-m", "publish baseline", cwd=central)
    publication_commit = _git("rev-parse", "HEAD", cwd=central)
    arguments = {
        "status": "quarantined",
        "reason": "quality failure",
        "detected_at": "2026-07-29T12:00:00Z",
        "detection_run_url": "https://github.example/runs/detection",
        "actor": "baseline-owner",
        "workflow_url": "https://github.example/runs/revocation",
    }

    with pytest.raises(ValueError, match="publication_commit does not match"):
        perfgate_baselines.record_revocation(
            central,
            _identity(),
            publication_commit="8" * 40,
            **arguments,
        )
    perfgate_baselines.record_revocation(
        central,
        _identity(),
        publication_commit=publication_commit,
        **arguments,
    )


def test_store_rejects_spec_and_companion_mismatch(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    with pytest.raises(ValueError, match="same-spec identity mismatch"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(spec_hash="b" * 64),
            _provenance(),
        )
    with pytest.raises(ValueError, match="plugin provenance mismatch"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(),
            _provenance(vllm_ascend_hust_sha="4" * 40),
        )


@pytest.mark.parametrize("error_rate", [None, 0.01, 1.0])
def test_validate_artifact_rejects_missing_or_nonzero_error_rate(
    tmp_path: Path, error_rate: float | None
) -> None:
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact, error_rate=error_rate)

    with pytest.raises(ValueError, match="invalid required metrics: error_rate"):
        perfgate_baselines.validate_artifact(artifact, _identity(), _provenance())


def test_validate_artifact_recomputes_resolved_spec_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["same_spec"]["resolved_server_parameters"]["max_num_seqs"] = 32
    artifact.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resolved spec hash mismatch"):
        perfgate_baselines.validate_artifact(artifact, _identity(), _provenance())


def test_store_rejects_symlinked_baselines_directory(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (central / "baselines").symlink_to(victim, target_is_directory=True)
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)

    with pytest.raises(ValueError, match="contains a symlink"):
        perfgate_baselines.store_baseline(central, artifact, _identity(), _provenance())
    assert list(victim.iterdir()) == []


def test_store_rejects_symlinked_latest_pointer(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    perfgate_baselines.store_baseline(central, artifact, _identity(), _provenance())
    pointer = central / perfgate_baselines.latest_pointer_relative_path(_identity())
    pointer.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.json"
    victim.write_text("preserve me\n", encoding="utf-8")
    pointer.symlink_to(victim)

    with pytest.raises(ValueError, match="contains a symlink"):
        perfgate_baselines.store_baseline(
            central,
            artifact,
            _identity(),
            _provenance(),
            update_latest_pointer=True,
        )
    assert victim.read_text(encoding="utf-8") == "preserve me\n"


def test_baseline_rejected_when_symlink_in_path(tmp_path: Path) -> None:
    repository_root = tmp_path / "central"
    repository_root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (repository_root / "baselines").symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        perfgate_baselines._reject_symlink_components(
            repository_root,
            perfgate_baselines.baseline_relative_dir(_identity()),
        )


def test_fetch_rejects_missing_exact_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact central baseline is unavailable"):
        perfgate_baselines.fetch_baseline(
            tmp_path,
            tmp_path / "output.json",
            _identity(),
        )


def test_fetch_rejects_tampered_artifact(tmp_path: Path) -> None:
    central = tmp_path / "central"
    central.mkdir()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    destination = perfgate_baselines.store_baseline(
        central, artifact, _identity(), _provenance()
    )
    destination.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        perfgate_baselines.fetch_baseline(
            central,
            tmp_path / "output.json",
            _identity(),
        )


def test_store_cli_requires_target_sha_on_main(tmp_path: Path) -> None:
    target = tmp_path / "target"
    actual_sha = _commit_target_main(target)
    artifact = tmp_path / "run_leaderboard.json"
    identity = _identity(target_sha=actual_sha)
    _write_artifact(artifact, identity=identity)
    central = tmp_path / "central"
    central.mkdir()
    measurement = tmp_path / "measurement.json"
    measurement.write_text(
        json.dumps(_measurement(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    common = [
        "store",
        "--repository-root",
        str(central),
        "--source",
        str(artifact),
        "--measurement-file",
        str(measurement),
        "--target-git-repository",
        str(target),
        "--main-ref",
        "main",
        "--target-repository",
        TARGET_REPOSITORY,
        "--target-sha",
        actual_sha,
        "--scenario",
        "random-online",
        "--spec-id",
        SPEC_ID,
        "--spec-hash",
        SPEC_HASH,
        "--vllm-hust-sha",
        actual_sha,
        "--vllm-ascend-hust-sha",
        PLUGIN_SHA,
        "--benchmark-runner-sha",
        BENCHMARK_SHA,
        "--runtime-manager-sha",
        "d" * 40,
        "--hardware-chip-model",
        "910B2",
        "--cann-version",
        "9.0.0",
        "--torch-version",
        "2.10.0",
        "--torch-npu-version",
        "2.10.0",
    ]

    assert perfgate_baselines.main(common) == 0
    assert (
        perfgate_baselines.main(
            [
                *common[: common.index("--target-sha") + 1],
                "f" * 40,
                *common[common.index("--target-sha") + 2 :],
            ]
        )
        == 2
    )


def test_latest_pointer_requires_current_main_tip(tmp_path: Path) -> None:
    target = tmp_path / "target"
    old_sha = _commit_target_main(target)
    (target / "README.md").write_text("new tip\n", encoding="utf-8")
    _git("add", "README.md", cwd=target)
    _git("commit", "-m", "advance target", cwd=target)
    new_sha = _git("rev-parse", "HEAD", cwd=target)

    perfgate_baselines.verify_main_commit(target, old_sha, "main")
    with pytest.raises(ValueError, match="current main tip"):
        perfgate_baselines.verify_main_commit(target, old_sha, "main", require_tip=True)
    perfgate_baselines.verify_main_commit(target, new_sha, "main", require_tip=True)


def test_latest_main_rejected_when_sha_not_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "target"
    main_sha = _commit_target_main(target)
    _git("checkout", "--orphan", "divergent", cwd=target)
    _git("rm", "-rf", "--ignore-unmatch", ".", cwd=target)
    (target / "DIVERGENT.md").write_text("divergent\n", encoding="utf-8")
    _git("add", "DIVERGENT.md", cwd=target)
    _git("commit", "-m", "divergent orphan", cwd=target)
    divergent_sha = _git("rev-parse", "HEAD", cwd=target)
    _git("checkout", "main", cwd=target)

    with pytest.raises(ValueError, match="is not an ancestor of main"):
        perfgate_baselines.verify_main_commit(target, divergent_sha, "main")
    perfgate_baselines.verify_main_commit(target, main_sha, "main")


def test_bare_git_branch_can_be_bootstrapped_and_updated(tmp_path: Path) -> None:
    remote = tmp_path / "central.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    _git("init", "--bare", str(remote), cwd=tmp_path)
    seed.mkdir()
    _git("init", "-b", "main", cwd=seed)
    _git("config", "user.name", "Test User", cwd=seed)
    _git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "README.md").write_text("central\n", encoding="utf-8")
    _git("add", "README.md", cwd=seed)
    _git("commit", "-m", "seed", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    _git("clone", str(remote), str(checkout), cwd=tmp_path)
    _git("config", "user.name", "Baseline Writer", cwd=checkout)
    _git("config", "user.email", "writer@example.com", cwd=checkout)
    _git("checkout", "--orphan", "benchmark-baselines", cwd=checkout)
    for child in checkout.iterdir():
        if child.name != ".git" and child.is_file():
            child.unlink()
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact)
    perfgate_baselines.store_baseline(checkout, artifact, _identity(), _provenance())
    _git("add", "baselines", cwd=checkout)
    _git("commit", "-m", "store first baseline", cwd=checkout)
    _git("push", "origin", "benchmark-baselines", cwd=checkout)

    second_sha = "4" * 40
    second_identity = _identity(target_sha=second_sha)
    second_artifact = tmp_path / "second-run-leaderboard.json"
    _write_artifact(second_artifact, identity=second_identity)
    perfgate_baselines.store_baseline(
        checkout,
        second_artifact,
        second_identity,
        _provenance(vllm_hust_sha=second_sha),
    )
    _git("add", "baselines", cwd=checkout)
    _git("commit", "-m", "store second baseline", cwd=checkout)
    _git("push", "origin", "benchmark-baselines", cwd=checkout)

    assert (
        _git(
            "ls-remote", "--heads", "origin", "benchmark-baselines", cwd=checkout
        ).split()[1]
        == "refs/heads/benchmark-baselines"
    )
    assert _git(
        "show",
        f"benchmark-baselines:{perfgate_baselines.baseline_relative_dir(second_identity)}/baseline-metadata.json",
        cwd=checkout,
    )


def test_publish_bootstraps_updates_and_is_idempotent(tmp_path: Path) -> None:
    remote = tmp_path / "central.git"
    seed = tmp_path / "seed"
    target = tmp_path / "target"
    target_sha = _commit_target_main(target)
    identity = _identity(target_sha=target_sha)
    provenance = _provenance(vllm_hust_sha=target_sha)
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact, identity=identity)

    _git("init", "--bare", str(remote), cwd=tmp_path)
    seed.mkdir()
    _git("init", "-b", "main", cwd=seed)
    _git("config", "user.name", "Test User", cwd=seed)
    _git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "README.md").write_text("central\n", encoding="utf-8")
    _git("add", "README.md", cwd=seed)
    _git("commit", "-m", "seed", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    first = perfgate_baselines.publish_baseline(
        str(remote),
        "benchmark-baselines",
        artifact,
        target,
        "main",
        identity,
        provenance,
        update_latest_pointer=True,
        measurement=_measurement(),
    )
    second = perfgate_baselines.publish_baseline(
        str(remote),
        "benchmark-baselines",
        artifact,
        target,
        "main",
        identity,
        provenance,
        update_latest_pointer=True,
        measurement=_measurement(),
    )

    assert first.startswith("published:")
    assert second.startswith("unchanged:")
    assert _git(
        "ls-remote", "--heads", str(remote), "benchmark-baselines", cwd=tmp_path
    )


def test_publish_requires_measurement_metadata(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target_sha = _commit_target_main(target)
    identity = _identity(target_sha=target_sha)
    provenance = _provenance(vllm_hust_sha=target_sha)
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact, identity=identity)

    with pytest.raises(ValueError, match="measurement metadata is required"):
        perfgate_baselines.publish_baseline(
            "unused-remote",
            "benchmark-baselines",
            artifact,
            target,
            "main",
            identity,
            provenance,
        )


def test_publish_retries_non_fast_forward_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "central.git"
    seed = tmp_path / "seed"
    target = tmp_path / "target"
    target_sha = _commit_target_main(target)
    identity = _identity(target_sha=target_sha)
    provenance = _provenance(vllm_hust_sha=target_sha)
    artifact = tmp_path / "run_leaderboard.json"
    _write_artifact(artifact, identity=identity)
    _git("init", "--bare", str(remote), cwd=tmp_path)
    seed.mkdir()
    _git("init", "-b", "main", cwd=seed)
    _git("config", "user.name", "Test User", cwd=seed)
    _git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "README.md").write_text("central\n", encoding="utf-8")
    _git("add", "README.md", cwd=seed)
    _git("commit", "-m", "seed", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    original_run_git = perfgate_baselines._run_git
    failed_once = False

    def fail_first_push(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal failed_once
        if arguments and arguments[0] == "push" and not failed_once:
            failed_once = True
            return subprocess.CompletedProcess(
                ["git", *arguments], 1, "", "simulated non-fast-forward"
            )
        return original_run_git(arguments, cwd=cwd, check=check)

    monkeypatch.setattr(perfgate_baselines, "_run_git", fail_first_push)
    result = perfgate_baselines.publish_baseline(
        str(remote),
        "benchmark-baselines",
        artifact,
        target,
        "main",
        identity,
        provenance,
        max_attempts=2,
        measurement=_measurement(),
    )

    assert failed_once is True
    assert result.startswith("published:")
