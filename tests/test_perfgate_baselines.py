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
SPEC_HASH = "a" * 64


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
        **overrides,
    }
    return perfgate_baselines.BaselineProvenance(**values)


def _write_artifact(
    path: Path,
    *,
    identity: perfgate_baselines.BaselineIdentity | None = None,
    throughput: float = 100.0,
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
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "throughput_tps": throughput,
                    "ttft_ms": 50.0,
                    "tbt_ms": 10.0,
                },
                "same_spec": {
                    "scenario": identity.scenario,
                    "spec_id": identity.spec_id,
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
    pointer = central / perfgate_baselines.latest_pointer_relative_path(_identity())
    assert json.loads(pointer.read_text(encoding="utf-8"))["identity"] == {
        "scenario": "random-online",
        "spec_hash": SPEC_HASH,
        "spec_id": SPEC_ID,
        "target_repository": TARGET_REPOSITORY,
        "target_sha": TARGET_SHA,
    }


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

    common = [
        "store",
        "--repository-root",
        str(central),
        "--source",
        str(artifact),
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
    )

    assert first.startswith("published:")
    assert second.startswith("unchanged:")
    assert _git(
        "ls-remote", "--heads", str(remote), "benchmark-baselines", cwd=tmp_path
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
    )

    assert failed_once is True
    assert result.startswith("published:")
