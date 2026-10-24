"""CPU contracts only: fixtures do not execute original games or model tasks."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_agentbench_alfworld_original.py"
spec = importlib.util.spec_from_file_location("agentbench_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_original_group_preparation_does_not_claim_new_native_execution():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/agentbench-original-groups-preparation-20261024.json"
        ).read_text("utf-8")
    )
    groups = receipt["groups"]
    assert [g["total_original_tasks"] for g in groups] == [9, 5, 27, 29, 18, 21]
    assert [i for g in groups for i in g["original_indices"]] == list(range(109))
    assert [i for g in groups for i in g["reused_native_indices"]] == [0, 1, 2]
    assert [i for g in groups for i in g["remaining_native_indices"]] == list(
        range(3, 109)
    )
    assert receipt["model_calls"] == receipt["native_tasks_executed_in_this_step"] == 0


def test_completed_group_receipt_matches_executed_runner_not_whole_suite():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/agentbench-original-complete-groups-audit-20261024.json"
        ).read_text("utf-8")
    )
    groups = receipt["completed_groups"]
    assert groups
    current_sha = hashlib.sha256(
        SCRIPT.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    for group in groups:
        assert group["helper_sha256"] == current_sha
        assert (
            group["official_rescore_equal"] and group["original_episode_assets_equal"]
        )
        assert len(group["native_tasks"]) == group["new_native_tasks"]
        assert runner.native_tasks_scored(group["native_tasks"])
        for task in group["native_tasks"]:
            assert task["model_calls"] > 0
            assert task["reward"] in (0, 1)
        overall = group["official_overall_including_reused"]["overall"]
        assert (
            overall["total"]
            == overall["pass"] + overall["wrong"]
            == group["total_original_tasks"]
        )
    assert receipt["verified_unique_native_tasks"] == 3 + sum(
        g["new_native_tasks"] for g in groups
    )
    assert receipt["audit_model_calls"] == 0
    assert not receipt["full_agentbench_suite_completed"]


@pytest.fixture(autouse=True)
def optional_data_provider_source(monkeypatch):
    # The root package deliberately does not require the optional data wheel.
    # Root-only CI checks the checked-in source; the dedicated wheel CI keeps
    # testing its installed distribution rather than shadowing it with source.
    if importlib.util.find_spec("vllm_hust_workload_data") is None:
        monkeypatch.syspath_prepend(str(ROOT / "workload-preparation/src"))


def fixture_inputs(tmp_path, monkeypatch):
    from vllm_hust_workload_data import provider

    paths = [f"json_2.1.1/valid_unseen/fixture-{i}/game.tw-pddl" for i in range(3)]
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "fixture/data/alfworld/new_std.json", json.dumps({"fixture": paths})
        )
    source_sha = runner.digest(archive)
    monkeypatch.setattr(provider, "AGENTBENCH_ARCHIVE_SHA256", source_sha)
    monkeypatch.setattr(runner, "SOURCE_SHA", source_sha)
    data = tmp_path / "data"
    archives = []
    hashes = {}
    for role in ("json", "pddl", "tw-pddl"):
        path = tmp_path / (role + ".zip")
        with zipfile.ZipFile(path, "w") as bundle:
            for game in paths:
                relative = (
                    str(Path(game).parent).replace("\\", "/")
                    + "/"
                    + {
                        "json": "traj_data.json",
                        "pddl": "game.pddl",
                        "tw-pddl": "game.tw-pddl",
                    }[role]
                )
                content = ("PRIVATE_FIXTURE_" + role + game).encode()
                target = data / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                bundle.writestr(relative, content)
        archives.append(path)
        hashes[role] = runner.digest(path)
    monkeypatch.setattr(provider, "AGENTBENCH_GAME_ARCHIVES", hashes)
    monkeypatch.setattr(runner, "GAME_SHAS", hashes)
    return provider, archive, archives, data


def test_data_provider_and_native_runner_share_reference_contract_without_payloads(
    tmp_path, monkeypatch
):
    provider, source, games, data = fixture_inputs(tmp_path, monkeypatch)
    references = provider.agentbench_task_references(
        official_archive=source, game_archives=games, data_root=data, indices=[0, 1, 2]
    )
    assert "PRIVATE_FIXTURE" not in json.dumps(references)
    assert all(row["preparation_only_not_execution"] for row in references)
    assert runner.validate_references(source, games, data, references) == ([0, 1, 2], 3)
    reordered = provider.agentbench_task_references(
        official_archive=source,
        game_archives=list(reversed(games)),
        data_root=data,
        indices=[0, 1, 2],
    )
    assert reordered == references


def test_duplicate_and_unknown_original_indices_are_rejected(tmp_path, monkeypatch):
    provider, source, games, data = fixture_inputs(tmp_path, monkeypatch)
    for indices in ([0, 0], [True], [-1], [3]):
        with pytest.raises(ValueError):
            provider.agentbench_task_references(
                official_archive=source,
                game_archives=games,
                data_root=data,
                indices=indices,
            )


def test_changing_game_and_claimed_hash_cannot_bypass_publisher_bytes(
    tmp_path, monkeypatch
):
    provider, source, games, data = fixture_inputs(tmp_path, monkeypatch)
    refs = provider.agentbench_task_references(
        official_archive=source, game_archives=games, data_root=data, indices=[0]
    )
    tampered = copy.deepcopy(refs)
    path = data / refs[0]["input_ref"]["game_file"]
    path.write_text("modified task")
    tampered[0]["input_ref"]["game_sha256"] = runner.digest(path)
    with pytest.raises(ValueError, match="differs from release"):
        runner.validate_references(source, games, data, tampered)
    with pytest.raises(ValueError, match="differs from publisher release"):
        provider.agentbench_task_references(
            official_archive=source, game_archives=games, data_root=data, indices=[0]
        )


def test_fixed_source_identities_match_across_python_environments():
    from vllm_hust_workload_data import provider

    assert provider.AGENTBENCH_REVISION == runner.REVISION
    assert provider.AGENTBENCH_ARCHIVE_SHA256 == runner.SOURCE_SHA
    assert provider.AGENTBENCH_GAME_ARCHIVES == runner.GAME_SHAS


def test_native_help_without_optional_agentrl_and_external_endpoint_refusal(tmp_path):
    assert (
        subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True
        ).returncode
        == 0
    )
    args = [
        sys.executable,
        str(SCRIPT),
        "--official-archive",
        "missing.zip",
        "--game-archives",
        "a.zip",
        "b.zip",
        "c.zip",
        "--runtime-source-archives",
        "tw.zip",
        "alf.zip",
        "--worker-wheel",
        "missing.whl",
        "--data-root",
        str(tmp_path),
        "--task-references",
        "refs.json",
        "--output",
        str(tmp_path / "out"),
        "--endpoint",
        "https://example.org/v1",
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2
    assert "only local /v1 model endpoints" in result.stderr
    assert not (tmp_path / "out").exists()


def test_preparation_receipt_is_not_native_execution_or_scoring():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/agentbench-alfworld-original-preparation-20261022.json"
        ).read_text("utf-8")
    )
    assert receipt["selected_indices"] == [0, 1, 2]
    assert receipt["original_task_count"] == 109
    assert receipt["model_calls"] == receipt["native_tasks_executed"] == 0
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["scoring"] == "NOT_RUN"


def test_real_native_tasks_are_distinct_from_environment_probes_and_full_suite():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/agentbench-alfworld-native-npu-20261022.json"
        ).read_text("utf-8")
    )
    # Historical initial headless-runtime attempt; its exact entry is retained
    # privately. Final standard-OpenCV entry validation has a separate receipt.
    assert (
        receipt["helper_sha256"]
        == "135481c6255fefe95d1f6a3266f8569203896b02de01a6ed4d631182b9a6bdbc"  # pragma: allowlist secret - historical public helper checksum
    )  # pragma: allowlist secret - historical executed helper checksum
    assert (
        receipt["selected_indices"] == [0, 1, 2]
        and receipt["original_task_count"] == 109
    )
    assert all(task["status"] == "completed" for task in receipt["tasks"])
    assert [task["native_environment_steps"] for task in receipt["tasks"]] == [4, 7, 20]
    assert (
        sum(task["model_calls"] for task in receipt["tasks"])
        == receipt["total_model_requests"]
        == 31
    )
    assert receipt["official_overall"]["overall"]["pass"] == 3
    assert receipt["task_prompt_action_and_scoring_code_changed"] is False
    assert (
        receipt["full_suite_claim"] is False
        and receipt["other_agentbench_domains_validated"] is False
    )


def test_runtime_build_only_changes_packaging_not_input_arguments_or_task_source():
    build_spec = importlib.util.spec_from_file_location(
        "runtime_builder", ROOT / "scripts/build_agentbench_pddl_runtime_wheels.py"
    )
    builder = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(builder)
    original = {
        "cmdclass": {"build_py": "fixture", "install": "fixture", "develop": "fixture"},
        "packages": ["textworld"],
    }
    updated = builder.adapt_packaging("textworld", original)
    assert updated["cmdclass"] == {} and updated["packages"] == original["packages"]
    assert len(original["cmdclass"]) == 3
    adapted = builder.adapt_packaging(
        "alfworld",
        {
            "install_requires": [
                "torch==1.2.0",
                "torchvision==0.4.0",
                "opencv-python",
                "Click<7.2.0",
            ]
        },
    )
    assert "torch==1.2.0" not in adapted["install_requires"]
    assert "opencv-python==4.11.0.86" in adapted["install_requires"]
    assert "Click<7.2.0" in adapted["install_requires"]
    with pytest.raises(ValueError):
        builder.adapt_packaging("unsupported", {})
    assert (
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_agentbench_pddl_runtime_wheels.py"),
                "--help",
            ],
            capture_output=True,
        ).returncode
        == 0
    )


def test_historical_public_handoff_preserves_executed_entry_and_dependency_recipe():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/agentbench-alfworld-public-handoff-npu-20261023.json"
        ).read_text("utf-8")
    )

    def sha(path):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    # This receipt belongs to the actual 2026-10-23 run, not later runner fixes.
    assert (
        receipt["helper_sha256"]
        == (
            "1bdccaa85b4f692e3bf29ff997ebb6875b5faeb838e87ab7883c41a492e4622d"  # pragma: allowlist secret - historical public helper checksum
        )
    )
    assert receipt["runtime_recipe"]["build_helper_sha256"] == sha(
        ROOT / "scripts/build_agentbench_pddl_runtime_wheels.py"
    )
    assert receipt["runtime_recipe"]["dependency_snapshot_sha256"] == sha(
        ROOT / "configs/workload-native/agentbench-alfworld-py39-dependencies.txt"
    )
    assert (
        receipt["runtime_original_source_checks"]["opencv-python"][
            "headless_namespace_overlap"
        ]
        is False
    )
    assert receipt["new_unique_tasks_vs_initial_attempt"] == 0
    assert receipt["unique_original_tasks"] == 3 < receipt["original_task_count"]
    assert [row["reward"] for row in receipt["tasks"]] == [True, True, True]


@pytest.mark.parametrize(
    "status", ["completed", "task limit reached", "agent invalid action"]
)
def test_native_scored_failure_is_not_an_infrastructure_failure(status):
    task = {"status": status, "reward": 0, "model_calls": 20}
    assert runner.native_tasks_scored([task.copy() for _ in range(3)])


@pytest.mark.parametrize(
    "status",
    [
        "running",
        "unknown",
        "task error",
        "cancelled",
        "agent context limit",
        "agent validation failed",
    ],
)
def test_incomplete_or_unsupported_native_end_does_not_pass_delivery(status):
    task = {"status": status, "reward": 0, "model_calls": 1}
    assert not runner.native_tasks_scored([task.copy() for _ in range(3)])


def test_native_scoring_requires_actual_model_calls_and_reward():
    task = {"status": "completed", "reward": 0, "model_calls": 1}
    assert not runner.native_tasks_scored([task.copy() for _ in range(2)])
    for override in ({"reward": None}, {"model_calls": 0}):
        bad = dict(task, **override)
        assert not runner.native_tasks_scored([task, task, bad])


def test_bounded_model_output_is_forwarded_without_rewriting():
    message = {"role": "assistant", "content": "unfinished original output"}
    choice = {"finish_reason": "length", "message": message}
    assert runner.model_choice({"choices": [choice]}) is choice
    for response in (
        {"choices": []},
        {"choices": [dict(choice, finish_reason="error")]},
        {"choices": [dict(choice, message=None)]},
    ):
        with pytest.raises(ValueError):
            runner.model_choice(response)
