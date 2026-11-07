"""Inert software archive fixtures; no original tasks, Docker or models."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "swe_source", SCRIPTS / "prepare_swebench_author_source.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_release_profile_plan_does_not_create_or_execute(
    tmp_path, monkeypatch, capsys
):
    import json
    import subprocess

    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_swebench_author_source.py"),
            "--profile",
            "v4.1.0",
            "--author-root",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 87 and report["bytes"] == 594174
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert not output.exists()


def test_cpu_recovery_does_not_qualify_repository_tasks_or_new_sources():
    receipt = json.loads(
        (
            SCRIPTS.parent
            / "docs/workload-data/validation/swebench-original-readiness-20261029.json"
        ).read_text(encoding="utf-8")
    )
    recovery = receipt["independent_cpu_recovery"]
    assert recovery["original_daemon_start_ticks_before_and_after_equal"]
    assert not recovery["native_benchmark_runtime_images_qualified"]
    assert recovery["native_benchmark_tasks_executed"] == 0
    assert recovery["source_checkbox_count_increment"] == 0
    components = recovery["restored_render_components"]
    for name in ("matplotlib", "browser"):
        assert components[name]["exact_config_and_all_diff_ids_preserved"]
        assert components[name]["environment_verifier_exit_code"] == 0
    assert len(components["public_controller_controls"]) == 4
    timeout = components["public_controller_controls"][-1]
    assert timeout["status"] == "BLOCKED_COMPONENT_TIME_LIMIT_NOT_MODEL_SCORE"
    assert timeout["render_score"] is None
    assert components["new_daemon_running_containers_after_controls"] == 0
    assert components["model_calls"] == components["native_tasks_executed"] == 0
    assert components["source_checkbox_increment"] == 0
    assert (
        recovery["new_backing_file_capacity_bytes"] + 1024**3
        <= recovery["approved_total_extra_budget_bytes"]
    )


def test_classic_original_asset_plan_is_separate_from_verified():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_reused_research_assets.py"),
            "SWE_BENCH",
            "--mode",
            "plan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["source_id"] == "SWE_BENCH"
    assert plan["bytes"] == 12097227
    assert len(plan["files"]) == 1
    assert plan["files"][0]["rows"] == 2294
    assert len(plan["files"][0]["fields"]) == 12
    assert "difficulty" not in plan["files"][0]["fields"]
    assert plan["files"][0]["role"] == "data"
    assert plan["model_calls"] == 0
    assert not plan["source_checkbox_eligible"]
    assert not plan["native_execution_complete"]


def test_classic_native_delivery_keeps_original_failures_and_research_scope():
    """Receipt consistency only, not native execution or external scoring."""
    receipt = json.loads(
        (
            SCRIPTS.parent
            / "docs/workload-data/validation/swebench-classic-native-delivery-20261030.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["source_id"] == "SWE_BENCH" and receipt["source_checkbox_eligible"]
    assert receipt["original_dataset"]["rows"] == 2294
    assert receipt["original_dataset"]["fields"] == 12
    assert receipt["fixed_original_ids"] == [
        "astropy__astropy-11693",
        "astropy__astropy-12057",
        "astropy__astropy-12318",
    ]
    native = receipt["native_execution"]
    assert native["total_backend_requests_including_incomplete_original_attempt"] == (
        native["original_attempt_backend_requests_http200"]
        + native["same_original_transport_recovery_requests_http200"]
    )
    assert native["original_transport_failures_retained"] == 2
    assert not native["first_original_case_rerun"]
    assert native["quality_failure_reruns"] == 0
    assert native[
        "composite_sample_outcomes_not_single_uninterrupted_official_score"
    ] == {
        "agent_submitted": 3,
        "official_tests_completed": 2,
        "resolved": 0,
        "unresolved": 2,
        "invalid_patch_before_tests": 1,
    }
    assert (
        receipt["saved_result_audit"]["new_model_calls_or_functional_test_reruns"] == 0
    )
    assert receipt["data_and_research_consumer"][
        "full2294_original_records_fields_and_index_verified"
    ]
    assert not receipt["raw_dataset_model_weights_or_native_outputs_published"]


def test_verified_completion_cannot_complete_classic_or_hide_negative_outcomes():
    """Receipt consistency only; this test does not execute or rescore tasks."""
    folder = SCRIPTS.parent / "docs/workload-data"
    receipt = json.loads(
        (folder / "validation/swebench-original-readiness-20261029.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (folder / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["delivery_inventory"]
    assert receipt["source_ids"] == ["SWE_VERIFIED"]
    assert receipt["shared_preparation_source_ids"] == ["SWE_BENCH", "SWE_VERIFIED"]
    assert receipt["source_checkbox_eligible"]
    assert inventory["SWE_VERIFIED"]["checked"]
    if inventory["SWE_BENCH"]["checked"]:
        assert (
            inventory["SWE_BENCH"]["delivery_receipt"]
            != inventory["SWE_VERIFIED"]["delivery_receipt"]
        ), "Classic must close with its own source-scoped evidence"
    assert receipt["public_entry_code_ci"]["both_conclusions"] == "success"
    assert receipt["official_grading_attempts"] == {
        "gold_control": 3,
        "model_predictions": 3,
    }
    assert receipt["original_test_executions_and_score_reports"] == 5
    assert "not all cases passing" in receipt["completion_scope"]
    assert "not SWE_BENCH completion" in receipt["completion_scope"]


@pytest.mark.parametrize("profile,count", [("core", 76), ("agent", 120)])
def test_public_cpu_dependency_locks_are_unique_hash_bound_and_not_inference_packages(
    profile, count
):
    import re

    path = (
        SCRIPTS.parent
        / f"configs/workload-native/swebench-original-{profile}-py312-aarch64.lock"
    )
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(rows) == count
    names = set()
    for line in rows:
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+) --hash=sha256:([0-9a-f]{64})", line
        )
        assert match
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        assert name not in names and name not in {
            "torch",
            "torch-npu",
            "triton",
            "flash-attn",
        }
        names.add(name)
    assert "pyarrow" in names
    if profile == "agent":
        assert {"mcp", "litellm", "jinja2"} <= names


def test_official_agent_profile_is_plan_only_and_pins_all_runtime_configs(tmp_path):
    import subprocess

    output = tmp_path / "not-created-agent"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_swebench_author_source.py"),
            "--profile",
            "mini-swe-agent",
            "--author-root",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 78 and report["bytes"] == 302226
    assert report["status"] == "PLAN_ONLY"
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not output.exists()
    pins = json.loads(
        (
            SCRIPTS.parent / "configs/workload-native/mini-swe-agent-author-source.json"
        ).read_text(encoding="utf-8")
    )
    assert pins["repository"] == "SWE-agent/mini-swe-agent"
    assert {"SWE_BENCH", "SWE_VERIFIED"} == set(pins["shared_source_ids"])
    names = {item["path"] for item in pins["files"]}
    assert {
        "LICENSE.md",
        "pyproject.toml",
        "src/minisweagent/environments/docker.py",
        "src/minisweagent/config/benchmarks/swebench.yaml",
        "src/minisweagent/run/benchmarks/swebench.py",
    } <= names


def fixture(tmp_path, names=None, content=b"inert original software\n", kind=None):
    archive = tmp_path / "source.tar.gz"
    names = names or ["publisher-fixed/swebench/__init__.py"]
    with tarfile.open(archive, "w:gz") as handle:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            if kind is not None:
                info.type = kind
            handle.addfile(info, io.BytesIO(content))
    pins = {
        "files": [
            {
                "path": "swebench/__init__.py",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest(),
            }
        ]
    }
    return archive, pins


def test_original_software_preparation_and_identical_resume(tmp_path):
    archive, pins = fixture(tmp_path)
    output = tmp_path / "external"
    assert module.prepare(archive, output, pins) == 1
    assert module.prepare(archive, output, pins) == 1
    assert (
        output / "swebench/__init__.py"
    ).read_bytes() == b"inert original software\n"


@pytest.mark.parametrize(
    "names",
    [
        ["../swebench/__init__.py"],
        ["/publisher/swebench/__init__.py"],
        ["publisher-fixed/missing.py"],
        ["publisher-fixed/swebench/__init__.py"] * 2,
        ["publisher-fixed/swebench/__init__.py", "other-root/README.md"],
    ],
)
def test_unsafe_duplicate_missing_or_multiple_root_archives_rejected(tmp_path, names):
    archive, pins = fixture(tmp_path, names)
    with pytest.raises(ValueError):
        module.prepare(archive, tmp_path / "external", pins)


@pytest.mark.parametrize("field", ["sha256", "git_blob", "bytes"])
def test_identity_mismatch_cannot_publish_file(tmp_path, field):
    archive, pins = fixture(tmp_path)
    pins["files"][0][field] = (
        0 if field == "bytes" else "0" * len(pins["files"][0][field])
    )
    output = tmp_path / "external"
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert not (output / "swebench/__init__.py").exists()


def test_nonregular_selected_member_rejected(tmp_path):
    archive, pins = fixture(tmp_path, kind=tarfile.SYMTYPE)
    with pytest.raises(ValueError):
        module.prepare(archive, tmp_path / "external", pins)


def test_existing_different_file_never_overwritten(tmp_path):
    archive, pins = fixture(tmp_path)
    output = tmp_path / "external"
    target = output / "swebench/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing unrelated bytes")
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert target.read_bytes() == b"existing unrelated bytes"


def test_checkout_output_rejected_before_writes(tmp_path):
    archive, pins = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the checkout"):
        module.prepare(archive, module.ROOT / "never-create-swebench-source", pins)


def lfs_fixture(tmp_path):
    archive, pins = fixture(tmp_path)
    item = pins["files"][0]
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{item['sha256']}\nsize {item['bytes']}\n".encode()
    item["git_lfs"] = {"pointer_bytes": len(pointer), "oid_sha256": item["sha256"]}
    item["git_blob"] = hashlib.sha1(
        b"blob " + str(len(pointer)).encode() + b"\0" + pointer
    ).hexdigest()
    item["git_mode"] = "100755"
    return archive, pins


def test_original_lfs_payload_preserved_and_identical_resume(tmp_path):
    import os

    archive, pins = lfs_fixture(tmp_path)
    output = tmp_path / "external"
    assert module.prepare(archive, output, pins) == 1
    target = output / "swebench/__init__.py"
    assert target.read_bytes() == b"inert original software\n"
    module.verify_git_blob(target, pins["files"][0])
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o755
    assert module.prepare(archive, output, pins) == 1
    target.write_bytes(b"altered original bytes\n")
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert target.read_bytes() == b"altered original bytes\n"


@pytest.mark.parametrize(
    "tamper", ["oid", "pointer_size", "git_blob", "unknown_field", "git_mode"]
)
def test_lfs_identity_corruption_fails_without_writing_payload(tmp_path, tamper):
    archive, pins = lfs_fixture(tmp_path)
    item = pins["files"][0]
    if tamper == "oid":
        item["git_lfs"]["oid_sha256"] = "0" * 64
    elif tamper == "pointer_size":
        item["git_lfs"]["pointer_bytes"] += 1
    elif tamper == "git_blob":
        item["git_blob"] = "0" * 40
    elif tamper == "git_mode":
        item["git_mode"] = "120000"
    else:
        item["git_lfs"]["unexpected"] = True
    output = tmp_path / "external"
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert not (output / "swebench/__init__.py").exists()


def test_lfs_pointer_only_download_is_not_expanded_payload(tmp_path):
    from prepare_crosscodeeval_original_assets import acquire

    _, pins = lfs_fixture(tmp_path)
    with pytest.raises(ValueError, match="original publisher archive"):
        acquire(tmp_path / "not-created", pins["files"][0])
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    "profile,source,count,size",
    [
        ("webarena", "WEBARENA", 88, 7550641),
        ("mlebench", "MLEBENCH", 758, 7501882),
        ("visualwebarena", "VISUALWEBARENA", 478, 200942354),
        ("osworld", "OSWORLD", 1324, 21403779),
        ("paperbench", "PAPERBENCH", 195, 2114907),
    ],
)
def test_reused_archive_profiles_plan_only(tmp_path, profile, source, count, size):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_swebench_author_source.py"),
            "--profile",
            profile,
            "--author-root",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["source_ids"] == [source]
    assert plan["files"] == count and plan["bytes"] == size
    assert plan["model_calls"] == 0 and not plan["source_checkbox_eligible"]
    assert not output.exists()


def test_visualwebarena_source_plan_exposes_software_not_native_completion():
    from vllm_hust_workload_data.provider import delivery_source_plan

    plan = delivery_source_plan("VISUALWEBARENA")
    assert any(
        entry["path"] == "scripts/prepare_swebench_author_source.py"
        and "--profile visualwebarena" in entry["scope"]
        for entry in plan["entrypoints"]
    )
    catalog = json.loads(
        (
            SCRIPTS.parent
            / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )
    # Original task custody is now readable; it is not native Agent completion.
    sources = catalog.get("sources", catalog)
    source = sources["VISUALWEBARENA"]
    assert source["native_software_profile"]["profile"] == "visualwebarena"
    pins = json.loads(
        (SCRIPTS.parent / source["native_software_profile"]["manifest"]).read_text(
            "utf-8"
        )
    )
    assert source["revision"] == pins["revision"]
    expected_tasks = {
        "LICENSE",
        "README.md",
        "config_files/vwa/test_classifieds.raw.json",
        "config_files/vwa/test_reddit.raw.json",
        "config_files/vwa/test_shopping.raw.json",
    }
    assert {row["path"] for row in source["files"]} == expected_tasks
    original = {row["path"]: row for row in pins["files"]}
    for row in source["files"]:
        assert row["role"] == "raw_source"
        assert row["bytes"] == original[row["path"]]["bytes"]
        assert row["sha256"] == original[row["path"]]["sha256"]
    assert any(row["role"] == "data_reading_only" for row in plan["entrypoints"])
    paths = {item["path"] for item in pins["files"]}
    assert {
        "run.py",
        "LICENSE",
        "requirements.txt",
        "scripts/generate_test_data.py",
        "config_files/vwa/test_classifieds.raw.json",
        "config_files/vwa/test_shopping.raw.json",
        "config_files/vwa/test_reddit.raw.json",
    } <= paths
    assert not any(path.startswith(("media/", "tests/")) for path in paths)


def test_mlebench_cli_recipe_keeps_installation_separate_from_native_tasks():
    recipe = (
        SCRIPTS.parent / "configs/workload-native/mlebench-arm64-cli.Dockerfile"
    ).read_text("utf-8")
    assert "FROM quay.io/ascend/vllm-ascend@sha256:" in recipe
    assert "python3 -m venv /opt/mlebench-client" in recipe
    assert "--system-site-packages" not in recipe
    assert 'PYTHONPATH=""' in recipe and "PYTHONNOUSERSITE=1" in recipe
    assert "mlebench-arm64-client-py312.txt" in recipe
    assert "--no-deps --no-build-isolation /opt/mlebench-author" in recipe
    assert recipe.count("python -m pip check") == 2
    assert "mlebench --help" in recipe
    assert 'CMD ["--help"]' in recipe
    assert "WORKDIR /tmp" in recipe
    assert all(
        term not in recipe
        for term in ("kaggle.json", "KAGGLE_KEY", "grade-sample", "--all")
    )


def test_osworld_source_plan_does_not_hide_submodule_and_vm_gaps():
    from vllm_hust_workload_data.provider import delivery_source_plan

    entries = delivery_source_plan("OSWORLD")["entrypoints"]
    assert any("--profile osworld" in entry["scope"] for entry in entries)
    assert any(
        "submodules" in entry["scope"] and "VM" in entry["scope"] for entry in entries
    )
    pins = json.loads(
        (
            SCRIPTS.parent / "configs/workload-native/osworld-author-source.json"
        ).read_text("utf-8")
    )
    paths = {item["path"] for item in pins["files"]}
    assert {
        "desktop_env/providers/docker/provider.py",
        "desktop_env/providers/docker/manager.py",
        "evaluation_examples/test_all.json",
        "run.py",
    } <= paths
    assert "separate unqualified" in pins["scope"]


def test_paperbench_keeps_required_original_splits_not_historical_results():
    pins = json.loads(
        (
            SCRIPTS.parent / "configs/workload-native/paperbench-author-source.json"
        ).read_text("utf-8")
    )
    experiment_files = {
        x["path"] for x in pins["files"] if "/experiments/" in x["path"]
    }
    assert experiment_files == {
        "project/paperbench/experiments/splits/" + name + ".txt"
        for name in ["all", "debug", "dev", "human", "lite", "testing"]
    }
