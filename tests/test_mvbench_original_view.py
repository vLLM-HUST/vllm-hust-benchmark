import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/prepare_mvbench_original_view.py"
SPEC = importlib.util.spec_from_file_location("mvbench_original_view", ENTRY)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_does_not_require_files_or_optional_video_packages(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ENTRY),
            "--official-checkout",
            str(tmp_path / "absent-code"),
            "--snapshot",
            str(tmp_path / "absent-data"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["original_tasks"] == 4000
    assert plan["model_calls"] == 0
    assert list(tmp_path.iterdir()) == []


def test_unverified_source_cannot_execute(tmp_path):
    marker = tmp_path / "unexpected"
    source = f"open({str(marker)!r}, 'w').write('unsafe')".encode()
    with pytest.raises(ValueError, match="Unverified original"):
        MODULE.original_generator(source)
    assert not marker.exists()


def test_fixed_sample_covers_all_twenty_original_types():
    profile = json.loads(
        (ROOT / "configs/workload-native/mvbench-original-protocol.json").read_text(
            encoding="utf-8"
        )
    )
    chosen = profile["fixed_selection"]
    assert len(chosen) == 60
    tasks = {row["task"] for row in chosen}
    assert len(tasks) == 20
    for task in tasks:
        assert [row["original_offset"] for row in chosen if row["task"] == task] == [
            0,
            1,
            2,
        ]
    assert profile["dataset_revision"] == MODULE.REVISION


@pytest.mark.parametrize("full", [False, True])
def test_native_runner_plan_without_video_or_model_access(tmp_path, full):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_mvbench_original.py"),
            "--official-checkout",
            str(tmp_path / "missing"),
            "--snapshot",
            str(tmp_path / "missing"),
            "--frame-cache",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
            "--model",
            "example",
            *(["--full"] if full else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["selected_original_tasks"] == (4000 if full else 60)
    assert plan["frames_each"] == 16
    assert plan["model_calls"] == 0
    assert list(tmp_path.iterdir()) == []


def test_native_fixed_positions_preserves_all_groups_and_source_indices(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_mvbench_original as runner

    data = [{"task_type": f"task-{task}"} for task in range(20) for _ in range(4)]
    expected = [task * 4 + offset for task in range(20) for offset in range(3)]
    assert runner.fixed_positions(data) == expected
    with pytest.raises(ValueError, match="twenty-type"):
        runner.fixed_positions(data[:76])


def asset_module():
    spec = importlib.util.spec_from_file_location(
        "mvbench_original_assets", ROOT / "scripts/prepare_mvbench_original_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_publisher_file_manifest_is_not_duplicate_task_tables():
    module = asset_module()
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    files = manifest["files"]
    assert len(files) == len({item["path"] for item in files}) == 3895
    assert sum(item["bytes"] for item in files) == 15796297352
    assert [item["path"] for item in files if item["role"] == "data"] == ["test.json"]
    assert manifest["revision"] == MODULE.REVISION


def test_complete_asset_verification_and_regular_view_do_not_copy_media(tmp_path):
    import hashlib

    module = asset_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    raw = b"fixture-only-not-original-data"
    (snapshot / "fixture.bin").write_bytes(raw)
    manifest = {
        "files": [
            {
                "path": "fixture.bin",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
    }
    assert module.verify_snapshot(snapshot, manifest) == 1
    view = tmp_path / "regular-view"
    module.materialize(snapshot, view, manifest)
    assert (snapshot / "fixture.bin").samefile(view / "fixture.bin")
    with pytest.raises(FileExistsError):
        module.materialize(snapshot, view, manifest)
    (snapshot / "fixture.bin").write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.verify_snapshot(view, manifest)


def test_native_cache_ref_never_overwrites_other_revision(tmp_path):
    module = asset_module()
    manifest = {"revision": "fixture-revision"}
    module.ensure_native_ref(tmp_path, manifest)
    module.ensure_native_ref(tmp_path, manifest)
    with pytest.raises(ValueError, match="no overwrite"):
        module.ensure_native_ref(tmp_path, {"revision": "different-fixture"})


def test_source_delivery_requires_native_scoring_and_actual_ci_research_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    item = ledger["delivery_inventory"]["MVBENCH"]
    proof = json.loads((ROOT / item["delivery_receipt"]).read_text(encoding="utf-8"))
    assert item["status"] == "AVAILABLE" and item["checked"]
    assert proof["checked"] and proof["source_checkbox_eligible"]
    assert proof["native_completed_cases"] == proof["model_calls"] == 60
    assert proof["native_correct_cases"] == 34
    assert proof["frames_per_case"] == 16
    assert proof["native_child_returncode"] == 0
    assert proof["saved_native_audit"]["original_qa_template_wire_text_verified"]
    ci = proof["actual_ci_installed_evidence"]
    assert ci["verified_installed_payloads"] == 62
    assert ci["original_rows_verified"] == ci["indexed_original_records"] == 4000
    assert ci["fixed_original_provider_and_current_research_roundtrips"] == 60
    assert ci["additional_model_calls"] == 0
    assert ci["native_entrypoints_discoverable_from_source_plan"] == 4
    assert proof["remaining"] == [] and proof["validation_limits"]


def test_delivery_navigation_exposes_the_actual_public_entrypoints():
    from vllm_hust_workload_data.provider import delivery_source_plan

    source = delivery_source_plan("MVBENCH")
    entries = {(entry["role"], entry["path"]) for entry in source["entrypoints"]}
    assert ("prepare_assets", "scripts/prepare_mvbench_original_assets.py") in entries
    assert ("prepare_assets", "scripts/prepare_mvbench_original_view.py") in entries
    assert ("generation", "scripts/run_mvbench_original.py") in entries
    assert (
        "original_profile",
        "configs/workload-native/mvbench-original-protocol.json",
    ) in entries
