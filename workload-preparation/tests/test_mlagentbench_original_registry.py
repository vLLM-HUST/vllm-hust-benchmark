"""Original task registry/software relations, not scientific or Agent execution."""

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_workload_data.adapters import iter_workload
from vllm_hust_workload_data.mlagentbench import REGISTRY, original_rows, relation_spec
from vllm_hust_workload_data.registry import adapter_spec, source_spec


@pytest.fixture
def bundle(tmp_path):
    tmp_path = tmp_path / "source"
    tmp_path.mkdir()
    cases = {
        "debug": {
            "research_problem": "Original fixture task\nwith exact newline",
            "benchmark_folder_name": "cifar10",
        }
    }
    payloads = {
        REGISTRY: json.dumps(cases, ensure_ascii=False).encode(),
        "MLAgentBench/benchmarks/cifar10/env/train.py": b"# fixture starter, never executed\n",
        "MLAgentBench/benchmarks/cifar10/scripts/eval.py": b"# fixture evaluator, never executed\n",
        "MLAgentBench/benchmarks/cifar10/scripts/research_problem.txt": b"Different original folder default; do not replace registry task",
        "MLAgentBench/benchmarks/another/env/train.py": b"# native folder, not an invented registry case\n",
        "LICENSE": b"fixture software license metadata",
    }
    files = []
    for name, raw in payloads.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        role = (
            "data"
            if name == REGISTRY
            else "evaluation_asset"
            if name.endswith("/eval.py")
            else "license_asset"
            if name == "LICENSE"
            else "task_software"
        )
        files.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": role,
            }
        )
    manifest = {
        "source_id": "MLAGENTBENCH",
        "revision": "explicit-fixture-not-real-release",
        "unit": "original_registered_scientific_task",
        "provenance_kind": "scientific_task_environment",
        "files": files,
    }
    return tmp_path, cases, manifest


def test_packaged_manifest_matches_all_existing_author_pins():
    root = Path(__file__).resolve().parents[2]
    original = json.loads(
        (root / "configs/workload-native/mlagentbench-author-source.json").read_text(
            "utf-8"
        )
    )
    public = source_spec("MLAGENTBENCH")
    assert public["revision"] == original["revision"]
    assert len(public["files"]) == 138
    assert sum(item["bytes"] for item in public["files"]) == 670271
    keys = ("path", "bytes", "sha256", "git_blob")
    assert [{k: f[k] for k in keys} for f in public["files"]] == [
        {k: f[k] for k in keys} for f in original["files"]
    ]
    definitions = [item["path"] for item in public["files"] if item["role"] == "data"]
    assert REGISTRY in definitions
    assert (
        len(definitions) == 16
    )  # One alias registry +15 directories, not16 paper tasks.
    assert all(
        name == REGISTRY or name.endswith("/scripts/research_problem.txt")
        for name in definitions
    )


def test_original_registry_case_and_exact_material_relationships(bundle):
    root, cases, manifest = bundle
    spec = relation_spec(manifest, adapter_spec("MLAGENTBENCH"))
    rows = list(original_rows(root / REGISTRY, spec))
    assert len(rows) == 1  # Do not turn a native folder into an invented case.
    assert rows[0]["case_id"] == "debug"
    assert rows[0]["original_registry_case"] == cases["debug"]
    assert rows[0]["original_registry_text"].encode() == (root / REGISTRY).read_bytes()
    assert rows[0]["research_problem"] == cases["debug"]["research_problem"]
    assert all(
        item["path"].startswith("MLAgentBench/benchmarks/cifar10/")
        for item in rows[0]["task_software"]
    )


def test_public_records_separate_evaluator_and_do_not_fake_requests(bundle):
    root, cases, manifest = bundle
    records = list(iter_workload("MLAGENTBENCH", data_root=root, manifest=manifest))
    assert len(records) == 1
    record = records[0]
    assert record.task["task_id"] == "debug"
    assert record.task["input"] == cases["debug"]
    assert "evaluation_assets" not in record.input_record()["task"]["input"]
    assert record.outcome["evaluation_assets"][0]["path"].endswith("/scripts/eval.py")
    assert record.runtime["underlying_scientific_data_verified"] is False
    assert record.runtime["execution"] == "not_executed"
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_changed_evaluator_does_not_silently_adapt(bundle):
    root, _, manifest = bundle
    (root / "MLAgentBench/benchmarks/cifar10/scripts/eval.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset identity"):
        list(iter_workload("MLAGENTBENCH", data_root=root, manifest=manifest))


def test_partially_consumed_reader_rechecks_materials_on_close(bundle):
    root, _, manifest = bundle
    rows = original_rows(
        root / REGISTRY, relation_spec(manifest, adapter_spec("MLAGENTBENCH"))
    )
    next(rows)
    (root / "MLAgentBench/benchmarks/cifar10/env/train.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset identity"):
        rows.close()


def test_prepared_index_reconstructs_bound_original_case_and_evaluator(bundle):
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run

    root, cases, manifest = bundle
    manifest_path = root.parent / "fixture-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = root.parent / "index"
    run(
        {
            "sources": [
                {
                    "source_id": "MLAGENTBENCH",
                    "data_root": str(root),
                    "manifest": str(manifest_path),
                }
            ]
        },
        output=output,
    )
    database = output / "workload.sqlite"
    indexed = list(prepared_rows(database))
    assert len(indexed) == 1
    record = resolve_input(database, indexed[0]["record_id"], data_root=root)
    assert record.task["input"] == cases["debug"]
    assert record.outcome["evaluation_assets"][0]["path"].endswith("/scripts/eval.py")


def test_native_directory_case_uses_original_description_not_debug_alias(bundle):
    root, cases, manifest = bundle
    name = "MLAgentBench/benchmarks/cifar10/scripts/research_problem.txt"
    next(item for item in manifest["files"] if item["path"] == name)["role"] = "data"
    records = list(
        iter_workload("MLAGENTBENCH", data_root=root, manifest=manifest, files=[name])
    )
    assert len(records) == 1
    record = records[0]
    assert record.task["task_id"] == "cifar10"
    assert record.task["input"]["research_problem"] == (root / name).read_text("utf-8")
    assert (
        record.task["input"]["research_problem"] != cases["debug"]["research_problem"]
    )
    assert record.replay["definition_origin"] == "native_directory"
    assert record.replay["original_registry_case"] is None
    assert record.replay["original_registry_text"] is None
    assert (
        record.replay["original_definition_text"].encode() == (root / name).read_bytes()
    )


def test_released_directory_without_evaluator_is_not_marked_scored(bundle):
    root, _, manifest = bundle
    name = "MLAgentBench/benchmarks/another/scripts/research_problem.txt"
    path = root / name
    path.parent.mkdir(parents=True)
    raw = b"Original released example fixture, not a validated benchmark"
    path.write_bytes(raw)
    manifest["files"].append(
        {
            "path": name,
            "role": "data",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    record = next(
        iter_workload("MLAGENTBENCH", data_root=root, manifest=manifest, files=[name])
    )
    assert record.task["task_id"] == "another"
    assert record.outcome["evaluation_assets"] == []
    assert record.outcome["original_evaluator_available"] is False
    assert record.runtime["execution"] == "not_executed"


def test_real_default_case_receipts_keep_native_failure_and_artifact_scope():
    root = Path(__file__).resolve().parents[2]
    directory = root / "docs/workload-data/validation"
    native = json.loads(
        (directory / "mlagentbench-default-agent-native-audit-20261102.json").read_text(
            "utf-8"
        )
    )
    readback = json.loads(
        (
            directory / "mlagentbench-local-wheel-research-readback-20261102.json"
        ).read_text("utf-8")
    )
    assert native["task"] == readback["case_id"] == "debug"
    assert native["agent_exit_code"] == native["original_scorer_exit_code"] == 0
    assert native["submitted_final_answer"] is False
    assert "max_retries" in native["original_agent_final_message"]
    assert native["task_quality_success_claim"] is False
    assert readback["original_registry_cases"] == 1
    assert readback["public_and_research_input_outcome_equal"] is True
    assert "NOT_CI_ARTIFACT" in readback["status"]
    assert (
        native["source_checkbox_eligible"]
        is readback["source_checkbox_eligible"]
        is False
    )


def test_native_definition_evidence_and_fixed_samples_do_not_inflate_datasets():
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/mlagentbench-native-definitions-readback-20261102.json"
        ).read_text("utf-8")
    )
    profile = json.loads(
        (
            root / "configs/workload-native/mlagentbench-original-directory-fixed3.json"
        ).read_text("utf-8")
    )
    assert receipt["original_registry_aliases"] == 1
    assert receipt["original_native_directory_definitions"] == 15
    assert receipt["public_and_unchanged_research_records_equal"] == 16
    assert receipt["source_checkbox_eligible"] is False
    ci = receipt["actual_ci_followup"]
    assert ci["cpu_ci_conclusion"] == "success"
    assert ci["installed_payload_files_byte_verified"] == 81
    assert ci["all_original_software_assets_verified"] == 138
    assert ci["all_16_choices_match_author_get_task_info"] is True
    assert ci["unchanged_research_input_and_outcome_readbacks"] == 16
    assert ci["prepared_index_unchanged"] is True
    assert ci["model_calls"] == ci["scientific_programs_executed"] == 0
    assert ci["source_checkbox_eligible"] is False
    assert profile["task_ids"] == ["cifar10", "imdb", "ogbn-arxiv"]
    assert profile["default_debug_alias_is_not_one_of_selected_tasks"] is True
    assert not set(profile["task_ids"]) & set(
        receipt["released_definitions_without_evaluator"]
    )
    canonical = {item["path"]: item for item in source_spec("MLAGENTBENCH")["files"]}
    for task in profile["tasks"]:
        for item in task["original_files"]:
            assert item["sha256"] == canonical[item["path"]]["sha256"]
