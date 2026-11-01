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
    assert [item["path"] for item in public["files"] if item["role"] == "data"] == [
        REGISTRY
    ]


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
