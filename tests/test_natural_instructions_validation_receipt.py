"""Check historical receipt consistency, never rerun models or infer new acceptance."""

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def receipt():
    return json.loads(
        (
            ROOT
            / "docs/workload-data/validation/natural-instructions-original-npu-20261023.json"
        ).read_text(encoding="utf-8")
    )


def runner():
    path = ROOT / "scripts/run_natural_instructions_original.py"
    spec = importlib.util.spec_from_file_location("receipt_ni_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_runner_and_original_sources_are_pinned():
    evidence = receipt()
    code = ROOT / "scripts/run_natural_instructions_original.py"
    # Git canonical LF content, independent of Windows checkout line endings.
    assert (
        hashlib.sha256(code.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == evidence["final_public_runner_sha256"]
    )
    module = runner()
    assert module.REVISION == evidence["source_revision"]
    assert module.COLLATOR_SHA == evidence["collator_sha256"]
    assert module.SCORER_FILES["evaluation.py"] == evidence["scorer_sha256"]
    assert evidence["references_excluded_from_model_messages"] is True
    assert evidence["original_task_files_unchanged"] is True


def test_full_file_and_sampled_counts_are_not_confused():
    evidence = receipt()
    results = {row["task_file"]: row for row in evidence["task_results"]}
    module = runner()
    assert set(results) == set(module.TASKS)
    counts = []
    for name, (_, total) in module.TASKS.items():
        row = results[name]
        assert row["original_instances"] == total
        assert row["executed_unique"] == row["scored_unique"]
        assert row["all_normal_stop"] is True
        if total <= 250:
            assert row["full_released_task_file"] is True
            assert row["executed_unique"] == total
        else:
            assert row["full_released_task_file"] is False
            assert row["selected_original_indices"] == [0, 1, 2]
            assert row["executed_unique"] == 3
        assert all(0 <= score <= 100 for score in row["metrics"].values())
        counts.append(row["executed_unique"])
    assert sum(counts) == evidence["unique_original_instances_total"] == 355
    assert evidence["scoring_new_model_calls"] == 0
    assert evidence["entire_ni_suite_complete"] is False
    assert evidence["underlying_amazon_paws_xsum_benchmark_equivalence"] is False
    assert evidence["performance_claim"] is False


def test_catalog_data_identity_matches_runner():
    catalog = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text(encoding="utf-8")
    )
    source = catalog["sources"]["NATURAL_INSTRUCTIONS"]
    assert source["revision"] == runner().REVISION
    for row in source["files"]:
        assert row["url"] == (
            "https://raw.githubusercontent.com/allenai/natural-instructions/"
            + source["revision"]
            + "/tasks/"
            + row["path"]
        )
    editing_path = ROOT / "scripts/run_ni_editing_extraction_original.py"
    editing_spec = importlib.util.spec_from_file_location(
        "ni_additional_assets", editing_path
    )
    editing = importlib.util.module_from_spec(editing_spec)
    editing_spec.loader.exec_module(editing)
    assert not set(runner().TASKS) & set(editing.TASKS)
    assert {row["path"]: (row["sha256"], row["rows"]) for row in source["files"]} == {
        **runner().TASKS,
        **editing.TASKS,
    }
