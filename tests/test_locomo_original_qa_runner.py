"""Software fixtures only; no publisher tasks, model, or semantic score evidence."""

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "locomo_runner", Path(__file__).parents[1] / "scripts/run_locomo_original_qa.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_real_data_compatibility_audit_is_not_new_model_execution():
    root = Path(__file__).parents[1]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/locomo-missing-key-compatibility-20261025.json"
        ).read_text("utf-8")
    )
    assert receipt["original_existing_answer_reads_preserved"] == 1542
    assert receipt["only_missing_cat5_reads_fall_back"] == 444
    assert receipt["original_rows"] == 1986
    assert receipt["existing_two_conflicting_fields_keep_original_answer"] == 2
    assert receipt["publisher_generation_and_scoring_ast_answer_load_replacements"] == [
        4,
        2,
    ]
    assert receipt["historical_original_nine_native_scores_recomputed_equal"] == 9
    assert receipt["original_category1_to4_diagnostic_scorer_parity_rows"] == 1540
    assert receipt["new_model_calls"] == receipt["historical_model_calls_repeated"] == 0
    assert not receipt["diagnostic_placeholders_are_native_results"]
    assert not receipt["source_completion_checked"]
    snapshot = root / "configs/workload-native/locomo-text-qa-py312-dependencies.txt"
    assert (
        hashlib.sha256(snapshot.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["dependency_snapshot_sha256"]
    )
    handoff = json.loads(
        (
            root / "docs/workload-data/validation/locomo-research-handoff-20261025.json"
        ).read_text("utf-8")
    )
    assert handoff["installed_provider_payload_files_equal"] == 32
    assert handoff["public_prepare_resolve_selected_rows"] == 6
    assert handoff["research_direct_prepare_resolve_selected_rows"] == 6
    assert handoff["original_cat5_reference_rows_preserved"] == 3
    assert not handoff["missing_answer_field_added_to_raw_data_or_research_record"]
    assert handoff["new_model_calls"] == 0
    assert not handoff["native_benchmark_completed_by_this_audit"]


def test_missing_key_shim_never_rewrites_or_overrides_existing_answers():
    for category in range(1, 6):
        qa = {
            "category": category,
            "answer": ["fixture-existing"],
            "adversarial_answer": "fixture-other",
        }
        before = deepcopy(qa)
        assert runner.existing_answer_or_cat5_distractor(qa) is qa["answer"]
        assert qa == before
    qa = {"category": 5, "adversarial_answer": "fixture-distractor"}
    assert runner.existing_answer_or_cat5_distractor(qa) == "fixture-distractor"
    assert "answer" not in qa


@pytest.mark.parametrize(
    "qa", [{"category": 1, "adversarial_answer": "fixture"}, {"category": 5}]
)
def test_missing_nonadversarial_or_both_fields_remains_an_error(qa):
    with pytest.raises(KeyError, match="answer"):
        runner.existing_answer_or_cat5_distractor(qa)


def test_ast_patch_changes_only_answer_loads_not_source_or_other_fields(tmp_path):
    source = b"def fixture(qa):\n    return qa['answer'], qa['question']\n"
    path = tmp_path / "source.py"
    path.write_bytes(source)
    args = path, hashlib.sha256(source).hexdigest(), {"fixture"}, set()
    plain, count = runner.selected_definitions(*args, {}, False)
    assert count == 0
    with pytest.raises(KeyError):
        plain["fixture"](
            {"category": 5, "question": "fixture", "adversarial_answer": "other"}
        )
    patched, count = runner.selected_definitions(*args, {}, True)
    assert count == 1 and path.read_bytes() == source
    assert patched["fixture"](
        {"category": 5, "question": "fixture", "adversarial_answer": "other"}
    ) == ("other", "fixture")
    for category in range(1, 6):
        qa = {
            "category": category,
            "question": "fixture",
            "answer": "existing",
            "adversarial_answer": "other",
        }
        assert plain["fixture"](qa) == patched["fixture"](qa)


def test_source_digest_and_definition_set_fail_closed(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"def fixture(qa): return qa['answer']\n")
    with pytest.raises(ValueError, match="identity"):
        runner.selected_definitions(path, "0" * 64, {"fixture"}, set(), {}, True)
    with pytest.raises(ValueError, match="definitions"):
        runner.selected_definitions(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            {"missing"},
            set(),
            {},
            True,
        )


def test_store_access_and_unrelated_subscripts_are_not_patched():
    tree = ast.parse("qa['answer'] = other['answer']; value = line['answer']")
    patch = runner.MissingCat5Answer()
    changed = patch.visit(tree)
    assert patch.replacements == 1
    assert isinstance(changed.body[0].targets[0], ast.Subscript)
    assert isinstance(changed.body[0].value, ast.Subscript)


def test_selection_keeps_original_global_positions_and_all_category_members():
    data = [
        {"qa": [{"category": c} for c in [2, 5, 3]]},
        {"qa": [{"category": c} for c in [3, 5, 1]]},
    ]
    assert [r[0] for r in runner.select_questions(data)] == [0, 1, 2]
    assert [r[0] for r in runner.select_questions(data, category=3, full=True)] == [
        2,
        3,
    ]
    assert [r[:3] for r in runner.select_questions(data, indices=[1, 4])] == [
        (1, 0, 1),
        (4, 1, 1),
    ]
    for indices in [[1, 1], [4, 1], [-1], [6], [True], []]:
        with pytest.raises(ValueError):
            runner.select_questions(data, indices=indices)
    with pytest.raises(ValueError):
        runner.select_questions(data, indices=[1], full=True)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://localhost/",
        "http://localhost/v1?key=x",
        "http://localhost/v1#x",
    ],
)
def test_only_local_uncredentialed_native_endpoint_allowed(url):
    with pytest.raises(ValueError):
        runner.local_endpoint(url)
    assert (
        runner.local_endpoint("http://127.0.0.1:19466/v1/")
        == "http://127.0.0.1:19466/v1"
    )
