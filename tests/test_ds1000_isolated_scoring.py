"""Inert CPU fixtures only: never run a candidate or a namespace in CI."""

import gzip
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "isolated_ds1000", ROOT / "scripts/run_ds1000_isolated_scoring.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inputs(tmp_path):
    author, prepared = tmp_path / "author", tmp_path / "prepared"
    (author / "data").mkdir(parents=True)
    prepared.mkdir()
    module = author / "test_ds1000.py"
    module.write_text("# inert fixture: must not be imported\n")
    rows = [
        {
            "metadata": {"problem_id": n, "library": "Pandas"},
            "prompt": "original",
            "code_context": "original oracle",
        }
        for n in range(1000)
    ]
    data = author / "data/ds1000.jsonl.gz"
    with gzip.open(data, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    original = prepared / "original-dataset-private.jsonl"
    original.write_text("".join(json.dumps(row) + "\n" for row in rows))
    answers = prepared / "answers-private.jsonl"
    answers.write_text(
        "".join(
            json.dumps({"id": n, "code": "invalid code retained", "metadata": {}})
            + "\n"
            for n in range(3)
        )
    )
    receipt = {
        "source_id": "DS1000",
        "status": "AUTHOR_FUNCTIONAL_EVALUATOR_INPUT_PREPARED_NOT_EXECUTED",
        "code_revision": "fixture",
        "original_dataset_sha256_required_by_cli": runner.DATA_SHA,
        "selected_problem_ids": [0, 1, 2],
        "all_original_rows_preserved": 1000,
        "prepared_file_sha256": {p.name: runner.digest(p) for p in (original, answers)},
    }
    (prepared / "receipt-private.json").write_text(json.dumps(receipt))
    pins = {
        "revision": "fixture",
        "files": {
            "test_ds1000.py": runner.digest(module),
            "data/ds1000.jsonl.gz": runner.digest(data),
        },
    }
    return author, prepared, pins


def test_original1000_and_fixed_answers_verified_without_execution(tmp_path):
    author, prepared, pins = inputs(tmp_path)
    samples, selected, libraries = runner.validate_inputs(author, prepared, pins)
    assert samples == prepared / "answers-private.jsonl"
    assert selected == [0, 1, 2] and libraries == ["Pandas"]
    assert "invalid code retained" in (prepared / "answers-private.jsonl").read_text()


@pytest.mark.parametrize(
    "target", ["answers-private.jsonl", "original-dataset-private.jsonl"]
)
def test_tampered_saved_input_rejected(tmp_path, target):
    author, prepared, pins = inputs(tmp_path)
    path = prepared / target
    path.write_text(path.read_text() + "tampered")
    with pytest.raises(ValueError, match="identity"):
        runner.validate_inputs(author, prepared, pins)


def test_author_source_change_rejected(tmp_path):
    author, prepared, pins = inputs(tmp_path)
    (author / "test_ds1000.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="identity"):
        runner.validate_inputs(author, prepared, pins)


def test_non_linux_cannot_execute_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="no host fallback"):
        runner.launch(SimpleNamespace(work_root=tmp_path / "not-created"))
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    "layout,filename",
    [
        ("pandas-1.5.3.dist-info", "METADATA"),
        ("pandas-1.5.3-py3.10.egg-info", "PKG-INFO"),
    ],
)
def test_environment_uses_standard_metadata_for_conda_or_wheel(
    tmp_path, layout, filename
):
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python3").write_text("inert fixture, not executable")
    info = runtime / "lib/python3.10/site-packages" / layout
    info.mkdir(parents=True)
    (info / filename).write_text(
        "Metadata-Version: 2.1\nName: pandas\nVersion: 1.5.3\n"
    )
    assert (
        runner.validate_environment(runtime, {"package_versions": {"pandas": "1.5.3"}})
        == runtime
    )
    with pytest.raises(ValueError, match="environment"):
        runner.validate_environment(runtime, {"package_versions": {"pandas": "2.0.0"}})


def test_worker_retains_official_driver_and_checks_isolation_before_import():
    code = (ROOT / "scripts/ds1000_isolated_worker.py").read_text()
    assert code.index('"network reachable"') < code.index(
        "import test_ds1000 as author"
    )
    assert "author.postprocess(r[" in code and "author.eval_ds1000(answers)" in code
    assert "author.ds1000 = selected" in code
    import ast

    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "eval_ds1000"
        ):
            assert not node.keywords  # retain unchanged author's default120 seconds
    pins = json.loads(
        (ROOT / "configs/workload-native/ds1000-author-scoring-pins.json").read_text()
    )
    assert pins["original_problem_ids"] == [0, 1, 2]
    assert pins["package_versions"]["pandas"] == "1.5.3"


def test_real_profile_receipt_does_not_claim_complete_dataset_capability():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    source = ledger["delivery_inventory"]["DS1000"]
    receipt = json.loads(
        (ROOT / source["public_profile_and_consumer_receipt"]).read_text("utf-8")
    )
    assert receipt["profile_available"]
    assert not receipt["source_checkbox_eligible"]
    assert (
        receipt["consumer"][
            "full_original_records_prepared_and_public_research_direct_equal"
        ]
        == 1000
    )
    assert receipt["official_summary_count"] == 3
    assert receipt["official_summary_mean_rounded"] == 0.667
    assert receipt["official_timeout_seconds"] == 120
    assert receipt["new_model_calls"] == 0
    assert source["checked"]  # later seven-library proof, not this old profile alone
    delivery = json.loads((ROOT / source["delivery_receipt"]).read_text("utf-8"))
    assert len(delivery["fixed_sample_per_library"]) == 7


def test_author_selection_iterates_subset_but_indexes_original_metadata():
    spec = importlib.util.spec_from_file_location(
        "ds1000_worker_fixture", ROOT / "scripts/ds1000_isolated_worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    original = [
        {"metadata": {"problem_id": i}, "reference": f"reference-{i}"}
        for i in range(1000)
    ]
    selection = worker.OriginalTaskSelection(original, [291, 511, 932])
    assert [r["metadata"]["problem_id"] for r in selection] == [291, 511, 932]
    assert selection[932] == original[932]
    assert selection[291] == original[291]
    assert len(selection) == 3
    answers = {i: f"candidate-{i}" for i in [291, 511, 932]}
    # Same data-access pattern as the unchanged author's loop/result formatter.
    for row in selection:
        native_id = row["metadata"]["problem_id"]
        assert answers[native_id] == f"candidate-{native_id}"
        assert selection[native_id]["reference"] == f"reference-{native_id}"


def test_only_standard_character_devices_are_allowed_not_host_dev_tree():
    spec = importlib.util.spec_from_file_location(
        "namespace_fixture", ROOT / "scripts/native_python_namespace.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.SAFE_CHAR_DEVICES == {
        "null": (1, 3),
        "zero": (1, 5),
        "random": (1, 8),
        "urandom": (1, 9),
    }
    code = (ROOT / "scripts/native_python_namespace.py").read_text()
    assert 'bind("/dev"' not in code
