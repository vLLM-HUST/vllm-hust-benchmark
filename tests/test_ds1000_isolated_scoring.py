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
    assert (
        runner.validate_inputs(author, prepared, pins)
        == prepared / "answers-private.jsonl"
    )
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
    assert "timeout=3" not in code
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
    assert receipt["profile_available"] and not source["checked"]
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
