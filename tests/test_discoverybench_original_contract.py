"""Filesystem fixtures only, no original task or model execution."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "discovery_contract", ROOT / "scripts/verify_discoverybench_original_contract.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_task(tmp_path, *, name="input.csv", malformed=False):
    folder = tmp_path / "test/fixture"
    folder.mkdir(parents=True)
    metadata = folder / "metadata_0.json"
    metadata.write_bytes(
        b"\xff"
        if malformed
        else json.dumps(
            {
                "queries": [[{"qid": 0, "question": "fixture question"}]],
                "datasets": [{"name": name}],
            }
        ).encode()
    )
    (folder / "input.csv").write_text("x\n1\n", encoding="utf-8")
    return {
        "dataset": "fixture",
        "metadataid": "0",
        "query_id": "0",
        "gold_hypo": "fixture reference",
    }


def test_original_reference_not_rendered_or_mutated(tmp_path):
    row = fixture_task(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = MODULE.check_relations(tmp_path, [row])
    assert result["verified_task_references"] == 1 and result["errors"] == []
    assert result["source_checkbox_eligible"] is False and result["model_calls"] == 0
    assert "fixture reference" not in json.dumps(result)
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("name", ["input-file.csv", "../../../outside.csv"])
def test_no_filename_alias_or_path_escape(tmp_path, name):
    row = fixture_task(tmp_path, name=name)
    result = MODULE.check_relations(tmp_path, [row])
    assert result["verified_task_references"] == 0
    assert result["error_type_counts"] == {"ValueError": 1}
    assert not (tmp_path / "test/fixture" / name).exists()


def test_encoding_failure_retained_without_guess(tmp_path):
    row = fixture_task(tmp_path, malformed=True)
    result = MODULE.check_relations(tmp_path, [row])
    assert result["error_type_counts"] == {"UnicodeDecodeError": 1}
    assert (tmp_path / "test/fixture/metadata_0.json").read_bytes() == b"\xff"


def test_duplicate_task_is_not_silently_filtered(tmp_path):
    row = fixture_task(tmp_path)
    result = MODULE.check_relations(tmp_path, [row, row])
    assert result["task_references"] == 2 and result["verified_task_references"] == 1
    assert result["error_type_counts"] == {"ValueError": 1}


def test_answer_key_identity_required(tmp_path):
    with pytest.raises(ValueError, match="identity mismatch"):
        from unittest.mock import patch

        (tmp_path / "answer_key").mkdir()
        (tmp_path / "answer_key/answer_key_real.csv").write_text("not original")
        with patch.object(MODULE, "prepare_publisher_index"):
            MODULE.verify(tmp_path, tmp_path)


@pytest.mark.parametrize("field", ["metadataid", "query_id"])
def test_zero_padded_identity_is_same_original_task(tmp_path, field):
    row = fixture_task(tmp_path)
    padded = {**row, field: "00"}
    result = MODULE.check_relations(tmp_path, [row, padded])
    assert result["task_references"] == 2
    assert result["verified_task_references"] == 1
    assert result["error_type_counts"] == {"ValueError": 1}
    assert result["errors"][0][field] == "00"
    assert result["errors"][0]["error"] == "duplicate original task identity"
    assert result["source_checkbox_eligible"] is False
