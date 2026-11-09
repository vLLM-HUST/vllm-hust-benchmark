"""Inert task-shaped fixtures; never official execution or scoring evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "lcb_prepare", ROOT / "scripts/prepare_livecodebench_original_tasks.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.fixture
def originals(tmp_path):
    rows = [
        {
            "question_id": str(i),
            "question_content": "inert fixture",
            "public_test_cases": "[]",
            "private_test_cases": "opaque-not-decoded",
            "metadata": "{}",
            "extra_original_field": {"integer": i},
        }
        for i in range(4)
    ]
    data = tmp_path / "raw"
    path = data / "code_generation/test.jsonl"
    path.parent.mkdir(parents=True)
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    path.write_bytes(raw)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "sources": {
                    "LIVECODEBENCH": {
                        "files": [
                            {
                                "path": "code_generation/test.jsonl",
                                "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "publisher_revision": "fixture",
                            }
                        ]
                    }
                }
            }
        )
    )
    return data, catalog, rows


def test_first_three_keep_every_original_field_and_do_not_unpickle(
    originals, tmp_path, monkeypatch
):
    data, catalog, rows = originals

    def forbidden(*args, **kwargs):
        raise AssertionError("host pickle decoding is forbidden")

    monkeypatch.setattr(pickle, "loads", forbidden)
    output = tmp_path / "prepared"
    receipt = tool.prepare(data, output, catalog=catalog)
    actual = json.loads((output / "codegen-original-tasks.private.json").read_text())
    assert actual == rows[:3]
    assert receipt["selected_positions"] == [0, 1, 2]
    assert receipt["selected_question_ids"] == ["0", "1", "2"]
    assert not receipt["private_tests_decoded"]
    assert receipt["original_fields_unchanged"]
    assert receipt["model_calls"] == 0
    assert not receipt["native_execution_complete"]
    assert not receipt["source_checkbox_eligible"]


def test_identity_mismatch_does_not_create_output(originals, tmp_path):
    data, catalog, _ = originals
    (data / "code_generation/test.jsonl").write_bytes(b"changed")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="identity mismatch"):
        tool.prepare(data, output, catalog=catalog)
    assert not output.exists()


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_sample_count_rejected_before_io(originals, tmp_path, count):
    data, catalog, _ = originals
    with pytest.raises(ValueError, match="positive"):
        tool.prepare(data, tmp_path / "prepared", sample_count=count, catalog=catalog)


def test_no_replacement_if_fixed_sample_exceeds_source(originals, tmp_path):
    data, catalog, _ = originals
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="exceeds original"):
        tool.prepare(data, output, sample_count=5, catalog=catalog)
    assert not output.exists()


def test_existing_output_and_raw_nested_output_rejected(originals, tmp_path):
    data, catalog, _ = originals
    output = tmp_path / "existing"
    output.mkdir()
    for target in (output, data / "nested"):
        with pytest.raises(ValueError, match="fresh output"):
            tool.prepare(data, target, catalog=catalog)
