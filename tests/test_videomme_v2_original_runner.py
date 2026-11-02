"""Native runner input-boundary fixtures, without model or publisher imports."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "videomme2_runner", SCRIPTS / "run_videomme_v2_original.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def prepared_record(tmp_path):
    original = {
        "question_id": "001-1",
        "video_id": "001",
        "question": "Original question",
        "options": "A. One\nB. Two",
    }
    paths = []
    for index in range(64):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(f"fixture-{index}".encode())
        paths.append(path)
    record = {
        "source_index": 0,
        "original_question_id": "001-1",
        "video_id": "001",
        "frame_sha256": [entry.sha256(path) for path in paths],
        "original_prompt": [{"type": "image", "value": str(path)} for path in paths]
        + [{"type": "text", "value": "Original question\nA. One\nB. Two"}],
    }
    return original, record


def test_original_record_requires64_identified_frame_bytes(tmp_path):
    original, record = prepared_record(tmp_path)
    entry.validate_record(record, original, 0, tmp_path)
    Path(record["original_prompt"][0]["value"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        entry.validate_record(record, original, 0, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [("source_index", 1), ("original_question_id", "001-2"), ("video_id", "002")],
)
def test_no_task_identity_or_order_substitution(tmp_path, key, value):
    original, record = prepared_record(tmp_path)
    record[key] = value
    with pytest.raises(ValueError, match="identity"):
        entry.validate_record(record, original, 0, tmp_path)


def test_frame_hash_list_cannot_omit_an_image(tmp_path):
    original, record = prepared_record(tmp_path)
    record["frame_sha256"].pop()
    with pytest.raises(ValueError, match="all64"):
        entry.validate_record(record, original, 0, tmp_path)


def test_identical_frame_outside_prepared_root_not_accepted(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    original, record = prepared_record(owned)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(Path(record["original_prompt"][0]["value"]).read_bytes())
    record = copy.deepcopy(record)
    record["original_prompt"][0]["value"] = str(outside)
    with pytest.raises(ValueError, match="escaped"):
        entry.validate_record(record, original, 0, owned)
    with pytest.raises(ValueError, match="escaped"):
        entry.check_prompt(record["original_prompt"], original, owned_root=owned)


def test_default_plan_does_not_require_assets_or_call_model():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_videomme_v2_original.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "source_id": "VIDEOMME_V2",
        "status": "PLAN_ONLY",
        "original_questions": 24,
        "frames_each": 64,
        "model_calls": 0,
    }
