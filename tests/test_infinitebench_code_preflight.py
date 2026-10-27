"""Fake software code fixtures, not author execution or benchmark evidence."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ib_preflight", ROOT / "scripts/verify_infinitebench_original_code.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path, budget=50):
    files = []
    for name, content in {
        "src/eval_utils.py": f"raise RuntimeError('MUST NOT EXECUTE')\nDATA_NAME_TO_PATH = {{'fixture': 'fixture.jsonl'}}\nDATA_NAME_TO_MAX_NEW_TOKENS = {{'fixture': {budget}}}\n",
        "src/compute_scores.py": "raise RuntimeError('MUST NOT IMPORT SCORER')\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = content.encode()
        path.write_bytes(raw)
        files.append(
            {
                "path": name,
                "bytes": len(raw),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            }
        )
    return {"revision": "software-fixture", "files": files}


def test_verified_static_contract_without_executing_code(tmp_path):
    result = runner.verify(tmp_path, fixture(tmp_path))
    assert result["tasks"] == [
        {"task": "fixture", "file": "fixture.jsonl", "max_new_tokens": 50}
    ]
    assert not result["publisher_code_executed"]
    assert not result["native_execution_complete"]
    assert not result["source_checkbox_eligible"]


def test_modified_code_rejected(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / "src/eval_utils.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.verify(tmp_path, pins)


def test_invalid_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="contract incomplete"):
        runner.verify(tmp_path, fixture(tmp_path, budget=0))


def test_escaping_code_path_rejected(tmp_path):
    pins = fixture(tmp_path)
    pins["files"][0]["path"] = "../outside.py"
    with pytest.raises(ValueError, match="escaped"):
        runner.verify(tmp_path, pins)
