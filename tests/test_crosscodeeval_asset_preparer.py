"""Inert byte fixtures only; no original source, tasks, retrieval or model calls."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "crosscodeeval_preparer", SCRIPTS / "prepare_crosscodeeval_original_assets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pins_cover_original_native_author_entries_and_four_languages():
    pins = json.loads(module.PINS.read_text(encoding="utf-8"))
    paths = {item["path"] for item in pins["author_files"]}
    assert {
        "scripts/openai_inference.py",
        "scripts/vllm_inference.py",
        "scripts/eval_metric.py",
        "scripts/eval_utils.py",
        "scripts/build_ts_lib.py",
        "LICENSE",
        "THIRD_PARTY_LICENSES",
    } <= paths
    assert len(paths) == 25
    assert len(pins["native_profile"]["languages"]) == 4
    assert pins["dataset"]["bytes"] == 41997264


def test_original_git_blob_and_sha256_both_required(tmp_path):
    path = tmp_path / "inert"
    raw = b"inert software fixture"
    path.write_bytes(raw)
    item = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
    }
    module.verify_git_blob(path, item)
    item["git_blob"] = "0" * 40
    with pytest.raises(ValueError, match="Git blob"):
        module.verify_git_blob(path, item)


def test_existing_partial_never_overwritten(tmp_path):
    final = tmp_path / "asset"
    partial = tmp_path / "asset.partial"
    partial.write_bytes(b"owned by an unknown previous process")
    with pytest.raises(ValueError, match="never overwritten"):
        module.acquire(final, {})
    assert partial.read_bytes() == b"owned by an unknown previous process"
