"""Invented Unicode fixtures, never modified original dataset content."""

import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from marble_config_serialization import native_config_json  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "fixture\x7f",
        "fixture\x85line",
        "fixture\x80\x9f",
        "line\u2028paragraph\u2029",
        "astral\U0001f525",
        "surrogate\ud800",
        "noncharacters\ufffe\uffff",
        "literal\\u007f",
        "中文\x7f\U0001f525",
    ],
)
def test_json_and_original_yaml_parser_return_identical_codepoints(text):
    original = {
        "task": {"content": text},
        "agents": [{"profile": text}],
        "iterations": 5,
    }
    serialized = native_config_json(original)
    serialized.encode("utf-8")
    assert json.loads(serialized) == original
    assert yaml.safe_load(serialized) == original
