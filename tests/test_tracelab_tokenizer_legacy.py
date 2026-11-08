"""Synthetic serialization fixtures; no original model/task or native replay."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "tracelab_legacy", ROOT / "scripts/prepare_tracelab_tokenizer_legacy.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def document():
    return {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4},
            "merges": [["a", "b"], ["ab", "c"]],
            "ignore_merges": False,
        },
        "added_tokens": [{"id": 8, "content": "<fixture>", "special": True}],
        "unknown_extension": {"preserve": [1, 2]},
    }


def test_only_serialization_changes_and_input_object_stays_unchanged():
    original = document()
    saved = deepcopy(original)
    result = entry.legacy_view(original)
    assert original == saved
    assert result["model"]["merges"] == ["a b", "ab c"]
    result["model"]["merges"] = deepcopy(saved["model"]["merges"])
    assert result == saved


@pytest.mark.parametrize(
    "merges", [[], [["a", "b c"]], [["a"]], [["a", 2]], [["a", "\tb"]]]
)
def test_reject_ambiguous_or_invalid_serialization(merges):
    original = document()
    original["model"]["merges"] = merges
    with pytest.raises(ValueError):
        entry.legacy_view(original)


def test_plan_then_write_preserves_source_and_rejects_overwrite(tmp_path):
    source = tmp_path / "original.json"
    raw = json.dumps(document()).encode()
    source.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    target = tmp_path / "legacy.json"
    planned = entry.prepare(source, expected, target)
    assert planned["status"] == "PLAN_ONLY" and not target.exists()
    written = entry.prepare(source, expected, target, write=True)
    assert (
        written["compatibility_tokenizer_sha256"]
        == hashlib.sha256(target.read_bytes()).hexdigest()
    )
    assert source.read_bytes() == raw
    assert not written["runtime_encoding_equivalence_verified"]
    assert not written["source_checkbox_eligible"]
    with pytest.raises(ValueError, match="fresh external"):
        entry.prepare(source, expected, target, write=True)
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.prepare(source, "0" * 64, tmp_path / "other.json", write=True)


def test_legacy_strings_are_idempotent_and_non_bpe_is_rejected():
    original = document()
    original["model"]["merges"] = ["a b", "ab c"]
    assert entry.legacy_view(original) == original
    original["model"]["type"] = "WordPiece"
    with pytest.raises(ValueError, match="BPE model"):
        entry.legacy_view(original)
