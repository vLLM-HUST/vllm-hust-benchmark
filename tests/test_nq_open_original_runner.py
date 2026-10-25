"""Inert software fixtures only; no real dataset, model, NPU or task grading."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "nq_original", ROOT / "scripts/run_nq_open_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_selection_is_fixed_original_positions_or_full_dev():
    assert runner.select() == [0, 1, 2]
    assert runner.select(start=3607) == [3607, 3608, 3609]
    assert runner.select(full=True) == list(range(3610))
    for kwargs in [
        {"count": 2},
        {"start": -1},
        {"start": 3608},
        {"start": True},
        {"full": True, "start": 1},
    ]:
        with pytest.raises(ValueError):
            runner.select(**kwargs)


def test_local_data_transport_preserves_original_native_protocol():
    original = {
        "task": "nq_open",
        "dataset_path": "official/fixture",
        "description": "original instruction",
        "doc_to_text": "unchanged question template",
        "doc_to_target": "all original aliases",
        "generation_kwargs": {"until": ["\n", ".", ","], "temperature": 0},
        "metric_list": [{"metric": "exact_match"}],
        "filter_list": [{"filter": [{"function": "remove_whitespace"}]}],
    }
    frozen = deepcopy(original)
    paths = {"train": "/private/train.jsonl", "validation": "/private/dev.jsonl"}
    result = runner.local_config(original, paths)
    assert original == frozen
    assert set(k for k in result if k not in original or result[k] != original[k]) == {
        "dataset_path",
        "dataset_name",
        "dataset_kwargs",
    }
    for key in (
        "generation_kwargs",
        "description",
        "doc_to_text",
        "doc_to_target",
        "filter_list",
        "metric_list",
    ):
        assert result[key] == frozen[key]


@pytest.mark.parametrize(
    "question", [" unmodified fixture question? ", "original\u2028embedded-separator"]
)
def test_original_file_fidelity_and_answer_aliases(tmp_path, monkeypatch, question):
    rows = [
        {
            "question": question,
            "answer": ["alias one", "alias two"],
        }
    ]
    path = tmp_path / "inert.jsonl"
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "FILES",
        {
            "validation": (
                path.name,
                path.stat().st_size,
                1,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        },
    )
    paths, restored = runner.originals(tmp_path)
    assert paths == {"validation": str(path)}
    assert restored == {"validation": rows}
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned original"):
        runner.originals(tmp_path)


@pytest.mark.parametrize(
    "failure", [None, "missing", "duplicate", "reference", "prompt"]
)
def test_native_saved_sample_identity_is_not_quality_selection(failure):
    rows = [{"question": f"inert {i}", "answer": ["a", "alias"]} for i in range(3)]
    contexts = {i: f"official fixture prefix {i}" for i in range(3)}
    samples = [
        {
            "doc_id": i,
            "doc": deepcopy(rows[i]),
            "arguments": [[contexts[i], {}]],
            "resps": [[""]],
        }
        for i in range(3)
    ]
    if failure == "missing":
        samples.pop()
    elif failure == "duplicate":
        samples[2] = deepcopy(samples[1])
    elif failure == "reference":
        samples[0]["doc"]["answer"] = ["changed"]
    elif failure == "prompt":
        samples[0]["arguments"][0][0] = "altered request"
    if failure:
        with pytest.raises(ValueError):
            runner.verify_native_samples(samples, [0, 1, 2], rows, contexts)
    else:
        # Empty/wrong answers remain available to the upstream scorer, not swapped.
        runner.verify_native_samples(samples, [0, 1, 2], rows, contexts)


def test_help_does_not_require_harness_or_a_model():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_nq_open_original.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "--full" in result.stdout and "--max-length" in result.stdout
