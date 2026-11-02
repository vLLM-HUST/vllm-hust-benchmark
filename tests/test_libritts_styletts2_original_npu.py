"""CPU-only protocol selection tests, not original synthesis evidence."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "libritts_native", ROOT / "scripts/run_libritts_styletts2_original_npu.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_distinct_reference_even_with_dot_prefix():
    members = [
        "./LibriTTS/test-other/1688/1/1688_1_000000_000000.wav",
        "./LibriTTS/test-other/1688/1/1688_1_000001_000000.wav",
        "./LibriTTS/test-other/9999/1/9999_1_000099_000000.wav",
    ]
    assert entry.reference_for("1688_1_000001_000000", members) == members[0]
    assert entry.reference_for("1688_1_000000_000000", members) == members[1]


def test_no_reference_substitution_from_other_speaker():
    with pytest.raises(ValueError, match="distinct"):
        entry.reference_for("1688_1_0_0", ["LibriTTS/test-other/9999/1/9999_1_0_0.wav"])


def test_notebook_definitions_preserved_not_rewritten():
    preprocess = "to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80)\n"
    inference = "def inference(text, ref_s):\n    return text\n"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["not executable"]},
            {"cell_type": "code", "source": [preprocess]},
            {"cell_type": "code", "source": [inference]},
        ]
    }
    assert entry.notebook_definitions(notebook) == (preprocess, inference)
    notebook["cells"][2]["source"] = [inference + "other_statement = True\n"]
    with pytest.raises(ValueError, match="single unchanged"):
        entry.notebook_definitions(notebook)
