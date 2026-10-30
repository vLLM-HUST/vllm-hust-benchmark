"""Pure configuration preparation fixtures, not native task results."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/prepare_marble_original_configs.py"


def test_default_plan_has_no_reads_downloads_or_output(tmp_path):
    report = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                str(ENTRY),
                "--data-root",
                str(tmp_path / "missing-data"),
                "--author-root",
                str(tmp_path / "missing-source"),
                "--output",
                str(tmp_path / "absent"),
            ],
            text=True,
        )
    )
    assert report["status"] == "PLAN_ONLY" and report["per_file_limit"] == 3
    assert report["model_calls"] == 0 and report["source_checkbox_eligible"] is False
    assert not list(tmp_path.iterdir())


def test_native_loader_comparison_includes_references_and_routing():
    spec = importlib.util.spec_from_file_location("marble_configs", ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    original = {field: {} for field in entry.FIELDS}
    original.update(
        task={"content": "fixture", "labels": ["scoring-only"]},
        coordinate_mode="fixture-mode",
    )
    values = {key: value for key, value in original.items() if key != "coordinate_mode"}
    loaded = SimpleNamespace(**values, coordination_mode="fixture-mode")
    entry.validate_loaded_config(original, loaded)
    loaded.task = {"content": "fixture"}
    with pytest.raises(ValueError, match="field changed"):
        entry.validate_loaded_config(original, loaded)


def test_cleared_runtime_fields_are_reported_without_default_invention():
    spec = importlib.util.spec_from_file_location("marble_empty", ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    original = {
        "environment": {"type": "", "max_iterations": ""},
        "llm": "",
        "task": {"content": "unchanged fixture"},
    }
    before = json.loads(json.dumps(original))
    assert entry.empty_runtime_fields(original) == [
        "llm",
        "environment.type",
        "environment.max_iterations",
    ]
    assert original == before
    assert (
        entry.empty_runtime_fields(
            {
                "environment": {"type": "Minecraft", "max_iterations": 20},
                "llm": "original-model",
            }
        )
        == []
    )
