"""Artificial protocol metadata only, no renderer or native model execution."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "render_plan", ROOT / "scripts/prepare_structeval_rendering_plan.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_groups_keep_original_indices_without_nonrendering_reruns():
    rows = [
        {"rendering": False, "input_type": "Text", "output_type": "JSON"},
        {"rendering": True, "input_type": "Text", "output_type": "SVG"},
        {"rendering": True, "input_type": "Text", "output_type": "SVG"},
    ]
    assert MODULE.protocol_inventory(rows) == [
        {
            "input_type": "Text",
            "output_type": "SVG",
            "original_rows": 2,
            "fixed_first_three_original_indices": [1, 2],
        }
    ]


def test_runtime_binary_is_not_sandbox_qualification():
    result = MODULE.prerequisites(which=lambda name: "/fixture/" + name)
    assert result["executables"]["docker"]
    assert result["sandbox_qualified"] is False


def test_rejects_implicit_rendering_classification():
    with pytest.raises(ValueError):
        MODULE.protocol_inventory([{"input_type": "Text", "output_type": "SVG"}])


def test_author_dependency_scan_does_not_import_or_execute(tmp_path):
    folder = tmp_path / "render_engine"
    folder.mkdir()
    (folder / "render_fixture.py").write_text(
        'import nonexistent_fixture_module\nraise RuntimeError("must not execute")\nASSET="https://example.invalid/static-fixture.js"\n'
    )
    report = MODULE.author_render_dependencies(tmp_path)
    assert report["original_modules"][0]["imports"] == ["nonexistent_fixture_module"]
    assert report["original_modules"][0]["literal_http_assets"] == [
        "https://example.invalid/static-fixture.js"
    ]
    assert report["offline_network_qualified"] is False
