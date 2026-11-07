"""Fixed-plan export fixtures, no real media or benchmark execution."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "duplex_export",
    Path(__file__).resolve().parents[1]
    / "scripts/export_full_duplex_original_capture.py",
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def arguments(tmp_path, version):
    args = argparse.Namespace(
        version=version,
        capture_root=tmp_path / "capture",
        dataset_root=tmp_path / "data",
        output=tmp_path / "out",
        omni_root=tmp_path / "omni",
        python="python",
        engine="fixture-not-real",
    )
    args.capture_root.mkdir()
    rows = [
        {"version": version, "sample_id": f"{family}/{case}"}
        for version, families in ENTRY.FAMILIES.items()
        for family, ids in families.items()
        for case in ids
    ]
    for index, row in enumerate(rows):
        row["output"] = str(args.capture_root / f"case-{index:02d}")
    (args.capture_root / "fixed-plan-public.json").write_text(json.dumps(rows))
    return args


@pytest.mark.parametrize("version,count", [("v1.0", 15), ("v1.5", 12)])
def test_export_keeps_original_runs_windows_and_v15_explicit_selection(
    tmp_path, version, count
):
    args = arguments(tmp_path, version)
    command, selected = ENTRY.export_command(args)
    assert len(selected) == command.count("--run") == count
    assert command.count("--only") == (12 if version == "v1.5" else 0)
    assert "--timeout" not in command and "--limit" not in command
    assert not args.output.exists()


def test_changed_original_selection_is_not_accepted(tmp_path):
    args = arguments(tmp_path, "v1.5")
    path = args.capture_root / "fixed-plan-public.json"
    rows = json.loads(path.read_text())
    rows[0]["sample_id"] = "invented/replacement"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="fixed original"):
        ENTRY.export_command(args)


def test_capture_plan_cannot_redirect_to_other_run(tmp_path):
    args = arguments(tmp_path, "v1.0")
    path = args.capture_root / "fixed-plan-public.json"
    rows = json.loads(path.read_text())
    rows[0]["output"] = str(tmp_path / "unrelated-run")
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="original case"):
        ENTRY.export_command(args)
