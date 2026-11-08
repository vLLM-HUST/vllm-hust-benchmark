"""Public setup contracts only; no model, original task or score evidence."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ib_metric", ROOT / "scripts/prepare_infinitebench_original_metric.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plan_does_not_create_files_or_fetch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path / "out")])
    monkeypatch.setattr(
        module, "acquire", lambda **_: pytest.fail("unrequested acquisition")
    )
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLAN_ONLY" and report["bytes"] == 6140
    assert report["model_calls"] == 0 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", ["--execute", "--verify-only"])
def test_explicit_modes_reuse_pinned_shared_preparer(monkeypatch, tmp_path, mode):
    args = ["prepare", "--output", str(tmp_path), mode]
    monkeypatch.setattr(sys, "argv", args)
    calls = []
    monkeypatch.setattr(
        module, "acquire", lambda **kw: calls.append((kw, list(sys.argv)))
    )
    module.main()
    assert sys.argv is args
    assert calls[0][0] == {
        "pins_path": module.PINS,
        "source_id": "INFINITEBENCH",
        "parts": ("scoring",),
    }
    assert ("--verify-only" in calls[0][1]) == (mode == "--verify-only")


def test_metric_identity_matches_already_qualified_source():
    pins = json.loads(module.PINS.read_text("utf-8"))["scoring_files"]
    xsum = json.loads(
        (ROOT / "configs/workload-native/xsum-author-pins.json").read_text("utf-8")
    )
    original = next(x for x in xsum["scoring_files"] if x["path"] == "metric/rouge.py")
    assert len(pins) == 1
    assert (pins[0]["sha256"], pins[0]["bytes"]) == (
        original["sha256"],
        original["bytes"],
    )
