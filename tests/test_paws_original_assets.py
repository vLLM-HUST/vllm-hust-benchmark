"""Publisher-asset preparation only; no classification runtime claim."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "paws_assets", ROOT / "scripts/prepare_paws_original_assets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_published_configurations():
    result = module.plan()
    assert result["files"] == 6
    assert result["bytes"] == 129285432
    assert result["configurations"] == [
        "labeled_final",
        "labeled_swap",
        "unlabeled_final",
    ]
    assert not result["source_checkbox_eligible"]


def test_default_registry_preserves_all_files_and_reference_provenance():
    source = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text(encoding="utf-8")
    )["sources"]["PAWS"]
    pins = json.loads(module.PINS.read_text())["dataset"]
    assert source["revision"] == pins["revision"]
    assert len(source["files"]) == 6
    assert all(item["role"] == "data" for item in source["files"])
    assert sum(item["rows"] for item in source["files"]) == 751450
    assert {
        (item["path"], item["sha256"], item["bytes"]) for item in source["files"]
    } == {(item["path"], item["sha256"], item["bytes"]) for item in pins["files"]}
    assert all(
        item["reference_kind"]
        == (
            "noisy_auxiliary" if item["configuration"] == "unlabeled_final" else "human"
        )
        for item in source["files"]
    )


def test_plan_never_creates_output(monkeypatch, tmp_path, capsys):
    output = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(output)])
    monkeypatch.setattr(
        module, "acquire", lambda **kwargs: pytest.fail("Unexpected download")
    )
    module.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


def test_resume_needs_explicit_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path), "--resume"])
    with pytest.raises(SystemExit):
        module.main()


@pytest.mark.parametrize("mode", ["--execute", "--verify-only"])
def test_fixed_source_delegation_and_argument_restoration(monkeypatch, tmp_path, mode):
    original = ["prepare", "--output", str(tmp_path), mode]
    monkeypatch.setattr(sys, "argv", original)
    calls = []
    monkeypatch.setattr(
        module, "acquire", lambda **kwargs: calls.append((kwargs, list(sys.argv)))
    )
    module.main()
    assert calls[0][0]["source_id"] == "PAWS"
    assert calls[0][0]["parts"] == ("data",)
    assert ("--verify-only" in calls[0][1]) == (mode == "--verify-only")
    assert sys.argv is original
