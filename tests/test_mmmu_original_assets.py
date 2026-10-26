"""Offline contracts: original acquisition is not benchmark completion."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "mmmu_assets", ROOT / "scripts/prepare_mmmu_original_assets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_original_inventory():
    result = module.plan()
    assert result["files"] == 92
    assert result["subjects"] == 30
    assert result["bytes"] == 3655862232
    assert result["source_checkbox_eligible"] is False
    files = json.loads(module.PINS.read_text())["dataset"]["files"]
    for subject in ("Agriculture", "Art_Theory"):
        assert (
            len([item for item in files if item["path"].startswith(subject + "/test-")])
            == 2
        )


def test_public_registry_registers_every_acquired_data_file():
    pins = json.loads(module.PINS.read_text())["dataset"]
    catalog = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text(encoding="utf-8")
    )
    source = catalog["sources"]["MMMU"]
    assert source["revision"] == pins["revision"]
    assert len(source["files"]) == 92
    assert all(item["role"] == "data" for item in source["files"])

    def fields(rows):
        return {(row["path"], row["bytes"], row["sha256"]) for row in rows}

    assert fields(source["files"]) == fields(pins["files"])


def test_default_no_download_or_directory(monkeypatch, tmp_path, capsys):
    output = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(output)])
    monkeypatch.setattr(
        module, "acquire", lambda **kwargs: pytest.fail("Unexpected acquisition")
    )
    module.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


@pytest.mark.parametrize("flag", ["--execute", "--verify-only"])
def test_explicit_mode_delegates_fixed_source(monkeypatch, tmp_path, flag):
    original = ["prepare", "--output", str(tmp_path), flag]
    monkeypatch.setattr(sys, "argv", original)
    calls = []
    monkeypatch.setattr(
        module, "acquire", lambda **kwargs: calls.append((kwargs, list(sys.argv)))
    )
    module.main()
    assert calls[0][0]["source_id"] == "MMMU"
    assert calls[0][0]["parts"] == ("data",)
    assert ("--verify-only" in calls[0][1]) == (flag == "--verify-only")
    assert sys.argv is original


def test_resume_requires_execute(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path), "--resume"])
    with pytest.raises(SystemExit):
        module.main()
