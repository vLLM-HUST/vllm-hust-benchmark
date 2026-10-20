import gzip
import importlib
import io
import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("inventory_recovered_capsules")


def test_compressed_credential_pattern_is_rejected(module):
    raw = b"Authorization: Bearer " + b"x" * 30
    with pytest.raises(ValueError, match="values suppressed"):
        module.screen(gzip.compress(raw))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.pkl", raw)
    with pytest.raises(ValueError, match="values suppressed"):
        module.screen(buffer.getvalue())


def test_bounded_decompression(module, monkeypatch):
    monkeypatch.setattr(module, "LIMIT", 100)
    with pytest.raises(ValueError, match="Decompressed"):
        module.screen(gzip.compress(b"x" * 101))


def test_inventory_excludes_models_and_never_loads_pickle(module, tmp_path):
    source = tmp_path / "source"
    phase = source / "phase1"
    phase.mkdir(parents=True)
    (phase / "calibration.pt").write_bytes(b"not executable as a pickle")
    model = phase / "model"
    model.mkdir()
    (model / "weights.pt").write_bytes(b"excluded")
    output = tmp_path / "inventory.json"
    module.inventory(source, output)
    result = json.loads(output.read_text())
    assert [row["original_path"] for row in result["files"]] == [
        "phase1/calibration.pt"
    ]
    assert result["excluded_directories"] == ["phase1/model"]
    with pytest.raises(ValueError, match="overwrite"):
        module.inventory(source, output)
