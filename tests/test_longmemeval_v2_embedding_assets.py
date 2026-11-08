"""Model asset identity fixtures, no weights, encoder calls or device imports."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
wrapper = importlib.import_module("prepare_longmemeval_v2_embedding")
assets = importlib.import_module("prepare_xsum_original_assets")


def test_original_embedding_role_and_complete_asset_identity():
    pins = json.loads(wrapper.PINS.read_text("utf-8"))
    model = pins["model"]
    assert model["repo_id"] == "Qwen/Qwen3-Embedding-8B"
    assert len(model["revision"]) == 40
    files = model["files"]
    assert len(files) == len({row["path"] for row in files}) == 17
    assert sum(row["bytes"] for row in files) == 15150575778
    assert sum(row["path"].endswith(".safetensors") for row in files) == 4
    assert all(len(row["sha256"]) == 64 for row in files)
    assert not pins["model_service_started"] and not pins["source_checkbox_eligible"]


def test_acquisition_fixture_never_starts_embedding_service(tmp_path, monkeypatch):
    raw = b"inert model metadata fixture"
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "LONGMEMEVAL_V2_EMBEDDING",
                "model": {
                    "repo_id": "fixture/original-model",
                    "revision": "fixed-fixture",
                    "files": [
                        {
                            "path": "config.json",
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    ],
                },
            }
        )
    )

    def fetch(path, item, url):
        assert "/fixture/original-model/resolve/fixed-fixture/config.json" in url
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        assets.verify(path, item)

    monkeypatch.setattr(assets, "fetch", fetch)
    output = tmp_path / "assets"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(output)])
    assets.main(pins_path=pins, source_id="LONGMEMEVAL_V2_EMBEDDING", parts=("model",))
    receipt = json.loads((output / "preparation.private.json").read_text())
    assert receipt["model_calls"] == 0 and not receipt["source_checkbox_eligible"]
    assert {row["part"] for row in receipt["files"]} == {"model"}
