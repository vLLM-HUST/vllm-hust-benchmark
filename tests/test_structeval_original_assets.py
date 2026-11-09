"""Original identity and inert shared-getter contracts, not native task execution."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
wrapper = importlib.import_module("prepare_structeval_original_assets")
assets = importlib.import_module("prepare_xsum_original_assets")


def test_original_dataset_and_complete_author_source_identities():
    pins = json.loads(wrapper.PINS.read_text("utf-8"))
    assert pins["source_id"] == "STRUCTEVAL"
    assert pins["dataset"]["repo_id"] == "TIGER-Lab/StructEval"
    assert pins["dataset"]["files"][0]["rows"] == 2035
    assert pins["dataset"]["files"][0]["bytes"] == 2334418
    assert len(pins["scoring_files"]) == 1
    assert pins["scoring_files"][0]["bytes"] == 3302422
    assert "TIGER-AI-Lab/StructEval/legacy.zip/" in pins["scoring_files"][0]["url"]
    assert not pins["source_checkbox_eligible"]


def test_shared_getter_preserves_originals_and_rejects_corruption(
    tmp_path, monkeypatch
):
    raw = b"inert fixture, not original questions or source code"
    spec = {
        "path": "original.bin",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "STRUCTEVAL",
                "dataset": {
                    "repo_id": "fixture",
                    "revision": "fixture",
                    "files": [spec],
                },
                "scoring_files": [{**spec, "url": "https://publisher.invalid/source"}],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fetch(path, item, url):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        assets.verify(path, item)

    monkeypatch.setattr(assets, "fetch", fetch)
    output = tmp_path / "assets"
    for flags in ([], ["--resume"], ["--verify-only"]):
        monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(output), *flags])
        assets.main(pins_path=pins, source_id="STRUCTEVAL", parts=("data", "scoring"))
    assert len(calls) == 2
    assert (output / "data/original.bin").read_bytes() == raw
    assert (output / "scoring/original.bin").read_bytes() == raw
    (output / "scoring/original.bin").write_bytes(b"corrupt")
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--output", str(output), "--verify-only"]
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        assets.main(pins_path=pins, source_id="STRUCTEVAL", parts=("data", "scoring"))
