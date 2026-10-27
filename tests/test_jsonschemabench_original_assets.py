"""Metadata and inert acquisition fixtures; no original schemas or NPU calls."""

import fnmatch
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
wrapper = importlib.import_module("prepare_jsonschemabench_original_assets")
assets = importlib.import_module("prepare_xsum_original_assets")


def test_official_config_snapshot_has_all_referenced_splits_without_payloads():
    pins = json.loads(wrapper.PINS.read_text("utf-8"))
    dataset = pins["dataset"]
    assert dataset["repo_id"] == "epfl-dlab/JSONSchemaBench"
    assert len(dataset["revision"]) == 40
    assert len(pins["config_names"]) == 11
    assert pins["config_names"].count("default") == 1
    files = dataset["files"]
    assert len(files) == len({r["path"] for r in files}) == 34
    assert sum(r["bytes"] for r in files) == 40827041
    for pattern in pins["config_file_patterns"]:
        assert any(fnmatch.fnmatchcase(r["path"], pattern) for r in files)
    assert len([r for r in files if "/test-" in r["path"]]) == 11
    assert all(
        len(r["sha256"]) == 64 and "/resolve/" + dataset["revision"] + "/" in r["url"]
        for r in files
    )
    assert (
        not pins["native_execution_complete"] and not pins["source_checkbox_eligible"]
    )


def test_data_only_default_reuses_preparer_without_model_download(
    tmp_path, monkeypatch
):
    raw = b"inert file fixture"
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "JSONSCHEMABENCH",
                "dataset": {
                    "repo_id": "epfl-dlab/JSONSchemaBench",
                    "revision": "fixture-fixed",
                    "files": [
                        {
                            "path": "fixture.parquet",
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    ],
                },
            }
        )
    )
    seen = []

    def fetch(path, entry, url):
        seen.append(url)
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        assets.verify(path, entry)

    monkeypatch.setattr(assets, "fetch", fetch)
    out = tmp_path / "external"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(out)])
    assets.main(pins_path=pins, source_id="JSONSCHEMABENCH", parts=("data",))
    assert len(seen) == 1 and "/datasets/epfl-dlab/JSONSchemaBench/" in seen[0]
    receipt = json.loads((out / "preparation.private.json").read_text())
    assert receipt["model_calls"] == 0 and not receipt["source_checkbox_eligible"]
    assert {r["part"] for r in receipt["files"]} == {"data"}
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(out), "--verify-only"])
    assets.main(pins_path=pins, source_id="JSONSCHEMABENCH", parts=("data",))
    assert len(seen) == 1
    (out / "data/fixture.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        assets.main(pins_path=pins, source_id="JSONSCHEMABENCH", parts=("data",))


def test_help_without_optional_data_or_inference_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_original_assets.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--parts {data}" in result.stdout
    assert "--verify-only" in result.stdout
