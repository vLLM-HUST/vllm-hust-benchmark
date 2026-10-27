"""Metadata and inert preparation only; not MBPP task or scoring evidence."""

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
wrapper = importlib.import_module("prepare_mbpp_original_assets")
assets = importlib.import_module("prepare_xsum_original_assets")


def test_both_author_variants_and_all_original_splits_are_pinned():
    pins = json.loads(wrapper.PINS.read_text("utf-8"))
    dataset = pins["dataset"]
    assert dataset["repo_id"] == "google-research-datasets/mbpp"
    assert len(dataset["revision"]) == 40
    assert len(dataset["files"]) == 9
    assert sum(item["bytes"] for item in dataset["files"]) == 360549
    assert {item["path"] for item in dataset["files"]} == {
        f"{config}/{split}-00000-of-00001.parquet"
        for config in ("full", "sanitized")
        for split in ("train", "test", "validation", "prompt")
    } | {"README.md"}
    assert pins["config_split_rows"]["full"]["test"] == 500
    assert pins["config_split_rows"]["sanitized"]["test"] == 257
    assert pins["paper_fewshot_task_ids"] == [2, 3, 4]
    assert pins["official_test_task_id_range"] == [11, 510]
    assert not pins["source_checkbox_eligible"]
    assert not pins["original_github_payload_parity_verified"]


def test_data_only_acquisition_resume_and_readonly_verify(tmp_path, monkeypatch):
    raw = b"inert asset fixture, not publisher data"
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "MBPP",
                "dataset": {
                    "repo_id": "google-research-datasets/mbpp",
                    "revision": "fixture",
                    "files": [
                        {
                            "path": "full/fixture.parquet",
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fetch(path, spec, url):
        calls.append(url)
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        assets.verify(path, spec)

    monkeypatch.setattr(assets, "fetch", fetch)
    output = tmp_path / "out"
    for extra in ([], ["--resume"], ["--verify-only"]):
        monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(output), *extra])
        assets.main(pins_path=pins, source_id="MBPP", parts=("data",))
    assert len(calls) == 1
    assert "/datasets/google-research-datasets/mbpp/resolve/fixture/" in calls[0]
    receipt = json.loads((output / "preparation.private.json").read_text("utf-8"))
    assert receipt["model_calls"] == 0
    assert not receipt["source_checkbox_eligible"]
    (output / "data/full/fixture.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        assets.main(pins_path=pins, source_id="MBPP", parts=("data",))


def test_help_requires_no_dataset_or_model_packages():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_mbpp_original_assets.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--parts {data}" in result.stdout
