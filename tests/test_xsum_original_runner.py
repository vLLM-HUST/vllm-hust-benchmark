"""CPU fixtures only; not model execution or summarization quality evidence."""

from copy import deepcopy
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "xsum_runner", Path(__file__).resolve().parents[1] / "scripts/run_xsum_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

asset_spec = importlib.util.spec_from_file_location(
    "xsum_assets",
    Path(__file__).resolve().parents[1] / "scripts/prepare_xsum_original_assets.py",
)
assets = importlib.util.module_from_spec(asset_spec)
asset_spec.loader.exec_module(assets)


def test_publisher_split_and_native_generation_pins():
    pins = json.loads(runner.PINS.read_text("utf-8"))
    assert pins["model"]["repo_id"] == "facebook/bart-large-xsum"
    assert {entry["split"]: entry["rows"] for entry in pins["dataset"]["files"]} == {
        "test": 11334,
        "validation": 11332,
        "train": 204045,
    }
    assert pins["protocol"]["encoder_maximum_tokens"] == 1024
    assert pins["generation_config"]["max_length"] == 62
    assert pins["generation_config"]["min_length"] == 11
    assert pins["generation_config"]["num_beams"] == 6
    assert pins["generation_config"]["no_repeat_ngram_size"] == 3
    assert len(pins["scoring_files"]) == 3


def test_original_selection_does_not_replace_failed_positions():
    assert runner.selected_indices(11334) == [0, 1, 2]
    assert runner.selected_indices(2) == [0, 1]
    assert runner.selected_indices(4, full=True) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        runner.selected_indices(3, indices=[0, 0, 2])
    with pytest.raises(ValueError):
        runner.selected_indices(3, indices=[3])


def test_saved_case_requires_complete_original_article_and_reference():
    row = {
        "document": "Article\nOriginal paragraph",
        "summary": "Reference",
        "id": "42",
    }
    case = {
        "original_index": 2,
        "original_row": deepcopy(row),
        "original_document_utf8_sha256": hashlib.sha256(
            row["document"].encode()
        ).hexdigest(),
        "native_result": [{"summary_text": "Prediction"}],
    }
    assert runner.validate_saved_case(case, row, 2) == "Prediction"
    for field in ["document", "summary", "id"]:
        altered = deepcopy(row)
        altered[field] += "changed"
        with pytest.raises(ValueError, match="original article"):
            runner.validate_saved_case(case, altered, 2)


def test_saved_run_must_bind_original_model_and_full_generation_protocol():
    pins = json.loads(runner.PINS.read_text("utf-8"))
    receipt = {
        "source_id": "XSUM",
        "status": "ORIGINAL_XSUM_GENERATION_COMPLETED",
        "model_revision": pins["model"]["revision"],
        "dataset_revision": pins["dataset"]["revision"],
        "versions": {"transformers": pins["framework_version"]},
        "device": "npu:1",
        "native_generation_config": deepcopy(pins["generation_config"]),
        "all_model_parameters_verified_on_npu": True,
    }
    runner.validate_saved_run(receipt, pins)
    receipt["native_generation_config"]["max_length"] = 10
    with pytest.raises(ValueError, match="model/protocol binding"):
        runner.validate_saved_run(receipt, pins)


def test_private_asset_resolution_does_not_escape(tmp_path):
    with pytest.raises(ValueError, match="contained"):
        runner.contained(tmp_path, "../outside")


@pytest.mark.parametrize("api", [False, True])
def test_asset_fetch_preserves_bytes_and_checks_publisher_identity(
    tmp_path, monkeypatch, api
):
    raw = b"original-software-fixture\n"
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    payload = (
        json.dumps(
            {
                "sha": blob,
                "encoding": "base64",
                "content": base64.b64encode(raw).decode(),
            }
        ).encode()
        if api
        else raw
    )
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    spec = {
        "path": "fixture",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": blob,
    }
    path = assets.bound_path(tmp_path, "scoring", spec)
    assets.fetch(
        path,
        spec,
        "https://api.github.com/example" if api else "https://publisher.example/file",
    )
    assert path.read_bytes() == raw
    assets.verify(path, spec)
    with pytest.raises(ValueError, match="contained"):
        assets.bound_path(tmp_path, "scoring", {"path": "../../escape"})


def test_shared_asset_entry_keeps_profile_identity_separate(
    tmp_path, monkeypatch, capsys
):
    raw = tmp_path / "data" / "fixture"
    raw.parent.mkdir()
    raw.write_bytes(b"software-fixture")
    pins = tmp_path / "profile.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "MULTINEWS",
                "dataset": {
                    "files": [
                        {
                            "path": "fixture",
                            "bytes": raw.stat().st_size,
                            "sha256": assets.sha(raw),
                        }
                    ],
                    "repo_id": "publisher/source",
                    "revision": "fixture",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--output", str(tmp_path), "--parts", "data", "--verify-only"],
    )
    assets.main(pins_path=pins, source_id="MULTINEWS")
    assert json.loads(capsys.readouterr().out)["source_id"] == "MULTINEWS"
    with pytest.raises(ValueError, match="source identity"):
        assets.main(pins_path=pins, source_id="XSUM")
