"""Historical metadata and tiny software fixtures, not new raw-data validation."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reuse_assets", ROOT / "scripts/prepare_reused_research_assets.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
EXPECTED = {
    "WILDCHAT": 14,
    "ULTRACHAT": 8,
    "INFINITEBENCH": 12,
    "LMSYS_CHAT": 6,
    "LONGALPACA": 1,
    "NARRATIVEQA": 3,
    "NATURALQUESTIONS": 2,
    "DUREADER": 2,
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source key")
        result[key] = value
    return result


def test_original_research_assets_are_registered_without_native_completion_claim():
    sources = json.loads(
        runner.CATALOG.read_text("utf-8"), object_pairs_hook=unique_object
    )["sources"]
    assert len(sources) == 100
    assert sum(EXPECTED.values()) == 48
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    audit = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/research-acquisition-reuse-audit-20261025.json"
        ).read_text("utf-8")
    )
    assert audit["migrated_sources"] == 8 and audit["migrated_raw_file_specs"] == 48
    assert not audit["raw_data_downloaded_this_phase"] and audit["model_calls"] == 0
    assert not audit["research_observation_or_analysis_migrated"]
    assert not audit["source_checkbox_eligible"]
    for source_id, count in EXPECTED.items():
        source, files = runner.selection(source_id)
        assert len(files) == count and source == sources[source_id]
        assert all(item["role"] == "raw_source" and item["bytes"] > 0 for item in files)
        assert all(
            len(item["sha256"]) == 64 and item["url"].startswith("https://")
            for item in files
        )
        evidence = source["historical_acquisition_reuse"]
        assert evidence["research_repository"] == "intellistream/llm-workload-suite"
        assert not evidence["current_source_delivery_complete"]
        assert evidence["receipt_paths"]
        current = ledger["delivery_inventory"][source_id]
        # The frozen acquisition-only phase cannot promote a source. Later
        # independently evidenced native delivery must not be forbidden forever.
        if current["checked"]:
            assert current["status"] == "AVAILABLE"
            path = current["delivery_receipt"]
            assert (
                path
                != "docs/workload-data/validation/research-acquisition-reuse-audit-20261025.json"
            )
            delivery = json.loads((ROOT / path).read_text("utf-8"))
            assert delivery["source_id"] == source_id
            assert delivery["source_checkbox_eligible"] is True
        else:
            assert current["status"] != "AVAILABLE"
            assert current["delivery_receipt"] is None
    assert sources["NARRATIVEQA"]["repo_id"] == "google-deepmind/narrativeqa"
    assert (
        sources["NATURALQUESTIONS"]["repo_id"]
        == "google-research-datasets/natural-questions"
    )
    assert sources["INFINITEBENCH"]["repo_id"] == "xinrongzhang2022/InfiniteBench"


def test_empty_or_unregistered_selections_never_fall_back_to_new_data():
    for files in [[], ["unregistered.json"], ["data/train-00000-of-00014.parquet"] * 2]:
        with pytest.raises(ValueError, match="select distinct"):
            runner.selection("WILDCHAT", files)
    with pytest.raises(ValueError, match="audited historical"):
        runner.selection("not-an-audited-source")


@pytest.mark.parametrize("failure", [None, "host", "revision", "path", "size", "type"])
def test_explicit_github_transport_preserves_pinned_file(failure):
    source, files = runner.selection("NATURALQUESTIONS")
    item = dict(files[1])
    info = {
        "type": "file",
        "path": item["path"],
        "size": item["bytes"],
        "sha": "a" * 40,
    }
    calls = []
    if failure == "host":
        item["url"] = item["url"].replace(
            "raw.githubusercontent.com", "example.invalid"
        )
    elif failure == "revision":
        item["url"] = item["url"].replace(source["revision"], "main")
    elif failure in {"path", "size", "type"}:
        info[failure] = {
            "path": "different.jsonl",
            "size": item["bytes"] + 1,
            "type": "dir",
        }[failure]

    def loader(url):
        calls.append(url)
        return info

    if failure:
        with pytest.raises(ValueError, match="original GitHub"):
            runner.github_blob_transport(source, item, metadata_loader=loader)
    else:
        spec, url = runner.github_blob_transport(source, item, metadata_loader=loader)
        assert spec["sha256"] == item["sha256"] and spec["bytes"] == item["bytes"]
        assert spec["git_blob"] == "a" * 40
        assert (
            url
            == f"https://api.github.com/repos/{source['repo_id']}/git/blobs/" + "a" * 40
        )
        assert calls == [
            f"https://api.github.com/repos/{source['repo_id']}/contents/{item['path']}?ref={source['revision']}"
        ]


def fixture(tmp_path, monkeypatch, gated=False):
    raw = b'{"software_fixture":true}\n'
    source = {
        "source_id": "WILDCHAT",
        "revision": "fixture-revision",
        "access": {"gated": gated},
        "historical_acquisition_reuse": {"fixture_only": True},
        "files": [
            {
                "path": "original.jsonl",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": "raw_source",
                "url": "https://example.invalid/original",
            }
        ],
    }
    catalog = tmp_path / "fixture-catalog.json"
    catalog.write_text(json.dumps({"sources": {"WILDCHAT": source}}), encoding="utf-8")
    monkeypatch.setattr(runner, "CATALOG", catalog)
    monkeypatch.setattr(runner.IO, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    return raw


def test_offline_plan_never_downloads_or_creates_storage(tmp_path, monkeypatch, capsys):
    fixture(tmp_path, monkeypatch)
    output = tmp_path / "unused"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "WILDCHAT", "--data-root", str(output)]
    )
    monkeypatch.setattr(
        runner.IO,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("plan must not connect"),
    )
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "OFFLINE_ORIGINAL_ASSET_PLAN"
    assert result["model_calls"] == 0 and not result["source_checkbox_eligible"]
    assert not output.exists()


def test_verify_existing_assets_is_read_only_even_for_gated_source(
    tmp_path, monkeypatch, capsys
):
    raw = fixture(tmp_path, monkeypatch, gated="auto")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "original.jsonl").write_bytes(raw)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "WILDCHAT", "--mode", "verify", "--data-root", str(output)],
    )
    monkeypatch.setattr(
        runner.IO,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("verify must not connect"),
    )
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert (
        result["status"] == "ORIGINAL_RAW_ASSETS_VERIFIED_READ_ONLY_NOT_NATIVE_DELIVERY"
    )
    assert not result["native_execution_complete"]
    assert not (output / "source-preparation.private.json").exists()


@pytest.mark.parametrize(
    "gated,budget,error", [("auto", 1000, "access gate"), (False, 1, "byte budget")]
)
def test_download_gate_and_budget_are_checked_before_filesystem_changes(
    tmp_path, monkeypatch, gated, budget, error
):
    fixture(tmp_path, monkeypatch, gated=gated)
    output = tmp_path / "fresh"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "WILDCHAT",
            "--mode",
            "download",
            "--data-root",
            str(output),
            "--max-bytes",
            str(budget),
        ],
    )
    with pytest.raises(ValueError, match=error):
        runner.main()
    assert not output.exists()


def test_explicit_download_and_resume_reuse_verified_bytes_without_native_claim(
    tmp_path, monkeypatch
):
    raw = fixture(tmp_path, monkeypatch)
    output = tmp_path / "fresh"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "WILDCHAT", "--mode", "download", "--data-root", str(output)],
    )
    runner.main()
    assert (output / "original.jsonl").read_bytes() == raw
    receipt = json.loads(
        (output / "source-preparation.private.json").read_text("utf-8")
    )
    assert (
        receipt["status"] == "PINNED_ORIGINAL_RAW_ASSETS_ACQUIRED_NOT_NATIVE_DELIVERY"
    )
    assert not receipt["source_checkbox_eligible"] and receipt["model_calls"] == 0
    with pytest.raises(ValueError, match="fresh root"):
        runner.main()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--resume"])
    monkeypatch.setattr(
        runner.IO,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("existing bytes must be reused"),
    )
    runner.main()
    assert json.loads((output / "source-preparation.private.json").read_text("utf-8"))[
        "verified_files"
    ] == [{"path": "original.jsonl", "reused": True}]
