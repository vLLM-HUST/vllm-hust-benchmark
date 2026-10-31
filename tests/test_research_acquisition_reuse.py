"""Historical metadata and tiny software fixtures, not new raw-data validation."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reuse_assets", ROOT / "scripts/prepare_reused_research_assets.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_existing_public_source_baseline_positions_match_unchanged_digests():
    """Bookkeeping check, not a substitute for the actual secret scanner."""
    path = "workload-preparation/data/catalogs/source_manifests_v1.json"
    lines = (ROOT / path).read_text("utf-8").splitlines()
    baseline = json.loads((ROOT / ".secrets.baseline").read_text("utf-8"))
    for item in baseline["results"][path]:
        assert item["type"] == "Hex High Entropy String"
        assert item["is_secret"] is False
        assert type(item["is_verified"]) is bool
        line = lines[item["line_number"] - 1]
        tokens = re.findall(r"(?<![0-9a-f])[0-9a-f]{40,64}(?![0-9a-f])", line)
        # SHA1 is the scanner's existing identifier, not a security guarantee.
        assert item["hashed_secret"] in {
            hashlib.sha1(token.encode()).hexdigest() for token in tokens
        }, "Existing source digest moved: synchronize its baseline position only"


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


def test_mind2web_original_assets_use_existing_acquisition_roles():
    source, files = runner.selection("MIND2WEB")
    assert [item["path"] for item in files] == ["test.zip", "scores_all_data.pkl"]
    assert sum(item["bytes"] for item in files) == 812936103
    assert {item["role"] for item in files} == {"data", "raw_source"}
    assert all(source["revision"] in item["url"] for item in files)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source key")
        result[key] = value
    return result


def test_videomme_v2_small_assets_do_not_claim_video_or_native_delivery():
    source, files = runner.selection("VIDEOMME_V2")
    assert {item["path"] for item in files} == {
        "README.md",
        "eval.yaml",
        "test.parquet",
        "subtitle.zip",
    }
    assert sum(item["bytes"] for item in files) == 9046844
    assert all(source["revision"] in item["url"] for item in files)
    assert not any(item["path"].startswith("videos/") for item in files)
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/videomme-v2-original-preparation-20261029.json"
        ).read_text("utf-8")
    )
    assert receipt["original_rows"] == receipt["original_groups"] * 4 == 3200
    assert len(receipt["original_fields"]) == 11 and receipt["subtitle_files"] == 800
    assert sum(receipt["group_types"].values()) == 800
    assert receipt["reference_controls_are_not_model_scores"]
    assert (
        receipt["original_metric_software_controls_not_model_scores"][
            "first_wrong_sequential_logic"
        ]
        == 0
    )
    assert (
        receipt["original_metric_software_controls_not_model_scores"][
            "three_correct_relevance"
        ]
        == 56.25
    )
    assert (
        receipt["official_recommended_frames"] > receipt["current_service_image_limit"]
    )
    assert receipt["model_calls"] == receipt["local_windows_raw_files_written"] == 0
    assert not receipt["source_checkbox_eligible"] and not receipt["videos_acquired"]
    inventory = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["VIDEOMME_V2"]
    assert not inventory["checked"] and inventory["delivery_receipt"] is None


def test_livecodebench_original_assets_are_pinned_per_dataset_not_code_revision():
    source, files = runner.selection("LIVECODEBENCH")
    assert len(files) == 3
    assert sum(item["bytes"] for item in files) == 9_375_789_400
    assert source["access"]["license"] is None
    assert all(item["publisher_revision"] in item["url"] for item in files)
    assert all(item["publisher_repo_id"] in item["url"] for item in files)
    assert all(source["revision"] != item["publisher_revision"] for item in files)
    assert not any("lite" in item["url"] for item in files)
    _, small = runner.selection(
        "LIVECODEBENCH",
        [
            "execution-v2/data/test-00000-of-00001.parquet",
            "test_generation/test.parquet",
        ],
    )
    assert sum(item["bytes"] for item in small) == 144_814


def test_livecodebench_default_byte_budget_rejects_full_download_before_io(
    tmp_path, monkeypatch
):
    target = tmp_path / "must-not-be-created"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "LIVECODEBENCH", "--mode", "download", "--data-root", str(target)],
    )
    with pytest.raises(ValueError, match="byte budget"):
        runner.main()
    assert not target.exists()


def test_livecodebench_plan_is_offline_and_not_native_delivery(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare", "LIVECODEBENCH"])
    runner.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "OFFLINE_ORIGINAL_ASSET_PLAN"
    assert plan["bytes"] == 9_375_789_400
    assert plan["model_calls"] == 0
    assert not plan["native_execution_complete"]
    assert not plan["source_checkbox_eligible"]


def test_repobench_uses_fixed_original_resource_files():
    source, selected = runner.selection("REPOBENCH")
    assert source["source_id"] == "REPOBENCH"
    assert len(selected) == 10
    assert sum(item["rows"] for item in selected) == 49684
    assert {item["language"] for item in selected} == {"python", "java"}
    assert all(item["dataset_revision"] in item["url"] for item in selected)


def test_resource_identity_mismatch_rejected(tmp_path, monkeypatch):
    source = {
        "source_id": "fixture",
        "repo_id": "publisher/fixture",
        "revision": "fixed",
        "publisher_metadata_acquisition": True,
        "files": [],
        "files_resource": "files.json",
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"sources": {"fixture": source}}))
    (tmp_path / "files.json").write_text(
        json.dumps(
            {
                "schema_version": "source-file-manifest/v1",
                "source_id": "wrong-source",
                "repo_id": "publisher/fixture",
                "revision": "fixed",
                "files": [{"path": "fixture.jsonl", "role": "raw_source"}],
            }
        )
    )
    monkeypatch.setattr(runner, "CATALOG", catalog)
    with pytest.raises(ValueError, match="identity"):
        runner.selection("fixture")


def test_original_research_assets_are_registered_without_native_completion_claim():
    sources = json.loads(
        runner.CATALOG.read_text("utf-8"), object_pairs_hook=unique_object
    )["sources"]
    assert (
        len(sources) == 110
    )  # Existing109 plus already-planned OVO-Bench, not new delivery scope.
    assert "TEMPORALWIKI" in sources
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


def test_azure_lmm_has_explicit_publisher_acquisition_profile():
    source, files = runner.selection("AZURE_LMM_2025")
    assert (
        source["publisher_metadata_acquisition"]["kind"]
        == "original_publisher_gzip_metadata_profile"
    )
    assert len(files) == 1 and files[0]["rows"] == 1000000
    assert files[0]["path"].endswith(".csv.gz")


def test_empty_or_unregistered_selections_never_fall_back_to_new_data():
    for files in [[], ["unregistered.json"], ["data/train-00000-of-00014.parquet"] * 2]:
        with pytest.raises(ValueError, match="select distinct"):
            runner.selection("WILDCHAT", files)
    with pytest.raises(ValueError, match="audited historical"):
        runner.selection("not-an-audited-source")


def test_new_publisher_pinned_source_is_not_fake_historical_acquisition():
    source, files = runner.selection("TULU3")
    assert not source.get("historical_acquisition_reuse")
    assert source["publisher_metadata_acquisition"]["date"] == "2026-10-26"
    assert source["access"]["gated"] is False
    assert len(files) == 6
    assert sum(item["bytes"] for item in files) == 1412954868
    assert all(
        len(item["sha256"]) == 64 and item["role"] == "raw_source" for item in files
    )


def test_openthoughts_all_original_shards_registered_but_selection_is_explicit():
    source, all_files = runner.selection("OPENTHOUGHTS3")
    assert source["provenance_kind"] == "synthetic_reasoning_corpus"
    assert not source.get("historical_acquisition_reuse")
    assert len(all_files) == 120
    assert sum(item["bytes"] for item in all_files) == 28188197544
    assert all(
        item["role"] == "raw_source" and len(item["sha256"]) == 64 for item in all_files
    )
    names = [f"data/train-{index:05d}-of-00120.parquet" for index in [0, 25, 110]]
    _, selected = runner.selection("OPENTHOUGHTS3", names)
    assert [item["path"] for item in selected] == names
    assert len(selected) == 3  # Not a claim that all120 originals are downloaded.


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


def test_original_publisher_path_may_differ_from_local_jsonl_name():
    source, files = runner.selection("DUREADER")
    item = files[0]
    publisher_path = item["path"].removesuffix("l")
    calls = []

    def loader(url):
        calls.append(url)
        return {
            "type": "file",
            "path": publisher_path,
            "size": item["bytes"],
            "sha": "b" * 40,
        }

    spec, url = runner.github_blob_transport(source, item, metadata_loader=loader)
    assert spec["path"] == item["path"]
    assert spec["publisher_path"] == publisher_path
    assert spec["sha256"] == item["sha256"]
    assert publisher_path in calls[0] and item["path"] not in calls[0]
    assert url.endswith("b" * 40)


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


def test_explicit_publisher_data_profile_reuses_existing_asset_preparer():
    source, selected = runner.selection("CODEJUDGE_EVAL")
    assert source["publisher_metadata_acquisition"]["registered_file_roles"] == ["data"]
    assert len(selected) == 6
    assert all(item["role"] == "data" for item in selected)


def test_bigcodebench_getter_preserves_fixed_original_parquet_only():
    source, selected = runner.selection("BIGCODEBENCH")
    assert source["publisher_metadata_acquisition"]["registered_file_roles"] == ["data"]
    assert len(selected) == 1
    assert selected[0]["path"] == "data/v0.1.4-00000-of-00001.parquet"
    assert selected[0]["bytes"] == 2362110
    assert selected[0]["rows"] == 1140
    assert source["revision"] in selected[0]["url"]


def test_invalid_publisher_acquisition_roles_fail_closed(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    source = {
        "publisher_metadata_acquisition": {
            "identity_source": "fixture",
            "registered_file_roles": [{}],
        },
        "files": [{"path": "fixture.json", "role": "data"}],
    }
    catalog.write_text(json.dumps({"sources": {"fixture": source}}))
    monkeypatch.setattr(runner, "CATALOG", catalog)
    with pytest.raises(ValueError, match="asset roles"):
        runner.selection("fixture")
    source["publisher_metadata_acquisition"].pop("registered_file_roles")
    catalog.write_text(json.dumps({"sources": {"fixture": source}}))
    with pytest.raises(ValueError, match="distinct original"):
        runner.selection("fixture")
