"""Tiny corpus fixtures only; no actual ToolACE dataset/model/tool execution."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "workload-preparation/src"))
runner = importlib.import_module("audit_toolace_original_corpus")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {
            "system": "PRIVATE_SYSTEM_SENTINEL",
            "conversations": [
                {"from": "human", "value": f"fixture question {i}"},
                {"from": "gpt", "value": "PRIVATE_REFERENCE_SENTINEL"},
                {"from": "function", "value": "historical fixture only"},
            ],
        }
        for i in range(8)
    ]
    raw = json.dumps(rows).encode()
    (data / "data.json").write_bytes(raw)
    card = b"# Publisher card fixture, not real dataset content\n"
    (data / "README.md").write_bytes(card)
    monkeypatch.setattr(runner, "DATA_SHA", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(runner, "CARD_SHA", hashlib.sha256(card).hexdigest())
    catalog = tmp_path / "catalog"
    folder = catalog / "data/catalogs"
    folder.mkdir(parents=True)
    for name in (
        "source_manifests_v1.json",
        "adapter_registry_v1.json",
        "adapter_extensions_v1.json",
    ):
        payload = json.loads(
            (ROOT / "workload-preparation/data/catalogs" / name).read_text("utf-8")
        )
        if name == "source_manifests_v1.json":
            source = payload["sources"]["TOOLACE"]
            source["files"] = [
                {
                    "path": "data.json",
                    "role": "data",
                    "bytes": len(raw),
                    "rows": 8,
                    "sha256": runner.DATA_SHA,
                },
                {
                    "path": "README.md",
                    "role": "documentation",
                    "bytes": len(card),
                    "sha256": runner.CARD_SHA,
                    "url": "https://example.invalid/publisher-card",
                },
            ]
        (folder / name).write_text(json.dumps(payload), encoding="utf-8")
    return data, tmp_path / "audit", catalog, card


def test_full_original_input_and_reference_only_roundtrips(inputs):
    data, output, catalog, _ = inputs
    result = runner.audit(data, output, catalog_root=catalog)
    assert result["original_rows_equal"] == result["prepared_rows"] == 8
    assert result["selected_original_indices"] == [0, 1, 2, 5, 6, 7]
    assert result["original_input_roundtrips"] == 6
    assert result["known_speaker_contract_counts"] == {
        "human": 8,
        "gpt": 8,
        "function": 8,
    }
    assert not result["request_renderer_verified"]
    assert (
        result["provider_artifact"]["status"]
        == "CURRENT_IMPORT_ONLY_NO_INSTALLED_WHEEL_PROOF"
    )
    assert result["model_calls"] == result["tool_calls_executed"] == 0
    assert not result["source_checkbox_eligible"]
    database = (output / "prepared/workload.sqlite").read_bytes()
    assert b"PRIVATE_SYSTEM_SENTINEL" not in database
    assert b"PRIVATE_REFERENCE_SENTINEL" not in database


def test_missing_card_requires_explicit_fetch_and_only_fetches_card(
    inputs, monkeypatch
):
    data, output, catalog, card = inputs
    (data / "README.md").unlink()
    with pytest.raises(FileNotFoundError):
        runner.audit(data, output, catalog_root=catalog)
    assert (
        json.loads((output / "receipt-private.json").read_text("utf-8"))["status"]
        == "FAILED_NOT_COMPLETE"
    )
    calls = []

    def fetch(path, spec, url):
        calls.append((path.name, url))
        path.write_bytes(card)
        runner.verify(path, spec)

    monkeypatch.setattr(runner, "fetch", fetch)
    result = runner.audit(
        data,
        output.with_name("audit-fetch"),
        catalog_root=catalog,
        fetch_missing_card=True,
    )
    assert calls == [("README.md", "https://example.invalid/publisher-card")]
    assert result["publisher_card_verified"]


def test_unknown_partial_card_is_not_overwritten(inputs, monkeypatch):
    data, output, catalog, _ = inputs
    (data / "README.md").unlink()
    (data / "README.md.partial").write_bytes(b"prior failed fixture")
    with pytest.raises(ValueError, match="partial"):
        runner.audit(data, output, catalog_root=catalog, fetch_missing_card=True)
    assert (data / "README.md.partial").read_bytes() == b"prior failed fixture"


def test_existing_outputs_and_changed_originals_are_not_overwritten(inputs):
    data, output, catalog, _ = inputs
    output.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        runner.audit(data, output, catalog_root=catalog)
    (data / "data.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        runner.audit(data, output.with_name("other"), catalog_root=catalog)


def test_source_egg_metadata_cannot_be_called_installed_wheel(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda name: None),
    )
    with pytest.raises(ValueError, match="installed-wheel"):
        runner.installed_wheel_audit(tmp_path / "unused.whl")


@pytest.mark.parametrize("failure", [None, "changed-payload", "editable"])
def test_wheel_payload_binding_with_inert_distribution_fixture(
    tmp_path, monkeypatch, failure
):
    import vllm_hust_workload_data as provider

    site = tmp_path / "site"
    package = site / "vllm_hust_workload_data"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_bytes(b"# inert installed fixture\n")
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vllm_hust_workload_data/__init__.py", init.read_bytes())
    monkeypatch.setattr(provider, "__file__", str(init))

    def metadata_text(name):
        if name == "WHEEL":
            return "Wheel-Version: 1.0\n"
        if name == "direct_url.json" and failure == "editable":
            return json.dumps({"dir_info": {"editable": True}})
        return None

    distribution = SimpleNamespace(
        locate_file=lambda name: site / name, read_text=metadata_text
    )
    monkeypatch.setattr(runner.metadata, "distribution", lambda name: distribution)
    if failure == "changed-payload":
        init.write_bytes(b"changed")
    if failure:
        with pytest.raises(ValueError):
            runner.installed_wheel_audit(wheel)
    else:
        result = runner.installed_wheel_audit(wheel)
        assert result["payloads"] == 1
        assert result["wheel_sha256"] == runner.sha(wheel)
