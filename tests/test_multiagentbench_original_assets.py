"""Preparation fixtures, never native multi-agent execution claims."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_multiagentbench_original_assets as entry  # noqa: E402 - standalone script path


def test_default_plan_is_inert_and_preserves_five_original_scenarios(tmp_path):
    destination = tmp_path / "absent"
    report = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/prepare_multiagentbench_original_assets.py"),
                "--data-root",
                str(destination),
            ],
            text=True,
        )
    )
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert len(report["files"]) == 5
    assert report["total_bytes"] == 5369142
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not destination.exists()


def test_preparation_requires_explicit_budget_before_acquisition(tmp_path, monkeypatch):
    def forbidden(*_args):
        pytest.fail("budget rejection must precede acquisition")

    monkeypatch.setattr(entry, "acquire", forbidden)
    with pytest.raises(ValueError, match="max-bytes"):
        entry.prepare(tmp_path / "absent", action="prepare", max_bytes=1)


def test_missing_originals_not_replaced_by_synthetic_prompts(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        entry.prepare(tmp_path, action="verify")
    assert not list(tmp_path.iterdir())


def test_repository_is_not_a_data_destination():
    with pytest.raises(ValueError, match="outside"):
        entry.prepare(ROOT / "not-created")


@pytest.mark.parametrize("corrupt", [False, True])
def test_existing_fixture_bytes_are_reused_or_rejected_without_download(
    tmp_path, monkeypatch, corrupt
):
    # Artificial records only test file custody, never benchmark validity.
    raw = b'{"fixture_only":true}\n' * 100
    item = {
        "path": "multiagentbench/fixture/fixture_main.jsonl",
        "bytes": len(raw),
        "git_blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
    }
    ledger = tmp_path / "fixture-ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "delivery_inventory": {
                    "MULTIAGENTBENCH": {
                        "protocol_review": {
                            "revision": "fixture-only",
                            "original_assets": [item],
                        },
                        "original_asset_acquisition": {
                            "sha256_by_scenario": {
                                "fixture": hashlib.sha256(raw).hexdigest()
                            }
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setattr(entry, "LEDGER", ledger)
    data = tmp_path / "data"
    target = data / item["path"]
    target.parent.mkdir(parents=True)
    saved = raw + b"corruption" if corrupt else raw
    target.write_bytes(saved)

    def forbidden(*_args):
        pytest.fail("existing data must not trigger redownload")

    monkeypatch.setattr(entry, "acquire", forbidden)
    if corrupt:
        with pytest.raises(ValueError, match="identity"):
            entry.prepare(data, action="prepare", max_bytes=len(raw))
    else:
        report = entry.prepare(data, action="prepare", max_bytes=len(raw))
        assert report["files"][0]["rows"] == 100
        assert report["source_checkbox_eligible"] is False
    assert target.read_bytes() == saved
