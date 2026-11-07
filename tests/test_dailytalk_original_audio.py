"""Offline acquisition contracts only; fixtures are not native speech evidence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/prepare_dailytalk_original_audio.py"


def module():
    spec = importlib.util.spec_from_file_location("dailytalk_audio", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_default_plan_no_data_network_or_model(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--data-root",
            str(tmp_path / "audio"),
            "--metadata-root",
            str(tmp_path / "metadata"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(r.stdout)
    assert report["fixed_dialogue_ids"] == ["23", "30", "59"]
    assert report["original_selected_files"] == 59
    assert report["model_calls"] == report["network_bytes"] == 0
    assert not report["source_checkbox_eligible"] and not list(tmp_path.iterdir())


class Response:
    def __init__(self, status, headers, raw):
        self.status, self.headers, self.raw = status, headers, raw
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n):
        self.read_called = True
        return self.raw[:n]


@pytest.mark.parametrize(
    "status,range_value,modified",
    [
        (200, "bytes 0-3/100", "original"),
        (206, "bytes 0-9/100", "original"),
        (206, "bytes 0-3/100", "changed"),
    ],
)
def test_range_rejects_full_download_or_version_change_before_body(
    monkeypatch, status, range_value, modified
):
    m = module()
    response = Response(
        status, {"Content-Range": range_value, "Last-Modified": modified}, b"abcd"
    )
    monkeypatch.setattr(m, "urlopen", lambda *a, **k: response)
    remote = m.OriginalRanges(
        {
            "url": "https://example.invalid/archive",
            "bytes": 100,
            "last_modified": "original",
        },
        100,
    )
    with pytest.raises(ValueError, match="Range/version"):
        remote.read(4)
    assert not response.read_called


def test_exact_range_cache_and_budget(monkeypatch):
    m = module()
    calls = []

    def fetch(request, **kwargs):
        calls.append(request)
        return Response(
            206,
            {"Content-Range": "bytes 10-19/100", "Last-Modified": "original"},
            b"0123456789",
        )

    monkeypatch.setattr(m, "urlopen", fetch)
    remote = m.OriginalRanges(
        {
            "url": "https://example.invalid/archive",
            "bytes": 100,
            "last_modified": "original",
        },
        10,
    )
    remote.prefetch(10, 20)
    remote.seek(12)
    assert remote.read(4) == b"2345" and len(calls) == 1 and remote.transferred == 10
    remote.seek(0)
    with pytest.raises(ValueError, match="budget"):
        remote.read(1)
    assert len(calls) == 1


def fixture(tmp_path, monkeypatch, action="verify"):
    m = module()
    metadata = tmp_path / "metadata"
    data = tmp_path / "audio"
    val = metadata / "preprocessed_data/DailyTalk/val_phone.txt"
    val.parent.mkdir(parents=True)
    raw = b"0_0_d1|speaker|fixture\n0_0_d2|speaker|fixture\n0_0_d3|speaker|fixture\n"
    val.write_bytes(raw)
    content = {
        f"dailytalk/data/{n}/0_0_d{n}.{suffix}": b"offline fixture"
        for n in range(1, 4)
        for suffix in ["wav", "txt"]
    }
    content["dailytalk/metadata.json"] = b'{"offline_fixture":true}'
    files = [
        {
            "path": name,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "crc32": zlib.crc32(value),
        }
        for name, value in content.items()
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, value in content.items():
            archive.writestr(name, value)
    profile = {
        "selection": {
            "dialogue_ids": ["1", "2", "3"],
            "validation_manifest_bytes": len(raw),
            "validation_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "archive": {"bytes": len(buffer.getvalue())},
        "files": files,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(m, "PROFILE", path)
    args = SimpleNamespace(
        action=action,
        data_root=data,
        metadata_root=metadata,
        max_extracted_bytes=1024,
        max_network_bytes=65536,
        reserve_free_bytes=0,
    )
    return m, args, content, profile, buffer.getvalue()


def test_verify_reuses_complete_files_without_network(tmp_path, monkeypatch):
    m, args, content, _, _ = fixture(tmp_path, monkeypatch)
    for name, value in content.items():
        p = args.data_root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value)
    monkeypatch.setattr(
        m, "OriginalRanges", lambda *a: pytest.fail("verify must not request network")
    )
    report = m.prepare(args)
    assert (
        report["original_audio_files"] == 3
        and report["complete_original_dialogue_turns_preserved"]
    )
    assert report["network_bytes"] == report["model_calls"] == 0
    assert not report["native_synthesis_or_scoring_executed"]


def test_prepare_publishes_exact_fixture_members_then_reuses(tmp_path, monkeypatch):
    m, args, content, _, raw = fixture(tmp_path, monkeypatch, "prepare")

    class OfflineArchive(io.BytesIO):
        def __init__(self, *a):
            super().__init__(raw)
            self.transferred = 0

        def prefetch(self, *a):
            pass

    monkeypatch.setattr(m, "OriginalRanges", OfflineArchive)
    m.prepare(args)
    assert all(
        (args.data_root / name).read_bytes() == value for name, value in content.items()
    )
    assert not list(args.data_root.rglob("*.partial"))
    monkeypatch.setattr(
        m, "OriginalRanges", lambda *a: pytest.fail("complete data must be reused")
    )
    assert m.prepare(args)["original_audio_files"] == 3


def test_cannot_drop_turn_to_make_sample_pass(tmp_path, monkeypatch):
    m, args, _, profile, _ = fixture(tmp_path, monkeypatch)
    profile["files"] = [
        x for x in profile["files"] if x["path"] != "dailytalk/data/2/0_0_d2.wav"
    ]
    m.PROFILE.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="turns missing"):
        m.prepare(args)
    assert not args.data_root.exists()


def test_missing_and_changed_files_do_not_trigger_replacement_download(
    tmp_path, monkeypatch
):
    m, args, content, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        m, "OriginalRanges", lambda *a: pytest.fail("no replacement download")
    )
    with pytest.raises(ValueError, match="missing original"):
        m.prepare(args)
    name = next(iter(content))
    p = args.data_root / name
    p.parent.mkdir(parents=True)
    p.write_bytes(b"changed")
    args.action = "prepare"
    with pytest.raises(ValueError, match="size differs"):
        m.prepare(args)
    assert p.read_bytes() == b"changed"
