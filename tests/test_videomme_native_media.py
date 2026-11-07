"""Safe extraction software fixtures only; no original media or model calls."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def media(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("prepare_videomme_native_media")


def descriptor(path, kind="videos"):
    return {"path": path, "role": "source_archive", "archive_kind": kind}


def test_native_media_plan_has_no_reads_downloads_or_output(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_videomme_native_media.py"),
            "--official-checkout",
            str(tmp_path / "absent-code"),
            "--snapshot",
            str(tmp_path / "absent-data"),
            "--output",
            str(tmp_path / "absent-receipt"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY" and plan["model_calls"] == 0
    assert not plan["downloads"] and not plan["source_checkbox"]
    assert list(tmp_path.iterdir()) == []


def test_original_unpacker_requires_exact_source_identity(media):
    with pytest.raises(ValueError, match="Unverified original"):
        media.original_unpacker(b"raise RuntimeError('must not execute')")


def test_archive_members_and_extracted_bytes_are_related_exactly(tmp_path, media):
    raw = b"video-fixture-only"
    with zipfile.ZipFile(tmp_path / "video.zip", "w") as packed:
        packed.writestr("original/clip.mp4", raw)
    layout, count = media.audit_archives(tmp_path, [descriptor("video.zip")])
    assert count == len(raw)
    (tmp_path / "video").mkdir()
    (tmp_path / "video/clip.mp4").write_bytes(raw)
    assert len(media.verify_extracted(tmp_path, layout)) == 1
    (tmp_path / "video/clip.mp4").write_bytes(b"x" * len(raw))
    with pytest.raises(ValueError, match="differ"):
        media.verify_extracted(tmp_path, layout)


@pytest.mark.parametrize(
    "name", ["../escape.mp4", "/absolute.mp4", "x\\bad.mp4", "C:/drive.mp4"]
)
def test_unsafe_original_members_are_never_extracted(tmp_path, media, name):
    with zipfile.ZipFile(tmp_path / "video.zip", "w") as packed:
        info = zipfile.ZipInfo("fixture-placeholder")
        info.filename = name  # Preserve raw ZIP header bytes on Windows too.
        packed.writestr(info, b"fixture")
    with pytest.raises(ValueError, match="Unsafe"):
        media.audit_archives(tmp_path, [descriptor("video.zip")])
    assert not (tmp_path / "video").exists()


def test_basename_overwrites_and_unpacked_budget_are_rejected(tmp_path, media):
    for index in (1, 2):
        with zipfile.ZipFile(tmp_path / f"video{index}.zip", "w") as packed:
            packed.writestr(f"folder{index}/same.mp4", b"fixture")
    with pytest.raises(ValueError, match="overwrite"):
        media.audit_archives(
            tmp_path, [descriptor("video1.zip"), descriptor("video2.zip")]
        )
    with pytest.raises(ValueError, match="budget"):
        media.audit_archives(tmp_path, [descriptor("video1.zip")], max_unpacked_bytes=1)


def test_existing_partial_media_are_retained_without_source_execution(tmp_path, media):
    snapshot = tmp_path / media.REVISION
    video = snapshot / "video"
    video.mkdir(parents=True)
    sentinel = video / "partial.bin"
    sentinel.write_bytes(b"retain")
    with pytest.raises(ValueError, match="Existing native directories"):
        media.prepare(
            tmp_path / "absent-code", snapshot, tmp_path / "receipt.json", {"files": []}
        )
    assert sentinel.read_bytes() == b"retain"
    assert not (snapshot / ".videomme-native-extraction.lock").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_only_own_extraction_lease_is_released(tmp_path, media):
    lock = tmp_path / ".videomme-native-extraction.lock"
    with media.extraction_lease(tmp_path):
        assert lock.exists()
        with pytest.raises(FileExistsError):
            with media.extraction_lease(tmp_path):
                pytest.fail("A second extractor entered the same snapshot")
    assert not lock.exists()


def test_incomplete_release_fails_before_hashing_large_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    assets = importlib.import_module("prepare_videomme_original_assets")
    (tmp_path / "first").write_bytes(b"fixture")

    def forbidden_hash():
        pytest.fail("Incomplete release must fail before hashing")

    monkeypatch.setattr(assets.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="missing"):
        assets.verify_files(
            tmp_path,
            [
                {"path": "first", "bytes": 7, "sha256": "fixture"},
                {"path": "absent", "bytes": 1, "sha256": "fixture"},
            ],
        )


def test_memory_headroom_respects_container_limit(tmp_path, media, monkeypatch):
    paths = {
        "/proc/meminfo": "MemAvailable: 8388608 kB\n",
        "/sys/fs/cgroup/memory.max": "4294967296",
        "/sys/fs/cgroup/memory.current": "3221225472",
    }
    monkeypatch.setattr(
        media.Path, "read_text", lambda path: paths[str(path).replace("\\", "/")]
    )
    monkeypatch.setattr(
        media.Path, "is_file", lambda path: str(path).replace("\\", "/") in paths
    )
    assert media.available_memory_bytes() == 1073741824


@pytest.mark.parametrize(
    "stats,expected",
    [
        (
            "file 3072\nshmem 0\ninactive_file 2048\nfile_dirty 0\nfile_writeback 0\n",
            3072,
        ),
        (
            "file 3072\nshmem 2048\ninactive_file 2048\nfile_dirty 100\nfile_writeback 100\n",
            1848,
        ),
        ("file 3072\nshmem 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n", 1024),
        (
            "file 3072\nshmem 0\ninactive_file 2048\nfile_dirty 2048\nfile_writeback 1\n",
            1024,
        ),
        (
            "file -1\nshmem 0\ninactive_file 2048\nfile_dirty 0\nfile_writeback 0\n",
            1024,
        ),
        ("file 3072\ninactive_file 2048\n", 1024),
    ],
)
def test_clean_inactive_cache_is_reclaimable_without_touching_model_pages(
    media, monkeypatch, stats, expected
):
    paths = {
        "/proc/meminfo": "MemAvailable: 8388608 kB\n",
        "/sys/fs/cgroup/memory.max": "4096",
        "/sys/fs/cgroup/memory.current": "3072",
        "/sys/fs/cgroup/memory.stat": stats,
    }
    monkeypatch.setattr(
        media.Path, "read_text", lambda path: paths[str(path).replace("\\", "/")]
    )
    monkeypatch.setattr(
        media.Path, "is_file", lambda path: str(path).replace("\\", "/") in paths
    )
    assert media.available_memory_bytes() == expected
