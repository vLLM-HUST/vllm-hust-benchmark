"""ZIP/identity boundary fixtures, not original media or video decoding."""

import importlib.util
from pathlib import Path
import zipfile
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "videomme2_media",
    Path(__file__).resolve().parents[1] / "scripts/prepare_videomme_v2_fixed_media.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def fixture(tmp_path):
    path = tmp_path / "original.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("videos/001.mp4", b"fixture-not-real-video")
        archive.writestr("videos/002.mp4", b"not selected")
    profile = {
        "archive_bytes": path.stat().st_size,
        "archive_sha256": entry.digest(path),
        "selected_video_ids": ["001"],
    }
    return path, profile


def test_exact_member_copy_preserves_bytes_and_no_extra_video(tmp_path):
    path, profile = fixture(tmp_path)
    output = tmp_path / "prepared"
    report = entry.extract(profile, path, output, budget=1024)
    assert (output / "001.mp4").read_bytes() == b"fixture-not-real-video"
    assert not (output / "002.mp4").exists()
    assert report["all_selected_member_crc_verified"]
    assert report["source_checkbox_eligible"] is False
    assert report["status"].endswith("NOT_DECODED_OR_EXECUTED")


def test_archive_identity_mismatch_preserves_no_false_success(tmp_path):
    path, profile = fixture(tmp_path)
    profile["archive_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="whole-archive"):
        entry.extract(profile, path, tmp_path / "output", budget=1024)


def test_budget_checked_before_output_created(tmp_path):
    path, profile = fixture(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="budget"):
        entry.extract(profile, path, output, budget=1)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../001.mp4", "/001.mp4", "dir\\001.mp4"])
def test_unsafe_member_rejected(tmp_path, name):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        item = zipfile.ZipInfo("placeholder")
        item.filename = item.orig_filename = name
        archive.writestr(item, b"fixture")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError):
        entry.selected_members(archive, ["001"])


def test_no_missing_task_replacement(tmp_path):
    path, _ = fixture(tmp_path)
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="missing"):
        entry.selected_members(archive, ["003"])


def test_duplicate_same_video_member_is_not_silently_chosen(tmp_path):
    path, _ = fixture(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("another/001.mp4", b"ambiguous")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="ambiguous"):
        entry.selected_members(archive, ["001"])


def test_saved_media_hash_audit_does_not_require_deleted_archive(tmp_path):
    archive, profile = fixture(tmp_path)
    media = tmp_path / "media"
    entry.extract(profile, archive, media, budget=1024)
    archive.unlink()
    args = SimpleNamespace(
        verify_media_root=media, output=tmp_path / "audit", decode=False
    )
    report = entry.verify_media(profile, args)
    assert report["status"] == "SAVED_ORIGINAL_MEDIA_HASHES_VERIFIED_NOT_DECODED"
    assert report["model_calls"] == 0
    assert report["source_checkbox_eligible"] is False


def test_saved_media_change_is_not_silently_accepted(tmp_path):
    archive, profile = fixture(tmp_path)
    media = tmp_path / "media"
    entry.extract(profile, archive, media, budget=1024)
    (media / "001.mp4").write_bytes(b"modified")
    args = SimpleNamespace(
        verify_media_root=media, output=tmp_path / "audit", decode=False
    )
    with pytest.raises(ValueError, match="identity differs"):
        entry.verify_media(profile, args)


def test_decoder_command_is_full_cpu_video_to_null_not_frame_sampling():
    command = entry.decode_command(Path("/trusted/ffmpeg"), Path("/media/001.mp4"))
    assert command[command.index("-threads") + 1] == "2"
    assert command[command.index("-f") + 1] == "null"
    assert command[command.index("-vsync") + 1] == "0"
    assert command[-1] == "-"
    assert not any(
        flag in command for flag in ("-ss", "-t", "-frames:v", "-vf", "-hwaccel")
    )
    assert (
        entry.decoded_frames("frame=25\nprogress=continue\nframe=100\nprogress=end\n")
        == 100
    )


@pytest.mark.parametrize(
    "progress", ["frame=0\nprogress=end", "frame=100\nprogress=continue"]
)
def test_decoder_requires_terminal_positive_frames(progress):
    with pytest.raises(ValueError, match="terminal"):
        entry.decoded_frames(progress)
