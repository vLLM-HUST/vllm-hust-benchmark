"""Inert archive-byte fixtures only; no original speech, network or model calls."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

path = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_ljspeech_original_assets.py"
)
specification = importlib.util.spec_from_file_location("lj_assets", path)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)

RAW = b"BZh-inert-byte-fixture-not-real-speech"
SPEC = {
    "path": "LJSpeech-1.1.tar.bz2",
    "url": "https://example.invalid/original",
    "bytes": len(RAW),
    "sha256": hashlib.sha256(RAW).hexdigest(),
}


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def acquire(root, *, opener=None, resume=False, spec=None, max_bytes=None):
    return module.download(
        root,
        spec or SPEC,
        max_bytes=len(RAW) if max_bytes is None else max_bytes,
        resume=resume,
        opener=opener or (lambda request, timeout: Response(RAW)),
    )


@pytest.mark.parametrize("directory", ["ascii", "语音语料"])
def test_whole_original_identity_and_verified_resume_without_network(
    tmp_path, directory
):
    tmp_path = tmp_path / directory
    output = acquire(tmp_path)
    assert output.read_bytes() == RAW
    assert not output.with_name(output.name + ".partial").exists()

    def forbidden(*args, **kwargs):
        raise AssertionError("complete valid archive must not be redownloaded")

    assert acquire(tmp_path, opener=forbidden, resume=True) == output
    with pytest.raises(ValueError, match="inspect existing"):
        acquire(tmp_path)


def test_partial_transport_failure_resumes_only_exact_original_range(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        acquire(tmp_path, opener=lambda *args, **kwargs: Response(RAW[:8]))
    partial = tmp_path / (SPEC["path"] + ".partial")
    assert partial.read_bytes() == RAW[:8]

    def resumed(request, timeout):
        assert request.get_header("Range") == "bytes=8-"
        return Response(
            RAW[8:],
            status=206,
            headers={"Content-Range": f"bytes 8-{len(RAW) - 1}/{len(RAW)}"},
        )

    assert acquire(tmp_path, opener=resumed, resume=True).read_bytes() == RAW


@pytest.mark.parametrize(
    "headers,status", [({}, 200), ({}, 206), ({"Content-Range": "bytes 0-1/2"}, 206)]
)
def test_ignored_or_mismatched_range_never_overwrites_partial(
    tmp_path, headers, status
):
    with pytest.raises(ValueError, match="incomplete"):
        acquire(tmp_path, opener=lambda *args, **kwargs: Response(RAW[:8]))
    with pytest.raises(ValueError, match="exact original resume range"):
        acquire(
            tmp_path,
            resume=True,
            opener=lambda *args, **kwargs: Response(
                RAW, status=status, headers=headers
            ),
        )
    assert (tmp_path / (SPEC["path"] + ".partial")).read_bytes() == RAW[:8]


def test_empty_owned_partial_can_resume_from_zero(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        acquire(tmp_path, opener=lambda *args, **kwargs: Response(b""))
    assert acquire(tmp_path, resume=True).read_bytes() == RAW


@pytest.mark.parametrize(
    "body,error",
    [
        (b"<html>not an archive</html>", "magic"),
        (RAW + b"extra", "exceeds"),
        (b"BZh" + b"x" * (len(RAW) - 3), "checksum"),
    ],
)
def test_html_overlong_and_wrong_original_hash_cannot_publish(tmp_path, body, error):
    with pytest.raises(ValueError, match=error):
        acquire(tmp_path, opener=lambda *args, **kwargs: Response(body))
    assert not (tmp_path / SPEC["path"]).exists()


def test_resume_requires_matching_owned_record(tmp_path):
    with pytest.raises(FileNotFoundError):
        acquire(tmp_path, resume=True)
    with pytest.raises(ValueError, match="incomplete"):
        acquire(tmp_path, opener=lambda *args, **kwargs: Response(RAW[:8]))
    with pytest.raises(ValueError, match="matching private"):
        acquire(tmp_path, resume=True, spec={**SPEC, "sha256": "0" * 64})


def test_descriptive_note_changes_do_not_change_original_file_identity(tmp_path):
    acquire(tmp_path, spec={**SPEC, "note": "old provenance wording"})
    marker = tmp_path / "ljspeech-acquisition.private.json"
    original = json.loads(marker.read_text(encoding="utf-8"))
    original["dataset"]["note"] = "historical record retains descriptive provenance"
    marker.write_text(json.dumps(original), encoding="utf-8")
    raw_marker = marker.read_bytes()
    assert (
        acquire(
            tmp_path, resume=True, spec={**SPEC, "note": "new provenance wording"}
        ).read_bytes()
        == RAW
    )
    assert marker.read_bytes() == raw_marker


def test_insufficient_budget_or_bad_filename_never_starts_download(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        acquire(tmp_path, max_bytes=1)
    with pytest.raises(ValueError, match="literal"):
        acquire(tmp_path, spec={**SPEC, "path": "../other.tar.bz2"})
    assert list(tmp_path.iterdir()) == []


def test_checkout_output_is_rejected_before_write():
    with pytest.raises(ValueError, match="outside the checkout"):
        acquire(module.ROOT / "never-create-ljspeech")


def test_default_cli_plan_does_not_create_data_directory(tmp_path):
    output = tmp_path / "never-create"
    result = subprocess.run(
        [sys.executable, str(path), "--data-root", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert report["bytes"] == 2748572632 and not output.exists()
