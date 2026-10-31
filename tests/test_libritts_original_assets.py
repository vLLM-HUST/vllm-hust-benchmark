"""Inert transport fixtures, not original speech or native model evidence."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_libritts_original_assets.py"
)
specification = importlib.util.spec_from_file_location("libritts_assets", path)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
RAW = b"\x1f\x8b-inert-transport-fixture"
SPEC = {
    "path": "test-other.tar.gz",
    "url": "https://example.invalid/original",
    "bytes": len(RAW),
    "sha256": hashlib.sha256(RAW).hexdigest(),
    "md5": hashlib.md5(RAW).hexdigest(),
}


class Response(io.BytesIO):
    status = 200
    headers = {}


def acquire(root, opener=None, **kwargs):
    return module.download(
        root,
        SPEC,
        max_bytes=len(RAW),
        opener=opener or (lambda *a, **k: Response(RAW)),
        **kwargs,
    )


def test_original_identity_and_no_duplicate_download(tmp_path):
    result = acquire(tmp_path)
    assert result.read_bytes() == RAW
    assert (
        acquire(tmp_path, opener=lambda *a, **k: pytest.fail("redownload"), resume=True)
        == result
    )
    with pytest.raises(ValueError, match="existing"):
        acquire(tmp_path)


def test_interrupted_range_requires_matching_owner_and_exact_server_range(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        acquire(tmp_path, opener=lambda *a, **k: Response(RAW[:8]))
    with pytest.raises(ValueError, match="exact original resume"):
        acquire(tmp_path, resume=True)

    class Partial(Response):
        status = 206
        headers = {"Content-Range": f"bytes 8-{len(RAW) - 1}/{len(RAW)}"}

    def response(request, **kwargs):
        assert request.get_header("Range") == "bytes=8-"
        return Partial(RAW[8:])

    assert acquire(tmp_path, opener=response, resume=True).read_bytes() == RAW


def test_checksum_failure_retains_original_partial(tmp_path):
    bad = b"\x1f\x8b" + b"x" * (len(RAW) - 2)
    with pytest.raises(ValueError, match="checksum"):
        acquire(tmp_path, opener=lambda *a, **k: Response(bad))
    assert (tmp_path / "test-other.tar.gz.partial").read_bytes() == bad
    assert not (tmp_path / "test-other.tar.gz").exists()


def test_budget_and_checkout_guards_before_any_network(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        module.download(tmp_path, SPEC, max_bytes=0)
    with pytest.raises(ValueError, match="outside"):
        acquire(module.ROOT / "unused-fixture-root")
