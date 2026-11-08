"""Transport guards for the split NOMAD validator, without grading labels."""

from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mlebench_nomad_validation.py"
EXTRACT = runpy.run_path(str(SCRIPT))["extract_submission"]
CONTENT_TYPE = "multipart/form-data; boundary=test-boundary"


def multipart(*parts):
    result = b""
    for name, filename, data in parts:
        result += (
            b"--test-boundary\r\n"
            + f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            + b"\r\n"
            + data
            + b"\r\n"
        )
    return result + b"--test-boundary--\r\n"


def test_extract_preserves_bytes_and_does_not_use_client_path():
    csv = b"id,value\r\n1,7\r\n"
    assert EXTRACT(CONTENT_TYPE, multipart(("file", "../../input.csv", csv))) == csv


@pytest.mark.parametrize(
    "parts",
    [
        (("wrong", "file.csv", b"csv"),),
        (("file", "file.csv", b""),),
        (("file", "one.csv", b"one"), ("file", "two.csv", b"two")),
    ],
)
def test_invalid_multipart_is_rejected(parts):
    with pytest.raises(ValueError):
        EXTRACT(CONTENT_TYPE, multipart(*parts))
