import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_mind2web_author_source import prepare  # noqa: E402


def test_default_plan_is_inert(tmp_path):
    output = tmp_path / "absent"
    raw = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_mind2web_author_source.py"),
            "--author-root",
            str(output),
        ]
    )
    assert json.loads(raw)["files"] == 31
    assert not output.exists()


@pytest.mark.parametrize("tampered", [False, True])
def test_source_extraction_checks_identity_before_publishing(tmp_path, tampered):
    raw = b"original publisher fixture\n"
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        entry = tarfile.TarInfo("publisher-root/module.py")
        entry.size = len(raw)
        handle.addfile(entry, io.BytesIO(raw))
    pins = {
        "files": [
            {
                "path": "module.py",
                "bytes": len(raw),
                "sha256": "0" * 64 if tampered else hashlib.sha256(raw).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            }
        ]
    }
    output = tmp_path / "prepared"
    if tampered:
        with pytest.raises(ValueError, match="identity mismatch"):
            prepare(archive, output, pins)
        assert not (output / "module.py").exists()
    else:
        assert prepare(archive, output, pins) == 1
        assert (output / "module.py").read_bytes() == raw
