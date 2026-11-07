"""Entry/metadata fixtures, not real waveform or NPU evidence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ljspeech_native", ROOT / "scripts/run_ljspeech_vits_original_npu.py"
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


@pytest.mark.parametrize("full,expected", [(False, 3), (True, 13100)])
def test_plan_needs_no_assets_and_creates_nothing(tmp_path, full, expected):
    args = [
        sys.executable,
        str(ROOT / "scripts/run_ljspeech_vits_original_npu.py"),
        "--archive",
        str(tmp_path / "absent.tar.bz2"),
        "--author-root",
        str(tmp_path / "author"),
        "--checkpoint",
        str(tmp_path / "absent.pth"),
        "--output",
        str(tmp_path / "output"),
    ]
    if full:
        args.append("--full")
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["rows"] == expected
    assert plan["model_calls"] == 0 and not plan["output_truncated"]
    assert not plan["source_checkbox_eligible"]
    assert list(tmp_path.iterdir()) == []


def test_metadata_preserves_both_original_text_columns(tmp_path):
    rows = [
        f'LJ001-{index:04d}|raw "quotation"|normalized text' for index in range(1, 4)
    ]
    rows += [f"fixture-{index}|raw|normalized" for index in range(13097)]
    raw = ("\n".join(rows) + "\n").encode()
    path = tmp_path / "fixture.tar.bz2"
    with tarfile.open(path, "w:bz2") as packed:
        member = tarfile.TarInfo("LJSpeech-1.1/metadata.csv")
        member.size = len(raw)
        packed.addfile(member, io.BytesIO(raw))
    parsed = native.metadata_rows(path, expected_digest=hashlib.sha256(raw).hexdigest())
    assert len(parsed) == 13100
    assert parsed[0] == ["LJ001-0001", 'raw "quotation"', "normalized text"]
    with pytest.raises(ValueError, match="digest differs"):
        native.metadata_rows(path)


def test_checkpoint_rejects_unrelated_or_partial_weights(tmp_path):
    partial = tmp_path / "pretrained_ljs.pth"
    partial.write_bytes(b"fixture only")
    with pytest.raises(ValueError, match="original regular"):
        native.verify_checkpoint(partial)
