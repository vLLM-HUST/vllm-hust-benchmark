from pathlib import Path
import subprocess
import sys


def test_rejects_wrong_asset_without_creating_output(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare_stabletoolbench_fixed_g1.py"
    )
    asset = tmp_path / "wrong"
    asset.write_bytes(b"wrong original bytes")
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--instructions",
            str(asset),
            "--toolenv",
            str(asset),
            "--cache",
            str(asset),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unexpected original asset bytes" in result.stderr
    assert not output.exists()
