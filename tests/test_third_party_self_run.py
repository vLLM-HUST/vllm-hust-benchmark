"""CPU admission checks for the test-specific third-party reproduction tools."""

from pathlib import Path
import hashlib
import subprocess
import sys


def test_version_timing_and_orphan_cleanup():
    kit = Path(__file__).resolve().parents[1] / "scripts" / "third_party_self_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "test_version_gate",
            "test_native_timing",
            "test_process_cleanup",
        ],
        cwd=kit,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_current_source_inventory_and_original_bytes():
    kit = Path(__file__).resolve().parents[1] / "scripts" / "third_party_self_run"

    def manifest(path):
        return {
            name.removeprefix("./"): digest
            for digest, name in (
                line.split(maxsplit=1) for line in path.read_text().splitlines() if line
            )
        }

    for name, expected in manifest(kit / "SHA256SUMS").items():
        path = kit / name
        assert path.resolve().is_relative_to(kit.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    original = kit / "original-kit"
    expected = manifest(original / "SHA256SUMS")
    snapshots = list(original.glob("*.py.source"))
    assert len(snapshots) == 3
    for path in snapshots:
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == expected[path.name.removesuffix(".source")]
        )
