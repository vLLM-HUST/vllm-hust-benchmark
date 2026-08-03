"""Tests for ``scripts/verify_submission_checksums.py``.

The checksum verifier is the evidence self-verification layer: every
submission directory that ships a ``checksums.sha256`` manifest must pass
``sha256sum -c`` for every file listed in it. These tests cover the happy
path, stale checksums, missing files, malformed lines, and the quarantine
skip rule.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "verify_submission_checksums.py"
)


def _load_module():
    """Load the script as a module (it has no package parent)."""
    spec = importlib.util.spec_from_file_location(
        "verify_submission_checksums", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def verifier():
    return _load_module()


def _write_file(path: Path, content: str = "hello\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_checksums(directory: Path, entries: list[tuple[str, str]]) -> None:
    """Write ``checksums.sha256`` with ``./`` prefix like sha256sum does."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"{hex_digest}  ./{name}" for hex_digest, name in entries]
    (directory / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_directory_without_checksums_skipped(verifier, tmp_path: Path) -> None:
    """Directories without ``checksums.sha256`` are not flagged."""
    sub = tmp_path / "submissions" / "no-checksums"
    _write_file(sub / "run_leaderboard.json", "{}")
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_valid_checksums_pass(verifier, tmp_path: Path) -> None:
    """A correctly generated ``checksums.sha256`` verifies cleanly."""
    sub = tmp_path / "submissions" / "valid"
    content_a = '{"entry_id": "a"}\n'
    content_b = "env-info\n"
    _write_file(sub / "run_leaderboard.json", content_a)
    _write_file(sub / "env-manifest.json", content_b)
    _write_checksums(
        sub,
        [
            (_sha256(content_a), "run_leaderboard.json"),
            (_sha256(content_b), "env-manifest.json"),
        ],
    )
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_binary_mode_checksums_pass(verifier, tmp_path: Path) -> None:
    """Binary mode ``sha256sum -b`` output (``<hex> *./file``) parses and
    verifies correctly, not just text mode (``<hex>  ./file``)."""
    sub = tmp_path / "submissions" / "binary-mode"
    content_a = '{"entry_id": "bin"}\n'
    _write_file(sub / "run_leaderboard.json", content_a)
    sub.mkdir(parents=True, exist_ok=True)
    hex_a = _sha256(content_a)
    # Binary mode format: "<hex> *./run_leaderboard.json"
    (sub / "checksums.sha256").write_text(
        f"{hex_a} *./run_leaderboard.json\n", encoding="utf-8"
    )
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_stale_checksum_rejected(verifier, tmp_path: Path) -> None:
    """A stale checksum (file edited after manifest generation) is a failure."""
    sub = tmp_path / "submissions" / "stale"
    _write_file(sub / "run_leaderboard.json", "original-content\n")
    _write_checksums(
        sub,
        [(_sha256("different-content\n"), "run_leaderboard.json")],
    )
    failures = verifier.verify_directory(sub)
    assert len(failures) == 1
    assert "checksum mismatch" in failures[0]
    assert "run_leaderboard.json" in failures[0]


def test_missing_file_rejected(verifier, tmp_path: Path) -> None:
    """A file listed in the manifest but absent from disk is a failure."""
    sub = tmp_path / "submissions" / "missing-file"
    _write_checksums(
        sub,
        [(_sha256("x\n"), "run_leaderboard.json")],
    )
    failures = verifier.verify_directory(sub)
    assert len(failures) == 1
    assert "missing file" in failures[0]
    assert "run_leaderboard.json" in failures[0]


def test_malformed_line_rejected(verifier, tmp_path: Path) -> None:
    """A line that does not parse as ``<hex>  <path>`` is a failure."""
    sub = tmp_path / "submissions" / "malformed"
    sub.mkdir(parents=True)
    (sub / "checksums.sha256").write_text("not-a-valid-line\n", encoding="utf-8")
    failures = verifier.verify_directory(sub)
    assert len(failures) == 1
    assert "malformed line" in failures[0]


def test_empty_manifest_rejected(verifier, tmp_path: Path) -> None:
    """An empty ``checksums.sha256`` is a failure (manifest is required once
    the file exists)."""
    sub = tmp_path / "submissions" / "empty-manifest"
    sub.mkdir(parents=True)
    (sub / "checksums.sha256").write_text("", encoding="utf-8")
    failures = verifier.verify_directory(sub)
    assert len(failures) == 1
    assert "empty checksum manifest" in failures[0]


def test_quarantine_directories_skipped(verifier, tmp_path: Path) -> None:
    """Directories prefixed with ``quarantine`` or ``.pre105-backup`` are
    skipped to avoid noise from work-in-progress quarantine areas."""
    root = tmp_path / "submissions"
    bad = root / "quarantine-bad"
    _write_file(bad / "run_leaderboard.json", "x\n")
    _write_checksums(bad, [("0" * 64, "run_leaderboard.json")])

    backup = root / ".pre105-backup-20260731"
    _write_file(backup / "run_leaderboard.json", "y\n")
    _write_checksums(backup, [("0" * 64, "run_leaderboard.json")])

    results = verifier.verify_root(root)
    assert results == []


def test_verify_root_reports_failures(verifier, tmp_path: Path) -> None:
    """``verify_root`` returns ``(dir, failures)`` tuples for bad dirs."""
    root = tmp_path / "submissions"
    bad = root / "bad-dir"
    _write_file(bad / "run_leaderboard.json", "x\n")
    _write_checksums(bad, [("0" * 64, "run_leaderboard.json")])

    good = root / "good-dir"
    _write_file(good / "run_leaderboard.json", "ok\n")
    _write_checksums(good, [(_sha256("ok\n"), "run_leaderboard.json")])

    results = verifier.verify_root(root)
    assert len(results) == 1
    bad_dir, failures = results[0]
    assert bad_dir.name == "bad-dir"
    assert len(failures) == 1


def test_main_returns_zero_when_clean(verifier, tmp_path: Path) -> None:
    """``main`` exits 0 when every checksum verifies."""
    root = tmp_path / "submissions"
    good = root / "good"
    _write_file(good / "run_leaderboard.json", "ok\n")
    _write_checksums(good, [(_sha256("ok\n"), "run_leaderboard.json")])
    assert verifier.main(["--root", str(root)]) == 0


def test_main_returns_one_on_failure(verifier, tmp_path: Path) -> None:
    """``main`` exits 1 when any checksum fails."""
    root = tmp_path / "submissions"
    bad = root / "bad"
    _write_file(bad / "run_leaderboard.json", "real\n")
    _write_checksums(bad, [(_sha256("fake\n"), "run_leaderboard.json")])
    assert verifier.main(["--root", str(root)]) == 1
