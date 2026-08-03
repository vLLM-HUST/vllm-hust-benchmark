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


def test_non_submission_directory_without_checksums_skipped(
    verifier, tmp_path: Path
) -> None:
    """Directories without ``run_leaderboard.json`` are not formal submissions
    and are skipped even without ``checksums.sha256``."""
    sub = tmp_path / "submissions" / "no-artifact"
    _write_file(sub / "README.md", "just a readme")
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_formal_submission_without_checksums_skipped(verifier, tmp_path: Path) -> None:
    """A formal submission without ``checksums.sha256`` is skipped by the CI
    script.  The admission gate (``_verify_admission_checksums`` in
    ``integration.py``) enforces the mandatory checksum manifest; this CI
    script focuses on verifying existing manifests."""
    sub = tmp_path / "submissions" / "missing-checksums"
    _write_file(sub / "run_leaderboard.json", "{}")
    _write_file(sub / "leaderboard_manifest.json", "{}")
    _write_file(sub / "env-manifest.json", "{}")
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_valid_checksums_pass(verifier, tmp_path: Path) -> None:
    """A correctly generated ``checksums.sha256`` verifies cleanly."""
    sub = tmp_path / "submissions" / "valid"
    content_a = '{"entry_id": "a"}\n'
    content_b = "env-info\n"
    _write_file(sub / "data.json", content_a)
    _write_file(sub / "info.txt", content_b)
    _write_checksums(
        sub,
        [
            (_sha256(content_a), "data.json"),
            (_sha256(content_b), "info.txt"),
        ],
    )
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_binary_mode_checksums_pass(verifier, tmp_path: Path) -> None:
    """Binary mode ``sha256sum -b`` output (``<hex> *./file``) parses and
    verifies correctly, not just text mode (``<hex>  ./file``)."""
    sub = tmp_path / "submissions" / "binary-mode"
    content_a = '{"entry_id": "bin"}\n'
    _write_file(sub / "data.json", content_a)
    sub.mkdir(parents=True, exist_ok=True)
    hex_a = _sha256(content_a)
    # Binary mode format: "<hex> *./data.json"
    (sub / "checksums.sha256").write_text(f"{hex_a} *./data.json\n", encoding="utf-8")
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_stale_checksum_rejected(verifier, tmp_path: Path) -> None:
    """A stale checksum (file edited after manifest generation) is a failure."""
    sub = tmp_path / "submissions" / "stale"
    _write_file(sub / "data.json", "original-content\n")
    _write_checksums(
        sub,
        [(_sha256("different-content\n"), "data.json")],
    )
    failures = verifier.verify_directory(sub)
    assert len(failures) == 1
    assert "checksum mismatch" in failures[0]
    assert "data.json" in failures[0]


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
    _write_file(bad / "data.json", "x\n")
    _write_checksums(bad, [("0" * 64, "data.json")])

    good = root / "good-dir"
    _write_file(good / "data.json", "ok\n")
    _write_checksums(good, [(_sha256("ok\n"), "data.json")])

    results = verifier.verify_root(root)
    assert len(results) == 1
    bad_dir, failures = results[0]
    assert bad_dir.name == "bad-dir"
    assert len(failures) == 1


def test_main_returns_zero_when_clean(verifier, tmp_path: Path) -> None:
    """``main`` exits 0 when every checksum verifies."""
    root = tmp_path / "submissions"
    good = root / "good"
    _write_file(good / "data.json", "ok\n")
    _write_checksums(good, [(_sha256("ok\n"), "data.json")])
    assert verifier.main(["--root", str(root)]) == 0


def test_main_returns_one_on_failure(verifier, tmp_path: Path) -> None:
    """``main`` exits 1 when any checksum fails."""
    root = tmp_path / "submissions"
    bad = root / "bad"
    _write_file(bad / "data.json", "real\n")
    _write_checksums(bad, [(_sha256("fake\n"), "data.json")])
    assert verifier.main(["--root", str(root)]) == 1


def test_formal_submission_missing_required_entry_rejected(
    verifier, tmp_path: Path
) -> None:
    """A formal submission whose checksum manifest omits a required entry
    (``run_leaderboard.json``, ``leaderboard_manifest.json``, or
    ``env-manifest.json``) must be rejected.  Removing a single line from
    the manifest must not bypass verification for that evidence file."""
    sub = tmp_path / "submissions" / "incomplete-checksums"
    content_a = '{"entry_id": "a"}\n'
    content_b = '{"manifest": true}\n'
    _write_file(sub / "run_leaderboard.json", content_a)
    _write_file(sub / "leaderboard_manifest.json", content_b)
    _write_file(sub / "env-manifest.json", "env\n")
    # Manifest covers run_leaderboard.json and leaderboard_manifest.json
    # but deliberately omits env-manifest.json
    _write_checksums(
        sub,
        [
            (_sha256(content_a), "run_leaderboard.json"),
            (_sha256(content_b), "leaderboard_manifest.json"),
        ],
    )
    failures = verifier.verify_directory(sub)
    assert len(failures) == 1
    assert "missing required entry" in failures[0]
    assert "env-manifest.json" in failures[0]


def test_formal_submission_all_required_entries_present_passes(
    verifier, tmp_path: Path
) -> None:
    """A formal submission with all three required entries and correct
    checksums passes cleanly."""
    sub = tmp_path / "submissions" / "complete-checksums"
    content_a = '{"entry_id": "ok"}\n'
    content_b = '{"manifest": true}\n'
    content_c = '{"env": true}\n'
    _write_file(sub / "run_leaderboard.json", content_a)
    _write_file(sub / "leaderboard_manifest.json", content_b)
    _write_file(sub / "env-manifest.json", content_c)
    _write_checksums(
        sub,
        [
            (_sha256(content_a), "run_leaderboard.json"),
            (_sha256(content_b), "leaderboard_manifest.json"),
            (_sha256(content_c), "env-manifest.json"),
        ],
    )
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_formal_submission_missing_two_required_entries_rejected(
    verifier, tmp_path: Path
) -> None:
    """A formal submission whose manifest only covers run_leaderboard.json
    (missing leaderboard_manifest.json and env-manifest.json) is rejected
    with two failure messages."""
    sub = tmp_path / "submissions" / "only-run-checksums"
    content_a = '{"entry_id": "x"}\n'
    _write_file(sub / "run_leaderboard.json", content_a)
    _write_file(sub / "leaderboard_manifest.json", "{}")
    _write_file(sub / "env-manifest.json", "{}")
    _write_checksums(sub, [(_sha256(content_a), "run_leaderboard.json")])
    failures = verifier.verify_directory(sub)
    assert len(failures) == 2
    messages = "\n".join(failures)
    assert "leaderboard_manifest.json" in messages
    assert "env-manifest.json" in messages


def test_non_submission_with_checksums_still_verified(verifier, tmp_path: Path) -> None:
    """A directory without ``run_leaderboard.json`` but with a checksum
    manifest is still verified (the manifest must be correct), but required
    entry coverage is not enforced."""
    sub = tmp_path / "submissions" / "non-submission-with-checksums"
    content = "some data\n"
    _write_file(sub / "data.txt", content)
    _write_checksums(sub, [(_sha256(content), "data.txt")])
    failures = verifier.verify_directory(sub)
    assert failures == []


def test_verify_root_skips_missing_checksum_manifest(verifier, tmp_path: Path) -> None:
    """``verify_root`` skips directories without ``checksums.sha256``;
    the admission gate handles the mandatory manifest enforcement."""
    root = tmp_path / "submissions"
    no_checksums = root / "formal-no-checksums"
    _write_file(no_checksums / "run_leaderboard.json", "{}")
    _write_file(no_checksums / "leaderboard_manifest.json", "{}")
    _write_file(no_checksums / "env-manifest.json", "{}")

    good = root / "formal-complete"
    content_a = '{"ok": true}\n'
    content_b = '{"m": true}\n'
    content_c = '{"e": true}\n'
    _write_file(good / "run_leaderboard.json", content_a)
    _write_file(good / "leaderboard_manifest.json", content_b)
    _write_file(good / "env-manifest.json", content_c)
    _write_checksums(
        good,
        [
            (_sha256(content_a), "run_leaderboard.json"),
            (_sha256(content_b), "leaderboard_manifest.json"),
            (_sha256(content_c), "env-manifest.json"),
        ],
    )

    results = verifier.verify_root(root)
    assert results == []
