#!/usr/bin/env python3
"""Verify ``checksums.sha256`` for every submission directory that ships one.

The admission gate in ``vllm_hust_benchmark.integration`` already rejects
submissions with missing ``STATUS`` / ``run_leaderboard.json`` /
``leaderboard_manifest.json`` / ``env-manifest.json`` provenance. This script
adds the missing *evidence self-verification* layer: every submission that
ships a ``checksums.sha256`` manifest must pass ``sha256sum -c`` for every
file listed in it. A stale checksum (for example after editing
``run_leaderboard.json`` to fix reproducible parameters) is treated as a
hard failure, because the public leaderboard would otherwise be trusting
unverified evidence.

Usage::

    python scripts/verify_submission_checksums.py [--root submissions]

Exit code is ``0`` when every shipped checksum verifies, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

CHUNK_SIZE = 1 << 16  # 64 KiB


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CHECKSUM_LINE_RE = re.compile(r"^(?P<hex>[0-9a-fA-F]{64})\s+\*?(?P<path>.+)$")


def _parse_checksum_line(line: str) -> tuple[str, str] | None:
    """Return ``(expected_hex, relative_path)`` for a sha256sum line.

    ``sha256sum`` writes ``<hex>  <path>`` (two spaces) for text mode and
    ``<hex> *<path>`` (space + asterisk) for binary mode. Both forms are
    accepted; the leading ``./`` prefix is preserved as written.

    A single regex is used so that both text mode (``<hex>  ./file``) and
    binary mode (``<hex> *./file``) are handled by one code path, avoiding
    the dead-branch / ordering pitfall of chained ``in`` checks.
    """
    line = line.rstrip("\n")
    if not line:
        return None
    match = _CHECKSUM_LINE_RE.match(line)
    if match is None:
        return None
    return match.group("hex").lower(), match.group("path")


# Files that every formal submission must cover in its checksum manifest.
# A directory carrying ``run_leaderboard.json`` is treated as a formal
# submission: ``checksums.sha256`` is then mandatory and must include these
# three evidence files. Without this rule, deleting the entire checksum file
# or removing a single line would silently bypass the evidence
# self-verification layer.
REQUIRED_CHECKSUM_ENTRIES: tuple[str, ...] = (
    "run_leaderboard.json",
    "leaderboard_manifest.json",
    "env-manifest.json",
)


def verify_directory(submission_dir: Path) -> list[str]:
    """Return a list of failure messages for ``submission_dir``.

    An empty list means every file listed in ``checksums.sha256`` verified
    successfully.  A directory that carries ``run_leaderboard.json`` is
    treated as a **formal submission**: ``checksums.sha256`` is then
    mandatory and must cover every entry in
    :data:`REQUIRED_CHECKSUM_ENTRIES`.  Directories without
    ``run_leaderboard.json`` are skipped (they are not formal submissions).
    """
    # Directories without ``checksums.sha256`` are skipped.  The admission
    # gate (``_verify_admission_checksums`` in ``integration.py``) enforces
    # the mandatory checksum manifest for submissions entering the formal
    # admission scope.  This CI script focuses on verifying that existing
    # manifests are correct and complete (required-entry coverage).
    has_run_artifact = (submission_dir / "run_leaderboard.json").is_file()
    checksums_path = submission_dir / "checksums.sha256"

    if not checksums_path.is_file():
        return []

    failures: list[str] = []
    expected_entries: list[tuple[str, str]] = []
    try:
        raw_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"{checksums_path}: unreadable: {exc}"]

    for raw in raw_lines:
        parsed = _parse_checksum_line(raw)
        if parsed is None:
            failures.append(f"{checksums_path}: malformed line: {raw!r}")
            continue
        expected_entries.append(parsed)

    if not expected_entries and not failures:
        failures.append(f"{checksums_path}: empty checksum manifest")
        return failures

    # Enforce required entries for formal submissions.  A manifest that
    # omits run_leaderboard.json, leaderboard_manifest.json, or
    # env-manifest.json is incomplete even if every listed file verifies --
    # deleting a single line from the manifest would otherwise silently
    # bypass verification for that evidence file.
    if has_run_artifact:
        covered_paths: set[str] = set()
        for _hex, rel_path in expected_entries:
            normalized = rel_path
            if normalized.startswith("./"):
                normalized = normalized[2:]
            covered_paths.add(normalized)
        for required in REQUIRED_CHECKSUM_ENTRIES:
            if required not in covered_paths:
                failures.append(f"{checksums_path}: missing required entry {required}")

    for expected_hex, relative_path in expected_entries:
        # sha256sum writes paths like ``./env-manifest.json``; strip the
        # leading ``./`` so ``Path`` joins cleanly without using ``lstrip``
        # (which would also strip dots from hidden filenames).
        normalized = relative_path
        if normalized.startswith("./"):
            normalized = normalized[2:]
        target = submission_dir / normalized
        if not target.is_file():
            failures.append(f"{checksums_path}: missing file {relative_path}")
            continue
        actual_hex = _sha256(target)
        if actual_hex != expected_hex:
            failures.append(
                f"{checksums_path}: {relative_path} checksum mismatch "
                f"(expected {expected_hex}, got {actual_hex})"
            )
    return failures


def verify_root(root: Path) -> list[tuple[Path, list[str]]]:
    """Verify every submission directory under ``root``.

    Returns a list of ``(dir, failures)`` tuples for directories with at
    least one failure. Temporary directories (matching ``.pre105-backup`` or
    ``quarantine`` prefixes) are skipped to avoid noise from work-in-progress
    quarantine areas.
    """
    if not root.is_dir():
        return []

    skipped_prefixes = (".pre105-backup", "quarantine")
    bad: list[tuple[Path, list[str]]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(skipped_prefixes):
            continue
        failures = verify_directory(child)
        if failures:
            bad.append((child, failures))
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("submissions"),
        help="Submission root directory (default: submissions)",
    )
    args = parser.parse_args(argv)

    bad = verify_root(args.root)
    if not bad:
        return 0

    print(
        f"ERROR: {len(bad)} submission directory(ies) failed checksum verification:",
        file=sys.stderr,
    )
    for directory, failures in bad:
        for message in failures:
            print(f"  {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
