"""Shared helpers for bash-based tests.

macOS ships bash 3.2 (2007) at ``/bin/bash``; the test suite exercises
scripts that use bash 4+ features (``mapfile``, ``source <(...)``, etc.).
This helper discovers a modern bash (Homebrew on macOS, or the system
bash on Linux) so tests run identically across platforms without
requiring contributors to mess with ``$PATH``.

The CI runners use Ubuntu (bash 5+) where ``/bin/bash`` is fine, so the
override is a no-op there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

# Candidate bash paths in priority order. On macOS, Homebrew installs bash
# to /opt/homebrew/bin/bash (Apple Silicon) or /usr/local/bin/bash (Intel).
# On Linux, /bin/bash is already bash 5+ so we fall back to PATH lookup.
_BASH_CANDIDATES = (
    "/opt/homebrew/bin/bash",
    "/usr/local/bin/bash",
    "bash",  # PATH lookup; resolves to /bin/bash on Linux CI.
)


def _is_bash_version_4_or_newer(path: str) -> bool:
    """Return ``True`` if ``path`` is bash >= 4.0 (supports mapfile)."""
    try:
        # Force English locale so the version string is always
        # "GNU bash, version X.Y.Z..." regardless of the user's LANG.
        env = {**os.environ, "LC_ALL": "C"}
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    # e.g. "GNU bash, version 5.3.3(1)-release (aarch64-apple-darwin24.4.0)"
    if "version " not in first_line:
        return False
    version_part = first_line.split("version ", 1)[1]
    try:
        major = int(version_part.split(".", 1)[0])
    except ValueError:
        return False
    return major >= 4


@lru_cache(maxsize=1)
def find_modern_bash() -> str:
    """Return a path to bash >= 4.0, or ``"bash"`` as a fallback.

    Resolution order:
    1. Explicit ``BASH_TEST_BIN`` env var (escape hatch for unusual setups).
    2. Homebrew bash at ``/opt/homebrew/bin/bash`` or ``/usr/local/bin/bash``.
    3. ``bash`` from ``$PATH`` if it is >= 4.0 (typical on Linux).
    4. Last resort: ``"bash"`` (will use whatever is on ``$PATH``; tests
       that need bash 4+ features may still fail on stock macOS).
    """
    override = os.environ.get("BASH_TEST_BIN")
    if override and _is_bash_version_4_or_newer(override):
        return override

    for candidate in _BASH_CANDIDATES:
        resolved = shutil.which(candidate) if candidate == "bash" else candidate
        if not resolved:
            continue
        if _is_bash_version_4_or_newer(resolved):
            return resolved

    return "bash"


def bash_executable() -> str:
    """Stable alias for :func:`find_modern_bash`; cached after first call."""
    return find_modern_bash()
