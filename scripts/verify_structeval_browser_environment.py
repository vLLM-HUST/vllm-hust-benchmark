"""Verify fixed browser dependencies/assets, not a renderer or native task run."""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


NAMES = {
    "playwright",
    "pyee",
    "greenlet",
    "typing-extensions",
    "pyyaml",
    "xmltodict",
    "toml",
}


def verify_pins(lock):
    pins = [
        line.strip().split("==")
        for line in Path(lock).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if (
        len(pins) != 7
        or any(len(pair) != 2 for pair in pins)
        or {pair[0].lower() for pair in pins} != NAMES
    ):
        raise ValueError("complete observed seven-package extension required")
    for name, expected in pins:
        if metadata.version(name) != expected:
            raise ValueError("observed browser dependency differs: " + name)
    return dict(pins)


def verify(lock):
    if sys.version_info[:2] != (3, 11):
        raise ValueError("observed Python3.11 profile required")
    pins = verify_pins(lock)
    import yaml
    import xmltodict
    import toml
    from playwright.async_api import async_playwright

    assert (
        yaml is not None
        and xmltodict is not None
        and toml is not None
        and async_playwright is not None
    )
    manifest = Path(
        metadata.distribution("playwright").locate_file(
            "playwright/driver/package/browsers.json"
        )
    )
    browsers = json.loads(manifest.read_bytes())["browsers"]
    shell = next(row for row in browsers if row["name"] == "chromium-headless-shell")
    cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/playwright"))
    shell_root = cache / ("chromium_headless_shell-" + str(shell["revision"]))
    binaries = [
        p
        for p in shell_root.rglob("chrome-headless-shell")
        if p.is_file() and not p.is_symlink() and os.access(p, os.X_OK)
    ]
    if len(binaries) != 1:
        raise ValueError("matching installed original headless browser asset required")
    with binaries[0].open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "status": "BROWSER_SOFTWARE_IMPORTS_AND_ASSET_IDENTITY_VERIFIED_NOT_NATIVE_TASK",
        "observed_versions": pins,
        "browser_revision": shell["revision"],
        "browser_version": shell.get("browserVersion"),
        "observed_browser_binary_sha256": digest,
        "author_lock": False,
        "sandbox_qualified": False,
        "original_renderer_executed": False,
        "original_task_executed": False,
        "vqa_executed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().lock), indent=2))


if __name__ == "__main__":
    main()
