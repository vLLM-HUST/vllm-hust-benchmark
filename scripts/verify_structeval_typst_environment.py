"""Check the original Typst renderer's declared CLI prerequisites, not tasks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


POLICY = Path("/etc/ImageMagick-6/policy.xml")


def pdf_policy(text):
    # Some distribution policy files contain nested comment openings. Like the
    # converter, terminate comments at the first -->, but never modify them.
    active = re.sub(r"<!--.*?-->", lambda match: " " * len(match[0]), text, flags=re.S)
    ET.fromstring(active)  # Reject other malformed syntax instead of guessing.
    found = []
    for match in re.finditer(r"<policy\b[^>]*?/>", active):
        element = ET.fromstring(match[0])
        if element.get("domain") == "coder" and element.get("pattern") == "PDF":
            found.append((match, element))
    if len(found) != 1:
        raise ValueError("expected one explicit PDF policy; inspect environment")
    return found[0]


def allow_pdf_read(path=POLICY):
    """Build-time only: change exactly one active PDF rights attribute."""
    text = path.read_bytes().decode("utf-8")
    match, _ = pdf_policy(text)
    replacement, count = re.subn(
        r'\brights\s*=\s*([\'"])[^\'"]*\1', 'rights="read"', match[0]
    )
    if count != 1:
        raise ValueError("expected one PDF rights attribute; inspect environment")
    changed = text[: match.start()] + replacement + text[match.end() :]
    if pdf_policy(changed)[1].get("rights") != "read":
        raise ValueError("PDF policy update did not validate")
    path.write_bytes(changed.encode("utf-8"))


def verify():
    paths = {name: shutil.which(name) for name in ("typst", "convert", "gs", "fc-list")}
    if not all(paths.values()):
        raise ValueError("original compiler/converter and fonts prerequisites missing")
    typst = subprocess.check_output(
        [paths["typst"], "--version"], text=True, timeout=10
    ).strip()
    if not typst.startswith("typst 0.13.1 "):
        raise ValueError("explicitly selected compiler differs")
    if pdf_policy(POLICY.read_bytes().decode("utf-8"))[1].get("rights") != "read":
        raise ValueError("original PDF conversion read policy not ready")
    fonts = subprocess.check_output(
        [paths["fc-list"]], text=True, timeout=10
    ).splitlines()
    if not fonts:
        raise ValueError("no fonts available")
    packages = subprocess.check_output(
        [
            "dpkg-query",
            "-W",
            "-f=${Package}=${Version}\n",
            "imagemagick",
            "ghostscript",
            "fontconfig",
            "fonts-dejavu-core",
        ],
        text=True,
        timeout=10,
    ).splitlines()
    return {
        "status": "TYPST_CLI_PREREQUISITES_VERIFIED_NOT_RENDER_OR_VQA",
        "typst_version": typst,
        "binary_sha256": hashlib.sha256(Path(paths["typst"]).read_bytes()).hexdigest(),
        "observed_os_packages": packages,
        "original_author_dependency_lock": False,
        "available_font_entries": len(fonts),
        "pdf_policy_rights": "read",
        "original_task_executed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-pdf-read-policy", action="store_true")
    args = parser.parse_args()
    if args.prepare_pdf_read_policy:
        if os.geteuid() != 0:
            raise ValueError("explicit root build stage required")
        allow_pdf_read()
    else:
        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
