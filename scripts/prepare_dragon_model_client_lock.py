"""Freeze a reviewed DRAGON pip resolution; never install or run a model.

These are observed integration dependencies, not an author lock. Protected NPU
runtime distributions must be inherited unchanged, never silently replaced.
"""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def canonical(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("plain distribution name required")
    return re.sub(r"[-_.]+", "-", name).lower()


def hashed_requirements(resolution, protected):
    protected = {canonical(name) for name in protected}
    records = resolution.get("install")
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty reviewed pip install plan required")
    seen, lines = set(), []
    for item in records:
        name = canonical(item["metadata"]["name"])
        version = item["metadata"]["version"]
        if name in seen or name in protected:
            raise ValueError("duplicate package or protected serving-stack replacement")
        if not isinstance(version, str) or not re.fullmatch(
            r"[A-Za-z0-9.!+_-]+", version
        ):
            raise ValueError("literal resolved version required")
        info = item["download_info"]
        url = urlsplit(info["url"])
        if (
            url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("only credential-free official PyPI artifacts permitted")
        sha = info.get("archive_info", {}).get("hashes", {}).get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("resolved original artifact SHA256 required")
        lines.append(f"{name}=={version} --hash=sha256:{sha}")
        seen.add(name)
    if "sentence-transformers" not in seen | protected:
        raise ValueError("HuggingFace full extra missing sentence-transformers")
    return sorted(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = json.loads(args.protected.read_bytes())
    protected = protected.get("serving_versions_protected", protected)
    lines = hashed_requirements(json.loads(args.resolution.read_bytes()), protected)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# Observed DRAGON integration resolution, not an author lock.\n")
        stream.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "HASHED_SOFTWARE_PLAN_ONLY_NOT_INSTALLED",
                "packages": len(lines),
                "protected_packages": len(protected),
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )
