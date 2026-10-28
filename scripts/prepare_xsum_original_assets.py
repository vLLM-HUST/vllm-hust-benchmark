"""Fetch fixed publisher assets externally; never creates prompts or model results."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/xsum-author-pins.json"
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_path(root, part, spec):
    root = Path(root).resolve()
    path = (root / part / spec["path"]).resolve()
    if Path(spec["path"]).is_absolute() or not path.is_relative_to(root / part):
        raise ValueError("contained original asset path required")
    return path


def verify(path, spec):
    if path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
        raise ValueError("original publisher asset identity mismatch; do not overwrite")


def fetch(path, spec, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with urlopen(
        Request(url, headers={"User-Agent": "original-XSum-asset-preparation"}),
        timeout=60,
    ) as response:
        if url.startswith("https://api.github.com/"):
            blob = json.load(response)
            if blob.get("encoding") != "base64" or blob.get("sha") != spec["git_blob"]:
                raise ValueError("publisher Git object response mismatch")
            temporary.write_bytes(
                base64.b64decode("".join(blob["content"].split()), validate=True)
            )
        else:
            with temporary.open("wb") as destination:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    destination.write(block)
    verify(temporary, spec)
    temporary.rename(path)


def main(
    *,
    pins_path=PINS,
    source_id="XSUM",
    parts=("data", "model", "scoring"),
    hf_download_query=False,
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=list(parts),
        default=list(parts),
    )
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="read-only verification of existing original files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only a matching preparer's private output",
    )
    args = parser.parse_args()
    pins_path = Path(pins_path)
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    if pins.get("source_id") != source_id:
        raise ValueError("asset source identity differs from selected profile")
    jobs = []
    for part in dict.fromkeys(args.parts):
        files = (
            pins["dataset"]["files"]
            if part == "data"
            else pins["model"]["files"]
            if part == "model"
            else pins["scoring_files"]
        )
        for spec in files:
            if part == "scoring":
                url = (
                    spec["url"]
                    .replace("https://hf-mirror.com", args.hf_endpoint)
                    .replace("https://huggingface.co", args.hf_endpoint)
                )
            else:
                source = pins["dataset"] if part == "data" else pins["model"]
                prefix = "datasets/" if part == "data" else ""
                url = f"{args.hf_endpoint}/{prefix}{source['repo_id']}/resolve/{source['revision']}/{spec['path']}"
                if hf_download_query:
                    url += "?download=true"
            jobs.append((part, spec, url))
    if args.verify_only:
        for part, spec, _ in jobs:
            verify(bound_path(args.output, part, spec), spec)
        print(
            json.dumps(
                {
                    "source_id": source_id,
                    "status": "ORIGINAL_ASSETS_VERIFIED_READ_ONLY",
                    "files": len(jobs),
                    "model_calls": 0,
                }
            )
        )
        return
    previous = args.output / "preparation.private.json"
    if args.output.exists():
        if not args.resume or not previous.is_file():
            raise ValueError("fresh output or explicit matching resume required")
        prior = json.loads(previous.read_text(encoding="utf-8"))
        if prior.get("source_id") != source_id or prior.get("pins_sha256") != sha(
            pins_path
        ):
            raise ValueError("resume source binding differs")
    else:
        args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": source_id,
        "pins_sha256": sha(pins_path),
        "status": "PREPARING_ORIGINAL_ASSETS",
        "files": [],
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }

    def save():
        previous.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for part, spec, url in jobs:
            path = bound_path(args.output, part, spec)
            reused = path.exists()
            if reused:
                verify(path, spec)
            else:
                fetch(path, spec, url)
            receipt["files"].append(
                {"part": part, "path": spec["path"], "verified": True, "reused": reused}
            )
            save()
        receipt["status"] = "ORIGINAL_ASSETS_PREPARED_NOT_NATIVE_EXECUTION"
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
