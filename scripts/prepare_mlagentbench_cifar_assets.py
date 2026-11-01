"""Acquire or verify default-task original assets, never run Agent/model code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/mlagentbench-cifar10-original-assets.json"


def verify(path, spec):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError("exact original regular asset and byte count required")
    algorithm = "sha256" if "sha256" in spec else "md5"
    checksum, sha = hashlib.new(algorithm), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            checksum.update(chunk)
            sha.update(chunk)
    if checksum.hexdigest() != spec[algorithm]:
        raise ValueError(
            "official publisher asset checksum differs; retain original failure"
        )
    return {
        "path": spec["path"],
        "bytes": spec["bytes"],
        "verified_algorithm": algorithm,
        "sha256": sha.hexdigest(),
    }


def acquire(root, spec, *, max_bytes, resume=False, opener=urlopen):
    root = Path(root).resolve()
    if root.is_relative_to(ROOT) or Path(spec["path"]).name != spec["path"]:
        raise ValueError("external data root and original basename required")
    if max_bytes < spec["bytes"]:
        raise ValueError("explicit byte budget must cover original asset")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = root / spec["path"]
    partial = final.with_name(final.name + ".partial")
    marker = final.with_name(final.name + ".acquisition.private.json")
    if final.exists() or final.is_symlink():
        verify(final, spec)
        return final
    identity = {"source_id": "MLAGENTBENCH", "root": str(root), "asset": spec}
    if resume:
        if marker.is_symlink() or json.loads(marker.read_text("utf-8")) != identity:
            raise ValueError("matching owned acquisition marker required")
    else:
        if any(path.exists() or path.is_symlink() for path in (partial, marker)):
            raise ValueError("inspect existing partial; never overwrite or restart it")
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream)
    if partial.is_symlink():
        raise ValueError("partial cannot be a symlink")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec["bytes"]:
        raise ValueError("partial exceeds original asset length")
    if shutil.disk_usage(root).free < spec["bytes"] - offset + 256 * 1024**2:
        raise ValueError("insufficient free space and reserve")
    if offset < spec["bytes"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with opener(Request(spec["url"], headers=headers), timeout=60) as response:
            if offset:
                expected = f"bytes {offset}-{spec['bytes'] - 1}/{spec['bytes']}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected
                ):
                    raise ValueError(
                        "server did not honor the exact original resume range"
                    )
            elif response.status != 200:
                raise ValueError("whole original response required")
            with partial.open("ab" if partial.exists() else "xb") as output:
                size = offset
                while chunk := response.read(1024**2):
                    size += len(chunk)
                    if size > spec["bytes"]:
                        raise ValueError("original response exceeds byte budget")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size != spec["bytes"]:
                raise ValueError("incomplete original response retained")
    verify(partial, spec)
    os.link(partial, final)
    partial.unlink()
    return final


def run(args):
    profile = json.loads(PROFILE.read_text("utf-8"))
    selected = [(part, profile[part]) for part in args.parts]
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_DOWNLOAD",
        "selected_parts": args.parts,
        "selected_bytes": sum(item["bytes"] for _, item in selected),
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("keep original assets outside the repository")
    results = []
    for part, item in selected:
        target = args.output / item["path"]
        if args.action == "prepare":
            if args.max_bytes is None or args.max_bytes < report["selected_bytes"]:
                raise ValueError("explicit selected-asset byte budget required")
            acquire(args.output, item, max_bytes=args.max_bytes, resume=args.resume)
        results.append({"part": part, **verify(target, item)})
    report.update(
        status="ORIGINAL_TASK_ASSETS_VERIFIED_NOT_NATIVE_AGENT_DELIVERY",
        assets=results,
        software_installed=False,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parts", nargs="+", choices=["software", "data"], default=["software", "data"]
    )
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.parts)) != len(args.parts):
        parser.error("distinct asset parts required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
