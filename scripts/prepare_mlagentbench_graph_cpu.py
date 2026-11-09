"""Acquire pinned official software and emit a fresh offline graph CPU context."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mlagentbench_cifar_assets import acquire, verify

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/mlagentbench-graph-cpu-software.json"
DOCKERFILE = ROOT / "configs/workload-native/mlagentbench-graph-cpu.Dockerfile"


def prepare(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_DOWNLOAD_BUILD_OR_EXECUTION",
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    items = json.loads(PROFILE.read_text("utf-8"))["software"]
    software = args.software_root.resolve()
    if software.is_relative_to(ROOT):
        raise ValueError("external software cache required")
    for item in items:
        if Path(item["path"]).name != item["path"] or item["group"] not in {
            "core",
            "scatter",
            "sparse",
        }:
            raise ValueError("pinned simple software basename and build group required")
        if args.action == "acquire" and not (software / item["path"]).exists():
            total = sum(entry["bytes"] for entry in items)
            if args.max_bytes is None or args.max_bytes < total:
                raise ValueError("explicit byte budget must cover all pinned software")
            with urlopen(
                f"https://pypi.org/pypi/{item['project']}/{item['version']}/json",
                timeout=30,
            ) as response:
                metadata = json.load(response)
            candidates = [
                row for row in metadata["urls"] if row["filename"] == item["path"]
            ]
            if len(candidates) != 1:
                raise ValueError("pinned official PyPI file unavailable")
            release = candidates[0]
            if (
                release["digests"]["sha256"] != item["sha256"]
                or release["size"] != item["bytes"]
                or not release["url"].startswith("https://files.pythonhosted.org/")
            ):
                raise ValueError("official software release identity differs")
            acquire(
                software,
                dict(item, url=release["url"]),
                max_bytes=args.max_bytes,
                resume=args.resume,
            )
        verify(software / item["path"], item)
    if args.action == "acquire":
        report.update(
            status="PINNED_OFFICIAL_GRAPH_SOFTWARE_VERIFIED_NOT_BUILT",
            software_files=len(items),
        )
        return report
    if args.output is None:
        raise ValueError("fresh external output context required")
    output = args.output.resolve()
    if (
        output.is_relative_to(ROOT)
        or output.is_relative_to(software)
        or software.is_relative_to(output)
    ):
        raise ValueError("context must be separate from repository and source software")
    files = {"Dockerfile": DOCKERFILE.read_text("utf-8").encode()}
    for group in ["core", "scatter", "sparse"]:
        files[f"requirements-{group}.txt"] = "".join(
            f"{item['project']}=={item['version']} --hash=sha256:{item['sha256']}\n"
            for item in items
            if item["group"] == group
        ).encode()
    if args.action == "prepare":
        if (
            not output.parent.is_dir()
            or shutil.disk_usage(output.parent).free < 64 * 1024**2
        ):
            raise ValueError("existing parent with64MiB context reserve required")
        output.mkdir(mode=0o700, exist_ok=False)
        (output / "software").mkdir()
        for name, raw in files.items():
            (output / name).open("xb").write(raw)
        for item in items:
            with (
                (software / item["path"]).open("rb") as source,
                (output / "software" / item["path"]).open("xb") as target,
            ):
                shutil.copyfileobj(source, target, length=1024**2)
    for name, raw in files.items():
        path = output / name
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("context differs from public graph environment recipe")
    for item in items:
        verify(output / "software" / item["path"], item)
    expected = set(files) | {"software/" + item["path"] for item in items}
    actual = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if actual != expected:
        raise ValueError("context contains unexpected files")
    report.update(
        status="OFFLINE_GRAPH_CPU_CONTEXT_VERIFIED_NOT_BUILT_OR_NATIVE_EXECUTION",
        software_files=len(items),
        build_network_required=False,
        explicit_qualified_base_required=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["plan", "acquire", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--software-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
