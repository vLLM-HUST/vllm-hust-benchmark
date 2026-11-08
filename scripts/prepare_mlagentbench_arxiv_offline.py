"""Run original graph preparation in isolation using verified offline transport."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mlagentbench_arxiv_assets import PROFILE
from prepare_mlagentbench_cifar_assets import verify
from run_mlagentbench_local_agent import PINS, check_isolation

PREFIX = "MLAgentBench/benchmarks/ogbn-arxiv/"


def verify_task(root):
    pins = json.loads(PINS.read_text("utf-8"))
    items = [item for item in pins["files"] if item["path"].startswith(PREFIX)]
    if not items:
        raise ValueError("pinned original graph task files required")
    for item in items:
        path = root / item["path"][len(PREFIX) :]
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("contained original graph task files required")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("original graph task software differs")
    return len(items)


def offline_transport(archive, networks, expected_url):
    def confirm(url):
        if url != expected_url:
            raise ValueError("only the pinned original graph URL is allowed")
        return True

    def download(url, folder, log=True):
        confirm(url)
        if Path(folder).resolve() != networks:
            raise ValueError("original task network directory required")
        target = networks / "arxiv.zip"
        with archive.open("rb") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024**2)
        return str(target)

    return confirm, download


def prepare(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "task": "ogbn-arxiv",
        "status": "PLAN_ONLY_NO_PREPARATION_OR_MODEL_EXECUTION",
        "native_agent_tasks_run": 0,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        return report
    if not Path("/.dockerenv").is_file():
        raise ValueError("original preparation requires an isolated task container")
    check_isolation(Path("/proc/self/status").read_text())
    root = args.task_root.resolve(strict=True)
    count = verify_task(root)
    profile = json.loads(PROFILE.read_text("utf-8"))
    archive = args.archive.resolve(strict=True)
    verify(archive, profile["data"])
    networks = root / "env/networks"
    if networks.exists() or networks.is_symlink():
        raise ValueError("fresh original graph cache required; no implicit overwrite")
    if not networks.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("graph cache must remain inside owned task workspace")
    if shutil.disk_usage(root).free < 1024**3:
        raise ValueError("one GiB preparation free-space reserve required")
    import ogb.nodeproppred.dataset_pyg as loader

    confirm, download = offline_transport(archive, networks, profile["publisher_url"])
    previous = (loader.decide_download, loader.download_url, Path.cwd())
    try:
        loader.decide_download, loader.download_url = confirm, download
        os.chdir(root / "scripts")
        state = runpy.run_path(str(root / "scripts/prepare.py"), run_name="__main__")
        dataset = state["dataset"]
        graph = dataset[0]
        split = dataset.get_idx_split()
    finally:
        loader.decide_download, loader.download_url = previous[:2]
        os.chdir(previous[2])
    report.update(
        status="ORIGINAL_GRAPH_PREPARATION_COMPLETED_NOT_AGENT_EXECUTION",
        original_task_files_verified=count,
        original_script_executed="scripts/prepare.py",
        offline_transport_only_no_task_or_processing_changes=True,
        original_nodes=int(graph.num_nodes),
        original_edges=int(graph.num_edges),
        original_classes=int(dataset.num_classes),
        original_split_sizes={key: int(value.numel()) for key, value in split.items()},
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
