"""Bind pinned original IMDb assets to a fresh isolated Agent workspace.

No task, prompt, scientific program or scorer is rewritten. The author's original
short-name dataset call resolves a read-only local directory. HF model aliases
resolve an explicitly local content cache, never a fabricated publisher commit.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_mlagentbench_local_agent import verify_port

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/mlagentbench-imdb-original-assets.json"
LOCAL_CACHE_LABEL = "local-pinned-content-not-publisher-git-revision"


def verify_assets(root, profile):
    root = root.resolve(strict=True)
    for name, identity in profile["files"].items():
        digest = identity["sha256"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("contained relative original asset path required")
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("contained original regular asset required")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"original asset digest differs: {name}")
    return root


def bind(args):
    profile = json.loads(PROFILE.read_text("utf-8"))
    assets = verify_assets(args.assets_root, profile)
    verify_port(args.port_root)
    env = args.port_root.resolve(strict=True) / "MLAgentBench/benchmarks/imdb/env"
    link = env / "imdb"
    cache = args.cache_root.resolve()
    port = args.port_root.resolve()
    if cache.exists() or any(
        cache.is_relative_to(root) or root.is_relative_to(cache)
        for root in [port, assets]
    ):
        raise ValueError("fresh cache separate from original Agent required")
    if link.exists() or link.is_symlink():
        raise ValueError("original dataset binding already exists")
    if not args.container_asset_root.is_absolute() or args.container_asset_root == Path(
        args.container_asset_root.anchor
    ):
        raise ValueError("absolute read-only container asset mount required")
    if (
        not args.container_cache_root.is_absolute()
        or args.container_cache_root == Path(args.container_cache_root.anchor)
        or args.container_cache_root == args.container_asset_root
    ):
        raise ValueError("separate absolute read-only container cache mount required")
    parquets = [
        name
        for name in profile["files"]
        if name.startswith("assets/") and name.endswith(".parquet")
    ]
    expected_splits = {
        f"assets/{split}-00000-of-00001.parquet"
        for split in ["train", "test", "unsupervised"]
    }
    if (
        set(parquets) != expected_splits
        or "dataset-card/README.md" not in profile["files"]
    ):
        raise ValueError("original dataset card and all three original splits required")
    cache.mkdir(parents=True, mode=0o700)
    # Retain the publisher's own metadata and layout; flattening Parquet files
    # lets automatic discovery silently omit the unsupervised split.
    dataset = cache / "dataset"
    (dataset / "plain_text").mkdir(parents=True)
    (dataset / "README.md").symlink_to(
        args.container_asset_root / "dataset-card/README.md"
    )
    for name in parquets:
        (dataset / "plain_text" / Path(name).name).symlink_to(
            args.container_asset_root / name
        )
    # Symlink paths name the container's declared read-only mount, not host paths.
    for alias in ["distilbert-base-uncased", "distilbert/distilbert-base-uncased"]:
        repo = cache / "hub" / ("models--" + alias.replace("/", "--"))
        snapshot = repo / "snapshots" / LOCAL_CACHE_LABEL
        snapshot.mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs/main").write_text(LOCAL_CACHE_LABEL, encoding="utf-8")
        for name in profile["files"]:
            if name.startswith("distilbert/"):
                (snapshot / Path(name).name).symlink_to(
                    args.container_asset_root / name
                )
    link.symlink_to(args.container_cache_root / "dataset", target_is_directory=True)
    verify_port(args.port_root)
    return {
        "status": "ORIGINAL_ASSETS_BOUND_NOT_AGENT_EXECUTION",
        "asset_files_verified": len(profile["files"]),
        "original_source_files_unchanged": True,
        "cache_label_is_not_publisher_revision": True,
        "original_dataset_card_and_three_split_layout_bound": True,
        "source_checkbox_eligible": False,
        "asset_root": str(assets),
        "container_asset_root": str(args.container_asset_root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--port-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--container-asset-root", type=Path, default=Path("/assets"))
    parser.add_argument("--container-cache-root", type=Path, default=Path("/hf"))
    print(json.dumps(bind(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
