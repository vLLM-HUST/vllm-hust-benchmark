"""Build a disposable analysis view from byte-preserved node capsules."""

import argparse
import shutil
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    view = root / "raw"
    view.mkdir(exist_ok=False)
    for role in ("P", "D"):
        target = root / ("raw-node-" + role)
        target.mkdir(exist_ok=False)
        with tarfile.open(root / f"replay-{role}-20261111.tar.gz") as archive:
            archive.extractall(target, filter="data")
        for family in ("services", "resources", "traces", "telemetry", "visibility"):
            source = target / family
            if not source.exists():
                continue
            output = view / family
            output.mkdir(exist_ok=True)
            for path in source.iterdir():
                destination = output / path.name
                if destination.exists():
                    raise FileExistsError(destination)
                if path.is_dir():
                    shutil.copytree(path, destination)
                else:
                    shutil.copyfile(path, destination)
        if role == "P":
            shutil.copytree(target / "runs", view / "runs")
    print(view)


if __name__ == "__main__":
    main()
