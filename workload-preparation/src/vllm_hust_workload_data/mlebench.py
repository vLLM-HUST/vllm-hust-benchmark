"""Expose the fixed NOMAD public competition bundle as one native task."""

from pathlib import Path

from .assets import asset_path, iter_rows, verify_file
from .registry import source_spec


COMPETITION = "nomad2018-predict-transparent-conductors"


def original_rows(path):
    path = Path(path).resolve(strict=True)
    if path.name != "description.md":
        raise ValueError("expected the prepared public NOMAD description")
    root = path.parent
    files = [verify_file(root, item) for item in source_spec("MLEBENCH")["files"]]
    geometry = {}
    for split, count in (("train", 2160), ("test", 240)):
        ids = [row["id"] for row in iter_rows(asset_path(root, f"{split}.csv"))]
        if (
            len(ids) != count
            or len(set(ids)) != count
            or not all(s.isascii() and s.isdecimal() for s in ids)
        ):
            raise ValueError("unexpected original NOMAD split identifiers")
        geometry[split] = [f"{split}/{identity}/geometry.xyz" for identity in ids]
        for reference in geometry[split]:
            asset_path(root, reference)
    yield {
        "competition": COMPETITION,
        "description": path.read_text(encoding="utf-8"),
        "public_files": files,
        "geometry_paths": geometry,
    }
