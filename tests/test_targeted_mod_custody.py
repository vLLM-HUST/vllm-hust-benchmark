import hashlib
import json
from pathlib import Path


def test_targeted_original_bytes():
    root = Path(__file__).resolve().parents[1] / "archive/mod-targeted-20261023"
    for name, count in (
        ("compress", 3),
        ("manager", 2),
        ("pyramid-current", 100),
        ("compress-history", 46),
        ("vspec-regression", 317),
        ("manager-paper", 7),
        ("manager-experiments", 208),
        ("pyramid-legacy", 7),
        ("manager-formal", 77),
        ("compress-mixed", 5),
        ("vspec-mixed", 5),
        ("manager-projections", 3),
        ("pyramid-mixed", 4),
        ("vspec-legacy-narratives", 4),
        ("compress-release", 3),
        ("compress-host", 3),
        ("pyramid-selected", 3),
        ("muxpd-documents", 3),
        ("muxpd-tp8-groupfx", 5),
        ("kv-school-document", 1),
        ("vspec-adaptive-narratives", 3),
        ("vspec-b16-frontier", 7),
    ):
        capsule = root / name
        manifest = json.loads((capsule / "manifest.json").read_text())
        assert manifest["source_visibility"] == "public"
        assert len(manifest["source_commit"]) == 40
        rows = manifest["files"]
        assert len(rows) == count
        assert {row["path"] for row in rows} == {
            p.relative_to(capsule / "snapshot").as_posix()
            for p in (capsule / "snapshot").rglob("*")
            if p.is_file()
        }
        for row in rows:
            raw = (capsule / "snapshot" / row["path"]).read_bytes()
            assert len(raw) == row["bytes"]
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
