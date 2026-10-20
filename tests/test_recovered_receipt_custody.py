import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "custody",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_recovered_receipt_custody.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dedup_preserves_invalid_status_and_has_no_local_archive_dependency(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for name, raw in [
        ("status.json", b""),
        ("duplicate.json", b""),
        ("known.json", b"{}\n"),
    ]:
        (source / name).write_bytes(raw)
        rows.append(
            {
                "original_path": name,
                "bytes": len(raw),
                "sha256": MODULE.sha(raw),
                "source": {"archive": "website-evidence"},
            }
        )
    rows[0]["run_status"] = {"parse_error": "invalid-json", "passed": None}
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_root": str(source),
                "files": rows,
                "selection": "test",
                "limitations": "test",
                "excluded_directories": [],
                "included_suffixes": [".json"],
            }
        )
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "url": "https://example.invalid/known.tar.gz",
                        "sha256": "known-archive",
                        "files": [
                            {"sha256": MODULE.sha(b"{}\n"), "member": "known.json"}
                        ],
                    }
                ]
            }
        )
    )
    destination = tmp_path / "out"
    MODULE.build(inventory, destination, index)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(manifest["files"]) == 3
    assert len(manifest["asset"]["members"]) == 1
    assert manifest["files"][0]["run_status"]["passed"] is None
    assert manifest["files"][0]["source"] == manifest["files"][1]["source"]
    assert all("archive" not in row["source"] for row in manifest["files"])
    with pytest.raises(ValueError, match="replace"):
        MODULE.build(inventory, destination, index)
    (source / "status.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        MODULE.build(inventory, tmp_path / "other", index)


@pytest.mark.parametrize("name", ["../escape", "/absolute", ""])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.safe_path(name)


@pytest.mark.parametrize(
    "tag,asset",
    [
        ("../tag", "safe.tar.gz"),
        ("safe", "manifest.json"),
        ("safe", "dir/file.tar.gz"),
        ("safe", "a?b.tar.gz"),
        ("safe", "a#b.tar.gz"),
    ],
)
def test_invalid_release_identity_rejected_before_reading(tmp_path, tag, asset):
    with pytest.raises(ValueError, match="Invalid"):
        MODULE.build(
            tmp_path / "missing", tmp_path / "out", tmp_path / "index", tag, asset
        )
