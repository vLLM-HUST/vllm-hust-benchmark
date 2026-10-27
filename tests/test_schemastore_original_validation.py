"""Synthetic tree contracts, never original asset or native runtime evidence."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ENTRY = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_schemastore_original_validation.py"
)


def load():
    spec = importlib.util.spec_from_file_location("schemastore_validation", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    content = b"synthetic schema fixture"
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas/a.json").write_bytes(content)
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    body = b"100644 a.json\0" + bytes.fromhex(blob)
    child = hashlib.sha1(f"tree {len(body)}\0".encode() + body).hexdigest()
    body = b"40000 schemas\0" + bytes.fromhex(child)
    root = hashlib.sha1(f"tree {len(body)}\0".encode() + body).hexdigest()
    tree = {
        "sha": "not_trusted_label",
        "tree": [
            {"path": "schemas", "mode": "040000", "type": "tree", "sha": child},
            {
                "path": "schemas/a.json",
                "type": "blob",
                "mode": "100644",
                "sha": blob,
                "size": len(content),
            },
        ],
    }
    return tree, root


def test_reconstructs_git_root_instead_of_trusting_manifest_label(tmp_path):
    tree, root = fixture(tmp_path)
    assert load().verify_tree(tmp_path, tree, root) == 1
    with pytest.raises(ValueError, match="root Git tree"):
        load().verify_tree(tmp_path, tree, "0" * 40)


def test_changed_original_file_rejects_before_execution(tmp_path):
    tree, root = fixture(tmp_path)
    (tmp_path / "schemas/a.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Git blob"):
        load().verify_tree(tmp_path, tree, root)


@pytest.mark.parametrize("change", ["duplicate", "traversal", "link", "disconnected"])
def test_unsafe_or_disconnected_tree_is_rejected(tmp_path, change):
    tree, root = fixture(tmp_path)
    if change == "duplicate":
        tree["tree"].append(tree["tree"][-1])
    elif change == "traversal":
        tree["tree"][-1]["path"] = "../escape"
    elif change == "link":
        tree["tree"][-1]["mode"] = "120000"
    else:
        tree["tree"] = tree["tree"][1:]
    with pytest.raises(ValueError):
        load().verify_tree(tmp_path, tree, root)
