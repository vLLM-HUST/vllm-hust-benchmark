import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "duplex_runtime_lock", ROOT / "scripts/verify_full_duplex_model_runtime.py"
)
ENTRY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ENTRY)
LOCK = (
    ROOT / "docs/workload-data/validation/full-duplex-model-runtime-lock-20261107.json"
)


def test_all_installed_packages_have_public_provenance_without_claiming_rebuild():
    lock = ENTRY.load_lock(LOCK)
    assert len(lock["packages"]) == 184
    assert lock["fresh_complete_environment_rebuild_verified"] is False
    assert lock["base_image_publicly_downloadable_claimed"] is False
    sglang = lock["sources"]["sglang-active-module-source"]
    assert sglang["metadata_is_not_active_source_revision"] is True


@pytest.mark.parametrize(
    "change", ["duplicate", "protected", "credential", "member", "missing", "editable"]
)
def test_rejects_incomplete_or_unsafe_bindings(tmp_path, change):
    lock = copy.deepcopy(json.loads(LOCK.read_text()))
    rows = lock["packages"]
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "protected":
        rows[0]["name"] = "torch"
    elif change == "credential":
        rows[0]["url"] = "https://token@files.pythonhosted.org/file.whl"
    elif change == "member":
        row = next(r for r in rows if r["binding"] == "PUBLISHER_ARCHIVE_WHEEL")
        row["sha256"] = "0" * 64
    elif change == "missing":
        rows.pop()
    else:
        row = next(r for r in rows if r["binding"] == "EDITABLE_SOURCE")
        row["source_binding"] = "unresolved"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        ENTRY.load_lock(path)


def test_installed_version_check_reports_missing_instead_of_accepting_inventory(
    monkeypatch,
):
    lock = ENTRY.load_lock(LOCK)
    expected = {row["name"]: row["version"] for row in lock["packages"]}
    expected.update(lock["protected_core_versions"])
    monkeypatch.setattr(ENTRY.metadata, "version", lambda name: expected[name])
    assert ENTRY.verify_versions(lock) == 190

    def missing(name):
        raise ENTRY.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ENTRY.metadata, "version", missing)
    with pytest.raises(ValueError, match="versions differ"):
        ENTRY.verify_versions(lock)
