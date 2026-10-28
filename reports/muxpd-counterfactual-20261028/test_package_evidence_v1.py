"""Synthetic custody checks only; no campaign payloads, network or devices."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "quality_review_packager", Path(__file__).with_name("package_evidence_v1.py")
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def make_run(root, label="failed-attempt", **changes):
    run = root / "runs" / label
    (run / "results/worker-observations").mkdir(parents=True)
    (run / "producer").mkdir()
    (run / "compiler-cache").mkdir()
    (run / "compiler-cache/never-package.bin").write_bytes(b"cache")
    (run / "results/worker-observations/vector.npy").write_bytes(b"fixture-vector")
    (run / "results/run.json").write_text('{"collection_complete": false}\n')
    (run / "producer/driver.py").write_text("# synthetic fixture\n")
    (run / "plan-v1.json").write_text("{}\n")
    (run / "process.log").write_text("original synthetic failure\n")
    controller = {
        "finished_at": "2026-10-28T00:00:00Z",
        "passed": False,
        "exit_code": 1,
        "owned_members_after_cleanup": [],
        "errors": ["synthetic failure"],
    }
    controller.update(changes)
    (run / "controller.json").write_text(json.dumps(controller))
    return run


def test_all_attempts_retained_cache_excluded_and_archive_repeatable(tmp_path):
    root = tmp_path / "campaign"
    make_run(root)
    make_run(root, "successful-attempt", passed=True, exit_code=0, errors=[])
    (root / "analysis-losses-v1").mkdir()
    (root / "analysis-losses-v1/review.json").write_text("{}\n")
    first = packager.build(root, tmp_path / "one")
    second = packager.build(root, tmp_path / "two")
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert [r["controller_passed"] for r in first["content_manifest"]["runs"]] == [
        False,
        True,
    ]
    names = {r["name"] for r in first["content_manifest"]["files"]}
    assert "runs/failed-attempt/results/worker-observations/vector.npy" in names
    assert "runs/failed-attempt/process.log" in names
    assert not any("compiler-cache" in name for name in names)
    assert packager.verify(
        tmp_path / "one" / packager.ARCHIVE_NAME,
        tmp_path / "one/MANIFEST-v1.json",
    )["passed"]
    with pytest.raises(FileExistsError):
        packager.build(root, tmp_path / "one")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"finished_at": None}, "every observed run"),
        ({"owned_members_after_cleanup": [1234]}, "surviving owned processes"),
        ({"owned_members_after_cleanup": None}, "final owned-process inventory"),
        ({"passed": None}, "explicit boolean outcome"),
    ],
)
def test_unsafe_terminal_record_rejected_before_hashing(
    tmp_path, monkeypatch, changes, match
):
    root = tmp_path / "campaign"
    make_run(root, **changes)
    monkeypatch.setattr(
        packager, "sha_file", lambda path: pytest.fail("must reject before hashing")
    )
    with pytest.raises(ValueError, match=match):
        packager.build(root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_no_support_selection_can_omit_runs_or_hash_payloads(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    make_run(root, "first")
    make_run(root, "second", passed=True)
    monkeypatch.setattr(
        packager, "sha_file", lambda path: pytest.fail("plan must not hash payloads")
    )
    selected = packager.selection(root, support_dirs=("runs",))
    assert [r["label"] for r in selected["runs"]] == ["first", "second"]
    with pytest.raises(ValueError, match="Run subsets"):
        packager.selection(root, support_dirs=("runs/second",))


def test_selected_symlinks_and_ancestor_symlinks_rejected(tmp_path):
    root = tmp_path / "campaign"
    run = make_run(root)
    (root / "analysis-losses-v1").symlink_to(run / "results", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        packager.selection(root)
    (root / "analysis-losses-v1").unlink()
    (run / "results/dangling").symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="regular files"):
        packager.selection(root)
    (run / "results/dangling").unlink()
    with pytest.raises(ValueError, match="Unsafe"):
        packager.selection(root, extra_files=("../outside",))


def test_archive_tampering_rejected(tmp_path):
    root = tmp_path / "campaign"
    make_run(root)
    output = tmp_path / "output"
    packager.build(root, output)
    archive = output / packager.ARCHIVE_NAME
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Archive bytes differ"):
        packager.verify(archive, output / "MANIFEST-v1.json")
