from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "audit_stateaxis_mod_onoff_readiness.py"
)
SPEC = importlib.util.spec_from_file_location("stateaxis_mod_onoff", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _repo(tmp_path: Path, *, active: bool) -> tuple[dict[str, object], Path]:
    slug = "stateaxis-example"
    repo = tmp_path / slug
    package = repo / "src" / "stateaxis_example"
    package.mkdir(parents=True)
    manifest = {
        "schema_version": "0.3-experimental",
        "implementation": [{"status": "active" if active else "import_only"}],
        "activation": {
            "entry_points": [{"group": "example", "name": "example"}] if active else []
        },
    }
    (package / "vllm-hust-extension-v0.3.json").write_text(json.dumps(manifest))
    (repo / "PROVENANCE.json").write_text(
        json.dumps({"implementation_extracted": active})
    )
    (repo / "MOD_METADATA.json").write_text(
        json.dumps(
            {
                "lifecycle": {
                    "activation_contract": "active implementation"
                    if active
                    else "Import-only descriptor"
                }
            }
        )
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    entry = {
        "mod_id": "org.vllm-hust.stateaxis-example",
        "repository": f"intellistream/{slug}",
        "commit": commit,
    }
    return entry, tmp_path


def test_import_only_descriptor_is_not_a_zero_percent_result(tmp_path: Path) -> None:
    entry, root = _repo(tmp_path, active=False)
    result = MODULE.inspect_repo(entry, root)
    assert result["on_status"] == "ON_NOT_RUNNABLE"
    assert result["off_status"] == "NOT_RUN_NO_PAIRED_ON_ARM"
    assert result["performance_result"] is None
    assert {row["code"] for row in result["blockers"]} == {
        "IMPLEMENTATION_NOT_ACTIVE",
        "NO_ACTIVATION_ENTRY_POINT",
        "IMPLEMENTATION_NOT_EXTRACTED",
        "ACTIVATION_CONTRACT_REFUSES_ON",
    }


def test_active_implementation_is_admitted_to_next_phase(tmp_path: Path) -> None:
    entry, root = _repo(tmp_path, active=True)
    result = MODULE.inspect_repo(entry, root)
    assert result["on_status"] == "READY_FOR_MATCHED_ON_OFF"
    assert result["off_status"] == "PENDING_PAIRED_EXECUTION"
    assert result["performance_result"] is None
    assert result["blockers"] == []


def test_checkout_must_match_the_catalog_pin(tmp_path: Path) -> None:
    entry, root = _repo(tmp_path, active=True)
    entry["commit"] = "0" * 40
    result = MODULE.inspect_repo(entry, root)
    assert result["on_status"] == "ON_NOT_RUNNABLE"
    assert {row["code"] for row in result["blockers"]} == {"PINNED_COMMIT_MISMATCH"}
