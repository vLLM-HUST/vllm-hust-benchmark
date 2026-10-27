"""Guard publication scope, preserved failures and visible token limitations."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path("reports/muxpd-followup-20261025/performance-lifecycle-v4")
DATA = ROOT / RELATIVE
SPEC = importlib.util.spec_from_file_location("muxpd_v4", DATA / "verify_v4.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def candidate(tmp_path):
    # Preserve the actual prior small metadata context; never copy archive payloads.
    shutil.copytree(
        ROOT / "reports/muxpd-followup-20261025",
        tmp_path / "reports/muxpd-followup-20261025",
    )
    return tmp_path / RELATIVE


def rehash_copy(root, relative, mutate):
    path = root / relative
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, indent=2) + "\n")
    mapping = root / "canonical-copy-map.json"
    value = json.loads(mapping.read_text())
    for row in value["copies"]:
        if row["canonical_path"] == relative:
            row["sha256"] = row["source"]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            row["bytes"] = row["source"]["bytes"] = path.stat().st_size
    mapping.write_text(json.dumps(value))


def test_all_terminal_roles_windows_and_historical_bytes():
    result = VERIFY.verify(DATA)
    assert (
        result["terminal_runs"],
        result["capsules"],
        result["original_files_preserved"],
    ) == (12, 13, 248)
    assert result["all_windows"] == 36
    assert result["failed_default_reference_preserved"]
    assert not result["paired_quality_accepted"]


def test_rehashed_summary_cannot_hide_native_plugin_token_difference(tmp_path):
    root = candidate(tmp_path)
    rehash_copy(
        root,
        "analysis/primary/group-abba-with-native.json",
        lambda value: value["comparisons"]["long-P-ingress"].update(
            all_generated_tokens_identical_across_arms=True
        ),
    )
    with pytest.raises(ValueError, match="token divergence hidden"):
        VERIFY.verify(root)


def test_preparation_cannot_be_relabelled_remote_verified(tmp_path):
    root = candidate(tmp_path)
    path = root / "release-assets-v4.json"
    value = json.loads(path.read_text())
    value["assets"][0]["remote_verified"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="remote availability"):
        VERIFY.verify(root)


def test_rehashed_metadata_cannot_turn_failed_reference_into_success(tmp_path):
    root = candidate(tmp_path)

    def hide_failure(value):
        row = next(
            row
            for row in value["runs"]
            if row["label"] == "perf-native-default-backend-01"
        )
        row.update(controller_passed=True, exit_code=0)

    rehash_copy(root, "terminal-selection-v4.json", hide_failure)
    with pytest.raises(ValueError, match="failure relabeled"):
        VERIFY.verify(root)
