"""Offline canonical-copy and bounded-claim verification; no archive downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def checked(path, sha, size=None):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha or (
        size is not None and len(data) != size
    ):
        raise ValueError(f"Canonical bytes differ: {path}")


def verify(directory):
    directory = Path(directory)
    copies = read(directory / "canonical-copy-map.json")["copies"]
    for row in copies:
        checked(directory / row["canonical_path"], row["sha256"], row["bytes"])
        if row["sha256"] != row["source"]["sha256"]:
            raise ValueError("Canonical copy differs from frozen source identity")
    index = read(directory / "archive-index-v2.json")
    old = read(directory / "archive-index-v1.json")
    if index["capsules"][: len(old["capsules"])] != old["capsules"]:
        raise ValueError("V2 changed an existing v1 capsule")
    if index["counts"]["terminal_runs"] != 11 or len(index["capsules"]) != 13:
        raise ValueError("Frozen v2 scope changed")
    assets = read(directory / "release-assets-v2.json")
    by_id = {row["capsule_id"]: row for row in assets["assets"]}
    if len(by_id) != 13 or assets["publication_state"] != "prepared_not_uploaded":
        raise ValueError("Missing assets or invented remote publication status")
    failed = 0
    members = total = compressed = npy = 0
    for row in index["capsules"]:
        asset = by_id[row["capsule_id"]]
        checked(directory / asset["manifest"], row["manifest"]["sha256"])
        checked(directory / asset["receipt"], row["receipt"]["sha256"])
        manifest = read(directory / asset["manifest"])
        receipt = read(directory / asset["receipt"])
        if not receipt["verified"] or receipt["archive"] != row["archive"]:
            raise ValueError("Local archive receipt mismatch")
        if (asset["sha256"], asset["bytes"]) != (
            row["archive"]["sha256"],
            row["archive"]["bytes"],
        ):
            raise ValueError("Asset identity mismatch")
        if (
            asset["remote_verified"]
            or asset["publication_state"] != "prepared_not_uploaded"
        ):
            raise ValueError("V2 contains an unsupported remote verification claim")
        if asset["planned_download_url"] != assets["release_base"] + asset["name"]:
            raise ValueError("Asset URL/name mismatch")
        paths = [item["member"] for item in manifest["members"]]
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate original member")
        members += len(paths)
        total += sum(item["bytes"] for item in manifest["members"])
        compressed += row["archive"]["bytes"]
        npy += sum(path.endswith(".npy") for path in paths)
        if manifest["kind"] == "terminal-run":
            terminal = manifest["terminal"]
            if not terminal["finished_at"] or type(terminal["exit_code"]) is not int:
                raise ValueError("Nonterminal run in canonical scope")
            if "quality" in terminal["label"] or "perf-" in terminal["label"]:
                raise ValueError("Later quality/performance evidence entered v2")
            failed += not terminal["controller_passed"]
    if (members, total, compressed, failed) != (1882, 1792926073, 725675722, 2):
        raise ValueError("Frozen original/failed-attempt accounting changed")
    summary = read(directory / "analysis/raw-vs-group-composite-summary-v2.json")
    comparison = summary["analyses"][0]
    if comparison["same_driver_source"] is not True:
        raise ValueError("Raw/group observer-source mismatch")
    for key in ("left_warm_pairs", "right_warm_pairs", "cross_pairs"):
        item = comparison[key]
        if any(
            item[name] != count
            for name, count in {
                "request_pairs": 30,
                "exact_token_sequence_pairs": 30,
                "compared_logit_vectors": 240,
                "exact_logit_value_equal_vectors": 240,
                "byte_identical_npy_files_on_compared_positions": 240,
            }.items()
        ):
            raise ValueError("Bounded raw/group numerical observation changed")
    v1 = read(directory / "analysis/composite-fusion-host-audit-v1.json")
    v2 = read(directory / "analysis/composite-fusion-host-audit-v2.json")
    for old_rank, rank in zip(
        v1["runs"][-1]["group_ranks"], v2["runs"][-1]["group_ranks"], strict=True
    ):
        if len(old_rank["failed_event_checks"]) != 128 or rank["failed_event_checks"]:
            raise ValueError("Validator false-positive history was lost")
        indices = [
            item["observed_group_indices"]
            for item in rank["coverage_by_actual_execute"]
        ]
        if rank["events"] != 128 or rank["actual_prefill_execute_count"] != 15:
            raise ValueError("Callback cap or actual cohort count changed")
        if indices != [list(range(10))] * 12 + [list(range(8)), [], []]:
            raise ValueError("Missing metadata coverage was concealed")
    if len(v2["runs"][-1]["group_ranks"]) != 4:
        raise ValueError("Missing group rank metadata")
    return {
        "passed": True,
        "byte_preserved_copies": len(copies),
        "terminal_runs": 11,
        "capsules": 13,
        "original_members": members,
        "npy_files": npy,
        "failed_attempts_preserved": failed,
        "complete_group_metadata_coverage": False,
        "remote_publication_verified": False,
        "quality_or_performance_qualified": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
