"""Offline custody and declared scope of v4; no network, devices or archive execution."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def file_ok(path, row):
    data = path.read_bytes()
    require(
        len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
        f"Frozen bytes changed: {path}",
    )


def verify(root):
    root = Path(root)
    copied = read(root / "canonical-copy-map.json")
    require(
        len({r["canonical_path"] for r in copied["copies"]}) == len(copied["copies"]),
        "Duplicate copied path",
    )
    for row in copied["copies"]:
        file_ok(root / row["canonical_path"], row)
    old = read(root / "preserved-originals-v4.json")
    repo = root.parents[2]
    for row in old["files"]:
        file_ok(repo / row["path"], row)
    require(len(old["files"]) == 248, "Historical snapshot scope changed")
    index = read(root / "archive-index-v4.json")
    require(
        index["status"] == "local_verified_not_uploaded",
        "Historical preparation status changed",
    )
    require(
        index["counts"]["terminal_runs"] == 12 and index["counts"]["capsules"] == 13,
        "Fixed v4 cutoff changed",
    )
    require(
        index["paired_quality_accepted"] is False
        and index["default_native_v2_measured"] is False
        and index["actual_mixed_profile_accepted"] is False,
        "Unsupported quality/reference/profile acceptance",
    )
    terms = read(root / "terminal-selection-v4.json")["runs"]
    expected = {
        "raw-cancel-partial-block22-hccl-true-01",
        "raw-cancel-full-block22-hccl-true-01",
        "group-cancel-partial-block22-hccl-true-01",
        "group-cancel-full-block22-hccl-true-01",
        "group-pressure-block20-hccl-true-01",
        "perf-f7-native-sync-01",
        "perf-f7-raw-01",
        "perf-f7-group-01",
        "perf-f7-group-02",
        "perf-f7-raw-02",
        "perf-f7-native-sync-02",
        "perf-native-default-backend-01",
    }
    require(
        {r["label"] for r in terms} == expected and len(terms) == 12,
        "Missing/extra terminal role",
    )
    failed = [r for r in terms if not r["controller_passed"]]
    require(
        len(failed) == 1
        and failed[0]["label"] == "perf-native-default-backend-01"
        and failed[0]["exit_code"] == 1,
        "Pre-window failure relabeled",
    )
    assets = read(root / "release-assets-v4.json")
    require(
        assets["publication_state"] == "prepared_not_uploaded"
        and len(assets["assets"]) == 13,
        "Prepared assets scope changed",
    )
    total_members = 0
    for asset in assets["assets"]:
        require(
            asset["publication_state"] == "prepared_not_uploaded"
            and asset["remote_verified"] is False,
            "Unproven remote availability",
        )
        manifest = read(root / asset["manifest"])
        receipt = read(root / asset["receipt"])
        require(
            receipt["verified"]
            and receipt["members_verified"] == len(manifest["members"]),
            "Missing complete local member proof",
        )
        require(
            receipt["archive"]["sha256"] == asset["sha256"]
            and receipt["archive"]["bytes"] == asset["bytes"],
            "Archive identity mismatch",
        )
        require(
            receipt["capsule_id"] == manifest["capsule_id"] == asset["capsule_id"],
            "Capsule ID mismatch",
        )
        total_members += receipt["members_verified"]
    require(
        total_members == index["counts"]["original_members"],
        "Member accounting changed",
    )
    primary = read(root / "analysis/primary/group-abba-with-native.json")
    raw_group = read(root / "analysis/primary/group-abba.json")
    require(
        primary["complete"] and raw_group["complete"],
        "Incomplete original primary analysis",
    )
    require(
        len(primary["input_files"]) == 6 and len(raw_group["input_files"]) == 4,
        "Primary role count changed",
    )
    require(
        primary["comparisons"]["decode-heavy"][
            "all_generated_tokens_identical_across_arms"
        ]
        is True,
        "Decode token finding changed",
    )
    require(
        primary["comparisons"]["long-P-ingress"][
            "all_generated_tokens_identical_across_arms"
        ]
        is False,
        "Mixed native/plugin token divergence hidden",
    )
    require(
        all(
            v["all_generated_tokens_identical_across_arms"] is True
            for v in raw_group["comparisons"].values()
        ),
        "Raw/group token finding changed",
    )
    windows = read(root / "analysis/primary/all-window-summaries.json")["windows"]
    require(
        len(windows) == 36 and sum(w["warmup"] is True for w in windows) == 12,
        "Window selection changed",
    )
    require(
        {w["role"] for w in windows} == {"N1", "R1", "G1", "G2", "R2", "N2"},
        "Missing primary role",
    )
    require(
        all(w["complete"] and w["error"] is None for w in windows), "Incomplete window"
    )
    location = read(root / "analysis/primary/stalls-release-location.json")
    require(
        location["in_git"] is False and location["original"]["bytes"] > 1000000,
        "Large original must stay in Release capsule",
    )
    require(
        location["member"]
        == "support/performance/group-matrix-analysis-v1/stalls-all-primary.json",
        "Wrong full gap original",
    )
    return {
        "passed": True,
        "terminal_runs": 12,
        "capsules": 13,
        "original_members": total_members,
        "original_files_preserved": 248,
        "primary_roles": 6,
        "all_windows": 36,
        "failed_default_reference_preserved": True,
        "paired_quality_accepted": False,
        "default_native_v2_measured": False,
        "actual_mixed_profile_accepted": False,
        "remote_download_verified": False,
        "scope": "Saved local custody and bounded findings only; no new archive or network validation.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
