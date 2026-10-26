"""Offline byte custody and bounded-claim checks; no downloads or model imports."""

import hashlib
import json
from pathlib import Path

LABELS = {
    "raw-fusion-composite-plain-hccl-true-01",
    "group-fusion-composite-plain-hccl-true-01",
    "source98-plain-raw-bridge-01",
    "source98-plain-group-bridge-01",
    "native-quality-hccl-true-01",
    "native-quality-v2-hccl-true-01",
    "raw-pressure-suite-hccl-true-01",
    "source98-manager-raw-bridge-01",
    "source98-manager-group-bridge-01",
}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, sha, size=None):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha, f"Byte identity changed: {path}")
    require(size is None or len(data) == size, f"Size changed: {path}")


def verify(directory):
    directory = Path(directory)
    copies = read(directory / "canonical-copy-map.json")["copies"]
    for row in copies:
        checked(directory / row["canonical_path"], row["sha256"], row["bytes"])
        require(row["source"]["sha256"] == row["sha256"], "Copy/source hash differs")
    index = read(directory / "archive-index-v3.json")
    require(
        index["status"] == "local_verified_not_uploaded",
        "Historical preparation state changed",
    )
    require(
        not index["current_performance_claim"] and not index["paired_quality_accepted"],
        "Unsupported performance/quality acceptance",
    )
    rows = index["capsules"]
    require(len(rows) == 10, "V3 fixed capsule selection changed")
    assets = read(directory / "release-assets-v3.json")
    by_id = {row["capsule_id"]: row for row in assets["assets"]}
    require(len(by_id) == 10, "Duplicate/missing asset")
    labels, failures, managers, members, raw_bytes, archive_bytes = set(), 0, 0, 0, 0, 0
    for row in rows:
        asset = by_id[row["capsule_id"]]
        checked(directory / asset["manifest"], row["manifest"]["sha256"])
        checked(directory / asset["receipt"], row["receipt"]["sha256"])
        manifest, receipt = (
            read(directory / asset["manifest"]),
            read(directory / asset["receipt"]),
        )
        require(
            receipt["verified"] and receipt["archive"] == row["archive"],
            "Archive receipt mismatch",
        )
        require(
            receipt["members_verified"] == len(manifest["members"]),
            "Member count mismatch",
        )
        require(
            len({item["member"] for item in manifest["members"]})
            == len(manifest["members"]),
            "Duplicate original member",
        )
        require(
            asset["sha256"] == row["archive"]["sha256"]
            and asset["bytes"] == row["archive"]["bytes"],
            "Asset identity mismatch",
        )
        require(
            not asset["remote_verified"]
            and asset["publication_state"] == "prepared_not_uploaded",
            "Historical remote state changed",
        )
        require(
            asset["planned_download_url"] == assets["release_base"] + asset["name"],
            "Asset URL mismatch",
        )
        members += len(manifest["members"])
        raw_bytes += sum(item["bytes"] for item in manifest["members"])
        archive_bytes += row["archive"]["bytes"]
        terminal = row["terminal"]
        if terminal:
            require(
                terminal == manifest["terminal"] and terminal["finished_at"],
                "Terminal identity mismatch",
            )
            labels.add(terminal["label"])
            failures += not terminal["controller_passed"]
            if "manager" in terminal["label"]:
                managers += 1
                require(
                    manifest["kind"] == "terminal-source98-manager-sigint-bridge",
                    "Manager terminal type lost",
                )
                require(
                    terminal["exit_code"] == 130 and terminal["controller_passed"],
                    "Actual Manager exit130 was changed",
                )
        else:
            require(
                manifest["kind"] == "explicit-support", "Unknown nonterminal capsule"
            )
    require(
        labels == LABELS and failures == 2 and managers == 2,
        "Fixed terminal/failure selection changed",
    )
    require(
        (members, raw_bytes, archive_bytes) == (786, 459357252, 65501327),
        "Frozen capsule accounting changed",
    )
    for path in sorted((directory / "analysis").glob("*plain-composite-v1.json")):
        comparison = read(path)
        for key in ("left_warm", "right_warm", "cross"):
            item = comparison[key]
            details = item["details"]
            require(
                len(details) == item["pairs"] == 30, "Comparison denominator changed"
            )
            require(
                sum(
                    d["left_token_ids"] == d["right_token_ids"]
                    and len(d["left_token_ids"]) == 8
                    for d in details
                )
                == item["exact_eight_token_matches"]
                == 30,
                "Token comparison changed",
            )
            require(
                sum(d["normalized_admission_match"] for d in details)
                == item["normalized_admission_matches"]
                == 30,
                "Admission comparison changed",
            )
    quality = read(directory / "analysis/native-control-01-description-v1.json")
    score = quality["score"]
    require(
        len(score["rows"]) == score["total"] == 200
        and sum(row["correct"] for row in score["rows"]) == score["correct"] == 165,
        "Single native quality count changed",
    )
    length = [row for row in score["rows"] if row["finish_reason"] == "length"]
    require(
        len(length) == 9
        and not any(row["correct"] for row in length)
        and not score["all_complete"],
        "Truncation failure was concealed",
    )
    require(
        quality["collection"]["passed"]
        and quality["admission_binding"]
        and quality["token_text_binding"]
        and quality["terminal_receipt"]["passed"],
        "Descriptive native provenance failed",
    )
    summary = read(directory / "analysis/device-input-reverification-v3.json")
    require(
        not summary["single_native_quality"]["paired_acceptance_computed"],
        "Single control was promoted to paired acceptance",
    )
    pressure = summary["pressure_suite"]
    require(
        not pressure["overall_passed"]
        and pressure["pressure_phase_passed"]
        and not pressure["partial_phase_passed"]
        and not pressure["abort_issued"]
        and not pressure["full_phase_admitted"],
        "Failed pressure coverage was concealed",
    )
    prior = read(directory / "preserved-originals-v3.json")
    repo = directory.parents[2]
    preserved = 0
    for row in prior["files"]:
        # Main README is navigation and may later link the separate public readback.
        if row["path"] == "reports/muxpd-followup-20261025/README.md":
            continue
        checked(repo / row["path"], row["sha256"], row["bytes"])
        preserved += 1
    return {
        "passed": True,
        "byte_preserved_copies": len(copies),
        "prior_evidence_files_unchanged": preserved,
        "terminal_runs": 9,
        "capsules": 10,
        "original_members": members,
        "failed_attempts_preserved": failures,
        "intentional_manager_exit130": managers,
        "remote_verified_by_this_check": False,
        "paired_quality_or_performance_accepted": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
