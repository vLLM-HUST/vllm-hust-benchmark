"""Verify saved v5 custody and bounded findings, without network or model execution."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_ok(path, identity):
    data = path.read_bytes()
    require(
        len(data) == identity["bytes"]
        and hashlib.sha256(data).hexdigest() == identity["sha256"],
        f"Frozen bytes changed: {path}",
    )


def validate_claims(claims):
    false = [
        "paired_quality_accepted",
        "strict_quality_gate_passed",
        "supplementary_quality_gate_passed",
        "reference_primary_abba_pooled",
        "native_original_offline_analyzer_passed",
        "native_kernel_launch_reduction",
        "http_goodput_measured",
        "physical_overlap_proven",
    ]
    require(
        all(claims[k] is False for k in false), "Unsupported acceptance or attribution"
    )
    require(
        claims["native_correct_counts"] == [165, 165]
        and claims["raw_group_correct_counts"] == [166, 166]
        and claims["native_stable_new_wrong"] == 3
        and claims["group_native_full_token_equal"] == 11,
        "Quality findings changed",
    )


def validate_profile(profile):
    require(profile["matched_capture"] is True, "Unmatched profile")
    rows = profile["comparisons"]
    require(
        len(rows) == 4 and {x["rank"] for x in rows} == set(range(4)),
        "Missing or duplicated profile rank",
    )
    for row in rows:
        api = row["apis"]["AscendCL@aclrtLaunchKernelWithHostArgs"]
        require(
            row["actual_D_graph_replays_each"] == 10
            and api["raw"]["count"] == 2854
            and api["group"]["count"] == 2924
            and api["group_minus_raw_exported_count"] == 70,
            "Profile launch finding changed",
        )


def verify(root):
    root = Path(root)
    count = 0
    seen = set()
    for name in [
        "canonical-copy-map-draft-v1.json",
        "canonical-copy-map-v5.json",
        "canonical-postselection-copy-map-v1.json",
    ]:
        for row in read(root / name)["copies"]:
            path = row["canonical_path"]
            require(path not in seen, "Duplicate copied path")
            seen.add(path)
            file_ok(root / path, row)
            count += 1
    old = read(root / "preserved-prior-files-v5.json")
    require(len(old["files"]) == 400, "Prior canonical scope changed")
    for row in old["files"]:
        file_ok(root.parents[2] / row["relative_path"], row)
    index = read(root / "archive-index-v5.json")
    require(
        index["status"] == "local_verified_not_uploaded",
        "Historical local status changed",
    )
    require(
        index["counts"]["terminal_runs"] == 7 and index["counts"]["capsules"] == 8,
        "Fixed publication cutoff changed",
    )
    validate_claims(index["claims"])
    expected = {
        "native-quality-v2-hccl-true-02",
        "raw-quality-v2-hccl-true-01",
        "group-quality-v2-hccl-true-01",
        "perf-native-default-backend-v2-01",
        "perf-native-default-backend-v2-02",
        "profile-f7-raw-mixed-01",
        "profile-f7-group-mixed-01",
    }
    terms = read(root / "terminal-selection-v5.json")["runs"]
    require(
        len(terms) == 7 and {r["label"] for r in terms} == expected,
        "Missing or extra terminal run",
    )
    require(
        all(r["controller_passed"] is True and r["exit_code"] == 0 for r in terms),
        "Terminal collection not passed",
    )
    assets = read(root / "release-assets-v5.json")
    require(
        assets["publication_state"] == "prepared_not_uploaded"
        and len(assets["assets"]) == 8,
        "Prepared asset scope changed",
    )
    total = 0
    for asset in assets["assets"]:
        require(asset["remote_verified"] is False, "Unproven historical public claim")
        manifest = read(root / asset["manifest"])
        receipt = read(root / asset["receipt"])
        require(
            receipt["verified"]
            and receipt["members_verified"] == len(manifest["members"]),
            "Incomplete local member receipt",
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
        total += receipt["members_verified"]
    require(total == index["counts"]["original_members"], "Member total mismatch")
    quality = read(root / "analysis/group/review.json")
    require(
        quality["original_strict_gate"]["full_frozen_plan_passed"] is False
        and quality["supplementary_gate"][
            "supplementary_budgeted_mcq_v1_answer_nonregression_passed"
        ]
        is False
        and quality["supplementary_gate"][
            "supplementary_budgeted_mcq_v1_collection_passed"
        ]
        is True,
        "Quality failure or collection distinction changed",
    )
    require(
        quality["raw_group_comparison_counts"]
        == {
            "cases": 200,
            "raw_group_token_ids_equal": 200,
            "raw_group_text_equal": 200,
            "raw_group_termination_equal": 200,
        },
        "Raw/group comparison incomplete",
    )
    require(
        len(quality["new_wrong_on_both_native_correct"]) == 3, "Stable losses hidden"
    )
    nd = read(root / "analysis/native-reference/native-default-analysis.json")
    require(nd["complete"] and len(nd["inputs"]) == 2, "Native reference incomplete")
    require(
        nd["offline_default_expansion_adapter"]["exact_three_keys_only"]
        == {"fuse_muls_add": True, "fuse_norm_quant": True, "fuse_qknorm_rope": True},
        "Offline adaptation scope changed",
    )
    failed = read(
        root
        / "analysis/native-reference-original-failure/frozen-native-analysis.execution.json"
    )
    require(failed["exit_code"] != 0, "Original offline failure relabeled")
    windows = read(root / "analysis/native-reference/all-window-summaries.json")[
        "windows"
    ]
    require(
        len(windows) == 48 and sum(x["warmup"] for x in windows) == 16,
        "Window selection changed",
    )
    validate_profile(read(root / "analysis/mixed-profile/comparison-v1.json"))
    uploads = read(root / "upload-selection-v5.json")["files"]
    prior = read(root / "analysis/prior57-release-api-v5.json")["assets"]
    require(
        len(uploads) == 13
        and len(prior) == 57
        and not {Path(x["path"]).name for x in uploads}.intersection(
            x["name"] for x in prior
        ),
        "New uploads collide with prior57",
    )
    return {
        "passed": True,
        "copied_files": count,
        "preserved_prior_files": 400,
        "terminal_runs": 7,
        "capsules": 8,
        "original_members": total,
        "quality_accepted": False,
        "native_reference_measured": True,
        "mixed_profile_matched": True,
        "native_kernel_launch_reduction": False,
        "remote_download_verified": False,
        "scope": "Saved custody and bounded findings only; no rescoring, archive payload recheck, new model, or live network query.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
