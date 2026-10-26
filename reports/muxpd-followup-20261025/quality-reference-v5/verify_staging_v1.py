"""Check only draft copies, historical bytes and explicit non-final status."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def file_ok(path, row):
    value = path.read_bytes()
    require(
        len(value) == row["bytes"]
        and hashlib.sha256(value).hexdigest() == row["sha256"],
        f"Copied or preserved bytes changed: {path}",
    )


def verify(root):
    root = Path(root)
    repo = root.parents[2]
    copied = read(root / "canonical-copy-map-draft-v1.json")["copies"]
    require(
        len({r["canonical_path"] for r in copied}) == len(copied),
        "Duplicate copied path",
    )
    for row in copied:
        require(
            row["sha256"] == row["source"]["sha256"]
            and row["bytes"] == row["source"]["bytes"],
            "Original-copy identity mismatch",
        )
        file_ok(root / row["canonical_path"], row)
    prior = read(root / "preserved-prior-files-v5.json")
    require(
        prior["base"] == "ef849ef49577c491fee68317ffc96d52796cd379",
        "Wrong prior cutoff",
    )
    for row in prior["files"]:
        file_ok(repo / row["relative_path"], row)
    state = read(root / "staging-state-v1.json")
    require(
        state["status"] == "local_draft_not_final_selection_not_uploaded",
        "Draft relabeled",
    )
    for key in (
        "archive_selection_finalized",
        "support_selection_finalized",
        "upload_selection_created",
        "public_readback_created",
    ):
        require(state[key] is False, f"Premature finalization: {key}")
    require(
        {r["label"] for r in state["known_terminal_capsules"]}
        == {"native-quality-v2-hccl-true-02", "raw-quality-v2-hccl-true-01"},
        "Unexpected terminal draft input",
    )
    for capsule in state["known_terminal_capsules"]:
        require(
            not capsule["selected_for_final_v5"]
            and not capsule["public_download_verified_for_v5"],
            "Pending asset falsely finalized",
        )
        receipt = read(root / capsule["receipt"])
        manifest = read(root / capsule["manifest"])
        require(
            receipt["verified"]
            and receipt["members_verified"] == len(manifest["members"]),
            "Existing receipt/member count mismatch",
        )
        require(
            receipt["archive"] == capsule["archive_as_reported_by_existing_builder"],
            "Archive metadata mismatch",
        )
    native = read(root / "analysis/native-pair/paired-description.json")
    require(
        native["all_200_token_ids_equal"]
        and native["all_200_text_and_termination_equal"],
        "Native description changed",
    )
    require(
        native["both_native_correct_count"] == 165
        and native["candidate_evaluated"] is False,
        "Native-only scope changed",
    )
    strict = read(root / "analysis/raw/strict-result-v1.json")
    require(
        strict["full_frozen_plan_passed"] is False
        and strict["bounded_answer_quality_gate_passed"] is False,
        "Strict failure hidden",
    )
    require(
        set(strict["new_wrong_case_ids"])
        == {"mmlu-pro-03766", "mmlu-pro-10672", "mmlu-pro-01741"},
        "Stable-set losses changed",
    )
    supplement = read(root / "analysis/raw/supplementary-result-v1.json")[
        "supplementary_gate"
    ]
    require(
        supplement["supplementary_budgeted_mcq_v1_collection_passed"] is True
        and supplement["supplementary_budgeted_mcq_v1_answer_nonregression_passed"]
        is False
        and supplement["supplementary_budgeted_mcq_v1_mixed_scope_passed"] is False,
        "Supplementary collection/answer distinction changed",
    )
    require(len(state["pending"]) == 4, "Pending scope omitted")
    return {
        "schema": "muxpd-v5-staging-verification-v1",
        "passed": True,
        "copied_files": len(copied),
        "preserved_prior_files": len(prior["files"]),
        "known_local_terminal_capsules": len(state["known_terminal_capsules"]),
        "archive_payloads_reverified": False,
        "quality_rescored": False,
        "final_selection_or_publication_verified": False,
        "scope": "Small copied metadata and preserved old bytes only; not model/quality/publication acceptance.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
