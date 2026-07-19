from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/push-to-hf.yml"


def test_snapshot_only_push_uploads_canonical_snapshots_to_hf() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'snapshot_root = Path("leaderboard-data/snapshots")' in workflow_text
    assert "fh.write(f\"snapshots_changed={'true' if snapshots_changed else 'false'}\\n\")" in workflow_text
    assert "steps.resolve-submissions.outputs.snapshots_changed == 'true'" in workflow_text
    assert (
        "steps.resolve-submissions.outputs.count == '0' && "
        "steps.resolve-submissions.outputs.snapshots_changed == 'true'"
    ) in workflow_text
    assert "Verify HF snapshots match GitHub canonical snapshots" in workflow_text
    assert "HF snapshots match GitHub canonical snapshots byte-for-byte." in workflow_text


def test_submission_push_keeps_the_aggregation_path() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "steps.resolve-submissions.outputs.count != '0'" in workflow_text
    assert 'if [[ "$SKIP_AGGREGATION" != "true" ]]; then' in workflow_text
    assert 'submission_args+=(--submission-dir "$submission_dir")' in workflow_text
