"""Ignore policy only; do not create or inspect any private dataset payload."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_native_private_artifact_names_are_not_accidentally_staged():
    paths = [
        "temporary-native/row-0.request-private.json",
        "temporary-native/row-0.response-private.json",
        "temporary-native/raw-solutions-private.jsonl",
        "temporary-native/original-dataset-private.jsonl",
        "temporary-native/0.postprocessed.private.json",
        "temporary-native/HumanEval-0.private.json",
        "temporary-native/receipt-private.json",
        "temporary-native/failed.private-error.txt",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input=("\n".join(paths) + "\n").encode("utf-8"),
        capture_output=True,
    )
    assert result.returncode == 0
    assert set(result.stdout.decode("utf-8").splitlines()) == set(paths)


def test_public_custody_metadata_remains_visible():
    path = (
        "docs/workload-data/validation/enterprise-private-source-consumer-20261024.json"
    )
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
