import json
from pathlib import Path


def test_matrix_artifact_uses_verified_benchmark_release():
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads(
        (
            root
            / "leaderboard-data/dataset-validation/dataset_validation_qwen35_tp2_matrix.json"
        ).read_text()
    )
    verification = json.loads(
        (root / "archive/dev-hub-releases-20261020/final-verification.json").read_text()
    )
    release = next(
        row
        for row in verification["releases"]
        if row["tag"] == "szyn-swebench-qwen35-20261005-attempt3-evidence"
    )
    asset = next(
        row
        for row in release["assets"]
        if row["name"] == "szyn-attempt3-500-adjudication-addendum.tar.gz"
    )

    def provenance_records(value):
        if isinstance(value, dict):
            if "artifact_sha256" in value:
                yield value
            for child in value.values():
                yield from provenance_records(child)
        elif isinstance(value, list):
            for child in value:
                yield from provenance_records(child)

    matched = [
        row
        for row in provenance_records(matrix)
        if row["artifact_sha256"] == asset["digest"].removeprefix("sha256:")
    ]
    assert len(matched) == 1
    assert matched[0]["artifact"] == asset["browser_download_url_snapshot"]
    assert matched[0]["release_url"] == release["destination_html_url_snapshot"]
    assert release["destination_draft"] is False
