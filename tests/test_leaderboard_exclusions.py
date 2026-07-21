import json

import pytest

from vllm_hust_benchmark.leaderboard_exclusions import (
    load_leaderboard_exclusions,
    match_leaderboard_exclusion,
)


def test_load_and_match_plugin_commit_exclusion(tmp_path) -> None:
    path = tmp_path / "exclusions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "vllm-hust-leaderboard-exclusions/v1",
                "exclusions": [
                    {
                        "id": "bad-plugin",
                        "status": "excluded",
                        "match": {"runtime_provenance.plugin.commit": "a" * 40},
                        "reason": "known incorrect output path",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exclusions = load_leaderboard_exclusions(path)
    artifact = {"metadata": {"runtime_provenance": {"plugin": {"commit": "A" * 40}}}}

    assert match_leaderboard_exclusion(artifact, exclusions) == exclusions[0]


def test_invalid_exclusion_entry_fails_closed(tmp_path) -> None:
    path = tmp_path / "exclusions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "vllm-hust-leaderboard-exclusions/v1",
                "exclusions": [
                    {
                        "id": "bad-plugin",
                        "status": "excluded",
                        "match": {"runtime_provenance.plugin.commit": "short"},
                        "reason": "known incorrect output path",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid leaderboard exclusion entry"):
        load_leaderboard_exclusions(path)
