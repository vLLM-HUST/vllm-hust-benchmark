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


def test_load_target_misalignment_exclusion(tmp_path) -> None:
    path = tmp_path / "exclusions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "vllm-hust-leaderboard-exclusions/v1",
                "exclusions": [
                    {
                        "id": "misaligned-target",
                        "status": "excluded",
                        "match": {"match_type": "target_misalignment"},
                        "reason": "target ground truth misaligned",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exclusions = load_leaderboard_exclusions(path)
    assert len(exclusions) == 1
    exclusion = exclusions[0]
    assert exclusion.exclusion_id == "misaligned-target"
    assert exclusion.match_type == "target_misalignment"
    assert exclusion.plugin_commit == ""
    assert exclusion.reason == "target ground truth misaligned"


def test_match_target_misalignment_hit() -> None:
    from vllm_hust_benchmark.leaderboard_exclusions import LeaderboardExclusion

    exclusions = (
        LeaderboardExclusion(
            exclusion_id="misaligned-target",
            plugin_commit="",
            reason="target ground truth misaligned",
            match_type="target_misalignment",
        ),
    )
    artifact = {"entry_id": "entry-001"}

    matched = match_leaderboard_exclusion(
        artifact, exclusions, misaligned_entry_ids={"entry-001"}
    )
    assert matched is exclusions[0]


def test_match_target_misalignment_miss_when_entry_not_in_set() -> None:
    from vllm_hust_benchmark.leaderboard_exclusions import LeaderboardExclusion

    exclusions = (
        LeaderboardExclusion(
            exclusion_id="misaligned-target",
            plugin_commit="",
            reason="target ground truth misaligned",
            match_type="target_misalignment",
        ),
    )
    artifact = {"entry_id": "entry-001"}

    matched = match_leaderboard_exclusion(
        artifact, exclusions, misaligned_entry_ids={"entry-999"}
    )
    assert matched is None


def test_match_target_misalignment_miss_when_no_set_provided() -> None:
    from vllm_hust_benchmark.leaderboard_exclusions import LeaderboardExclusion

    exclusions = (
        LeaderboardExclusion(
            exclusion_id="misaligned-target",
            plugin_commit="",
            reason="target ground truth misaligned",
            match_type="target_misalignment",
        ),
    )
    artifact = {"entry_id": "entry-001"}

    # 未传入 misaligned_entry_ids（默认 None）→ 不命中
    assert match_leaderboard_exclusion(artifact, exclusions) is None
    # 显式传入空集合 → 也不命中
    assert (
        match_leaderboard_exclusion(artifact, exclusions, misaligned_entry_ids=set())
        is None
    )
