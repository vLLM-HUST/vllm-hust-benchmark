import json
from pathlib import Path

from vllm_hust_benchmark.official_baselines import get_canonical_submission_dir
from vllm_hust_benchmark.official_baselines import get_primary_metric_name_for_benchmark_type
from vllm_hust_benchmark.official_baselines import has_canonical_run
from vllm_hust_benchmark.official_baselines import select_canonical_candidate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec() -> dict:
    return {
        "id": "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "scenario": "random-online",
    }


def test_get_canonical_submission_dir_uses_spec_id(tmp_path: Path) -> None:
    canonical_dir = get_canonical_submission_dir(_spec(), submissions_root=tmp_path)
    assert canonical_dir == tmp_path / _spec()["id"]


def test_perfgate_ascend_smoke_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-qwen25-3b-910b3.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))

    assert spec["id"] == "perfgate-ascend-qwen25-3b-910b3"
    assert spec["scenario"] == "random-online"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B3"
    assert spec["server_parameters"]["max_model_len"] == 256
    assert spec["client_parameters"]["input_len"] == 64
    assert spec["client_parameters"]["output_len"] == 16


def test_has_canonical_run_requires_matching_spec_id_and_submitter(tmp_path: Path) -> None:
    canonical_dir = tmp_path / _spec()["id"]
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "metadata": {"submitter": "official-ascend-baseline"},
                "same_spec": {"spec_id": _spec()["id"]},
            }
        ),
        encoding="utf-8",
    )
    (canonical_dir / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"leaderboard_artifact": "run_leaderboard.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert has_canonical_run(_spec(), submissions_root=tmp_path)


def test_has_canonical_run_rejects_mismatched_submitter(tmp_path: Path) -> None:
    canonical_dir = tmp_path / _spec()["id"]
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "metadata": {"submitter": "someone-else"},
                "same_spec": {"spec_id": _spec()["id"]},
            }
        ),
        encoding="utf-8",
    )
    (canonical_dir / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"leaderboard_artifact": "run_leaderboard.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert not has_canonical_run(_spec(), submissions_root=tmp_path)


def _write_result_artifact(
    result_dir: Path,
    *,
    ttft_ms: float | None,
    throughput_tps: float | None,
    error_rate: float = 0.0,
) -> None:
    submission_dir = result_dir / "submission"
    submission_dir.mkdir(parents=True)
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "ttft_ms": ttft_ms,
                    "throughput_tps": throughput_tps,
                    "error_rate": error_rate,
                }
            }
        ),
        encoding="utf-8",
    )


def test_get_primary_metric_name_for_benchmark_type() -> None:
    assert get_primary_metric_name_for_benchmark_type("serve") == "ttft_ms"
    assert get_primary_metric_name_for_benchmark_type("latency") == "ttft_ms"
    assert get_primary_metric_name_for_benchmark_type("throughput") == "throughput_tps"


def test_select_canonical_candidate_prefers_median_ttft(tmp_path: Path) -> None:
    repeat_a = tmp_path / "repeat-a"
    repeat_b = tmp_path / "repeat-b"
    repeat_c = tmp_path / "repeat-c"
    _write_result_artifact(repeat_a, ttft_ms=120.0, throughput_tps=200.0)
    _write_result_artifact(repeat_b, ttft_ms=100.0, throughput_tps=220.0)
    _write_result_artifact(repeat_c, ttft_ms=140.0, throughput_tps=180.0)

    payload = select_canonical_candidate(
        [repeat_a, repeat_b, repeat_c], benchmark_type="serve"
    )

    assert payload["primary_metric_name"] == "ttft_ms"
    assert payload["median_value"] == 120.0
    assert Path(payload["selected_result_dir"]) == repeat_a.resolve()


def test_select_canonical_candidate_uses_throughput_metric(tmp_path: Path) -> None:
    repeat_a = tmp_path / "repeat-a"
    repeat_b = tmp_path / "repeat-b"
    repeat_c = tmp_path / "repeat-c"
    _write_result_artifact(repeat_a, ttft_ms=0.0, throughput_tps=190.0)
    _write_result_artifact(repeat_b, ttft_ms=0.0, throughput_tps=210.0)
    _write_result_artifact(repeat_c, ttft_ms=0.0, throughput_tps=230.0)

    payload = select_canonical_candidate(
        [repeat_a, repeat_b, repeat_c], benchmark_type="throughput"
    )

    assert payload["primary_metric_name"] == "throughput_tps"
    assert payload["median_value"] == 210.0
    assert Path(payload["selected_result_dir"]) == repeat_b.resolve()


def test_select_canonical_candidate_prefers_lower_error_rate(tmp_path: Path) -> None:
    repeat_a = tmp_path / "repeat-a"
    repeat_b = tmp_path / "repeat-b"
    _write_result_artifact(repeat_a, ttft_ms=100.0, throughput_tps=220.0, error_rate=0.1)
    _write_result_artifact(repeat_b, ttft_ms=110.0, throughput_tps=215.0, error_rate=0.0)

    payload = select_canonical_candidate(
        [repeat_a, repeat_b], benchmark_type="serve"
    )

    assert Path(payload["selected_result_dir"]) == repeat_b.resolve()


def test_public_official_baseline_specs_are_v0180_910b2_fp16() -> None:
    spec_dir = REPO_ROOT / "docs" / "official-baselines"
    spec_paths = [
        path
        for path in spec_dir.glob("*.json")
        if path.name != "official-ascend-constraints.stub.json"
        and not path.name.startswith("perfgate-")
    ]

    assert spec_paths
    for path in spec_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_id = str(payload.get("id") or "")
        assert "v0180" in path.name or "v0.18.0" in spec_id
        assert payload.get("hardware_chip_model") == "910B2"
        assert payload.get("model_precision") == "FP16"
