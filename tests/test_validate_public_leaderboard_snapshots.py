from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from vllm_hust_benchmark.same_spec import build_same_spec_payload

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_public_leaderboard_snapshots.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_public_snapshots", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_spec() -> dict:
    return build_same_spec_payload(
        {
            "id": "test-spec",
            "label": "test",
            "scenario": "random-online",
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "model_parameters": "14B",
            "model_precision": "FP16",
            "hardware_vendor": "Huawei",
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
            "server_parameters": {"tensor_parallel_size": 1},
            "client_parameters": {
                "backend": "vllm",
                "dataset_name": "random",
                "num_prompts": 1,
                "input_len": 16,
                "output_len": 8,
            },
        }
    )


def entry(entry_id: str, payload: dict) -> dict:
    return {
        "entry_id": entry_id,
        "engine": "test-engine",
        "engine_version": "1.0",
        "workload": {"name": "test"},
        "model": {"name": "test", "precision": "FP16"},
        "hardware": {"chip_model": "910B2"},
        "same_spec": payload,
    }


def test_rejects_one_recorded_hash_for_different_effective_parameters(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    first = same_spec()
    second = json.loads(json.dumps(first))
    second["resolved_server_parameters"]["enable_prefix_caching"] = "true"
    (tmp_path / "leaderboard_single.json").write_text(
        json.dumps([entry("first", first), entry("second", second)]),
        encoding="utf-8",
    )
    (tmp_path / "leaderboard_multi.json").write_text("[]", encoding="utf-8")
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_public_leaderboard_snapshots.py", "--snapshot-dir", str(tmp_path)],
    )

    assert module.main() == 1
    assert "maps to different effective parameters" in capsys.readouterr().out


def test_future_official_entry_requires_effective_config_contract() -> None:
    module = load_module()
    payload = same_spec()
    payload["spec_id"] = (
        "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
    )
    payload["scenario"] = "random-online"
    candidate = entry("future", payload)
    candidate["engine"] = "vllm-hust"
    candidate["workload"] = {
        "name": "random-online",
        "input_length": 1024,
        "output_length": 256,
        "batch_size": None,
        "concurrent_requests": None,
        "dataset": "random",
    }
    candidate["model"]["name"] = "Qwen/Qwen2.5-14B-Instruct"
    candidate["metadata"] = {"submitted_at": "2026-07-25T00:00:00Z"}

    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )

    assert any("workload config contract" in error for error in errors)
    assert any("workload_config_contract" in error for error in errors)


def _historical_unverified_official_entry() -> dict:
    payload = same_spec()
    payload["spec_id"] = (
        "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
    )
    payload["scenario"] = "random-online"
    payload["resolved_server_parameters"].update(
        gpu_memory_utilization=0.6,
        max_model_len=32768,
    )
    payload["resolved_client_parameters"]["no_stream"] = False
    candidate = entry("historical-unverified", payload)
    candidate["engine"] = "vllm-hust"
    candidate["workload"] = {
        "name": "random-online",
        "input_length": 16,
        "output_length": 8,
        "batch_size": None,
        "concurrent_requests": None,
        "dataset": "random",
    }
    candidate["model"]["name"] = "Qwen/Qwen2.5-14B-Instruct"
    candidate["metadata"] = {
        "submitted_at": "2026-08-17T00:00:00Z",
        "workload_config_contract": "explicit-effective/v1",
        "verified": False,
        "official_admission_status": "historical-unverified",
        "official_admission_reason": "engine does not match target baseline runtime",
    }
    return candidate


def test_historical_unverified_marker_preserves_config_validation_without_target_claim() -> (
    None
):
    module = load_module()
    errors = module.validate_entry(
        _historical_unverified_official_entry(),
        source=Path("leaderboard_single.json"),
    )
    assert errors == []


def test_historical_unverified_marker_cannot_claim_target_metadata() -> None:
    module = load_module()
    candidate = _historical_unverified_official_entry()
    candidate["metadata"]["target_id"] = candidate["same_spec"]["spec_id"]
    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )
    assert any("cannot claim target_id" in error for error in errors)


def test_quarantined_entry_ids_are_rejected() -> None:
    module = load_module()
    for entry_id in module.QUARANTINED_ENTRY_IDS:
        candidate = entry(entry_id, same_spec())
        errors = module.validate_entry(
            candidate,
            source=Path("leaderboard_single.json"),
        )
        assert any("quarantined entry" in error for error in errors), (
            f"entry_id {entry_id} should be rejected as quarantined"
        )
        assert any("issue #79" in error for error in errors)


def test_non_quarantined_entry_id_passes_quarantine_gate() -> None:
    module = load_module()
    candidate = entry("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", same_spec())
    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )
    assert not any("quarantined entry" in error for error in errors)


def test_unregistered_specialty_prefix_does_not_bypass_official_spec_gate() -> None:
    """A bare specialty- string prefix must not exempt an official workload from
    the v0.18.0 same-spec pairing check — the spec must be registry-verified
    (PR #172 review round 2)."""
    module = load_module()
    payload = same_spec()
    payload["spec_id"] = "specialty-fake-unregistered-qwen25-14b-910b3"
    payload["hardware_chip_model"] = "910B3"
    candidate = entry("bypass-attempt", payload)
    candidate["engine"] = "vllm-hust"
    candidate["workload"] = {
        "name": "random-online",
        "input_length": 1024,
        "output_length": 256,
        "batch_size": None,
        "concurrent_requests": None,
        "dataset": "random",
    }
    candidate["model"]["name"] = "Qwen/Qwen2.5-14B-Instruct"
    candidate["hardware"] = {"chip_model": "910B3"}

    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )
    assert any("must use official v0.18.0 same_spec" in error for error in errors)


def test_registered_specialty_spec_passes_official_spec_gate() -> None:
    """A registry-verified specialty spec (the 7 real 910B3 dual-stream specs)
    is exempt from the v0.18.0 pairing check (issue #178)."""
    module = load_module()
    assert (
        "specialty-ascend-full-graph-parallel-inplace-random-online-qwen25-14b-910b3"
        in module.specialty_spec_ids()
    )
    payload = same_spec()
    payload["spec_id"] = (
        "specialty-ascend-full-graph-parallel-inplace-random-online-qwen25-14b-910b3"
    )
    payload["hardware_chip_model"] = "910B3"
    candidate = entry("registered-specialty", payload)
    candidate["engine"] = "vllm-hust"
    candidate["workload"] = {
        "name": "random-online",
        "input_length": 1024,
        "output_length": 256,
        "batch_size": None,
        "concurrent_requests": None,
        "dataset": "random",
    }
    candidate["model"]["name"] = "Qwen/Qwen2.5-14B-Instruct"
    candidate["hardware"] = {"chip_model": "910B3"}

    errors = module.validate_entry(
        candidate,
        source=Path("leaderboard_single.json"),
    )
    assert not any("must use official v0.18.0 same_spec" in error for error in errors)


def _suspect_section() -> dict:
    return {
        "schema_version": "issue-146-suspect/v2",
        "conclusion": "no_regression_reproduced",
        "action": "mark_suspect_noise",
        "analysis_provenance": "reports/issue_146_retest_analysis.json",
        "raw_evidence_dir": "reports/issue_146_retest_raw_results/",
        "note": "controlled re-test reproduces neither regression",
        "entries": [
            {
                "git_commit": "7a63f81e86bd71e980adb635870ff56c9e23b545",  # pragma: allowlist secret
                "workload": "sonnet-throughput",
                "workload_params": {
                    "input_length": 1024,
                    "output_length": 256,
                    "batch_size": None,
                    "dataset": "sonnet",
                },
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "original_value": 1589.93,
                "original_value_unit": "tok/s",
                "retest_median": 2898.8,
                "retest_median_unit": "tok/s",
                "threshold_pct": 10.0,
                "status": "invalid-suspect-noise",
                "retest_base_commit": "2206f1f7b7212801187bc001c5f6cb86b2289214",  # pragma: allowlist secret
                "retest_delta_vs_base_commit_pct": 0.24,
            }
        ],
    }


def _make_snapshot_dir(tmp_path: Path, *, public_entries=None) -> Path:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "leaderboard_single.json").write_text(
        json.dumps(public_entries or []) + "\n", encoding="utf-8"
    )
    (snapshot_dir / "leaderboard_multi.json").write_text("[]\n", encoding="utf-8")
    return snapshot_dir


class TestQuarantineSuspectEntriesValidation:
    def test_valid_suspect_section_passes(self, tmp_path):
        snapshot_dir = _make_snapshot_dir(tmp_path)
        (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
            json.dumps({"issue_146_suspect_entries": _suspect_section()}) + "\n",
            encoding="utf-8",
        )
        module = load_module()
        assert module.validate_quarantine_suspect_entries(snapshot_dir) == []

    def test_wrong_schema_version_fails(self, tmp_path):
        snapshot_dir = _make_snapshot_dir(tmp_path)
        section = _suspect_section()
        section["schema_version"] = "issue-146-suspect/v1"
        (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
            json.dumps({"issue_146_suspect_entries": section}) + "\n",
            encoding="utf-8",
        )
        module = load_module()
        errors = module.validate_quarantine_suspect_entries(snapshot_dir)
        assert any("schema_version" in e for e in errors)

    def test_missing_retest_base_commit_fails(self, tmp_path):
        snapshot_dir = _make_snapshot_dir(tmp_path)
        section = _suspect_section()
        del section["entries"][0]["retest_base_commit"]
        (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
            json.dumps({"issue_146_suspect_entries": section}) + "\n",
            encoding="utf-8",
        )
        module = load_module()
        errors = module.validate_quarantine_suspect_entries(snapshot_dir)
        assert any("retest_base_commit" in e for e in errors)

    def test_suspect_commit_reappearing_in_public_snapshot_fails(self, tmp_path):
        # [blocking] review: suspect entries flagged invalid-suspect-noise must be
        # consumed by snapshot exclusion logic, i.e. they must not re-enter public
        # snapshots. A public entry carrying the suspect (commit, workload) pair
        # must fail.
        snapshot_dir = _make_snapshot_dir(
            tmp_path,
            public_entries=[
                {
                    "entry_id": "0d86eb2b-0000-0000-0000-000000000000",
                    "workload": {"name": "sonnet-throughput"},
                    "metadata": {
                        "engine_version": "7a63f81",
                        "git_commit": "7a63f81e86bd71e980adb635870ff56c9e23b545",  # pragma: allowlist secret
                    },
                }
            ],
        )
        (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
            json.dumps({"issue_146_suspect_entries": _suspect_section()}) + "\n",
            encoding="utf-8",
        )
        module = load_module()
        errors = module.validate_quarantine_suspect_entries(snapshot_dir)
        assert any("must stay excluded" in e for e in errors)

    def test_suspect_commit_under_other_workload_passes(self, tmp_path):
        # [minor] review: the exclusion match is workload-aware. The same commit
        # under a non-suspect workload must NOT be flagged, otherwise valid
        # public entries are rejected (cf. leaderboard_compare.json carrying
        # 7a63f81e under random-latency/random-online while only
        # sonnet-throughput@7a63f81e is suspect).
        snapshot_dir = _make_snapshot_dir(
            tmp_path,
            public_entries=[
                {
                    "entry_id": "0d86eb2b-0000-0000-0000-000000000000",
                    "workload": {"name": "random-latency"},
                    "metadata": {
                        "engine_version": "7a63f81",
                        "git_commit": "7a63f81e86bd71e980adb635870ff56c9e23b545",  # pragma: allowlist secret
                    },
                }
            ],
        )
        (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
            json.dumps({"issue_146_suspect_entries": _suspect_section()}) + "\n",
            encoding="utf-8",
        )
        module = load_module()
        errors = module.validate_quarantine_suspect_entries(snapshot_dir)
        assert not any("must stay excluded" in e for e in errors)
