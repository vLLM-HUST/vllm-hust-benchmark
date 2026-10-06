from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_issue214_ppt_safe_summary.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("issue214_ppt_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def summary_module():
    return _load_module()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _repeat(
    root: Path,
    index: int,
    value: float,
    *,
    metric: str = "ttft_ms",
    status: str = "OK",
    identity_suffix: str = "",
) -> Path:
    repeat_dir = root / f"repeat-{index:02d}"
    submission = repeat_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "STATUS").write_text(f"{status}\n", encoding="utf-8")
    run = {
        "workload": {
            "name": "agent-research-online",
            "input_length": 1024,
            "output_length": 256,
            "batch_size": None,
            "concurrent_requests": None,
            "dataset": "custom",
        },
        "model": {
            "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
            "parameters": "14B",
            "precision": "FP16",
            "quantization": None,
        },
        "hardware": {"vendor": "Huawei", "chip_model": "910B2", "chip_count": 1},
        "same_spec": {
            "scenario": "agent-research-online",
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "model_parameters": "14B",
            "model_precision": "FP16",
            "model_quantization": "",
            "hardware_vendor": "Huawei",
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
            "resolved_server_parameters": {
                "tensor_parallel_size": 1,
                "port": 8000 + index,
                "model": f"/models/local-{index}",
            },
            "resolved_client_parameters": {
                "dataset_name": "custom",
                "dataset_path": "scripts/traces/input.jsonl",
                "num_prompts": 32,
                "request_rate": 1,
                "host": "127.0.0.1",
                "port": 8000 + index,
                "identity_suffix": identity_suffix,
            },
        },
        "metrics": {metric: value, "error_rate": 0},
        "metadata": {
            "runtime_provenance": {
                "engine": {"commit": "c" * 40},
                "plugin": {"commit": "d" * 40},
            }
        },
    }
    run_path = submission / "run_leaderboard.json"
    _write_json(run_path, run)
    digest = hashlib.sha256(run_path.read_bytes()).hexdigest()
    (submission / "checksums.sha256").write_text(
        f"{digest}  ./run_leaderboard.json\n", encoding="utf-8"
    )
    return repeat_dir


def _manifest(
    tmp_path: Path,
    repeat_dirs: list[Path],
    *,
    input_identity_status: str = "verified",
    reference_metric: str = "ttft_ms",
    candidate_metric: str = "ttft_ms",
    reference_evidence_root: Path | None = None,
) -> Path:
    evidence_root = reference_evidence_root or tmp_path / "missing-reference-evidence"
    reference_path = tmp_path / "artifact_suites.json"
    _write_json(
        reference_path,
        {
            "evidence_root": str(evidence_root),
            "pairs": {
                "current": {"core": "a" * 40, "plugin": "b" * 40},
            },
            "workloads": {
                "agent-research-online": {
                    "primary_metric": reference_metric,
                    "current_values": [10.0, 20.0, 30.0],
                }
            },
        },
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "issue214-ppt-safe-manifest/v1",
            "reference": {
                "artifact_suites": str(reference_path),
                "values_key": "current_values",
                "label": "current-main",
                "commits": {"core": "a" * 40, "plugin": "b" * 40},
            },
            "candidate": {
                "label": "v0.18.0",
                "commits": {"core": "c" * 40, "plugin": "d" * 40},
                "workloads": {
                    "agent-research-online": {
                        "repeat_dirs": [str(path) for path in repeat_dirs],
                        "primary_metric": candidate_metric,
                        "unit": "ms",
                        "direction": "lower_is_better",
                        "input_identity_status": input_identity_status,
                    }
                },
            },
        },
    )
    return manifest_path


def test_linear_quartiles(summary_module):
    assert summary_module.compute_stats([1.0, 2.0, 3.0]) == {
        "n": 3,
        "median": 2.0,
        "q1": 1.5,
        "q3": 2.5,
        "iqr": 1.0,
    }


def test_generates_three_formats_and_marks_summary_only_reference_grade_b(
    tmp_path, summary_module
):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([1, 2, 100], 1)
    ]
    manifest_path = _manifest(tmp_path, repeats)
    output_dir = tmp_path / "out"

    result = summary_module.generate(manifest_path, output_dir)
    row = result["workloads"][0]

    assert row["status"] == "ready"
    assert row["reference"]["summary_values"] == [10.0, 20.0, 30.0]
    assert "raw_values" not in row["reference"]
    assert row["reference"]["evidence_grade"] == "B"
    assert row["candidate"]["raw_values"] == [1.0, 2.0, 100.0]
    assert row["candidate"]["evidence_grade"] == "A"
    assert row["delta_percent"] is None
    assert "reference raw evidence is not verified" in row["comparison_blockers"]
    assert row["variability"]["candidate"] == "high"
    assert row["stability_notice"] is not None
    assert {path.name for path in output_dir.iterdir()} == {
        "issue214_ppt_safe_summary.json",
        "issue214_ppt_safe_summary.csv",
        "issue214_ppt_safe_summary.md",
    }
    markdown = (output_dir / "issue214_ppt_safe_summary.md").read_text()
    assert "Reference summary values" in markdown
    assert "Candidate raw values" in markdown
    assert not any(
        word in markdown.casefold() for word in summary_module.FORBIDDEN_NAMES
    )


def test_unverified_input_suppresses_delta(tmp_path, summary_module):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([9, 10, 11], 1)
    ]
    result = summary_module.generate(
        _manifest(tmp_path, repeats, input_identity_status="unverified"),
        tmp_path / "out",
    )
    row = result["workloads"][0]
    assert row["status"] == "ready"
    assert row["delta_percent"] is None
    assert "input identity is not verified" in row["comparison_blockers"]


def test_existing_reference_directory_remains_summary_only_grade_b(
    tmp_path, summary_module
):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([9, 10, 11], 1)
    ]
    evidence_root = tmp_path / "reference-evidence"
    evidence_root.mkdir()

    result = summary_module.generate(
        _manifest(
            tmp_path,
            repeats,
            reference_evidence_root=evidence_root,
        ),
        tmp_path / "out",
    )
    row = result["workloads"][0]

    assert result["reference"]["evidence_grade"] == "B"
    assert result["reference"]["referenced_evidence_directory_available"] is True
    assert row["delta_percent"] is None
    assert "reference raw evidence is not verified" in row["comparison_blockers"]


def test_generated_paths_are_manifest_portable(tmp_path, summary_module):
    repeats = [
        _repeat(tmp_path / "runs", i, value)
        for i, value in enumerate([9, 10, 11], 1)
    ]
    manifest_path = _manifest(tmp_path, repeats)
    manifest = json.loads(manifest_path.read_text())
    manifest["reference"]["artifact_suites"] = "artifact_suites.json"
    manifest["candidate"]["workloads"]["agent-research-online"]["repeat_dirs"] = [
        path.relative_to(tmp_path).as_posix() for path in repeats
    ]
    _write_json(manifest_path, manifest)

    result = summary_module.generate(manifest_path, tmp_path / "out")

    assert result["reference"]["source"] == "artifact_suites.json"
    assert result["reference"]["evidence_root"] is None
    assert result["workloads"][0]["candidate"]["repeat_dirs"] == [
        "runs/repeat-01",
        "runs/repeat-02",
        "runs/repeat-03",
    ]


def test_identity_projection_omits_container_absolute_paths(
    tmp_path, summary_module
):
    repeats = [
        _repeat(tmp_path / "runs", i, value)
        for i, value in enumerate([9, 10, 11], 1)
    ]
    for repeat in repeats:
        submission = repeat / "submission"
        provenance_path = submission / "input_provenance.json"
        _write_json(
            provenance_path,
            {
                "dataset_sha256": "a" * 64,
                "frozen_dataset_path": "/root/container-only/dataset",
            },
        )
        run_path = submission / "run_leaderboard.json"
        (submission / "checksums.sha256").write_text(
            "".join(
                [
                    f"{hashlib.sha256(provenance_path.read_bytes()).hexdigest()}  ./input_provenance.json\n",
                    f"{hashlib.sha256(run_path.read_bytes()).hexdigest()}  ./run_leaderboard.json\n",
                ]
            ),
            encoding="utf-8",
        )

    result = summary_module.generate(_manifest(tmp_path, repeats), tmp_path / "out")
    rendered = json.dumps(result)

    assert "/root/container-only/dataset" not in rendered
    assert "<absolute-path-omitted>" in rendered


@pytest.mark.parametrize(
    "failure", ["too_few", "bad_status", "bad_checksum", "identity"]
)
def test_candidate_evidence_failures_are_blocked(tmp_path, summary_module, failure):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([9, 10, 11], 1)
    ]
    if failure == "too_few":
        repeats.pop()
    elif failure == "bad_status":
        (repeats[0] / "submission" / "STATUS").write_text("FAILED\n")
    elif failure == "bad_checksum":
        (repeats[0] / "submission" / "run_leaderboard.json").write_text("{}")
    else:
        repeats[-1] = _repeat(tmp_path / "other", 3, 11, identity_suffix="changed")

    row = summary_module.generate(_manifest(tmp_path, repeats), tmp_path / "out")[
        "workloads"
    ][0]
    assert row["status"] == "blocked"
    assert row["delta_percent"] is None
    assert row["candidate"]["evidence_grade"] == "blocked"


def test_metric_conflict_is_blocked(tmp_path, summary_module):
    repeats = [
        _repeat(tmp_path / "runs", i, value, metric="batch_latency_ms")
        for i, value in enumerate([9, 10, 11], 1)
    ]
    row = summary_module.generate(
        _manifest(tmp_path, repeats, candidate_metric="batch_latency_ms"),
        tmp_path / "out",
    )["workloads"][0]
    assert row["status"] == "blocked"
    assert row["delta_percent"] is None
    assert "primary metric conflict" in " ".join(row["blockers"])


def test_candidate_commit_mismatch_is_blocked(tmp_path, summary_module):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([9, 10, 11], 1)
    ]
    run_path = repeats[-1] / "submission" / "run_leaderboard.json"
    run = json.loads(run_path.read_text())
    run["metadata"]["runtime_provenance"]["engine"]["commit"] = "e" * 40
    _write_json(run_path, run)
    digest = hashlib.sha256(run_path.read_bytes()).hexdigest()
    (run_path.parent / "checksums.sha256").write_text(
        f"{digest}  ./run_leaderboard.json\n", encoding="utf-8"
    )

    row = summary_module.generate(_manifest(tmp_path, repeats), tmp_path / "out")[
        "workloads"
    ][0]
    assert row["status"] == "blocked"
    assert "candidate commits do not match" in " ".join(row["blockers"])


def test_unchecksummed_input_provenance_is_blocked(tmp_path, summary_module):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([9, 10, 11], 1)
    ]
    _write_json(
        repeats[0] / "submission" / "input_provenance.json",
        {"dataset_sha256": "a" * 64},
    )

    row = summary_module.generate(_manifest(tmp_path, repeats), tmp_path / "out")[
        "workloads"
    ][0]

    assert row["status"] == "blocked"
    assert row["delta_percent"] is None
    assert "input_provenance.json is not covered by checksums" in " ".join(
        row["blockers"]
    )


@pytest.mark.parametrize("forbidden", ["historical", "frontier", "历史"])
def test_forbidden_label_is_rejected(tmp_path, summary_module, forbidden):
    repeats = [
        _repeat(tmp_path / "runs", i, value) for i, value in enumerate([9, 10, 11], 1)
    ]
    manifest_path = _manifest(tmp_path, repeats)
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate"]["label"] = f"bad-{forbidden}"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="forbidden naming"):
        summary_module.generate(manifest_path, tmp_path / "out")
