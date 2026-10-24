"""Publish audited bounded reference measurements without inventing efficiency."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(source, output, archives):
    raw = source / "minimum-reference-publication-summary-r1.json"
    measured = json.loads(raw.read_text())
    assert measured["unique_original_cases_completed"] == 281
    assert measured["model_validation"]["fit_count"] == 1
    for name in (
        "reference_seconds",
        "reference_tflops_per_chip",
        "efficiency_percent",
        "attainment",
    ):
        assert measured[name] is None
    output.mkdir(parents=True, exist_ok=True)
    selected = [
        raw.name,
        "minimum-reference-closure-r1.json",
        "minimum-reference-batch-union-audit-r1.json",
        "minimum-reference-model-mismatch-audit-r1.json",
        "minimum-reference-supplement-plan-r2.json",
        "minimum-reference-model-update-method-r3.json",
    ]
    for name in selected:
        shutil.copyfile(source / name, output / name)
    summary = {
        key: measured[key]
        for key in (
            "status",
            "formal_result_unchanged",
            "batches",
            "totals",
            "unique_original_cases_completed",
            "conditional_cases_skipped_by_reuse",
            "authorized_metadata_revision_cases",
            "scenarios",
            "leaf_coverage_by_family",
            "reference_seconds",
            "reference_tflops_per_chip",
            "efficiency_percent",
            "attainment",
            "remaining_reasons_zh",
        )
    }
    summary["schema"] = "m01-reference-supplement/v1"
    summary["model_validation"] = {
        key: value
        for key, value in measured["model_validation"].items()
        if key != "actual_predictions_and_source_differences"
    }
    summary["model_validation"]["evidence"] = (
        "minimum-reference-model-mismatch-audit-r1.json"
    )
    summary["replaces_best_b1_performance"] = False
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    entries = []
    for path in archives:
        assert path.is_file() and path.stat().st_size < 100 * 1024 * 1024
        for suffix in ("-manifest.json", "-readback.json"):
            metadata = path.with_name(path.name.removesuffix(".tar.gz") + suffix)
            shutil.copyfile(metadata, output / metadata.name)
            selected.append(metadata.name)
        entries.append(
            dict(
                name=path.name,
                bytes=path.stat().st_size,
                sha256=digest(path),
                in_git=False,
                raw_url="https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/"
                + path.name,
            )
        )
    index = dict(
        schema="m01-reference-supplement-archive/v1",
        archives=entries,
        selected_sha256={
            name: digest(output / name) for name in selected + ["summary.json"]
        },
        raw_evidence_unchanged=True,
        efficiency_percent=None,
        attainment=None,
    )
    (output / "archive.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "m01-recovery/storage-reference-supplement-20261024",
    )
    args = parser.parse_args()
    package(args.source, args.output, args.archive)
