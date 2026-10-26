"""Combine immutable task observations without dropping scoring limitations."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def build():
    inputs = json.loads((HERE / "catalog-inputs.json").read_text())
    entries = []
    for item in inputs:
        path = ROOT / item["source"]
        value = json.loads(path.read_text())
        assert value["vllm"] == value["vllm_ascend"] == "0.18.0"
        agent = value["agent"]
        result = value["grading"]["official_result"]
        assert type(result) is bool
        limited = value["grading"].get("patch_application_attested") is False
        entries.append(
            dict(
                run_id=value["run_id"],
                task_identity_sha256=item["task_identity_sha256"],
                source=item["source"],
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                official_resolved=int(result),
                official_denominator=1,
                official_unresolved_grading=value.get(
                    "official_unresolved_grading", value.get("unresolved_grading")
                ),
                model_calls=agent["model_calls"],
                elapsed_seconds=agent["elapsed_seconds"],
                status=agent["status"],
                agent_submitted=agent["agent_submitted"],
                patch_bytes=agent["patch_bytes"],
                patch_sha256=agent["patch_sha256"],
                patch_application_evidence_limited=limited,
                parsed_test_counts=value["grading"].get("parsed_test_counts", {}),
            )
        )
    assert len({x["run_id"] for x in entries}) == len(entries)
    assert len({x["task_identity_sha256"] for x in entries}) == len(entries)
    assert all(x["official_unresolved_grading"] == 0 for x in entries)
    return dict(
        schema="swe-native018-incremental-catalog/v1",
        dataset="SWE-bench-Pro V1",
        model="Qwen3.5-35B-A3B",
        vllm="0.18.0",
        vllm_ascend="0.18.0",
        dataset_tasks=731,
        attempted_unique_tasks=len(entries),
        recorded_official_resolved=sum(x["official_resolved"] for x in entries),
        recorded_official_denominator=len(entries),
        official_unresolved_grading=0,
        application_evidence_limited_tasks=sum(
            x["patch_application_evidence_limited"] for x in entries
        ),
        unattempted_tasks=731 - len(entries),
        full_dataset_acceptance=False,
        entries=entries,
    )


if __name__ == "__main__":
    (HERE / "catalog.json").write_text(json.dumps(build(), indent=2) + "\n")
