"""Verify private source bindings and export only aggregate observation fields."""

import collections
import hashlib
import json
from pathlib import Path

W = Path("/swe-persist/swe-pro-shuhao-20261025")
R = W / "runs/swe018-first-missing-r3"


def read(name):
    return json.loads((R / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


binding = read("execution-binding.json")
for name, digest in binding["files"].items():
    assert sha(Path(name)) == digest
settings = read("settings.json")
agent = read("agent/task-000/report-private.json")
grade = read("grade/grading-status.json")
scores = read("grade/eval_results.json")
assert len(scores) == 1 and list(scores.values()) == [False]
assert grade["official_scores"] == scores and grade["unresolved_grading"] == []
assert grade["model_calls_in_this_grading"] == 0
assert agent["status"] == "Submitted" and agent["agent_submitted"] is True
assert agent["model_calls"] == 36 and agent["container_cleanup_exit_code"] == 0
assert read("run-exit.json")["exit_code"] == read("grade-exit.json")["exit_code"] == 0
patch = R / "agent/task-000/patch-private.diff"
assert sha(patch) == agent["patch_sha256"]
assert patch.stat().st_size == agent["patch_bytes"]
sources = read("grade/source-and-patch-bindings.json")
for name, digest in sources["official_sources"].items():
    assert sha(Path(settings["paths"]["official_v1_source"]) / name) == digest
assert sources["patches"][0]["patch_sha256"] == sha(patch)
snapshots = list((R / "grade").glob("*/agent_patch.diff"))
assert len(snapshots) == 1 and sha(snapshots[0]) == sha(patch)
assert read("grade/before-image-ids.json") == read("grade/after-image-ids.json")
outputs = list((R / "grade").glob("*/workspace/output.json"))
assert len(outputs) == 1
counts = dict(
    collections.Counter(
        x["status"] for x in json.loads(outputs[0].read_text())["tests"]
    )
)
assert counts == {"FAILED": 1}
trajectory = read("agent/task-000/trajectory-private.json")
assert trajectory["info"]["model_stats"]["api_calls"] == agent["model_calls"]
evidence = [
    "execution-binding.json",
    "settings.json",
    "run-exit.json",
    "grade-exit.json",
    "agent/batch-report.json",
    "agent/task-000/report-private.json",
    "agent/task-000/trajectory-private.json",
    "agent/task-000/patch-private.diff",
    "grade/grading-status.json",
    "grade/eval_results.json",
    "grade/source-and-patch-bindings.json",
]
value = dict(
    schema="swe-incremental-task-observation/v1",
    run_id=R.name,
    dataset="SWE-bench-Pro V1",
    total_dataset_tasks=731,
    new_task_attempts=1,
    recorded_official_resolved=0,
    recorded_official_denominator=1,
    official_unresolved_grading=0,
    full_dataset_acceptance=False,
    model="Qwen3.5-35B-A3B",
    vllm="0.18.0",
    vllm_ascend="0.18.0",
    qualification=dict(
        new_calls=0, reused_calls=5, source_run="swe018-single-agent-r1"
    ),
    agent={
        k: agent[k]
        for k in (
            "status",
            "agent_submitted",
            "elapsed_seconds",
            "model_calls",
            "patch_bytes",
            "patch_sha256",
            "container_cleanup_exit_code",
        )
    },
    budget=dict(
        step_limit=100,
        wall_seconds=1200,
        max_tokens=4096,
        temperature=0,
        seed=0,
        thinking=False,
        automatic_retries=0,
    ),
    grading=dict(
        official_result=False,
        process_exit_code=0,
        model_calls=0,
        source_files=len(sources["official_sources"]),
        parsed_test_counts=counts,
        patch_application_attested=False,
        failure_stage="before-all hook",
        limitation="Submission adds two text fixtures already present in the immutable grading image. Original git-apply status and stderr were not retained; official false does not independently prove testing of the applied patch.",
        patch_modified=False,
        grading_reexecuted=False,
    ),
    prior_infrastructure_failure=dict(
        run_id="swe018-first-missing-r2",
        model_calls=0,
        cause="Initial upstream fixture rejection before model construction; not another scored task",
    ),
    images=list(read("grade/before-image-ids.json").values()),
    evidence_sha256={name: sha(R / name) for name in evidence},
    private_task_answers_patch_or_trajectory_included=False,
)
out = R / "public-summary.json"
with out.open("x") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
files = sorted(x for x in R.rglob("*") if x.is_file() and x.name != "SHA256SUMS")
assert all(not x.is_symlink() for x in files)
with (R / "SHA256SUMS").open("x") as stream:
    stream.writelines(f"{sha(p)}  {p.relative_to(R)}\n" for p in files)
print(
    json.dumps(
        dict(
            status="exported_with_grading_limitation",
            public_summary_sha256=sha(out),
            seal_sha256=sha(R / "SHA256SUMS"),
            sealed_files=len(files),
        )
    )
)
