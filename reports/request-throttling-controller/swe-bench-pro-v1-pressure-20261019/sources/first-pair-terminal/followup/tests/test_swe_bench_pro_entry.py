"""Contract rejection and upstream byte-identity tests for reusable entry."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("swe_entry", ROOT / "experiments/workload/run_swe_bench_pro.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def configuration():
    return json.loads((ROOT / "config/swe-bench-pro-v1.example.json").read_text())


def test_engine_drain_waits_for_running_and_waiting_requests(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "experiments/workload"))
    import swe_bench_pro_batch as batch
    readings = iter([(16, 25), (0, 2), (0, 0)])

    def metrics(*args, **kwargs):
        running, waiting = next(readings)
        return (f'vllm:num_requests_running{{engine="0"}} {running}\n'
                f'vllm:num_requests_waiting{{engine="0"}} {waiting}\n')

    monkeypatch.setattr(batch, "http", metrics)
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    result = batch.wait_engine_idle(configuration(), tmp_path)
    assert set(result.values()) == {0}
    assert len((tmp_path / "engine-drain-check-private.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("metrics,exception", [
    ('vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n', TimeoutError),
    ('vllm:num_requests_running 0\n', ValueError),
])
def test_engine_drain_rejects_busy_or_missing_observation(tmp_path, monkeypatch, metrics, exception):
    import sys
    sys.path.insert(0, str(ROOT / "experiments/workload"))
    import swe_bench_pro_batch as batch
    monkeypatch.setattr(batch, "http", lambda *args, **kwargs: metrics)
    with pytest.raises(exception):
        batch.wait_engine_idle(configuration(), tmp_path, timeout=0)


@pytest.mark.parametrize("field,value", [("arrival_time", 1), ("client_concurrency_limit", 2), ("size", 0)])
def test_rejects_changed_batch_contract(field, value):
    c = configuration()
    c["batch"][field] = value
    with pytest.raises(ValueError):
        entry.validate(c)


def test_rejects_unregistered_retry_or_pair_protocol():
    c = configuration()
    c["agent"]["automatic_retries"] = 1
    with pytest.raises(ValueError):
        entry.validate(c)
    c = configuration()
    c["protocol"]["pair_order"].pop()
    with pytest.raises(ValueError):
        entry.validate(c)


def test_independent_agents_keep_original_tasks_and_unique_program_lifecycles():
    import sys
    sys.path.insert(0, str(ROOT / "experiments/workload"))
    from swe_bench_pro_batch import expand_agent_instances, program_identifier
    c = configuration()
    c["batch"].update(independent_agents_per_task=16, program_instances=128)
    entry.validate(c)
    rows = [{"instance_id": str(i), "base_commit": str(i)} for i in range(8)]
    images = [{"image_id": str(i)} for i in range(8)]
    expanded, mapped = expand_agent_instances(c, rows, images)
    assert len(expanded) == len(mapped) == 128
    assert len({x["instance_id"] for x in expanded}) == 8
    identifiers = [program_identifier(c, x, "pair-1-controller") for x in expanded]
    assert len(set(identifiers)) == 128
    assert all(x["instance_id"] == image["image_id"] for x, image in zip(expanded, mapped))
    assert all("execution_id" not in x for x in rows)
    assert program_identifier(configuration(), rows[0], "pair-1-native").endswith(":0")
    c["batch"]["program_instances"] = 8
    with pytest.raises(ValueError, match="unique tasks"):
        entry.validate(c)


def test_replica_grading_uses_every_saved_patch_once(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "experiments/workload"))
    import swe_bench_pro_batch as batch
    c = configuration()
    c["batch"].update(size=2, independent_agents_per_task=2, program_instances=4)
    inputs, patches = tmp_path / "inputs", tmp_path / "patches"
    inputs.mkdir()
    rows = [{"instance_id": x} for x in ["a", "b"]]
    (inputs / "cases-private.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (inputs / "input-selection.json").write_text(json.dumps({"tasks": [
        {"instance_id": x, "image_uri": x, "image_id": x} for x in ["a", "b"]]}))
    for i in range(4):
        d = patches / f"task-{i:03d}"
        d.mkdir(parents=True)
        (d / "patch-private.diff").write_text("patch-" + str(i))
    monkeypatch.setattr(batch, "validate_grading_sources", lambda *args: {})
    seen = []

    def command(arguments, output, **kwargs):
        if "inspect" in arguments:
            return json.dumps([{"Id": arguments[-1]}])
        output = Path(output).parent
        supplied = json.loads((output / "patches-private.json").read_text())
        seen.extend(x["patch"] for x in supplied)
        (output / "eval_results.json").write_text(json.dumps({"a": True, "b": False}))
        for identifier in ["a", "b"]:
            folder = output / identifier / "workspace"
            folder.mkdir(parents=True)
            (folder / "output.json").write_text("{}")

    class Observer:
        def __init__(self, *args, **kwargs):
            pass

        def terminate(self):
            pass

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(batch, "command", command)
    monkeypatch.setattr(batch.subprocess, "Popen", Observer)
    result = batch.grade_saved_batch(c, inputs, patches, tmp_path / "grade")
    assert seen == ["patch-0", "patch-1", "patch-2", "patch-3"]
    assert len(result["official_scores"]) == 4
    assert result["unresolved_grading"] == []


def test_original_shared_profile_bytes():
    lock = json.loads((ROOT / "config/swe-bench-pro-v1-lock.json").read_text())
    c = configuration()
    assert entry.validate(copy.deepcopy(c)) == c
    profile = ROOT / "config/unified-graph-profile-v1.json"
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == "5792631172af883cf247e05b214444c02a106831aa50162b8c6f3a0e63f24661"  # pragma: allowlist secret
    assert lock["official_v1"]["nodebb_task_source_sha256"]


def test_official_relative_dockerfile_requires_explicit_working_directory(tmp_path):
    import sys
    official = tmp_path / "official"
    relative = "dockerfiles/base_dockerfile/example/Dockerfile"
    source = official / relative
    source.parent.mkdir(parents=True)
    source.write_text("FROM frozen\n")
    arguments = [sys.executable, "-c", f"from pathlib import Path; print(Path({relative!r}).read_text())"]
    with pytest.raises(RuntimeError):
        entry.command(arguments, tmp_path / "wrong-cwd.json", cwd=tmp_path)
    assert entry.command(arguments, tmp_path / "right-cwd.json", cwd=official).strip() == "FROM frozen"
    assert json.loads((tmp_path / "right-cwd.json").read_text())["cwd"] == str(official)


def test_missing_or_changed_official_grading_dockerfile_is_rejected(tmp_path):
    rows = [{"instance_id": "example"}]
    paths = [f"dockerfiles/{kind}/example/Dockerfile" for kind in ["base_dockerfile", "instance_dockerfile"]]
    paths += [f"run_scripts/example/{name}" for name in ["run_script.sh", "parser.py"]]
    hashes = {}
    for relative in paths:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(relative)
        hashes[relative] = entry.digest(p)
    lock = {"official_v1": {"sha256": {}, "nodebb_task_source_sha256": hashes,
                            "grading_dockerfiles_sha256": {}}}
    assert entry.validate_grading_sources(tmp_path, rows, lock) == hashes
    (tmp_path / paths[0]).unlink()
    with pytest.raises(ValueError, match="Missing frozen official"):
        entry.validate_grading_sources(tmp_path, rows, lock)
    (tmp_path / paths[0]).write_text("changed")
    with pytest.raises(ValueError, match="source hash mismatch"):
        entry.validate_grading_sources(tmp_path, rows, lock)


def test_same_container_model_requires_ray_npu_reservation():
    plan = json.loads((ROOT / "config/swe-bench-pro-v1.ride-plan.example.json").read_text())
    services = json.loads((ROOT / "config/swe-bench-pro-v1.ride.example.json").read_text())
    assert entry.validate_deployment(plan, services)["model_job_npu_reservation"] == 2
    services["model_services"][0]["resources"].pop("NPU")
    with pytest.raises(ValueError, match="reserve NPU"):
        entry.validate_deployment(plan, services)


def test_graph_capture_alone_is_not_execution_proof():
    import sys
    sys.path.insert(0, str(ROOT / "experiments/workload"))
    from swe_bench_pro_batch import graph_replay_delta

    def witness(replays, pid_offset=0):
        return {"g7_shared_runtime_v1": {"results": [
            {"rank": rank, "pid": 100 + rank + pid_offset, "worker_epoch": str(rank),
             "enforce_eager": False, "cudagraph_mode": "FULL_DECODE_ONLY"} for rank in [0, 1]]},
                "test7_gdn_main_witness_v1": {"results": [
            {"rank": rank, "pid": 100 + rank + pid_offset,
             "capture_and_replay_by_graph": {"decode": {
                 "actual_replays": replays, "captured_single_token_view_eligible_calls": 0}}}
            for rank in [0, 1]]}}
    with pytest.raises(ValueError, match="No eligible"):
        graph_replay_delta(witness(0), witness(0))
    with pytest.raises(ValueError, match="epoch changed"):
        graph_replay_delta(witness(0), witness(2, 10))
    assert graph_replay_delta(witness(1), witness(4)) == {0: 3, 1: 3}
    eager = witness(4)
    eager["g7_shared_runtime_v1"]["results"][0]["enforce_eager"] = True
    with pytest.raises(ValueError, match="frozen decode graph mode"):
        graph_replay_delta(witness(1), eager)


def test_runtime_shim_preserves_paths_and_forwarded_arguments(tmp_path):
    import subprocess
    c = configuration()
    fake = tmp_path / "bootstrap python"
    fake.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    c["model_deployment"]["bootstrap_python"] = str(fake)
    c["paths"]["repository"] = "/absolute/repository with spaces"
    settings = tmp_path / "private config.json"
    p = tmp_path / "vllm"
    p.write_text(entry.runtime_shim(c, settings))
    actual = json.loads(subprocess.check_output(["bash", str(p), "serve", "model with spaces"]))
    assert actual == ["/absolute/repository with spaces/experiments/workload/swe_bench_pro_runtime.py",
                      "--settings", str(settings), "--", "serve", "model with spaces"]


def test_suite_rejects_a_deadline_shorter_than_its_frozen_bounds():
    import datetime
    c = configuration()
    now = datetime.datetime(2026, 10, 19, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValueError, match="actual Coder deadline"):
        entry.validate_workspace_lease(c, now)
    c["workspace"]["deadline"] = (now + datetime.timedelta(hours=4)).isoformat()
    with pytest.raises(ValueError, match="does not cover"):
        entry.validate_workspace_lease(c, now)
    c["workspace"]["deadline"] = (now + datetime.timedelta(hours=7)).isoformat()
    assert entry.validate_workspace_lease(c, now)["status"] == "LEASE_COVERS_FROZEN_BOUNDS"
