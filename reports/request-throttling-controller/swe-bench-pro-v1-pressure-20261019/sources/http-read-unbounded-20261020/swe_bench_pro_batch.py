"""Finite real-Agent batches. Only orchestration; official Agent/scorer unchanged."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import re
import runpy
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

from run_swe_bench_pro import (REPOSITORY, command, digest, http, read, save,
                              validate_grading_sources, validate_workspace_lease,
                              bind_agent_http_timeout)


def runtime_witness(c, folder, window):
    """Call frozen upstream read-only observers outside the measurement interval."""
    result = {}
    for method in ("test7_gdn_main_witness_v1", "g7_cumulative_witness_v1", "g7_shared_runtime_v1"):
        try:
            body = {"method": method, "timeout": 60}
            if method != "g7_shared_runtime_v1":
                body["kwargs"] = {"window_id": window}
            result[method] = http(c["endpoints"]["model"] + "/collective_rpc", body, timeout=75)
        except Exception as exc:
            result[method] = {"error": str(exc)}
    save(Path(folder) / (window + "-runtime-private.json"), result)
    return result


def graph_replay_delta(before, after):
    method = "test7_gdn_main_witness_v1"
    previous = {x["rank"]: x for x in before[method]["results"]}
    current = {x["rank"]: x for x in after[method]["results"]}
    if set(previous) != {0, 1} or set(current) != {0, 1}:
        raise ValueError("Both actual TP workers required")
    delta = {}
    for rank, row in current.items():
        if row["pid"] != previous[rank]["pid"]:
            raise ValueError("Worker epoch changed across the request window")
        actual = {x["rank"]: x for x in after["g7_shared_runtime_v1"]["results"]}[rank]
        original = {x["rank"]: x for x in before["g7_shared_runtime_v1"]["results"]}[rank]
        if actual["worker_epoch"] != original["worker_epoch"] or actual["pid"] != row["pid"]:
            raise ValueError("Runtime worker epoch does not match graph observer")
        if actual["enforce_eager"] is not False or actual["cudagraph_mode"] != "FULL_DECODE_ONLY":
            raise ValueError("Actual worker is not in the frozen decode graph mode")
        graphs = row["capture_and_replay_by_graph"]
        old_graphs = previous[rank]["capture_and_replay_by_graph"]
        # QKV single-token-view eligibility belongs to another MOD. Native keeps
        # that optimization disabled, so its eligibility counter is correctly zero.
        # Require real replay deltas plus actual FULL_DECODE_ONLY worker identity.
        delta[rank] = sum(
            x["actual_replays"] - old_graphs.get(key, {}).get("actual_replays", 0)
            for key, x in graphs.items())
        if delta[rank] <= 0:
            raise ValueError("No eligible decode graph replay in actual request window")
    return delta


def expand_agent_instances(c, rows, task_images):
    """Separate workload multiplicity from official task identity and image."""
    copies = c["batch"].get("independent_agents_per_task", 1)
    if len(rows) != len(task_images):
        raise ValueError("Task/image cardinality mismatch")
    expanded, images = [], []
    for replica in range(copies):
        for row, image in zip(rows, task_images):
            item = dict(row)
            if copies != 1:
                item["execution_id"] = row["instance_id"] + f":replica-{replica:03d}"
            expanded.append(item)
            images.append(image)
    return expanded, images


def program_identifier(c, row, batch_name):
    return c["run_id"] + ":" + batch_name + ":" + row.get("execution_id", row["instance_id"])


def agent_task(configuration, row, task_image, arm, folder, barrier):
    c, folder = configuration, Path(folder)
    folder.mkdir(exist_ok=False)
    os.environ.update(DOCKER_HOST=c["paths"]["docker_host"],
                      MSWEA_SILENT_STARTUP="1", MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="1",
                      MSWEA_GLOBAL_CONFIG_DIR=str(folder / "agent-config"),
                      LITELLM_LOCAL_MODEL_COST_MAP="True")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(Path(c["paths"]["agent_source"]) / "src"))
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.run.utils.save import save_traj

    cfg = yaml.safe_load((Path(c["paths"]["agent_source"]) /
                          "src/minisweagent/config/extra/swebench.yaml").read_text())
    agent_cfg = {k: v.replace("/testbed", "/app") if isinstance(v, str) else v
                 for k, v in cfg["agent"].items()}
    agent_cfg.update(step_limit=c["agent"]["step_limit"], cost_limit=0)
    formatter = runpy.run_path(str(Path(c["paths"]["official_v1_source"]) /
                                  "helper_code/create_problem_statement.py"))
    task = formatter["create_problem_statement"](row)
    (folder / "task-private.txt").write_text(task)
    program = program_identifier(c, row, folder.parent.name)
    model_cfg = {
        "model_name": "openai/Qwen3.5-35B-A3B", "cost_tracking": "ignore_errors",
        "model_kwargs": {"api_base": c["endpoints"][arm] + "/v1", "api_key": "EMPTY",
                         "temperature": c["agent"]["temperature"], "seed": c["agent"]["seed"],
                         "max_tokens": c["agent"]["max_tokens"],
                         "timeout": c["agent"]["http_timeout_seconds"], "num_retries": 0,
                         "extra_headers": {"X-Session-ID": program},
                         "extra_body": {"chat_template_kwargs": {
                             "enable_thinking": c["agent"]["thinking"]}}}}
    env = agent = None
    began = None
    result = ""
    status = "PREPARATION_FAILED"
    error = None
    patch = ""
    stop_observer = threading.Event()
    observer = None

    def observe():
        previous = -1
        while not stop_observer.wait(2):
            try:
                calls = agent.model.n_calls
                if calls != previous:
                    responses = [m for m in list(agent.messages) if m["role"] == "assistant"]
                    action = re.findall(r"```bash\s*\n(.*?)\n```", responses[-1]["content"], re.S) if responses else []
                    save(folder / "progress-private.json", {
                        "model_calls": calls, "elapsed_seconds": time.monotonic() - began,
                        "last_command": action[0][:1000] if action else ""})
                    save_traj(agent, folder / "trajectory-live-private.json", print_path=False)
                    print("AGENT_PROGRESS", row["instance_id"], calls, flush=True)
                    previous = calls
            except Exception:
                (folder / "observer-error.txt").write_text(traceback.format_exc())

    try:
        env = DockerEnvironment(
            image=task_image["image_id"], cwd="/app", executable=c["paths"]["docker"],
            timeout=c["agent"]["tool_timeout_seconds"], container_timeout="40m",
            run_args=["--rm", "--entrypoint", "", "--network", c["container"]["network"],
                      "--cpus", str(c["container"]["cpus"]), "--memory", c["container"]["memory"]])
        (folder / "container-id.txt").write_text(env.container_id)
        before = env.execute("git rev-parse HEAD && git status --porcelain")
        save(folder / "code-before.json", before)
        assert before["returncode"] == 0 and before["output"].splitlines()[0] == row["base_commit"]
        inspected = json.loads(subprocess.check_output(
            [c["paths"]["docker"], "inspect", env.container_id]))[0]
        assert inspected["Mounts"] == []
        save(folder / "container-inspect-private.json", inspected)
        model = LitellmModel(**model_cfg)
        timeout_receipt = bind_agent_http_timeout(model, c)
        save(folder / "http-client-configuration.json", timeout_receipt)
        agent = DefaultAgent(model, env, **agent_cfg)
        save(folder / "ready.json", {"instance_id": row["instance_id"], "ready_monotonic": time.monotonic()})
        barrier.wait(timeout=180)
        began = time.monotonic()
        save(folder / "arrival.json", {"scheduled_arrival_seconds": 0, "actual_start_monotonic": began})

        def timeout(signum, frame):
            raise TimeoutError("Frozen Agent wall budget reached")

        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(c["agent"]["wall_budget_seconds"])
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            status, result = agent.run(task)
        except Exception as exc:
            status, error = type(exc).__name__, str(exc)
            (folder / "agent-exception-private.txt").write_text(traceback.format_exc())
        finally:
            signal.alarm(0)
        if status == "Submitted":
            patch = result
        else:
            partial = env.execute("git diff --binary HEAD")
            save(folder / "partial-extraction-private.json", partial)
            patch = partial["output"] if partial["returncode"] == 0 else ""
        save(folder / "working-tree-status-private.json", env.execute("git status --porcelain"))
    except Exception as exc:
        error = str(exc)
        (folder / "failure-private.txt").write_text(traceback.format_exc())
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        signal.alarm(0)
        stop_observer.set()
        if observer:
            observer.join(timeout=5)
        ended = time.monotonic()
        if agent:
            save_traj(agent, folder / "trajectory-private.json", exit_status=status, result=result,
                      extra_info={"dollar_cost": "NOT_MEASURED"})
        (folder / "patch-private.diff").write_text(patch)
        report = {"instance_id": row["instance_id"], "program_id": program,
                  "status": status, "error": error, "agent_submitted": status == "Submitted",
                  "started_monotonic": began, "finished_monotonic": ended,
                  "elapsed_seconds": ended - began if began is not None else None,
                  "model_calls": agent.model.n_calls if agent else 0,
                  "patch_bytes": len(patch.encode()), "patch_sha256": digest(folder / "patch-private.diff")}
        try:
            report["program_release"] = http(c["endpoints"][arm] + "/programs/release",
                                             {"program_id": program})
        except Exception as exc:
            report["release_error"] = str(exc)
        if env and env.container_id:
            subprocess.run([c["paths"]["docker"], "rm", "-f", env.container_id],
                           capture_output=True, check=True)
            env.container_id = None
        save(folder / "report-private.json", report)
        print("AGENT_TERMINAL", json.dumps(report), flush=True)


def grade_saved_batch(c, inputs, batch_directory, output, task_offset=0):
    """Grade saved patches only, without a model call or modifying old evidence."""
    inputs, batch_directory, output = [Path(x).resolve() for x in (inputs, batch_directory, output)]
    os.environ["DOCKER_HOST"] = c["paths"]["docker_host"]
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in (inputs / "cases-private.jsonl").read_text().splitlines()]
    copies = c["batch"].get("independent_agents_per_task", 1)
    if copies > 1:
        # Official scoring keys by instance_id. Separate invocations prevent
        # independent patches for the same task from overwriting each other.
        single = json.loads(json.dumps(c))
        single["batch"]["independent_agents_per_task"] = 1
        scores, unresolved = {}, []
        for replica in range(copies):
            label = f"replica-{replica:03d}"
            graded = grade_saved_batch(single, inputs, batch_directory, output / label,
                                      task_offset=replica * len(rows))
            scores.update({label + ":" + k: v for k, v in graded["official_scores"].items()})
            unresolved.extend(label + ":" + k for k in graded["unresolved_grading"])
        result = {"official_scores": scores, "unresolved_grading": unresolved,
                  "grading_status": "OUTPUTS_PRESENT" if not unresolved else "UNRESOLVED_OUTPUTS",
                  "unique_official_tasks": len(rows), "independent_agent_instances": len(rows) * copies,
                  "model_calls_in_this_grading": 0}
        save(output / "grading-status.json", result)
        return result
    lock = read(REPOSITORY / "config/swe-bench-pro-v1-lock.json")
    official = Path(c["paths"]["official_v1_source"])
    sources = validate_grading_sources(official, rows, lock)
    patches, bindings = [], []
    for index, row in enumerate(rows):
        patch = batch_directory / f"task-{index + task_offset:03d}/patch-private.diff"
        patches.append({"instance_id": row["instance_id"], "patch": patch.read_text(), "prefix": "agent"})
        bindings.append({"instance_id": row["instance_id"], "patch_sha256": digest(patch)})
    save(output / "source-and-patch-bindings.json", {"official_sources": sources, "patches": bindings})
    save(output / "patches-private.json", patches)
    task_images = read(inputs / "input-selection.json")["tasks"]

    def check_images(label):
        images = {}
        for task in task_images:
            raw = command([c["paths"]["docker"], "image", "inspect", task["image_uri"]],
                          output / f"{label}-{len(images)}-image-private.json")
            image = json.loads(raw)[0]
            if image["Id"] != task["image_id"]:
                raise ValueError("Grading image changed from frozen Agent input: " + task["instance_id"])
            images[task["instance_id"]] = image["Id"]
        save(output / (label + "-image-ids.json"), images)

    check_images("before")
    arguments = [sys.executable, str(official / "swe_bench_pro_eval.py"),
                 "--raw_sample_path", str(inputs / "cases-private.jsonl"),
                 "--patch_path", str(output / "patches-private.json"), "--output_dir", str(output),
                 "--scripts_dir", str(official / "run_scripts"), "--dockerhub_username", "bdqnghi",
                 "--use_local_docker", "--docker_platform", "linux/arm64", "--num_workers", "1"]
    with (output / "docker-events-private.jsonl").open("w") as events:
        observer = subprocess.Popen([c["paths"]["docker"], "events", "--since", str(int(time.time())),
                                     "--filter", "type=container", "--format", "{{json .}}"],
                                    stdout=events, stderr=subprocess.STDOUT)
        try:
            command(arguments, output / "command-private.json", timeout=2400, cwd=official)
        finally:
            observer.terminate()
            observer.wait(timeout=10)
    check_images("after")
    scores = read(output / "eval_results.json")
    unresolved = [row["instance_id"] for row in rows
                  if not (output / row["instance_id"] / "workspace/output.json").is_file()]
    result = {"official_scores": scores, "unresolved_grading": unresolved,
              "grading_status": "OUTPUTS_PRESENT" if not unresolved else "UNRESOLVED_OUTPUTS",
              "model_calls_in_this_grading": 0}
    save(output / "grading-status.json", result)
    print("SAVED_PATCH_GRADING_FINISHED", json.dumps(result), flush=True)
    return result


def wait_engine_idle(c, output, timeout=180):
    """A released program is not proof that its timed-out engine request drained."""
    started = time.monotonic()
    path = Path(output) / "engine-drain-check-private.jsonl"
    with path.open("x") as stream:
        while True:
            text = http(c["endpoints"]["model"] + "/metrics", timeout=10)
            counts = {}
            for name in ["vllm:num_requests_running", "vllm:num_requests_waiting"]:
                values = re.findall(r"^" + re.escape(name) + r"(?:\{[^\n]*?\})? ([^\n ]+)", text, re.M)
                if not values:
                    raise ValueError("Missing engine drain metric: " + name)
                counts[name] = sum(float(value) for value in values)
            stream.write(json.dumps({"epoch": time.time(), "counts": counts}) + "\n")
            stream.flush()
            if all(value == 0 for value in counts.values()):
                return counts
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Prior engine requests remain; no new Agent batch started")
            time.sleep(1)


def run_batch(c, inputs, arm, output):
    output, inputs = Path(output), Path(inputs)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["DOCKER_HOST"] = c["paths"]["docker_host"]
    manifest = read(inputs / "input-selection.json")
    assert manifest["status"] == "PREPARED"
    assert digest(inputs / "cases-private.jsonl") == manifest["cases_sha256"]
    rows = [json.loads(x) for x in (inputs / "cases-private.jsonl").read_text().splitlines()]
    assert [x["instance_id"] for x in rows] == [x["instance_id"] for x in manifest["tasks"]]
    rows, task_images = expand_agent_instances(c, rows, manifest["tasks"])
    for relative, expected in manifest["official_sources"].items():
        assert digest(Path(c["paths"]["official_v1_source"]) / relative) == expected
    save(output / "settings-private.json", c)
    wait_engine_idle(c, output, c["protocol"].get("engine_drain_timeout_seconds", 180))
    save(output / "router-identity-before.json", http(c["endpoints"][arm] + "/migration-identity"))
    runtime_before = runtime_witness(c, output, "batch-before")
    command_reset = http(c["endpoints"]["model"] + "/reset_prefix_cache", {})
    save(output / "prefix-reset.json", {"response": command_reset, "epoch": time.time()})
    before_metrics = http(c["endpoints"]["model"] + "/metrics")
    save(output / "engine-metrics-before.json", before_metrics)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(rows) + 1)
    processes = []
    for index, row in enumerate(rows):
        p = context.Process(target=agent_task, args=(
            c, row, task_images[index], arm, output / f"task-{index:03d}", barrier))
        p.start()
        processes.append(p)
    stop = threading.Event()

    def observe():
        with (output / "observations-private.jsonl").open("w") as stream:
            while not stop.wait(1):
                record = {"epoch": time.time(), "monotonic": time.monotonic()}
                for name, url in [("scheduler", c["endpoints"][arm] + "/scheduler-diagnostics"),
                                  ("programs", c["endpoints"][arm] + "/programs"),
                                  ("engine", c["endpoints"]["model"] + "/metrics")]:
                    try:
                        record[name] = http(url, timeout=5)
                    except Exception as exc:
                        record[name + "_error"] = str(exc)
                stream.write(json.dumps(record) + "\n")
                stream.flush()

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        barrier.wait(timeout=180)
        start = time.monotonic()
        save(output / "batch-start.json", {"epoch": time.time(), "monotonic": start,
             "batch_size": len(rows), "client_concurrency_limit": None, "arrival_time": 0})
        print("BATCH_STARTED", arm, output.name, len(rows), flush=True)
        deadline = start + c["agent"]["wall_budget_seconds"] + 180
        while any(p.is_alive() for p in processes) and time.monotonic() < deadline:
            time.sleep(1)
        if any(p.is_alive() for p in processes):
            raise TimeoutError("Batch drain deadline exceeded; all failures retained")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)
        stop.set()
        watcher.join(timeout=15)
        # A killed preparation/Agent child cannot execute its own finally block.
        abandoned = []
        for index, row in enumerate(rows):
            folder = output / f"task-{index:03d}"
            if (folder / "report-private.json").exists():
                continue
            record = {"instance_id": row["instance_id"], "status": "INCOMPLETE_CHILD"}
            identifier = folder / "container-id.txt"
            if identifier.exists():
                removed = subprocess.run([c["paths"]["docker"], "rm", "-f", identifier.read_text()],
                                         capture_output=True, text=True)
                record["container_cleanup_exit"] = removed.returncode
                record["container_cleanup_stderr"] = removed.stderr
            try:
                record["program_release"] = http(c["endpoints"][arm] + "/programs/release",
                                                  {"program_id": program_identifier(c, row, output.name)})
            except Exception as exc:
                record["release_error"] = str(exc)
            abandoned.append(record)
        save(output / "incomplete-children-cleanup-private.json", abandoned)
    reports = [read(output / f"task-{i:03d}/report-private.json") for i in range(len(rows))]
    start = min(x["started_monotonic"] for x in reports if x["started_monotonic"] is not None)
    end = max(x["finished_monotonic"] for x in reports)
    runtime_after = runtime_witness(c, output, "batch-after")
    result = {"arm": arm, "batch_size": len(rows), "agent_batch_elapsed_seconds": end - start,
              "all_bounded_agents_terminal": all(x["status"] in ["Submitted", "LimitsExceeded"] for x in reports),
              "submitted": sum(x["agent_submitted"] for x in reports), "tasks": reports,
              "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
              "model_calls": 0, "grading_in_model_timing": False}
    try:
        result["eligible_decode_graph_replays_by_rank"] = graph_replay_delta(runtime_before, runtime_after)
    except (KeyError, ValueError, TypeError) as exc:
        result["graph_evidence_error"] = str(exc)
    for i in range(len(rows)):
        trajectory = read(output / f"task-{i:03d}/trajectory-private.json")
        for message in trajectory["messages"]:
            if message["role"] == "assistant":
                result["model_calls"] += 1
                for key in result["usage"]:
                    result["usage"][key] += message["extra"]["response"]["usage"][key]
    result["workflow_output_tokens_per_second"] = result["usage"]["completion_tokens"] / (end - start)
    result["npu_card_seconds"] = c["price"]["npu_cards"] * (end - start)
    rate = c["price"]["cny_per_card_hour"]
    result["scenario_cny"] = None if rate is None else result["npu_card_seconds"] * rate / 3600
    save(output / "batch-report.json", result)
    save(output / "engine-metrics-after.json", http(c["endpoints"]["model"] + "/metrics"))
    # Grading runs after inference and in new containers; full denominator retained.
    if c["protocol"].get("defer_grading", False):
        result.update(official_scores={},
                      unresolved_grading=[row.get("execution_id", row["instance_id"]) for row in rows],
                      grading_status="NOT_RUN_DEFERRED_OUTSIDE_PRESSURE_GATE")
    else:
        result.update(grade_saved_batch(c, inputs, output, output / "grade"))
    result["resolved_in_selected_batch"] = (
        None if result["unresolved_grading"] else sum(result["official_scores"].values()))
    save(output / "batch-report.json", result)
    print("BATCH_FINISHED", json.dumps({k: v for k, v in result.items() if k != "tasks"}), flush=True)
    return result


def run_suite(c, inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "settings-private.json", c)
    try:
        lease = validate_workspace_lease(c)
        save(output / "workspace-lease-check.json", lease)
    except ValueError as exc:
        save(output / "workspace-lease-check.json", {"status": "REJECTED", "error": str(exc)})
        raise
    # Same harmless warmup requests; never count them as Agent tasks or formal work.
    before = runtime_witness(c, output, "warmup-before")
    warmups = []
    for index in range(c["protocol"]["warmup_requests"]):
        warmups.append(http(c["endpoints"]["model"] + "/v1/chat/completions", {
            "model": "Qwen3.5-35B-A3B", "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 16, "temperature": 0, "seed": c["agent"]["seed"],
            "chat_template_kwargs": {"enable_thinking": False}}))
    save(output / "warmups-private.json", warmups)
    after = runtime_witness(c, output, "warmup-after")
    try:
        delta = graph_replay_delta(before, after)
        save(output / "warmup-graph-check.json", {"status": "ACTUAL_REPLAY_OBSERVED", "delta": delta})
    except (KeyError, ValueError, TypeError) as exc:
        save(output / "warmup-graph-check.json", {"status": "NOT_QUALIFIED", "error": str(exc)})
        raise RuntimeError("No new Agent batch started: actual unified graph proof is missing") from exc
    for pair, arms in enumerate(c["protocol"]["pair_order"]):
        for arm in arms:
            run_batch(c, inputs, arm, output / f"pair-{pair + 1}-{arm}")
            summarize(output)


def summarize(output):
    output = Path(output)
    batches = []
    for path in sorted(output.glob("pair-*/batch-report.json")):
        row = read(path)
        batches.append({"run": path.parent.name, **{k: v for k, v in row.items() if k != "tasks"}})
    result = {"scope": "bounded real-Agent workload comparison; not a full dataset or formal cost qualification",
              "batches": batches, "complete_paired_repetitions": min(
                  sum(x["arm"] == "native" for x in batches), sum(x["arm"] == "controller" for x in batches))}
    for arm in ("native", "controller"):
        selected = [x for x in batches if x["arm"] == arm]
        if selected:
            result[arm] = {"median_agent_batch_seconds": statistics.median(x["agent_batch_elapsed_seconds"] for x in selected),
                           "batch_count": len(selected),
                           "all_bounded_agents_terminal": all(x["all_bounded_agents_terminal"] for x in selected),
                           "total_model_calls": sum(x["model_calls"] for x in selected),
                           "total_output_tokens": sum(x["usage"]["completion_tokens"] for x in selected)}
    if "native" in result and "controller" in result:
        result["median_elapsed_change_percent"] = 100 * (
            result["controller"]["median_agent_batch_seconds"] / result["native"]["median_agent_batch_seconds"] - 1)
        result["causal_performance_qualification"] = "PENDING workload/quality/error/mechanism audit; elapsed change alone is insufficient"
    save(output / "summary.json", result)
    return result
