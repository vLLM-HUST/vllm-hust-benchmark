"""Internal Docker/Podman worker. Fail closed before importing official scoring."""

import ctypes
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def assert_isolation(host_namespaces):
    require(
        sys.platform == "linux" and os.getpid() == 1 and os.geteuid() != 0,
        "private nonroot PID1 required",
    )
    for name in ("user", "mnt", "net", "pid"):
        # Docker can share the host user namespace while running a nonroot UID.
        if name == "user":
            continue
        require(
            os.readlink(f"/proc/self/ns/{name}") != host_namespaces[name],
            "host namespace exposed",
        )
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    require(
        all(
            int(status[key].strip(), 16) == 0
            for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        ),
        "capabilities exposed",
    )
    require(status["NoNewPrivs"].strip() == "1", "no_new_privs required")
    require(ctypes.CDLL(None).prctl(39, 0, 0, 0, 0) == 1, "no_new_privs check failed")
    require(not any(Path("/dev").glob("davinci*")), "NPU devices exposed")
    for name in (
        "/inputs/SHOULD_NOT_EXIST",
        "/author/SHOULD_NOT_EXIST",
        "/SHOULD_NOT_EXIST",
    ):
        try:
            Path(name).write_text("probe")
        except OSError:
            pass
        else:
            raise RuntimeError("required read-only mount writable")
    connection = socket.socket()
    connection.settimeout(1)
    try:
        connection.connect(("1.1.1.1", 443))
    except OSError:
        pass
    else:
        raise RuntimeError("external network reachable")
    finally:
        connection.close()


def qualify_environment(requirements):
    require(sys.version_info[:2] == (3, 10), "official image Python3.10 required")
    versions = {}
    for line in Path(requirements).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, expected = line.partition("==")
        require(not any(c in name for c in " ;<>!@/"), "unsupported requirement syntax")
        actual = metadata.version(name)
        require(
            not separator or actual == expected,
            "official evaluation dependency version mismatch",
        )
        versions[name.lower().replace("_", "-")] = actual
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, timeout=120)
    return versions


def qualify_author_entrypoints():
    """Check real CLI imports inside the qualified container, not on the host."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; importlib.import_module('bigcodebench.sanitize'); "
            "importlib.import_module('bigcodebench.evaluate')",
        ],
        check=True,
        timeout=120,
    )


def validate_results(evaluation, metrics, ids):
    require(
        metrics.get("gt_pass_rate") == 1.0 and metrics.get("failed_tasks") == [],
        "reference implementation/environment checks failed",
    )
    results = evaluation.get("eval", {})
    require(set(results) == set(ids), "official result task coverage mismatch")
    for task in ids:
        require(
            isinstance(results[task], list) and len(results[task]) == 1,
            "one original completion per task required",
        )
        require(
            results[task][0].get("task_id") == task,
            "official result original ID mismatch",
        )
        require(
            results[task][0].get("status") in {"pass", "fail", "timeout"},
            "unknown official status",
        )
    value = metrics.get("pass@1")
    require(type(value) in (int, float) and 0 <= value <= 1, "official pass@1 missing")
    expected = sum(results[task][0]["status"] == "pass" for task in ids) / len(ids)
    require(abs(value - expected) < 1e-9, "official pass@1/result inconsistency")
    return {"selected_count": len(ids), "official_pass_at_1": value}


def validate_sanitized_selection(path, selected_ids):
    """Reject changed selection before executing official functional tests."""
    records = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    require(
        all(isinstance(record, dict) for record in records),
        "official sanitizer returned non-object records",
    )
    require(
        [record.get("task_id") for record in records] == selected_ids,
        "official sanitizer dropped/reordered original tasks",
    )


def run_official_steps(commands, selected_ids, sanitized):
    require(len(commands) == 2, "exact author sanitize/evaluate steps required")
    subprocess.run([sys.executable, *commands[0][1:]], check=True)
    validate_sanitized_selection(sanitized, selected_ids)
    subprocess.run([sys.executable, *commands[1][1:]], check=True)


def evaluate():
    control = json.loads(Path("/inputs/control-private.json").read_text())
    assert_isolation(control["host_namespaces"])
    plan = control["plan"]
    for name, expected in plan["publisher_source_checksums"].items():
        require(
            hashlib.sha256((Path("/author") / name).read_bytes()).hexdigest()
            == expected,
            "staged author source identity changed",
        )
    for name, expected in plan["prepared_file_sha256"].items():
        path = (
            Path("/inputs") / name
            if name.startswith("original-dataset")
            else Path("/output") / name
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            "staged original input identity changed",
        )
    versions = qualify_environment("/author/Requirements/requirements-eval.txt")
    qualify_author_entrypoints()
    require(
        not Path("/output/fresh-cache").exists(), "fresh groundtruth cache required"
    )
    sanitized = Path("/output/raw-solutions-private-sanitized.jsonl")
    selected_ids = plan["selected_task_ids"]
    run_official_steps(plan["commands"], selected_ids, sanitized)
    base = "/output/raw-solutions-private-sanitized"
    evaluation = json.loads(Path(base + "_eval_results.json").read_text())
    metrics = json.loads(Path(base + "_pass_at_k.json").read_text())
    summary = validate_results(evaluation, metrics, selected_ids)
    Path("/output/receipt-public.json").write_text(
        json.dumps(
            {
                "source_id": "BIGCODEBENCH",
                "status": "SELECTED_ORIGINAL_TASKS_OFFICIALLY_SCORED",
                "selected_task_ids": selected_ids,
                "official_sanitize_and_reference_checks_executed": True,
                "official_evaluation_dependency_versions": versions,
                "official_cli_imports_qualified_before_task_execution": True,
                "new_model_calls": 0,
                "source_checkbox_eligible": False,
                "delivery_note": "Native scoring only; final public-entry/research-consumer closure remains separate.",
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--evaluate"]:
        raise SystemExit("internal container worker requires explicit --evaluate")
    evaluate()
