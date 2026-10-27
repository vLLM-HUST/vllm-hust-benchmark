"""Internal MBPP namespace worker, never a host-side code evaluator."""

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys


def namespace_helper():
    path = Path(__file__).with_name("humaneval_isolated_worker.py")
    spec = importlib.util.spec_from_file_location("existing_namespace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def isolation_guard(helper):
    require = helper.require
    require(sys.platform == "linux" and os.getpid() == 1, "not Linux namespace PID1")
    require(
        not any(Path(p).exists() for p in ("/data", "/root", "/proc/self")),
        "host paths exposed",
    )
    require(not list(Path("/dev").glob("davinci*")), "NPU device exposed")
    libc = ctypes.CDLL(None)
    header, data = helper.capabilities()
    caps = (data * 2)()
    require(
        libc.capget(ctypes.byref(header(0x20080522, 0)), caps) == 0, "capget failed"
    )
    require(
        all(not x.effective and not x.permitted and not x.inheritable for x in caps),
        "capabilities nonempty",
    )
    require(
        all(libc.prctl(23, c, 0, 0, 0) in (0, -1) for c in range(64)),
        "bounding capabilities nonempty",
    )
    require(libc.prctl(39, 0, 0, 0, 0) == 1, "no_new_privs missing")
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("1.1.1.1", 443))
    except OSError:
        pass
    else:
        raise RuntimeError("network reachable")
    finally:
        sock.close()
    try:
        Path("/inputs/SHOULD_NOT_EXIST").write_text("probe")
    except OSError:
        pass
    else:
        raise RuntimeError("input mount writable")


def assemble(tasks):
    """Preserve full original setup/tests; no generated-code repair or extraction."""
    if not isinstance(tasks, list) or len(tasks) < 3:
        raise ValueError("at least three fixed original tasks required")
    ids = [item.get("task_id") for item in tasks]
    if len(ids) != len(set(ids)) or any(
        type(i) is not int or not 11 <= i <= 510 for i in ids
    ):
        raise ValueError("distinct original full test task IDs required")
    references, candidates, gold = [], [], []
    for item in tasks:
        task = item["original_task"]
        tests = task.get("test_list")
        if (
            task.get("task_id") != item["task_id"]
            or not isinstance(tests, list)
            or len(tests) != 3
            or any(not isinstance(test, str) for test in tests)
            or not isinstance(task.get("test_setup_code"), str)
            or not isinstance(task.get("code"), str)
            or not isinstance(item.get("completion"), str)
            or item.get("finish_reason") not in {"stop", "length"}
        ):
            raise ValueError("complete unchanged original code task required")
        setup = task["test_setup_code"] + "\n"
        references.append("\n".join(tests))
        candidates.append([setup + item["completion"]])
        gold.append([setup + task["code"]])
    return references, candidates, gold


def checked_results(results, count):
    if set(results) != set(range(count)):
        raise ValueError("CodeEval result coverage mismatch")
    flags = []
    for index in range(count):
        items = results[index]
        if (
            len(items) != 1
            or items[0][0] != 0
            or items[0][1].get("task_id") != index
            or items[0][1].get("completion_id") != 0
            or type(items[0][1].get("passed")) is not bool
        ):
            raise ValueError("one original unfiltered completion per task required")
        flags.append(items[0][1]["passed"])
    return flags


def evaluate():
    helper = namespace_helper()
    isolation_guard(helper)
    helper.require(
        sys.version_info[:2] == (3, 10), "dedicated Python3.10 scorer runtime required"
    )
    pins = json.loads(Path("/inputs/scoring-pins.json").read_text("utf-8"))
    modules = Path("/author/metrics")
    expected = {Path(name).name: digest for name, digest in pins["files"].items()}
    helper.require(
        set(expected) == {p.name for p in modules.glob("*.py")},
        "scorer module inventory mismatch",
    )
    for name, expected_sha in expected.items():
        helper.require(
            hashlib.sha256((modules / name).read_bytes()).hexdigest() == expected_sha,
            "scorer identity mismatch",
        )
    original = Path("/inputs/tasks-private.json")
    raw = original.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    preparation = json.loads(
        Path("/inputs/preparation-receipt.json").read_text("utf-8")
    )
    helper.require(
        digest == preparation["tasks_sha256"], "original scoring inputs differ"
    )
    tasks = json.loads(raw)
    helper.require(
        [item["task_id"] for item in tasks] == preparation["task_ids"],
        "task mapping differs",
    )
    references, candidates, gold = assemble(tasks)
    sys.path[:0] = ["/author", "/packages"]
    import numpy
    from metrics.code_eval import compute_code_eval

    helper.require(
        numpy.__version__ == pins["client_versions"]["numpy"],
        "scorer numpy version mismatch",
    )
    helper.require(
        str(Path(numpy.__file__).resolve()).startswith("/packages/numpy/"),
        "numpy import escaped",
    )
    helper.require(
        (pins["k"], pins["num_workers"], pins["timeout_seconds"]) == ([1], 1, 3.0),
        "unreviewed evaluator parameters",
    )
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    kwargs = {
        "references": references,
        "k": pins["k"],
        "num_workers": pins["num_workers"],
        "timeout": pins["timeout_seconds"],
    }
    gold_metric, gold_results = compute_code_eval(predictions=gold, **kwargs)
    gold_flags = checked_results(gold_results, len(tasks))
    Path("/output/reference-check-private.json").write_text(
        json.dumps({"metric": gold_metric, "results": gold_results}), encoding="utf-8"
    )
    helper.require(
        all(gold_flags), "original reference failed; no test edits or host fallback"
    )
    metric, results = compute_code_eval(predictions=candidates, **kwargs)
    passed = checked_results(results, len(tasks))
    helper.require(
        set(metric) == {"pass@1"}
        and abs(float(metric["pass@1"]) - sum(passed) / len(passed)) < 1e-12,
        "pass@1 parity mismatch",
    )
    helper.require(
        hashlib.sha256(original.read_bytes()).hexdigest() == digest,
        "scoring inputs changed",
    )
    Path("/output/code-eval-private.json").write_text(
        json.dumps({"metric": metric, "results": results}), encoding="utf-8"
    )
    receipt = {
        "source_id": "MBPP",
        "status": "SELECTED_ORIGINAL_TESTS_CODEEVAL_SCORED",
        "task_ids": [item["task_id"] for item in tasks],
        "tasks_sha256": digest,
        "evaluator_revision": pins["evaluator_revision"],
        "metric": {"pass@1": float(metric["pass@1"])},
        "passed": passed,
        "original_reference_checks_passed": True,
        "budget_terminated_task_ids": [
            item["task_id"] for item in tasks if item["finish_reason"] == "length"
        ],
        "challenge_tests_executed": False,
        "generated_code_executed": True,
        "isolation_checks_passed_before_import_and_execution": True,
        "new_model_calls": 0,
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "source_checkbox_eligible": False,
    }
    Path("/output/receipt-public.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--evaluate"]:
        evaluate()
    elif len(sys.argv) == 3 and sys.argv[1] == "--inside":
        namespace_helper().inside(json.loads(sys.argv[2]))
    else:
        raise SystemExit("internal worker; use run_mbpp_isolated_scoring.py")
