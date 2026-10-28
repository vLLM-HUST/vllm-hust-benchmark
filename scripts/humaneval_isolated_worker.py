"""Internal Linux namespace worker for run_humaneval_isolated_scoring.py.

Never executes candidate code unless the post-exec isolation checks pass.
This is a bounded local evaluation sandbox, not a security certification.
"""

import ctypes
import json
import os
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def capabilities():
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    return Header, Data


def inside(config):
    import resource
    import subprocess

    root, assets = Path(config["root"]), Path(config["assets"])

    def call(*args):
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

    call("mount", "--make-rprivate", "/")
    call("mount", "-t", "tmpfs", "-o", "size=64m,nosuid,nodev", "tmpfs", str(root))
    targets = [
        "usr/lib",
        "usr/lib64",
        "proc",
        "tmp",
        "dev/shm",
        "inputs",
        "author",
        "packages",
        "output",
        config["runtime"].lstrip("/"),
    ]
    for name in targets:
        (root / name).mkdir(parents=True, exist_ok=True)

    def bind(source, target, readonly=True):
        call("mount", "--bind", source, str(root / target))
        if readonly:
            call("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / target))

    for directory in ("/usr/lib", "/usr/lib64"):
        if Path(directory).is_dir():
            bind(directory, directory.lstrip("/"))
    bind(config["runtime"], config["runtime"].lstrip("/"))
    for directory in ("inputs", "author", "packages"):
        bind(str(assets / directory), directory)
    bind(str(assets / "output"), "output", readonly=False)
    (root / "lib").symlink_to("usr/lib")
    (root / "lib64").symlink_to("usr/lib64")
    call(
        "mount",
        "-t",
        "tmpfs",
        "-o",
        "size=16m,nosuid,nodev",
        "tmpfs",
        str(root / "dev/shm"),
    )
    for name, value in (
        ("RLIMIT_NPROC", 64),
        ("RLIMIT_CPU", 30),
        ("RLIMIT_AS", 2 * 1024**3),
        ("RLIMIT_FSIZE", 16 * 1024**2),
        ("RLIMIT_NOFILE", 128),
    ):
        resource.setrlimit(getattr(resource, name), (value, value))
    require(
        Path("/proc/self/uid_map").read_text().split()
        == ["0", str(config["uid"]), "1"],
        "unexpected host UID mapping",
    )
    os.chdir(root)
    os.chroot(".")
    os.chdir("/")
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(38, 1, 0, 0, 0) == 0, "cannot set no_new_privs")
    require(libc.prctl(28, 15, 0, 0, 0) == 0, "cannot lock root capability behavior")
    for cap in range(64):
        result = libc.prctl(24, cap, 0, 0, 0)
        require(
            result == 0 or ctypes.get_errno() == 22,
            "cannot drop capability bounding set",
        )
    header, data = capabilities()
    require(
        libc.capset(ctypes.byref(header(0x20080522, 0)), (data * 2)()) == 0,
        "cannot clear capabilities",
    )
    executable = config["runtime"] + "/bin/python3"
    os.execve(
        executable,
        [executable, "-I", "/inputs/worker.py", "--evaluate"],
        {
            "PATH": config["runtime"] + "/bin",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )


def evaluate():
    import hashlib
    import shutil
    import socket

    require(os.getpid() == 1, "not namespace PID1")
    require(
        not any(Path(p).exists() for p in ("/data", "/root", "/proc/self")),
        "host paths exposed",
    )
    require(not list(Path("/dev").glob("davinci*")), "NPU device exposed")
    libc = ctypes.CDLL(None)
    header, data = capabilities()
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
    # Test writes against a task-local read-only input mount, never a host runtime.
    try:
        Path("/inputs/SHOULD_NOT_EXIST").write_text("probe")
    except OSError:
        pass
    else:
        raise RuntimeError("input mount writable")
    sys.path[:0] = ["/author", "/packages"]
    from human_eval.data import read_problems, stream_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    original = Path("/inputs/samples-private.jsonl")
    samples = Path("/output/samples-private.jsonl")
    shutil.copyfile(original, samples)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    problems = read_problems("/inputs/problems.jsonl.gz")
    require(
        [r["task_id"] for r in stream_jsonl(str(samples))] == list(problems),
        "sample/problem selection mismatch",
    )
    metric = evaluate_functional_correctness(
        str(samples),
        k=[1],
        n_workers=1,
        timeout=3.0,
        problem_file="/inputs/problems.jsonl.gz",
    )
    results = list(stream_jsonl(str(samples) + "_results.jsonl"))
    require(len(results) == len(problems), "incomplete official results")
    require(
        hashlib.sha256(original.read_bytes()).hexdigest() == digest, "input changed"
    )
    receipt = {
        "source_id": "HUMANEVAL",
        "status": "SELECTED_ORIGINAL_TASKS_OFFICIALLY_SCORED",
        "original_task_ids": list(problems),
        "official_metric": metric,
        "passed": [r["passed"] for r in results],
        "result_categories": [
            (
                "passed"
                if r["passed"]
                else ("timed_out" if r["result"] == "timed out" else "failed")
            )
            for r in results
        ],
        "samples_sha256": digest,
        "generated_code_executed": True,
        "isolation_checks_passed_before_import_and_execution": True,
        "new_model_calls": 0,
        "official_timeout_seconds": 3.0,
        "official_n_workers": 1,
        "raw_payloads_published": False,
    }
    Path("/output/receipt-public.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--evaluate"]:
        evaluate()
    elif len(sys.argv) == 3 and sys.argv[1] == "--inside":
        inside(json.loads(sys.argv[2]))
    else:
        raise SystemExit("internal worker; use run_humaneval_isolated_scoring.py")
