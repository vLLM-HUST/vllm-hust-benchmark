"""Internal original-task evaluator; no candidate execution on host."""

import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


class OriginalTaskSelection:
    """Iterate selected tasks but resolve metadata by unchanged original ID."""

    def __init__(self, original, indices):
        if (
            len(indices) < 3
            or any(type(i) is not int or i < 0 or i >= len(original) for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("valid fixed original task positions required")
        self.original, self.indices = original, indices

    def __iter__(self):
        return (self.original[i] for i in self.indices)

    def __getitem__(self, original_id):
        return self.original[original_id]

    def __len__(self):
        return len(self.indices)


def _probe_append(values, number):
    values.append(number)


def _probe_initializer(barrier):
    global _probe_barrier
    _probe_barrier = barrier


def _probe_worker(number):
    with multiprocessing.Manager() as manager:
        values = manager.list()
        _probe_barrier.wait(timeout=30)
        process = multiprocessing.Process(target=_probe_append, args=(values, number))
        process.start()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()
            raise RuntimeError("native process preflight did not terminate")
        if process.exitcode != 0 or list(values) != [number]:
            raise RuntimeError("native process/thread preflight failed")
    return number


def concurrency_preflight():
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(16)
    with ProcessPoolExecutor(
        max_workers=16,
        mp_context=context,
        initializer=_probe_initializer,
        initargs=(barrier,),
    ) as executor:
        if sorted(executor.map(_probe_worker, range(16))) != list(range(16)):
            raise RuntimeError("author 16-worker process layout unavailable")


def evaluate():
    sys.path.insert(0, "/inputs")
    from native_python_namespace import capabilities, require

    require(os.getpid() == 1, "not namespace PID1")
    require(
        not any(
            Path(p).exists()
            for p in (
                "/data",
                "/root",
                "/proc/self/status",
                "/proc/self/environ",
                "/proc/1",
            )
        ),
        "host paths exposed",
    )
    require(
        Path("/proc/self/exe").is_symlink()
        and os.readlink("/proc/self/exe") == sys.executable,
        "runtime executable shim mismatch",
    )
    require(not list(Path("/dev").glob("davinci*")), "NPU devices exposed")
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
    require(libc.prctl(39, 0, 0, 0, 0) == 1, "no_new_privs missing")
    require(
        all(libc.prctl(23, c, 0, 0, 0) in (0, -1) for c in range(64)),
        "bounding capabilities nonempty",
    )
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
        raise RuntimeError("inputs writable")
    sys.path.insert(0, "/author")
    os.chdir("/author")
    import test_ds1000 as author
    import numpy
    import pandas
    import tqdm

    require(sys.version_info[:2] == (3, 10), "author Python3.10 required")
    require(
        numpy.__version__ == "1.26.4" and pandas.__version__ == "1.5.3",
        "author dependency mismatch",
    )
    require(tqdm.__version__ == "4.67.1", "qualified tqdm version mismatch")
    original = author.ds1000
    require(len(original) == 1000, "original1000 required")
    selection = json.loads(Path("/inputs/selection.json").read_text())
    indices = selection["problem_ids"]
    selected = OriginalTaskSelection(original, indices)
    require(
        [int(r["metadata"]["problem_id"]) for r in selected] == indices,
        "fixed task IDs changed",
    )
    require(
        sorted({r["metadata"]["library"] for r in selected}) == selection["libraries"],
        "selected native libraries changed",
    )
    qualified = []
    if set(selection["libraries"]) - {"Pandas", "Numpy"}:
        # Trusted dependency probes in separate namespace-local processes. Do
        # not initialize TensorFlow in the parent before the author's fork pool.
        # A broken framework fails the run before any candidate is evaluated.
        probes = {
            "scipy": "import scipy",
            "sklearn": "import sklearn",
            "matplotlib": "import matplotlib",
            "gensim": "import gensim",
            "seaborn": "import seaborn",
            "statsmodels": "import statsmodels.api",
            "xgboost": "import xgboost",
            "yaml": "import yaml",
            "torch": "import torch; assert (torch.tensor([1])+1).tolist()==[2]",
            "tensorflow": "import tensorflow as tf; assert (tf.constant([1])+1).numpy().tolist()==[2]",
        }
        for module, program in probes.items():
            subprocess.run(
                [sys.executable, "-I", "-c", program], check=True, timeout=60
            )
            qualified.append(module)
    samples = Path("/inputs/answers-private.jsonl")
    raw = samples.read_bytes()
    rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    require([r["id"] for r in rows] == indices, "answer binding mismatch")
    answers = {r["id"]: author.postprocess(r["code"]) for r in rows}
    concurrency_preflight()
    # Fixed native sampling: bind original tasks, never edit test contexts or
    # substitute placeholders for unselected tasks. Author functions unchanged.
    author.ds1000 = selected
    summary = author.eval_ds1000(answers)
    require(samples.read_bytes() == raw, "original completions changed")
    count = re.search(r"^count\s+([0-9.]+)\s*$", summary, re.MULTILINE)
    mean = re.search(r"^mean\s+([0-9.]+)\s*$", summary, re.MULTILINE)
    require(
        count is not None
        and float(count.group(1)) == len(indices)
        and mean is not None,
        "incomplete publisher summary",
    )
    Path("/output/author-summary-private.txt").write_text(summary)
    receipt = {
        "source_id": "DS1000",
        "status": "SELECTED_ORIGINAL_DS1000_TASKS_OFFICIALLY_SCORED",
        "original_problem_ids": indices,
        "selected_libraries": selection["libraries"],
        "qualified_dependency_probes": qualified,
        "original_rows_loaded": 1000,
        "official_summary_count": len(indices),
        "official_summary_mean_rounded": float(mean.group(1)),
        "official_summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
        "samples_sha256": hashlib.sha256(raw).hexdigest(),
        "official_timeout_seconds": 120,
        "official_max_workers": 16,
        "author_16_worker_concurrency_preflight_passed": True,
        "official_postprocess_used": True,
        "original_functions_modified": False,
        "isolation_checks_passed_before_author_import": True,
        "generated_code_executed": True,
        "new_model_calls": 0,
        "full1000_execution": indices == list(range(1000)),
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
    }
    Path("/output/receipt-public.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--evaluate"]:
        evaluate()
    elif len(sys.argv) == 3 and sys.argv[1] == "--inside":
        from native_python_namespace import inside

        inside(json.loads(sys.argv[2]))
    else:
        raise SystemExit("use run_ds1000_isolated_scoring.py")
