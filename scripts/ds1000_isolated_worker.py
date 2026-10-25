"""Internal fixed-original-Pandas evaluator; no candidate execution on host."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys


def evaluate():
    sys.path.insert(0, "/inputs")
    from native_python_namespace import capabilities, require

    require(os.getpid() == 1, "not namespace PID1")
    require(
        not any(Path(p).exists() for p in ("/data", "/root", "/proc/self")),
        "host paths exposed",
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
    selected = original[:3]
    require(
        [int(r["metadata"]["problem_id"]) for r in selected] == [0, 1, 2],
        "fixed task IDs changed",
    )
    require(
        all(r["metadata"]["library"] == "Pandas" for r in selected),
        "profile library changed",
    )
    samples = Path("/inputs/answers-private.jsonl")
    raw = samples.read_bytes()
    rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    require([r["id"] for r in rows] == [0, 1, 2], "answer binding mismatch")
    answers = [author.postprocess(r["code"]) for r in rows]
    # Fixed native sampling: bind original tasks, never edit test contexts or
    # substitute placeholders for unselected tasks. Author functions unchanged.
    author.ds1000 = selected
    summary = author.eval_ds1000(answers)
    require(samples.read_bytes() == raw, "original completions changed")
    count = re.search(r"^count\s+([0-9.]+)\s*$", summary, re.MULTILINE)
    mean = re.search(r"^mean\s+([0-9.]+)\s*$", summary, re.MULTILINE)
    require(
        count is not None and float(count.group(1)) == 3 and mean is not None,
        "incomplete publisher summary",
    )
    Path("/output/author-summary-private.txt").write_text(summary)
    receipt = {
        "source_id": "DS1000",
        "status": "ORIGINAL_FIRST3_PANDAS_OFFICIALLY_SCORED",
        "original_problem_ids": [0, 1, 2],
        "original_rows_loaded": 1000,
        "official_summary_count": 3,
        "official_summary_mean_rounded": float(mean.group(1)),
        "official_summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
        "samples_sha256": hashlib.sha256(raw).hexdigest(),
        "official_timeout_seconds": 120,
        "official_max_workers": 16,
        "official_postprocess_used": True,
        "original_functions_modified": False,
        "isolation_checks_passed_before_author_import": True,
        "generated_code_executed": True,
        "new_model_calls": 0,
        "full1000_or_all_libraries_claim": False,
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
