"""Original OVO scorer with a disclosed lossless singleton-text input adapter.

No model calls. Preserve raw predictions; no label-based extraction, joining,
stripping, answer rewriting, task filtering or author scorer changes.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare_jsonschemabench_author_source import verify_source
from prepare_ovobench_native_selection import ROOT
from run_ovobench_original_npu import scorer_text
from verify_ovobench_original_media import PROFILE


def projection(original):
    if not isinstance(original, dict) or not all(
        isinstance(original.get(key), list)
        for key in ("backward", "realtime", "forward")
    ):
        raise ValueError("original three-category prediction document required")
    result = deepcopy(original)
    slots = []
    for key in ("backward", "realtime"):
        slots.extend(result[key])
    for row in result["forward"]:
        slots.extend(row["test_info"])
    converted = 0
    for slot in slots:
        value = slot["response"]
        if value is None:
            continue  # Original failed response remains None and is scored as zero.
        text = scorer_text(value)
        converted += isinstance(value, list)
        slot["response"] = text
    return result, {
        "response_slots": len(slots),
        "singleton_lists_unwrapped": converted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["QWen2VL_7B", "QWen2VL_72B"], default="QWen2VL_7B"
    )
    args = parser.parse_args()
    author = verify_source(args.author_root, PROFILE)
    if args.prediction_file.stat().st_size > 64 * 1024**2:
        parser.error("unexpectedly large original prediction document")
    raw = args.prediction_file.read_bytes()
    result, counts = projection(json.loads(raw))
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("private original predictions must stay outside the repository")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    folder = output / "results" / args.model
    folder.mkdir(parents=True)
    (folder / "predictions.json").write_text(
        json.dumps(result, ensure_ascii=False), "utf-8"
    )
    command = [
        sys.executable,
        str(author / "score.py"),
        "--result_dir",
        str(output / "results"),
        "--model",
        args.model,
        "--mode",
        "offline",
    ]
    score = subprocess.run(
        command, cwd=author, capture_output=True, text=True, timeout=120
    )
    (output / "official-score-private.log").write_text(
        score.stdout + score.stderr, "utf-8"
    )
    verify_source(author, PROFILE)
    if args.prediction_file.read_bytes() != raw:
        raise RuntimeError("original predictions changed during scoring")
    report = {
        "source_id": "OVOBENCH",
        "official_score_exit_code": score.returncode,
        "original_prediction_sha256": hashlib.sha256(raw).hexdigest(),
        **counts,
        "original_prediction_unchanged": True,
        "author_scorer_unchanged": True,
        "adapter": "singleton decoded string list to identical string; None preserved",
        "new_model_calls": 0,
        "native_generation_verified_by_this_command": False,
        "source_checkbox_eligible": False,
    }
    (output / "qualification-public.json").write_text(
        json.dumps(report, indent=2), "utf-8"
    )
    print(json.dumps(report))
    raise SystemExit(score.returncode)


if __name__ == "__main__":
    main()
