"""Seal a completed candidate screen, keeping acceptance separate from screening."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from verify_aligned_pair import verify


def package(root, run, output):
    receipt = json.loads((root / run / "receipt.json").read_text())
    command_path = root / "launch-candidate-command.json"
    command = json.loads(command_path.read_text())
    assert command[command.index("--max-num-seqs") + 1] == "32"
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["state_resident_seats"] == 36
    assert additional["state_cache_host_bytes"] == 8589934592
    assert command[command.index("--kv-cache-memory-bytes") + 1] == "21474836480"
    result = {
        "schema": "b1-candidate-screen/v1",
        "project": "vLLM-HUST",
        "candidate": "BetterScale E32/R36, school runtime, MTP2 FULL APC async",
        "run_id": run,
        "status": receipt["status"],
        "selected_as_b1": False,
        "quality_rescored": False,
        "role": "Exploratory width screen; requires independent performance and quality confirmation",
        "receipt": receipt,
    }
    if receipt["status"] == "measured":
        measured = verify(root, run)
        control = verify(root, "mmlu-b1-r2")
        baseline = verify(root, "mmlu-b0-r2")
        raw = json.loads((root / run / "mmlu-pro.json").read_text())
        prior = json.loads((root / "mmlu-b1-r2/mmlu-pro.json").read_text())
        assert raw["input_lens"] == prior["input_lens"]
        result.update(
            measured=measured,
            prior_candidate_run="mmlu-b1-r2",
            prior_candidate_output_tps=control["output_tps"],
            baseline_run="mmlu-b0-r2",
            baseline_output_tps=baseline["output_tps"],
            throughput_change_vs_prior_pct=100
            * (measured["output_tps"] / control["output_tps"] - 1),
            throughput_change_vs_baseline_pct=100
            * (measured["output_tps"] / baseline["output_tps"] - 1),
        )
    files = sorted((root / run).iterdir())
    files += [root / name for name in receipt["input_hashes"]]
    files.append(command_path)
    archive = output / "b1-e32-screen-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(set(files)):
            if path.is_file():
                tar.add(
                    path,
                    arcname=str(
                        Path("20261019-midterm-aligned") / path.relative_to(root)
                    ),
                )
    result["archive"] = {
        "file": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))
            if p.is_file()
        },
    }
    (output / "b1-e32-screen.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    package(args.root, args.run, args.output)
