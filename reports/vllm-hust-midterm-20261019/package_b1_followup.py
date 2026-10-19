"""Seal a completed candidate screen, keeping acceptance separate from screening."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from verify_aligned_pair import verify


def package(root, run, output, candidate):
    assert candidate in {"e24", "kvmat"}
    receipt = json.loads((root / run / "receipt.json").read_text())
    command_path = root / f"launch-{candidate}-command.json"
    command = json.loads(command_path.read_text())
    assert command[command.index("--max-num-seqs") + 1] == (
        "24" if candidate == "e24" else "16"
    )
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["state_resident_seats"] == 36
    assert additional["state_cache_host_bytes"] == 8589934592
    assert command[command.index("--kv-cache-memory-bytes") + 1] == "21474836480"
    result = {
        "schema": "b1-candidate-screen/v1",
        "project": "vLLM-HUST",
        "candidate": (
            "E24/R36"
            if candidate == "e24"
            else "E16/R36 plus KV materialization recompute"
        )
        + ", school runtime, MTP2 FULL APC async",
        "run_id": run,
        "status": receipt["status"],
        "selected_as_b1": False,
        "quality_rescored": False,
        "role": "Same-workload candidate screen; quality is not rescored",
        "receipt": receipt,
    }
    if receipt["status"] == "measured":
        measured = verify(root, run)
        control = verify(root, "mmlu-b1-r2")
        baseline = verify(root, "mmlu-b0-r2")
        raw = json.loads((root / run / "mmlu-pro.json").read_text())
        prior = json.loads((root / "mmlu-b1-r2/mmlu-pro.json").read_text())
        assert raw["input_lens"] == prior["input_lens"]
        assert raw["output_lens"] == prior["output_lens"]
        assert len(raw["generated_texts"]) == len(prior["generated_texts"]) == 200
        result["bounded_output_comparison"] = {
            "exact_matches_vs_e16": sum(
                a == b for a, b in zip(raw["generated_texts"], prior["generated_texts"])
            ),
            "denominator": 200,
            "scope": "256-token output equality, not MMLU answer accuracy or quality equivalence",
        }
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
        if candidate == "kvmat":
            policy = [
                json.loads(line)
                for line in (root / run / "kvmat-policy.jsonl").read_text().splitlines()
            ]
            engine = [
                json.loads(line)
                for line in (root / run / "kvmat-engine.jsonl").read_text().splitlines()
            ]
            admitted = [
                row
                for row in engine
                if row.get("event") == "school028_request_salt_admitted"
            ]
            assert len(policy) == len(admitted) == 200
            assert len({row["request_id"] for row in admitted}) == 200
            assert (
                sum(row["prompt_tokens"] for row in admitted)
                == raw["total_input_tokens"]
            )
            assert all(
                row["runtime_effective_decision"] == "recompute" for row in policy
            )
            assert sum(row["reused_tokens"] for row in policy) == 0
            assert {
                hashlib.sha256(row["runtime_cache_salt"].encode()).hexdigest()
                for row in policy
            } == {row["cache_salt_sha256"] for row in admitted}
            result["control_evidence"] = {
                "policy_observations": len(policy),
                "engine_salt_admissions": len(admitted),
                "partial_reuse": "not-exercised",
                "policy_engine_salts_match": True,
                "engine_prompt_tokens": sum(row["prompt_tokens"] for row in admitted),
                "scope": "Request salt admission is not evidence of cache reuse or avoided computation",
            }
    files = sorted((root / run).iterdir())
    files += [root / name for name in receipt["input_hashes"]]
    files.append(command_path)
    archive = output / f"b1-{candidate}-screen-evidence.tar.gz"
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
    (output / f"b1-{candidate}-screen.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    parser.add_argument("candidate", choices=["e24", "kvmat"])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    package(args.root, args.run, args.output, args.candidate)
