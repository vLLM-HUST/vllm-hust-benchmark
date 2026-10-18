"""Verify public arithmetic and, optionally, frozen assembled MMLU records."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts
    return path


def verify(root, archive=None, plan_archive=None):
    root = Path(root)
    if (root / "SHA256SUMS").exists():
        for line in (root / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert sha((root / relative(name)).read_bytes()) == expected
    proof = json.loads((root / "checkpoint.json").read_text())
    assert proof["state"] == "CONTROLLER_QUALITY_COMPLETE_PAIR_PENDING"
    assert proof["stage_status"] == "SUCCEEDED"
    summary = proof["summary"]
    assert summary["cases"] == 12032 and summary["smoke_only"] is False
    assert 0 <= summary["correct"] <= 12032
    assert summary["accuracy"] == summary["correct"] / 12032
    assert proof["concurrency"] == 16
    missing_usage = None
    if archive:
        assert sha(Path(archive).read_bytes()) == proof["private_raw_sha256"]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with tarfile.open(archive) as tar:
                for item in tar.getmembers():
                    name = relative(item.name)
                    assert item.isdir() or item.isfile()
                    path = base / name
                    if item.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(tar.extractfile(item).read())
            manifest_bytes = (base / "artifact-manifest.json").read_bytes()
            assert sha(manifest_bytes) == proof["artifact_manifest_sha256"]
            manifest = json.loads(manifest_bytes)
            assert len(manifest) == proof["artifact_files"]
            for name, expected in manifest.items():
                assert sha((base / relative(name)).read_bytes()) == expected
            for name, expected in proof["client_code_sha256"].items():
                assert (
                    sha((base / "hust_eval_mmlu_pro" / relative(name)).read_bytes())
                    == expected
                )
            for name, key in [
                ("contract.json", "contract_sha256"),
                ("cases.json", "cases_sha256"),
                ("tasks.json", "tasks_sha256"),
            ]:
                assert sha((base / "plan" / name).read_bytes()) == proof[key]
            sys.path.insert(0, str(base))
            from hust_eval_mmlu_pro.mmlu_pro.plan import load_plan
            from hust_eval_mmlu_pro.mmlu_pro.scoring import score_results

            contract, cases = load_plan(base / "plan")
            rows = json.loads((base / "run/results.json").read_text())
            assert len(rows) == len(cases) == 12032
            assert len({row["case_id"] for row in rows}) == 12032
            by_case = {case["case_id"]: case for case in cases}
            assert {row["case_id"] for row in rows} == set(by_case)
            missing_usage = 0
            for row in rows:
                case = by_case[row["case_id"]]
                for key in [
                    "case_id",
                    "prompt_sha256",
                    "prompt_tokens",
                    "max_tokens",
                    "ignore_eos",
                ]:
                    assert row[key] == case[key]
                assert (
                    json.loads(
                        (base / "run/requests" / (row["case_id"] + ".json")).read_text()
                    )
                    == row
                )
                usage = row.get("usage")
                if usage:
                    assert all(
                        type(usage[key]) is int and usage[key] >= 0
                        for key in [
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        ]
                    )
                    assert (
                        usage["prompt_tokens"] + usage["completion_tokens"]
                        == usage["total_tokens"]
                    )
                    assert usage["completion_tokens"] <= row["max_tokens"]
                else:
                    missing_usage += 1
            ok = [row for row in rows if row["success"]]
            assert summary["mean_ttft_s"] == sum(
                row["ttft_seconds"] for row in ok if row["ttft_seconds"]
            ) / max(1, len(ok))
            assert summary["mean_e2e_s"] == sum(row["e2e_seconds"] for row in ok) / max(
                1, len(ok)
            )
            scored = score_results(rows, cases)
            assert (
                json.dumps(scored, indent=1, ensure_ascii=False).encode()
                == (base / "run/score.json").read_bytes()
            )
            assert json.loads((base / "run/summary.json").read_text()) == summary
            for key in [
                "correct",
                "accuracy",
                "invalid_answers",
                "transport_failures",
                "length_limited",
            ]:
                assert summary[key] == scored[key]
            metadata_files = list((base / "run").glob("run.*.json"))
            assert len(metadata_files) == 1
            meta = json.loads(metadata_files[0].read_text())
            assert (
                meta["n_cases"] == 12032
                and meta["limit"] is None
                and meta["resume"] is False
            )
            assert meta["concurrency"] == proof["concurrency"]
            assert meta["generation"] == proof["generation"] == contract["generation"]
            assert (
                meta["scorer_sha256"]
                == proof["client_code_sha256"]["mmlu_pro/scoring.py"]
            )
            assert (
                meta["runner_sha256"] == proof["client_code_sha256"]["mmlu_pro/run.py"]
            )
            assert meta["contract_sha256"] == proof["contract_sha256"]
    return {
        "checkpoint": "PASS",
        "private_archive_verified": bool(archive),
        "cases": 12032,
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "usage_missing": missing_usage,
        "evidence_boundary": proof["evidence_boundary"],
        "controller_comparison": "PENDING",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive")
    parser.add_argument("--plan-archive")
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).parent, args.archive, args.plan_archive)))
