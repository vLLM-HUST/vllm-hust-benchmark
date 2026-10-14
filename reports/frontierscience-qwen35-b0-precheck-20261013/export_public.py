"""Export only allowlisted evidence and outcome metadata from a sealed run."""

import argparse
import csv
import gzip
import hashlib
import io
import os
import json
import pathlib
import re
import sys
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args()
    run, out = args.run, args.output
    if out.resolve() == run.resolve() or run.resolve() in out.resolve().parents:
        raise ValueError("Publication output must be outside the sealed run")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run / "client_source/src"))
    from hust_eval_frontierscience.hashing import bundle_hash
    from hust_eval_frontierscience.seal import seal_check

    check = seal_check(run)
    assert check["ok"] and check["table_matches_manifest"] is True

    def load(rel):
        return json.loads((run / rel).read_text())

    bundle = load("EVIDENCE_BUNDLE.json")
    manifest = load("MANIFEST.json")
    assert bundle_hash(bundle["files"]) == manifest["evidence_bundle_sha256"]
    summary = load("full/report/summary.json")
    bench = load("full/bench/outputs/frontierscience.summary.json")
    contract = load("plan/contract.json")
    effective = load("receipt/effective.json")
    assert summary["scope"] == "full_gold"
    assert len(effective["verified"]) == 18 and not effective["unverified"]
    assert bench["completed"] == 160 and bench["failed"] == 0
    assert contract["scoring_policy"] == "mmlu_extract_first"
    rows = list(csv.DictReader((run / "full/report/per_case.csv").open()))
    outcomes = []
    for row in rows:
        case = row["case_id"]
        solver_rel = f"full/solver/requests/{case}.json"
        judge_rel = f"full/judge/requests/{case}.json"
        judge = load(judge_rel)
        grade = judge.get("grade") or {}
        outcomes.append(
            {
                "case_id": case,
                "track": row["track"],
                "subject": row["subject"],
                "solver_status": row["solver_status"],
                "judge_status": row["judge_status"],
                "solver_truncated": row["solver_truncated"] == "True",
                "judge_truncated": row["judge_truncated"] == "True",
                "correct": {"True": True, "False": False, "": None}[row["correct"]],
                "score": float(row["score"]) if row["score"] else None,
                "audit_flag_count": int(row["audit_flag_count"]),
                "verdict_count": grade.get("verdict_count"),
                "extraction": grade.get("extraction"),
                "solver_record_sha256": bundle["files"][solver_rel]["sha256"],
                "judge_record_sha256": bundle["files"][judge_rel]["sha256"],
            }
        )
    assert len(outcomes) == 160
    for track, stats in summary["tracks"].items():
        subset = [x for x in outcomes if x["track"] == track]
        assert len(subset) == stats["total"]
        assert sum(x["correct"] is True for x in subset) == stats["successes"]
        assert sum(x["judge_truncated"] for x in subset) == stats["judge_truncated"]
        assert sum(x["solver_truncated"] for x in subset) == stats["solver_truncated"]
    scores = [
        x["score"]
        for x in outcomes
        if x["track"] == "research" and x["score"] is not None
    ]
    assert len(scores) == 57
    assert (
        abs(
            sum(scores) / len(scores)
            - summary["tracks"]["research"]["mean_research_score_valid_only"]
        )
        < 1e-12
    )
    distribution = [
        {"lower": i, "upper": i + 1, "count": sum(i <= s < i + 1 for s in scores)}
        for i in range(10)
    ]
    distribution.append(
        {"lower": 10, "upper": 10, "count": sum(s == 10 for s in scores)}
    )
    runtime = {
        "run_id": run.name,
        "fields": {
            k: {"value": v["log"], "status": v["status"]}
            for k, v in effective["fields"].items()
            if v["status"] == "verified"
        },
        "effective_sha256": manifest["effective_json_sha256"],
        "startup_log_sha256": manifest["startup_log_sha256"],
        "service_origin": "Reused continuously running service from fs-qwen35-v02-20261013T081932Z; actual process arguments rechecked; no new startup claimed.",
    }
    report = {
        "schema": 1,
        "run_id": run.name,
        "model": "Qwen3.5-35B-A3B",
        "measurement_date_utc": "2026-10-13",
        "sealed_utc": manifest["sealed_utc"],
        "formal_acceptance": False,
        "paired_b1": None,
        "trial_count": 1,
        "summary": summary,
        "performance": bench,
        "runtime": runtime,
        "research_score_distribution": distribution,
        "generation": {
            role: {k: contract[role][k] for k in ("concurrency", "timeout", "request")}
            for role in ("solver", "judge")
        },
        "dataset": {
            k: contract[k] for k in ("dataset", "declared_revision", "sources")
        },
        "original_binding": {
            **manifest,
            "manifest_sha256": sha((run / "MANIFEST.json").read_bytes()),
            "contract_sha256": sha((run / "plan/contract.json").read_bytes()),
            "evidence_file_count": len(bundle["files"]),
        },
        "limitations": [
            "Official Appendix B prompts; Qwen judge replaces official judge; one trial, local extraction; not official leaderboard replication.",
            "First complete explicit VERDICT is used, including length-ended replies; invalid judges remain in denominator as nonsuccess.",
            "Invalid judge means unresolved evaluation, not proof of a scientifically incorrect solver answer.",
            "Audit flags are heuristic, nonexhaustive and do not change scores; human review remains required.",
            "Research mean uses 57 valid scores; missing grades are not assigned zero numeric points.",
            "Solver and judge use the same served model; no independent expert validation.",
            "Bench is separate: 160 prompts, max output 256, request-rate inf, temperature 0, seed 0, default server chat template.",
            "Single service lifecycle reused after smoke; not a cold-start run or three-repeat V5.5 acceptance. No B1 or improvement claims.",
            "Full weight shards were not hashed; image digest not captured. Only 11 small artifact files were hashed.",
        ],
    }
    selected = [
        "MANIFEST.json",
        "EVIDENCE_BUNDLE.json",
        "results/frontierscience-results.md",
        "profile.json",
        "plan/contract.json",
        "plan/solver-runtime-contract.json",
        "plan/judge-runtime-contract.json",
        "server/startup-command.sh",
        "server/SERVICE_ORIGIN.json",
        "server/current_argv.json",
        "server/receipt/argv_actual.json",
        "receipt/effective.json",
        "receipt/startup.log",
        "receipt/models.json",
        "server/receipt/versions.txt",
        "server/receipt/npu_smi.txt",
        "model_hashes/qwen35-small-after.json",
        "orchestration/prepare-commands.json",
        "orchestration/full-commands.json",
        "orchestration/full.py",
        "full/report/summary.json",
        "full/report/per_case.csv",
        "full/solver/run.json",
        "full/solver/index.json",
        "full/judge/run.json",
        "full/judge/index.json",
        "full/bench/outputs/frontierscience.json",
        "full/bench/outputs/frontierscience.summary.json",
        "full/bench/outputs/frontierscience.cmd.txt",
        "full/bench/inputs/frontierscience.materialized.sha256",
        "FULL_REVIEW.json",
    ]
    selected += [
        str(p.relative_to(run))
        for p in (run / "client_source").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    selected += [str(p.relative_to(run)) for p in (run / "model_hashes").glob("*.json")]
    selected = sorted(set(selected))
    publication = {}
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tf:
        for rel in selected:
            original = (run / rel).read_bytes()
            # The only intentional redaction: operator identity in host strings.
            operator = os.environ.get("PUBLICATION_OPERATOR", "").encode()
            public = original.replace(operator, b"OPERATOR") if operator else original
            assert not re.search(rb"-----BEGIN [A-Z ]+KEY-----", public)
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = len(public), 0, 0o644
            tf.addfile(info, io.BytesIO(public))
            publication[rel] = {
                "source_sha256": sha(original),
                "published_sha256": sha(public),
                "bytes": len(public),
                "operator_identity_redacted": public != original,
            }
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected-evidence.tar.gz").write_bytes(
        gzip.compress(tar_bytes.getvalue(), mtime=0)
    )
    write_json(
        out / "publication.json",
        {
            "run_id": run.name,
            "original_binding": report["original_binding"],
            "files": publication,
            "scope": "Allowlisted execution evidence and frozen client source. Full question/answer requests, dataset content and environment dumps are excluded.",
        },
    )
    write_json(out / "report.json", report)
    write_json(out / "per-case-outcomes.json", outcomes)
    write_json(out / "seal-check.json", check)
    print(
        json.dumps(
            {
                "cases": len(outcomes),
                "evidence_files": len(bundle["files"]),
                "published_members": len(selected),
                "archive_bytes": (out / "selected-evidence.tar.gz").stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
