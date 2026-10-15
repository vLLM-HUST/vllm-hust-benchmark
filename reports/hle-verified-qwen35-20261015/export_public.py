"""Export allowlisted HLE evidence from three sealed runs; never publish answers/images."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accuracy", type=Path)
    ap.add_argument("bench", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--client-root", type=Path, required=True)
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.client_root / "src"))
    from hust_eval_hle_verified.seal import seal_check

    runs = {t: a.accuracy / ("full-" + t) for t in ["gold", "revision"]}
    runs["bench"] = a.bench
    for run in runs.values():
        assert run.resolve() not in a.output.resolve().parents
    a.output.mkdir(parents=True, exist_ok=True)
    checks, bindings, publication, summaries, outcomes = {}, {}, {}, {}, []
    for label, run in runs.items():
        checks[label] = seal_check(run)
        assert checks[label]["ok"] and checks[label]["table_matches_manifest"] is True

        def load(rel):
            return json.loads((run / rel).read_text())

        man, bundle = load("MANIFEST.json"), load("EVIDENCE_BUNDLE.json")
        bindings[label] = dict(
            man, manifest_sha256=sha((run / "MANIFEST.json").read_bytes())
        )
        selected = [
            "MANIFEST.json",
            "EVIDENCE_BUNDLE.json",
            "results/" + man["table_path"],
            "profile.json",
            "receipt/effective.json",
            "receipt/startup.log",
            "receipt/models.json",
            "model-small-before.json",
            "model-small-after.json",
            "source.json",
        ]
        if label != "bench":
            summaries[label] = load("full/report/summary.json")
            assert summaries[label]["scope"] == "complete_selected_plan"
            selected += [
                "plan/contract.json",
                "plan/contract.sha256",
                "plan/inventory.json",
                "plan/client_source/pipeline.py",
                "plan/client_source/official_prompts.py",
                "runtime-contract.json",
                "server/argv_actual.json",
                "server/CONTINUITY.json",
                "server/SERVICE_ORIGIN.json",
                "server/versions.txt",
                "server/npu_smi.txt",
                "full/report/summary.json",
                "full/report/per_case.json",
                "full/solver/run.json",
                "full/solver/index.json",
                "full/judge/run.json",
                "full/judge/index.json",
                "client-commands.json",
                "client-configs/solver.json",
                "client-configs/judge.json",
            ]
            for row in load("full/report/per_case.json"):
                item = dict(row)
                for stage in ["solver", "judge"]:
                    rel = f"full/{stage}/requests/{row['case_id']}.json"
                    item[stage + "_record_sha256"] = bundle["files"][rel]["sha256"]
                outcomes.append(item)
        else:
            selected += [
                "summary.json",
                "tokenization-counts.json",
                "request-builder-check.json",
                "server-startup/argv_actual.json",
                "server-startup/versions.txt",
                "server-startup/npu_smi.txt",
                "client_source/SOURCE_SHA256SUMS",
                "client_source/src/hust_eval_hle_verified/bench.py",
                "client_source/src/hust_eval_hle_verified/bench_inputs.py",
            ]
            for folder in ["continuity", "commands"]:
                selected += [
                    str(p.relative_to(run)) for p in (run / folder).glob("*.json")
                ]
            for g in load("summary.json"):
                selected += [
                    f"exports/{g}/prompts.jsonl.manifest.json",
                    f"groups/{g}/token-accounting.json",
                    f"groups/{g}/inputs/{g}.semantics.json",
                    f"groups/{g}/outputs/{g}.summary.json",
                    f"groups/{g}/outputs/{g}.cmd.txt",
                    f"groups/{g}/outputs/{g}.argv.json",
                ]
        records, stream = {}, io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for rel in sorted(set(selected)):
                raw = (run / rel).read_bytes()
                # Strip only operator identity from public host/path strings, with explicit mapping.
                public = raw.replace(b"peifangrui", b"OPERATOR")
                assert not re.search(
                    rb"-----BEGIN [A-Z ]+KEY-----|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}",
                    public,
                )
                info = tarfile.TarInfo(rel)
                info.size, info.mode = len(public), 0o644
                archive.addfile(info, io.BytesIO(public))
                records[rel] = {
                    "source_sha256": sha(raw),
                    "published_sha256": sha(public),
                    "bytes": len(public),
                    "operator_identity_redacted": public != raw,
                }
        packed = gzip.compress(stream.getvalue(), mtime=0)
        (a.output / (label + "-evidence.tar.gz")).write_bytes(packed)
        publication[label] = {"files": records, "archive_sha256": sha(packed)}

    def bench_load(rel):
        return json.loads((a.bench / rel).read_text())

    contract = json.loads((runs["gold"] / "plan/contract.json").read_text())
    effective = bench_load("receipt/effective.json")
    assert (
        len(effective["verified"]) == 20
        and not effective["unverified"]
        and not effective["argv_only"]
    )
    report = {
        "schema": 1,
        "run_id": a.bench.name,
        "model": "Qwen3.5-35B-A3B",
        "measurement_date_utc": "2026-10-14/15",
        "formal_acceptance": False,
        "paired_b1": None,
        "trial_count": 1,
        "summaries": summaries,
        "performance": bench_load("summary.json"),
        "runtime": {
            "fields": {
                k: {"value": v["log"], "status": v["status"]}
                for k, v in effective["fields"].items()
                if v["status"] == "verified"
            },
            "effective_sha256": bindings["bench"]["effective_json_sha256"],
            "startup_log_sha256": bindings["bench"]["startup_log_sha256"],
        },
        "bindings": bindings,
        "generation": {
            role: {k: contract[role][k] for k in ["concurrency", "timeout", "request"]}
            for role in ["solver", "judge"]
        },
        "dataset": {k: contract[k] for k in ["dataset", "declared_revision"]},
        "client_versions": {
            label: json.loads((run / "source.json").read_text())["git_commit"]
            for label, run in runs.items()
        },
        "limitations": [
            "Author HLE-Verified Gold 668 and Revision 1143; Uncertain 689 excluded. Not the original unmodified HLE dataset.",
            "Original HLE system/judge prompts and schema; same Qwen solver and substitute judge, one trial. Not official leaderboard replication or V5.5 acceptance.",
            "Strict JSON grading requires normal completion; all 72 length-ended judges remain unresolved in the full denominator. No partial JSON rescue or retry.",
            "Solver length-ended outputs are sent to judge, not automatically marked incorrect. Gold 383 and Revision 688 solver truncations materially limit interpretation.",
            "Bench: raw user question plus real images; no accuracy system prompt; template once, thinking=false, max output256, temperature0, request seed0, EOS allowed, rate inf, no client concurrency cap; server max_num_seqs16.",
            "Text selection floor(i*N/200) from frozen modality-filtered order; images all93/all118, no repetitions. No correctness-based selection.",
            "Per-request input token counts match same-server /tokenize including images. No separate visual token count claimed. TTFT includes burst queueing.",
            "Reused service after accuracy, request validation and tokenization preflight; not a cold-cache run, not paired B1 and not three-repeat acceptance.",
            "Full weights were inventoried but not hashed; model small files matched. Image digest and verified local model revision are unavailable.",
            "Public evidence omits question/image/answer payloads and environment dumps. Original raw records remain private and cannot be independently regraded from this package.",
        ],
    }
    write(a.output / "report.json", report)
    (a.output / "per-case-outcomes.json.gz").write_bytes(
        gzip.compress(
            (json.dumps(outcomes, ensure_ascii=False, indent=2) + "\n").encode(),
            mtime=0,
        )
    )
    write(
        a.output / "publication.json",
        {
            "bindings": bindings,
            "runs": publication,
            "scope": "Selected execution evidence only; no question/image/answer payloads or environment dumps. Original-to-public hashes record operator-identity redactions.",
        },
    )
    write(a.output / "seal-check.json", checks)
    print(
        json.dumps(
            {
                "outcomes": len(outcomes),
                "archive_bytes": {
                    k: (a.output / (k + "-evidence.tar.gz")).stat().st_size
                    for k in runs
                },
            }
        )
    )


if __name__ == "__main__":
    main()
