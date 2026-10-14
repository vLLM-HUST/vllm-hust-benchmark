"""Verify public evidence without access to raw question/answer records."""

import hashlib
import json
from pathlib import Path
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / "report.json").read_text())
    publication = json.loads((root / "publication.json").read_text())
    outcomes = json.loads((root / "per-case-outcomes.json").read_text())
    with tarfile.open(root / "selected-evidence.tar.gz") as tf:
        files = {
            m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()
        }
    assert set(files) == set(publication["files"])
    for path, data in files.items():
        assert sha(data) == publication["files"][path]["published_sha256"], path
    manifest = json.loads(files["MANIFEST.json"])
    bundle = json.loads(files["EVIDENCE_BUNDLE.json"])
    listing = "".join(
        f"{entry['sha256']}  {path}\n"
        for path, entry in sorted(bundle["files"].items())
    )
    assert sha(listing.encode()) == manifest["evidence_bundle_sha256"]
    binding = report["original_binding"]
    assert sha(files["MANIFEST.json"]) == binding["manifest_sha256"]
    assert sha(files["results/frontierscience-results.md"]) == manifest["table_sha256"]
    for path, item in publication["files"].items():
        if path in bundle["files"]:
            assert item["source_sha256"] == bundle["files"][path]["sha256"], path
        if not item["operator_identity_redacted"]:
            assert item["source_sha256"] == item["published_sha256"], path
    for rel, key in [
        ("profile.json", "profile_sha256"),
        ("receipt/effective.json", "effective_json_sha256"),
        ("receipt/startup.log", "startup_log_sha256"),
    ]:
        assert publication["files"][rel]["source_sha256"] == manifest[key]
    assert sha(files["plan/contract.json"]) == binding["contract_sha256"]
    for name, digest in json.loads(files["plan/contract.json"])["client_code"].items():
        assert (
            sha(files["client_source/src/hust_eval_frontierscience/" + name]) == digest
        )
    assert len(outcomes) == 160
    for case in outcomes:
        for role in ["solver", "judge"]:
            rel = f"full/{role}/requests/{case['case_id']}.json"
            assert case[f"{role}_record_sha256"] == bundle["files"][rel]["sha256"]
    for track, stats in report["summary"]["tracks"].items():
        cases = [r for r in outcomes if r["track"] == track]
        assert len(cases) == stats["total"]
        assert sum(r["correct"] is True for r in cases) == stats["successes"]
        assert sum(r["judge_status"] != "ok" for r in cases) == stats["judge_invalid"]
        assert sum(r["judge_truncated"] for r in cases) == stats["judge_truncated"]
        assert sum(r["solver_truncated"] for r in cases) == stats["solver_truncated"]
    scores = [
        r["score"]
        for r in outcomes
        if r["track"] == "research" and r["score"] is not None
    ]
    assert len(scores) == 57
    assert (
        abs(
            sum(scores) / len(scores)
            - report["summary"]["tracks"]["research"]["mean_research_score_valid_only"]
        )
        < 1e-12
    )
    assert report["summary"] == json.loads(files["full/report/summary.json"])
    assert report["performance"] == json.loads(
        files["full/bench/outputs/frontierscience.summary.json"]
    )
    assert report["performance"]["completed"] == 160
    assert report["performance"]["failed"] == 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        assert sha((root / rel).read_bytes()) == digest, rel
    print(
        f"PASS: {len(files)} public members, {len(bundle['files'])} indexed evidence files, 160 outcomes, result/seal binding."
    )
    print(
        "Raw request hashes are bound; omitted raw contents cannot be regraded from this public package."
    )


if __name__ == "__main__":
    main()
