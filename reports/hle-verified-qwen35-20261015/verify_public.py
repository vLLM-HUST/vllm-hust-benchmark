"""Verify public HLE tables, outcomes and original-to-public evidence bindings."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    pub = json.loads((root / "publication.json").read_text())
    outcomes = json.loads(
        gzip.decompress((root / "per-case-outcomes.json.gz").read_bytes())
    )
    assert report["bindings"] == pub["bindings"]
    assert report["formal_acceptance"] is False and report["paired_b1"] is None
    assert report["trial_count"] == 1
    assert len(outcomes) == len({x["case_id"] for x in outcomes}) == 1811
    assert {x["track"] for x in outcomes} == {"gold", "revision"}
    for label in ["gold", "revision", "bench"]:
        archive = root / (label + "-evidence.tar.gz")
        assert sha(archive.read_bytes()) == pub["runs"][label]["archive_sha256"]
        with tarfile.open(archive) as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf if m.isfile()}
        assert set(members) == set(pub["runs"][label]["files"])

        def load(rel):
            return json.loads(members[rel])

        bundle, man = load("EVIDENCE_BUNDLE.json"), load("MANIFEST.json")
        binding = report["bindings"][label]
        assert sha(members["MANIFEST.json"]) == binding["manifest_sha256"]
        assert dict(man, manifest_sha256=binding["manifest_sha256"]) == binding
        assert sha(members["results/" + man["table_path"]]) == man["table_sha256"]
        digest = sha(
            "".join(
                bundle["files"][k]["sha256"] + "  " + k + "\n"
                for k in sorted(bundle["files"])
            ).encode()
        )
        assert (
            digest == man["evidence_bundle_sha256"] == bundle["evidence_bundle_sha256"]
        )
        for rel, raw in members.items():
            entry = pub["runs"][label]["files"][rel]
            assert sha(raw) == entry["published_sha256"] and len(raw) == entry["bytes"]
            if not entry["operator_identity_redacted"]:
                assert entry["source_sha256"] == entry["published_sha256"]
            if rel in bundle["files"]:
                assert entry["source_sha256"] == bundle["files"][rel]["sha256"]
        effective = load("receipt/effective.json")
        assert effective["ok"] and len(effective["verified"]) == 20
        assert not effective["unverified"] and not effective["argv_only"]
        assert (
            pub["runs"][label]["files"]["receipt/effective.json"]["source_sha256"]
            == man["effective_json_sha256"]
        )
        assert (
            pub["runs"][label]["files"]["receipt/startup.log"]["source_sha256"]
            == man["startup_log_sha256"]
        )
        if label != "bench":
            summary = load("full/report/summary.json")
            assert summary == report["summaries"][label]
            assert summary["scoring_policy"] == "strict_json"
            assert summary["scope"] == "complete_selected_plan"
            rows = [x for x in outcomes if x["track"] == label]
            original = load("full/report/per_case.json")
            assert len(rows) == len(original)
            for row, source in zip(rows, original):
                assert {k: row[k] for k in source} == source
                for stage in ["solver", "judge"]:
                    assert (
                        row[stage + "_record_sha256"]
                        == bundle["files"][
                            f"full/{stage}/requests/{row['case_id']}.json"
                        ]["sha256"]
                    )
            groups = [(rows, summary["tracks"][label])]
            groups += [
                ([r for r in rows if label + "/" + r["modality"] == key], value)
                for key, value in summary["groups"].items()
            ]
            groups += [
                ([r for r in rows if label + "/" + r["subject"] == key], value)
                for key, value in summary["subjects"].items()
            ]
            for selected, stats in groups:
                assert len(selected) == stats["total"]
                assert sum(x["correct"] is True for x in selected) == stats["successes"]
                assert (
                    sum(x["correct"] is None for x in selected) == stats["unresolved"]
                )
                assert stats["graded"] + stats["unresolved"] == stats["total"]
                for k in ["solver_truncated", "judge_truncated"]:
                    assert sum(x[k] for x in selected) == stats[k]
                assert (
                    abs(stats["accuracy"] - stats["successes"] / stats["total"]) < 1e-12
                )
        else:
            assert load("summary.json") == report["performance"]
            runtime = {
                k: {"value": v["log"], "status": v["status"]}
                for k, v in effective["fields"].items()
                if v["status"] == "verified"
            }
            assert runtime == report["runtime"]["fields"]
            for name, n in [
                ("gold-text", 200),
                ("revision-text", 200),
                ("gold-image", 93),
                ("revision-image", 118),
            ]:
                stats = report["performance"][name]
                assert stats["n"] == stats["completed"] == n and stats["failed"] == 0
                assert (
                    stats["result_sha256"]
                    == bundle["files"][f"groups/{name}/outputs/{name}.json"]["sha256"]
                )
                accounting = load(f"groups/{name}/token-accounting.json")
                assert accounting["per_request_input_matches_server_tokenize"] is True
                assert (
                    accounting["bench_input_lens"]
                    == accounting["server_tokenize_counts"]
                )
                assert len(accounting["bench_input_lens"]) == n
                assert (
                    sum(accounting["bench_input_lens"]) == stats["total_input_tokens"]
                )
                assert (
                    abs(
                        stats["output_throughput"]
                        - stats["total_output_tokens"] / stats["duration"]
                    )
                    < 1e-9
                )
                selection = load(f"exports/{name}/prompts.jsonl.manifest.json")
                track, mode = name.split("-")
                eligible = [
                    r for r in outcomes if r["track"] == track and r["modality"] == mode
                ]
                chosen = (
                    eligible
                    if mode == "image"
                    else [eligible[i * len(eligible) // n] for i in range(n)]
                )
                assert [x["case_id"] for x in selection["cases"]] == [
                    x["case_id"] for x in chosen
                ]
    checks = json.loads((root / "seal-check.json").read_text())
    assert all(x["ok"] and x["table_matches_manifest"] for x in checks.values())
    print(
        "Verified: 1811 outcomes, three original seals, four bench groups, 20 runtime fields."
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
