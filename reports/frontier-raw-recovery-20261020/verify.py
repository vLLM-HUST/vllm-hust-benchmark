#!/usr/bin/env python3
"""Recompute each archived stream window using only Python's standard library."""

import argparse
import hashlib
import gzip
import io
import json
import math
import tarfile
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    groups = {}
    for run in manifest["runs"]:
        rid = run["run_id"]
        assert rid not in seen, f"Duplicate run: {rid}"
        seen.add(rid)
        chunks = []
        for part in run["archive_parts"]:
            path = (root / part["path"]).resolve()
            assert path.is_relative_to(root), "Archive outside report"
            data = path.read_bytes()
            assert len(data) == part["bytes"]
            assert hashlib.sha256(data).hexdigest() == part["sha256"]
            chunks.append(data)
        archive = b"".join(chunks)
        assert len(archive) == run["archive_bytes"]
        assert hashlib.sha256(archive).hexdigest() == run["archive_sha256"], rid
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
            assert sorted(tf.getnames()) == [
                "config.json",
                "requests.jsonl",
                "summary.json",
            ], rid
            files = {}
            for name, expected in run["files"].items():
                member = tf.getmember(name)
                assert member.isfile(), (rid, name)
                data = tf.extractfile(member).read()
                assert len(data) == expected["bytes"], (rid, name)
                assert hashlib.sha256(data).hexdigest() == expected["sha256"], (
                    rid,
                    name,
                )
                files[name] = data
        config = json.loads(files["config.json"])
        summary = json.loads(files["summary.json"])
        requests = [
            json.loads(line)
            for line in files["requests.jsonl"].splitlines()
            if line.strip()
        ]
        assert config["run_id"] == rid
        assert config["duration"] == 900
        assert config["concurrency"] == run["concurrency"]
        assert config["chips"] == run["chips"]
        assert config["workload_sha256"] == run["workload_sha256"]
        assert config["tokenizer"]["fingerprint"] == run["tokenizer_fingerprint"]
        assert config["seed"] == run["client_seed"]
        assert len(requests) == run["request_records"]
        if run["family"] != "supplementary":
            assert summary["valid"] is True and summary["failed_requests"] == 0
        else:
            assert summary["valid"] == run["original_valid"]
            assert summary["failed_requests"] == run["failed_requests"]
            assert run["performance_eligible"] is False
        count = 0
        for request in requests:
            for elapsed, tokens in request.get("chunks", []):
                assert (
                    math.isfinite(elapsed) and isinstance(tokens, int) and tokens >= 0
                )
                if 0 <= elapsed < config["duration"]:
                    count += tokens
        assert (
            count
            == run["output_tokens_in_window"]
            == summary["observed_output_tokens_in_window"]
        )
        for rate in [
            run["expected_tps"],
            run["output_tokens_per_second"],
            summary["output_tokens_per_second"],
        ]:
            assert math.isclose(count / 900, rate, rel_tol=0, abs_tol=1e-8), (rid, rate)
        if run["family"] != "supplementary":
            groups.setdefault((run["family"], run["arm"]), []).append(
                run["concurrency"]
            )
    assert len(seen) == 56
    assert len(groups) == 9
    assert all(sorted(points) == [1, 2, 4, 8, 16] for points in groups.values())
    sums = "".join(
        p["sha256"] + "  " + p["path"] + "\n"
        for r in manifest["runs"]
        for p in r["archive_parts"]
    )
    assert (root / "SHA256SUMS").read_text() == sums
    workload_rows = json.loads((root / "workloads.json").read_text())
    for workload in workload_rows:
        path = (root / workload["path"]).resolve()
        assert path.is_relative_to(root)
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == workload["compressed_sha256"]
        content = gzip.decompress(blob)
        assert len(content) == workload["bytes"]
        assert hashlib.sha256(content).hexdigest() == workload["sha256"]
    assert {r["workload_sha256"] for r in manifest["runs"]} == {
        w["sha256"] for w in workload_rows
    }
    return {
        "runs": len(seen),
        "complete_series": len(groups),
        "raw_window_recomputation": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
