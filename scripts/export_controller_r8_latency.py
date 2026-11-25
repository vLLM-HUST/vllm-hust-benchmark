"""Recompute B1 r8 controller latency from the frozen November 18 archives."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/controller-r8-20261118"
OUTPUT = ROOT / "website-publication/data/controller-r8-latency-v1.json"


def checked_bytes(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Archive digest mismatch: {path}")
    return raw


def client_tpot(raw_root, arm):
    values = {}
    evidence = json.loads((REPORT / f"{arm}-requests.json").read_text())
    for record in evidence:
        dataset = record["dataset"]
        name = hashlib.sha256(record["case_id"].encode()).hexdigest()
        path = raw_root / arm / dataset / "requests" / f"{name}.json"
        row = json.loads(checked_bytes(path, record["original_record_sha256"]))
        tokens = row["usage"]["completion_tokens"]
        start = row["first_token"]["monotonic"]
        elapsed = row["end"]["monotonic"] - start
        if row["status"] != "ok" or tokens < 2 or elapsed <= 0:
            raise ValueError(f"Invalid latency sample: {path}")
        values.setdefault(dataset, []).append(1000 * elapsed / (tokens - 1))
    return {
        dataset: {"requests": len(samples), "mean_tpot_ms": mean(samples)}
        for dataset, samples in sorted(values.items())
    }


def engine_latency(path, expected, requests):
    raw = checked_bytes(path, expected)
    snapshots = gzip.decompress(raw).splitlines()
    before, after = (json.loads(snapshots[index])["samples"] for index in (0, -1))
    result = {"requests": requests, "source_sha256": expected}
    for metric, field in (
        ("e2e_request_latency_seconds", "mean_e2e_ms"),
        ("time_to_first_token_seconds", "mean_ttft_ms"),
        ("request_time_per_output_token_seconds", "mean_tpot_ms"),
    ):

        def sample(snapshot, suffix):
            values = [
                value
                for name, _, value in snapshot
                if name == f"vllm:{metric}_{suffix}"
            ]
            if len(values) != 1 or not math.isfinite(values[0]):
                raise ValueError(f"Invalid engine counter: {metric}_{suffix}")
            return values[0]

        count = sample(after, "count") - sample(before, "count")
        total = sample(after, "sum") - sample(before, "sum")
        if count != requests or total <= 0:
            raise ValueError(f"Engine counter does not cover the full pair: {metric}")
        result[field] = 1000 * total / count
    return result


def export(multi_turn, profiles):
    diagnostics = json.loads((REPORT / "multi-turn-diagnostics.json").read_text())
    pair = json.loads((REPORT / "multi-turn-paired-result.json").read_text())
    fixed = {}
    engine = {}
    for arm, source in (
        ("direct", "direct"),
        ("controller", "request_throttling_controller"),
    ):
        fixed[arm] = client_tpot(profiles / "client-raw/website-profiles", arm)
        filename = "engine_metrics_raw.jsonl.gz"
        diagnostic = diagnostics[source]
        if not diagnostic["histogram_same_business_coverage"]:
            raise ValueError("Unmatched engine histogram coverage")
        path = multi_turn / f"client-raw/recovered-{arm}-1" / filename
        engine[arm] = engine_latency(
            path,
            diagnostic["sources_sha256"][filename],
            pair[source]["successful_requests"],
        )
        if not math.isclose(
            engine[arm]["mean_e2e_ms"],
            1000 * diagnostic["engine_mean_seconds"]["engine_e2e"],
        ):
            raise ValueError("Engine window differs from the frozen diagnostics")
    return {
        "schema": "controller-r8-latency/v1",
        "engine_release": "full-answer-b1-20261117-r8",
        "report_path": "reports/controller-r8-20261118/README.md",
        "report_revision": "fba8140d027426dc66074ce381929f97b30eceec",  # pragma: allowlist secret -- public report commit
        "fixed_profiles_client": fixed,
        "multi_turn_engine": engine,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-turn", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT.write_text(
        json.dumps(export(args.multi_turn, args.profiles), indent=2) + "\n"
    )
