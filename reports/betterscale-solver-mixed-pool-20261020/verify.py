# SPDX-License-Identifier: Apache-2.0
"""Verify the payload-free measured report, not task accuracy or private payloads."""

import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).parent
receipt = json.loads((root / "receipts.json").read_text())
results = json.loads((root / "results.json").read_text())
assert results["complete"] and len(results["rows"]) == 12
arms = receipt["arms"]
assert len(arms) == 6
assert {
    (a["identity"]["concurrency"], a["identity"]["maximum_query_capacity"])
    for a in arms
} == {(c, t) for c in (8, 16, 32) for t in (4096, 8192)}
source = hashlib.sha256((root / "benchmark_client.py.txt").read_bytes()).hexdigest()
formal_count = warm_count = 0
identities = {d: set() for d in ("mmlu", "hle")}
for arm in arms:
    assert arm["exit"]["exit_code"] == 0 and arm["hardware"]["release_idle_verified"]
    assert not arm["identity"]["dummy_qualification"]
    for dataset, evidence in arm["datasets"].items():
        contract = evidence["contract"]
        measures = evidence["measurements"]
        assert contract["source_sha256"] == source
        assert (
            contract["transport"]
            == "fresh HTTP connection per request; no cross-burst keepalive reuse"
        )
        assert len(measures) == 4 and [m["repeat"] for m in measures] == list(range(4))
        identities[dataset].add(contract["dataset_requests_sha256"])
        for i, m in enumerate(measures):
            assert m["warmup"] == (i == 0)
            assert (
                m["completed"] == m["planned_requests"] == 200
                and m["failures"] == m["retries"] == 0
            )
            assert (
                m["logical_input_tokens"] == contract["input_tokens"]
                and m["actual_output_tokens"] > 0
            )
            assert m["cards"] == 2 and m["assumed_cny_per_card_hour"] == 4
            assert (
                m["seconds"] > 0
                and m["ttft"] is None
                and m["tpot"] is None
                and not m["slo_certified"]
            )
            expected = {
                "output_tokens_per_s": m["actual_output_tokens"] / m["seconds"],
                "steady_card_rental_cny": 8 * m["seconds"] / 3600,
                "cny_per_million_output": 8
                * m["seconds"]
                / 3600
                / m["actual_output_tokens"]
                * 1e6,
                "cny_per_million_total": 8
                * m["seconds"]
                / 3600
                / (m["actual_output_tokens"] + m["logical_input_tokens"])
                * 1e6,
            }
            for k, v in expected.items():
                assert math.isclose(m[k], v, rel_tol=1e-10)
            if i:
                formal_count += 200
            else:
                warm_count += 200
        row = next(
            r
            for r in results["rows"]
            if r["capsule"] == arm["capsule"] and r["dataset"] == dataset
        )
        for output, raw in (
            ("output_tps", "output_tokens_per_s"),
            ("cny_per_million_output", "cny_per_million_output"),
            ("cny_per_million_total", "cny_per_million_total"),
            ("seconds", "seconds"),
        ):
            assert math.isclose(
                row[output],
                statistics.median(m[raw] for m in measures[1:]),
                rel_tol=1e-10,
            )
assert formal_count == 7200 and warm_count == 2400
assert all(len(v) == 1 for v in identities.values())
print(
    "PASS: six real-service arms, 12 dataset groups, 7200 formal + 2400 warmup requests; accounting and client identity verified"
)
