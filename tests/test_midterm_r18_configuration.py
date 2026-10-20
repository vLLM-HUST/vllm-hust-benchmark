"""Recompute recovered configuration, scalar ledgers and complete r18 coverage."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019"
    / "m01-recovery/r18-loaded-configuration"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r18"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("r18")
    index = json.loads((BASE / "archive.json").read_text())
    archive_path = root / "evidence.tar.gz"
    digest = hashlib.sha256()
    with archive_path.open("wb") as stream:
        for part in index["parts"]:
            data = (BASE / part["file"]).read_bytes()
            assert len(data) == part["bytes"]
            assert hashlib.sha256(data).hexdigest() == part["sha256"]
            digest.update(data)
            stream.write(data)
    assert digest.hexdigest() == index["archive_sha256"]
    expected = {m["name"]: m for m in index["members"]}
    seen = set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in seen
            seen.add(member.name)
            data = archive.extractfile(member).read()
            assert len(data) == expected[member.name]["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    assert seen == set(expected)
    return root


def test_source_identity_and_lifecycle(evidence):
    for path in BASE.glob("*.json"):
        if path.name == "archive.json":
            continue
        assert path.read_bytes() == (evidence / SOURCE / path.name).read_bytes()
        for source, digest in json.loads(path.read_text())["source_sha256"].items():
            assert (
                hashlib.sha256((evidence / source.lstrip("/")).read_bytes()).hexdigest()
                == digest
            )
        for source, digest in (
            json.loads(path.read_text()).get("loaded_source_sha256", {}).items()
        ):
            assert (
                hashlib.sha256((evidence / source.lstrip("/")).read_bytes()).hexdigest()
                == digest
            )
    receipt = json.loads((evidence / RUN / "receipt.json").read_text())
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    for name, digest in receipt["input_hashes"].items():
        path = (
            evidence / name.lstrip("/")
            if name.startswith("/")
            else evidence / SOURCE / name
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_device_state_controls_ledger(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from audit_r18_gdn_publications import join, count
    from derive_r18_device_gdn import operations
    from derive_r18_gdn_prefill import chunks, ho, wy, inverse

    summary = json.loads((BASE / "r18-gdn-publication-summary.json").read_text())
    ledger = json.loads((BASE / "r18-device-gdn-ledger.json").read_text())
    prefill = json.loads((BASE / "r18-device-gdn-prefill.json").read_text())
    for expected, expected_ledger in zip(
        summary["ranks"], ledger["ranks"], strict=True
    ):
        pid = expected["pid"]
        capture = json.loads((evidence / RUN / f"moe-deferred-{pid}.json").read_text())
        publication = json.loads(
            (evidence / RUN / f"gdn-publication-{pid}.json").read_text()
        )
        counts, arithmetic = Counter(), Counter()
        histogram = Counter()
        skipped = 0
        for frame, pairs in join(capture["frames"], publication["records"]):
            for group, record in pairs:
                counts.update(count(group, record))
                arithmetic.update(operations(group, record))
                histogram.update(chunks(group, record))
                fields = record["device_fields"]
                for j, i in enumerate(record["metadata"]["verify_ids"]):
                    if fields["verify_slots"][j][0] == -1:
                        skipped += fields["cu"][i + 1][0] - fields["cu"][i][0]
        assert skipped == counts["device_skipped_verify_tokens"] == 387
        assert counts == expected["cohorts"]["inside_request_under_clock_tolerance"]
        arithmetic["convolution_basic_arithmetic"] = (
            arithmetic["convolution_multiply"] + arithmetic["convolution_add"]
        )
        assert (
            arithmetic
            == expected_ledger["cohorts"]["inside_request_under_clock_tolerance"]
        )
        expected_prefill = next(r for r in prefill["ranks"] if r["pid"] == pid)[
            "cohorts"
        ]["inside_request_under_clock_tolerance"]
        assert [
            {"length": n, "has_state": state, "chunk_layers": number}
            for (n, state), number in sorted(histogram.items())
        ] == expected_prefill["chunk_catalog"]
        totals = Counter()
        for (n, state), number in histogram.items():
            for prefix, values in [
                ("ho", ho(n, state)),
                ("wy", wy(n)),
                ("inverse", inverse(n)),
            ]:
                totals.update(
                    {
                        f"{prefix}_{key}": value * number * 16
                        for key, value in values.items()
                    }
                )
        assert totals == expected_prefill["operations_per_rank"]
    assert ledger["model_effective_flops"] is ledger["m01_attainment"] is None


def test_all_three_hardware_intervals_are_covered(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from integrate_hbm_samples import has_complete_hbm_coverage, integrate
    from integrate_r12_hardware import link_bounds

    summary = json.loads((BASE / "r18-hardware-intervals.json").read_text())
    rows = [
        json.loads(line)
        for line in (evidence / SOURCE / "r18-system-control/hccs64.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(summary["windows"]) == 3
    for window in summary["windows"]:
        start, end = window["start_wall"] * 1e9, window["end_wall"] * 1e9
        assert link_bounds(rows, start, end) == window["hccs"]
        coverage = []
        for device in window["hbm_devices"]:
            path = evidence / device["db"].lstrip("/")
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                samples = connection.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
            actual = integrate(samples, start, end)
            coverage.append(has_complete_hbm_coverage(actual))
            for key, value in actual.items():
                if key == "prorated_bytes":
                    assert abs(value - device[key]) <= 8 * max(
                        math.ulp(value), math.ulp(device[key])
                    )
                else:
                    assert value == device[key]
        assert all(coverage)
        assert window["hbm_bytes"] is not None
        for key, value in window["hbm_bytes"].items():
            assert math.isclose(
                value, sum(d[key] for d in window["hbm_devices"]), rel_tol=1e-14
            )


def test_loaded_configuration_matches_raw_workers(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from audit_r18_inventory import summarize, requirements

    result = json.loads(
        (evidence / SOURCE / "r18-loaded-configuration.json").read_text()
    )
    assert result["status"] == "loaded_configuration_recovered"
    for rank in result["ranks"]:
        path = evidence / rank["trace"].lstrip("/")
        raw = next(
            e
            for line in path.open()
            if (e := json.loads(line))["event"] == "loaded_model_inventory"
        )
        for role in ("target", "draft"):
            groups = summarize(raw[role])
            assert groups == rank["groups"][role]
            assert requirements(groups, role) == []
            for source, digest in raw[role]["source_sha256"].items():
                assert result["loaded_source_sha256"][source] == digest
                assert (
                    hashlib.sha256(
                        (evidence / source.lstrip("/")).read_bytes()
                    ).hexdigest()
                    == digest
                )


def test_device_lengths_reproduce_new_scalar_ledgers(evidence, monkeypatch):
    monkeypatch.syspath_prepend(str(evidence / SOURCE))
    from owned_fia_pairs import count_device_pairs
    from derive_r18_norm import cohort_entries
    from derive_r18_softmax import count_cohort
    from derive_r18_scalar import operations

    expected = json.loads((BASE / "r18-request-attention.json").read_text())["cohorts"]
    assert expected["boundary_or_tail"] == {}
    per_rank = []
    receipt = json.loads((evidence / RUN / "receipt.json").read_text())
    for export in receipt["owned_fia_exports"]:
        path = evidence / RUN / f"owned-fia-{export['pid']}.json"
        groups = {}
        for record in json.loads(path.read_text())["records"]:
            meta = record["metadata"]
            step = meta["draft_step"]
            role = "target" if step is None else f"draft_step{step}"
            values = count_device_pairs(
                record["device_query_lengths"],
                record["device_kv_lengths"],
                len(meta["host_planning_query_lengths"]["host_values"]),
                meta["tokens"],
            )
            groups.setdefault(role, Counter()).update({"publications": 1, **values})
        for role, counts in groups.items():
            counts["semantic_qk_av_arithmetic"] = (
                4 * counts["causal_pairs"] * 16 * 256 * (10 if role == "target" else 1)
            )
        per_rank.append(groups)
    assert len(per_rank) == 2
    assert (
        per_rank[0] == per_rank[1] == expected["inside_request_under_clock_tolerance"]
    )
    norm = json.loads((BASE / "r18-norm-semantics.json").read_text())
    softmax = json.loads((BASE / "r18-softmax-semantics.json").read_text())
    scalar = json.loads((BASE / "r18-scalar-semantics.json").read_text())
    for name, rows in expected.items():
        assert cohort_entries(rows) == norm["cohorts"][name]
        assert count_cohort(rows) == softmax["cohorts"][name]
        target = rows.get("target", {}).get("valid_query_tokens", 0)
        draft = sum(
            rows.get(f"draft_step{i}", {}).get("valid_query_tokens", 0) for i in (0, 1)
        )
        assert operations(target, draft) == scalar["cohorts"][name]
