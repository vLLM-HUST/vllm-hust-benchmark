"""Reproduce r14 interval accounting using previously sealed device evidence."""

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import sqlite3
import tarfile

import pytest

from test_midterm_vocab_prefill import BASE, archive_members

ROOT = BASE / "r14-interval-accounting"
RUN = "microflow-token-accounting-r14/"


@pytest.fixture(scope="module")
def evidence():
    return archive_members(BASE / "owned-fia-r13-r14")


@pytest.fixture
def sources(tmp_path, monkeypatch):
    with tarfile.open(
        fileobj=io.BytesIO((ROOT / "sources.tar.gz").read_bytes())
    ) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            target = tmp_path / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_request_partition_from_actual_lengths(evidence, sources):
    from derive_r14_attention_window import associate
    from integrate_r14_hardware import classify
    from owned_fia_pairs import count_device_pairs
    from recover_model_hardware import request_phase

    expected = json.loads((ROOT / "r14-request-attention.json").read_text())
    receipt = json.loads(evidence[RUN + "receipt.json"])
    result = json.loads(evidence[RUN + "mmlu-pro.json"])
    phase = request_phase(receipt, result)
    assert phase == expected["request_phase"]
    captures, joined = [], []
    for export in receipt["owned_fia_exports"]:
        pid = export["pid"]
        raw = json.loads(evidence[RUN + f"moe-deferred-{pid}.json"])
        owned = json.loads(evidence[RUN + f"owned-fia-{pid}.json"])
        captures.append(raw)
        joined.append(associate(raw["frames"], owned["records"]))
    totals = {key: {} for key in expected["cohorts"]}
    for i, frames in enumerate(zip(*(r["frames"] for r in captures), strict=True)):
        inside = []
        for raw, frame in zip(captures, frames, strict=True):
            a, b = raw["anchor_host_bracket"]
            start = [
                a + frame["start_ms"] / 1000 - 0.002,
                b + frame["start_ms"] / 1000 + 0.002,
            ]
            end = [
                a + frame["end_ms"] / 1000 - 0.002,
                b + frame["end_ms"] / 1000 + 0.002,
            ]
            inside.append(classify(start, end, phase) == "inside")
        cohort = totals[
            "inside_request_under_clock_tolerance"
            if all(inside)
            else "boundary_or_tail"
        ]
        for left, right in zip(joined[0][i], joined[1][i], strict=True):
            meta = left["metadata"]
            assert left["device_query_lengths"] == right["device_query_lengths"]
            assert left["device_kv_lengths"] == right["device_kv_lengths"]
            step = meta["draft_step"]
            role = "target" if step is None else f"draft_step{step}"
            counts = count_device_pairs(
                left["device_query_lengths"],
                left["device_kv_lengths"],
                len(meta["host_planning_query_lengths"]["host_values"]),
                meta["tokens"],
            )
            cohort.setdefault(role, Counter()).update({"publications": 1, **counts})
    for cohort in totals.values():
        for role, counts in cohort.items():
            counts["semantic_qk_av_arithmetic"] = (
                4 * counts["causal_pairs"] * 16 * 256 * (10 if role == "target" else 1)
            )
    assert totals == expected["cohorts"]
    complete = json.loads(evidence["source/r14-device-attention-pairs.json"])
    for role, counts in complete["ranks"][0]["roles"].items():
        for key in totals["boundary_or_tail"][role]:
            if key == "semantic_qk_av_arithmetic":
                continue
            full_key = key if key == "publications" else "device_" + key
            assert sum(c[role][key] for c in totals.values()) == counts[full_key]
    assert expected["model_effective_flops"] is expected["m01_attainment"] is None


def test_hardware_and_boundaries_from_raw_evidence(evidence, sources):
    from integrate_r14_hardware import boundaries
    from integrate_r12_hardware import link_bounds
    from integrate_hbm_samples import check_export, has_complete_hbm_coverage, integrate

    expected = json.loads((ROOT / "r14-hardware-intervals.json").read_text())
    run = sources / "run"
    run.mkdir()
    for name, content in evidence.items():
        if name.startswith(RUN) and any(
            x in name for x in ("moe-deferred-", "owned-fia-")
        ):
            (run / name.split("/")[-1]).write_bytes(content)
    ranks, windows, _ = boundaries(run, expected["request_phase"])
    assert ranks == expected["ranks"]
    raw_counter = evidence["system/r14-system-control/hccs64.jsonl"]
    rows = [json.loads(line) for line in raw_counter.splitlines()]
    for computed, window in zip(windows, expected["windows"], strict=True):
        assert all(window[k] == v for k, v in computed.items())
        assert window["duration_seconds"] == window["end_wall"] - window["start_wall"]
        start, end = window["start_wall"] * 1e9, window["end_wall"] * 1e9
        assert link_bounds(rows, start, end) == window["hccs"]
        totals = Counter()
        for device in window["hbm_devices"]:
            suffix = "/".join(device["db"].split("/")[-3:])
            matches = [
                data
                for key, data in evidence.items()
                if key.startswith("system/r14-profiler/") and key.endswith("/" + suffix)
            ]
            assert len(matches) == 1
            assert hashlib.sha256(matches[0]).hexdigest() == device["sha256"]
            db = sources / "hbm.db"
            db.write_bytes(matches[0])
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                samples = conn.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
                exported = conn.execute("SELECT * FROM HBMbwData").fetchall()
            assert check_export(samples, exported) == device["decoder_crosscheck"]
            actual = integrate(samples, start, end)
            assert has_complete_hbm_coverage(actual)
            for key, value in actual.items():
                if key == "prorated_bytes":
                    assert abs(value - device[key]) <= 8 * max(
                        math.ulp(value), math.ulp(device[key])
                    )
                else:
                    assert value == device[key]
            totals.update({k: actual[k] for k in window["hbm_bytes"]})
        for key, value in totals.items():
            recorded = window["hbm_bytes"][key]
            if key == "prorated_bytes":
                assert abs(value - recorded) <= 8 * max(
                    math.ulp(value), math.ulp(recorded)
                )
            else:
                assert value == recorded
    assert expected["model_effective_flops"] is expected["m01_attainment"] is None


def test_source_bindings(evidence, sources):
    for filename in ("r14-hardware-intervals.json", "r14-request-attention.json"):
        summary = json.loads((ROOT / filename).read_text())
        for path, digest in summary["source_sha256"].items():
            basename = path.split("/")[-1]
            if "/microflow-token-accounting-r14/" in path:
                data = evidence[RUN + basename]
            elif "/r14-system-control/" in path:
                data = evidence["system/r14-system-control/" + basename]
            elif "/hccs-capacity-control-r1/" in path:
                data = (sources / "hccs-capacity-control-r1" / basename).read_bytes()
            else:
                data = (sources / basename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
