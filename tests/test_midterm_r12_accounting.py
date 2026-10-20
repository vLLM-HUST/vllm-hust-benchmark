"""Check new r12 summaries against the already published raw capture."""

import hashlib
import io
import json
import math
import sqlite3
import tarfile
import types
import unittest

from test_midterm_vocab_prefill import BASE, archive_members


def test_r12_matrix_and_vocab_against_archived_rows(tmp_path, monkeypatch):
    root = BASE / "r12-same-run-accounting"
    raw = archive_members(BASE / "deferred-device-r12")
    with tarfile.open(
        fileobj=io.BytesIO((root / "sources.tar.gz").read_bytes())
    ) as archive:
        sources = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    for name, data in sources.items():
        assert "/" not in name
        (tmp_path / name).write_bytes(data)
    run = tmp_path / "microflow-token-accounting-r12"
    run.mkdir()
    for name, data in raw.items():
        if name.startswith("microflow-token-accounting-r12/"):
            if "/moe-deferred-" in name or "/token-accounting-" in name:
                (run / name.split("/")[-1]).write_bytes(data)
    matrix = json.loads((root / "r12-matrix-accounting.json").read_text())
    vocab = json.loads((root / "vocab-projection-summary-r12.json").read_text())
    for summary in (matrix, vocab):
        for path, expected in summary["source_sha256"].items():
            basename = path.split("/")[-1]
            if basename in sources:
                assert hashlib.sha256(sources[basename]).hexdigest() == expected
            raw_key = "microflow-token-accounting-r12/" + basename
            if raw_key in raw:
                assert hashlib.sha256(raw[raw_key]).hexdigest() == expected
    (tmp_path / "r12-matrix-accounting.json").write_text(json.dumps(matrix))
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace = types.ModuleType("r12_matrix_tests")
    namespace.__file__ = str(tmp_path / "test_r12_matrix_evidence.py")
    exec(
        compile(sources["test_r12_matrix_evidence.py"], namespace.__file__, "exec"),
        namespace.__dict__,
    )
    namespace.RUN = run
    case = namespace.MatrixEvidenceTests
    suite = unittest.TestSuite(
        case(name)
        for name in (
            "test_device_boundaries_and_all_token_rows_reconciled",
            "test_routing_not_double_counted_or_tail_discarded",
            "test_routed_projection_from_global_expert_shape",
        )
    )
    result = unittest.TestResult()
    suite.run(result)
    assert result.wasSuccessful(), (result.errors, result.failures)
    from audit_vocab_projection_r12 import count_rows, projection

    receipt = json.loads(raw["microflow-token-accounting-r12/receipt.json"])
    for rank in vocab["ranks"]:
        matches = []
        for path in run.glob("token-accounting-*.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            installed = next(e for e in events if e["event"] == "observer_installed")
            if installed["rank"] == rank["rank"]:
                matches.append(
                    count_rows(
                        events,
                        vocab["request_phase"]["start_wall"],
                        receipt["bench_ended_wall"],
                    )
                )
        assert len(matches) == 1
        assert matches[0]["rows"] == rank["rows"]
        assert matches[0]["contexts"] == rank["contexts"]
    assert vocab["semantic_projection_subtotal"] == sum(
        projection(rows, [[124160, 2048]] * 2)
        for rows in vocab["ranks"][0]["rows"].values()
    )
    assert vocab["model_effective_flops"] is vocab["m01_attainment"] is None


def test_r12_hardware_intervals_against_archived_hccs(tmp_path, monkeypatch):
    root = BASE / "r12-same-run-accounting"
    raw = archive_members(BASE / "deferred-device-r12")
    with tarfile.open(
        fileobj=io.BytesIO((root / "sources.tar.gz").read_bytes())
    ) as archive:
        for member in archive:
            assert "/" not in member.name
            (tmp_path / member.name).write_bytes(archive.extractfile(member).read())
    monkeypatch.syspath_prepend(str(tmp_path))
    from integrate_r12_hardware import link_bounds
    from integrate_hbm_samples import check_export, has_complete_hbm_coverage, integrate

    summary = json.loads((root / "r12-hardware-intervals.json").read_text())
    packed = b"".join(
        path.read_bytes() for path in sorted(root.glob("hbm-db.tar.gz.part-*"))
    )
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        databases = {
            m.name: archive.extractfile(m).read() for m in archive if m.isfile()
        }
    assert set(databases) == {"device_0/sqlite/hbm.db", "device_1/sqlite/hbm.db"}
    counter = raw["system/r12-system-control/hccs64.jsonl"]
    expected_hashes = summary["source_sha256"]
    assert hashlib.sha256(counter).hexdigest() == next(
        value for key, value in expected_hashes.items() if key.endswith("/hccs64.jsonl")
    )
    rows = [json.loads(line) for line in counter.splitlines()]
    for window in summary["windows"]:
        assert (
            link_bounds(rows, window["start_wall"] * 1e9, window["end_wall"] * 1e9)
            == window["hccs"]
        )
        values = window["hbm_bytes"]
        assert 0 < values["fully_contained_bytes"] <= values["prorated_bytes"]
        assert values["prorated_bytes"] <= values["intersecting_interval_bytes"]
        assert len(window["hbm_devices"]) == 2
        totals = dict.fromkeys(values, 0)
        for device in window["hbm_devices"]:
            relative = "/".join(device["db"].split("/")[-3:])
            content = databases[relative]
            assert hashlib.sha256(content).hexdigest() == device["sha256"]
            db = tmp_path / (relative.split("/")[0] + ".db")
            db.write_bytes(content)
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
                samples = connection.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
                exported = connection.execute("SELECT * FROM HBMbwData").fetchall()
            assert check_export(samples, exported) == device["decoder_crosscheck"]
            calculated = integrate(
                samples, window["start_wall"] * 1e9, window["end_wall"] * 1e9
            )
            assert has_complete_hbm_coverage(calculated)
            for key, value in calculated.items():
                if key == "prorated_bytes":
                    # Python 3.12 changed float sum accuracy; preserve exact integer checks.
                    assert abs(device[key] - value) <= 8 * max(
                        math.ulp(device[key]), math.ulp(value)
                    )
                else:
                    assert device[key] == value
            for key in totals:
                totals[key] += calculated[key]
        for key, value in totals.items():
            if key == "prorated_bytes":
                assert abs(values[key] - value) <= 8 * max(
                    math.ulp(values[key]), math.ulp(value)
                )
            else:
                assert values[key] == value
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None
