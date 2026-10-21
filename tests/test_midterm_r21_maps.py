"""Verify portable r21 raw maps and accounting without local experiment paths."""

import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET

import pytest

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/r21-device-maps"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r21"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("r21-maps")
    index = json.loads((BASE / "archive.json").read_text())
    packed = root / "evidence.tar.gz"
    digest = hashlib.sha256()
    with packed.open("wb") as stream:
        for part in index["parts"]:
            data = (BASE / part["file"]).read_bytes()
            assert len(data) == part["bytes"]
            assert hashlib.sha256(data).hexdigest() == part["sha256"]
            digest.update(data)
            stream.write(data)
    assert packed.stat().st_size == index["archive_bytes"]
    assert digest.hexdigest() == index["archive_sha256"]
    expected = {row["name"]: row for row in index["members"]}
    assert len(expected) == len(index["members"])
    seen = set()
    with tarfile.open(packed) as archive:
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
    for row in index["module_relative_source_resolutions"]:
        assert (
            hashlib.sha256(
                (root / row["resolved_source"].lstrip("/")).read_bytes()
            ).hexdigest()
            == row["sha256"]
        )
    return root


def test_device_maps_and_lifecycle(evidence):
    receipt = json.loads((evidence / RUN / "receipt.json").read_text())
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    for export in receipt["gdn_publication_exports"]:
        path = evidence / RUN / f"gdn-publication-{export['pid']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == export["sha256"]
        raw = json.loads(path.read_text())
        assert len(raw["records"]) == export["records"] == 882
        assert not raw["errors"]
        for row in raw["records"]:
            meta, maps = row["metadata"], row["device_maps"]
            assert len(maps["prefill_map"]) == len(maps["restore"]) == meta["tokens"]
            assert len(maps["verify_map"]) == meta["requests"] * 3
            if meta["decode_only"]:
                continue
            cu = meta["host_cu"]
            restored = [0] * meta["tokens"]
            for name, ids, offset in (
                ("prefill_map", meta["prefill_ids"], 0),
                ("verify_map", meta["verify_ids"], meta["tokens"]),
            ):
                wanted = [token for i in ids for token in range(cu[i], cu[i + 1])]
                assert maps[name] == wanted + [0] * (len(maps[name]) - len(wanted))
                for packed, token in enumerate(wanted):
                    restored[token] = offset + packed
            assert maps["restore"] == restored
    suite = ET.parse(evidence / SOURCE / "r21-device-map-tests-final.xml")
    assert len(suite.findall(".//testcase")) == 18
    assert not suite.findall(".//failure") and not suite.findall(".//error")


def test_accounting_sources_and_no_full_m01_claim(evidence):
    for name in (
        "r21-device-map-window.json",
        "r21-accounted-families.json",
        "r21-hardware-intervals.json",
    ):
        path = BASE / name
        assert path.read_bytes() == (evidence / SOURCE / name).read_bytes()
        data = json.loads(path.read_text())
        for source, digest in data["source_sha256"].items():
            assert (
                hashlib.sha256((evidence / source.lstrip("/")).read_bytes()).hexdigest()
                == digest
            )
    ledger = json.loads((BASE / "r21-accounted-families.json").read_text())
    assert ledger["model_effective_flops"] is ledger["m01_attainment"] is None
    assert (
        len({r["family"] for r in ledger["families"]}) == len(ledger["families"]) == 10
    )
    assert (
        sum(r["global_basic_operations"] for r in ledger["families"])
        == ledger["accounted_basic_operation_subtotal"]
    )


def test_graph_intervals_and_zero_batch_retained(evidence):
    path = evidence / SOURCE / "test_gmm_graph_device_r6.py"
    spec = importlib.util.spec_from_file_location("portable_r6_interval_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN = (
        evidence / "workspace/gennova-measurements/gmm-capacity-r6-graph-traced"
    )
    module.test_all_repetitions_and_independent_interval_sweep()
    module.test_zero_overlap_first_batch_not_dropped()
    receipt = json.loads((module.RUN / "receipt.json").read_text())
    assert receipt["exit_code"] == receipt["cleanup_exit_code"] == 0
    assert (
        receipt["all_four_idle_after"]
        and not receipt["protocol"]["prior_cache_imported"]
    )
    for rank in range(2):
        rows = [
            json.loads(line)
            for line in (module.RUN / f"graph-numerics-{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["case_index"] for r in rows] == list(range(146))
        assert all(
            r["checked_outputs"] == 8 and r["changed_input_scale"] == -1 for r in rows
        )
