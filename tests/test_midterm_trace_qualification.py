from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
FOLDER = REPORT / "m01-recovery/trace-qualification-20261021"


@lru_cache
def evidence():
    index = json.loads((FOLDER / "archive.json").read_text())
    result = {}
    for archive in index["archives"]:
        chunks = []
        for part in archive["parts"]:
            raw = (FOLDER / part["name"]).read_bytes()
            assert len(raw) == part["bytes"]
            assert hashlib.sha256(raw).hexdigest() == part["sha256"]
            chunks.append(raw)
        raw = b"".join(chunks)
        assert len(raw) == archive["bytes"]
        assert hashlib.sha256(raw).hexdigest() == archive["sha256"]
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
        assert set(members) == {m["name"] for m in archive["members"]}
        for member in archive["members"]:
            data = members[member["name"]]
            assert len(data) == member["bytes"]
            assert hashlib.sha256(data).hexdigest() == member["sha256"]
        result[archive["name"]] = members
    return index, result


def test_recovery_and_limits_remain_distinct():
    index, archives = evidence()
    assert index["whole_model_dependency_closure_proven"] is False
    assert index["model_effective_flops"] is index["m01_attainment"] is None
    mapping = json.loads(
        archives["m01-r7-map-reconciliation-20261021.tar.gz"][
            "m01-r7-map-reconciliation.json"
        ]
    )
    assert all(r["exact_previous_records_match"] for r in mapping["ranks"])
    assert len(mapping["negative_checks"]) == 2
    capability = json.loads(
        archives["m01-r7-trace-capability-20261021.tar.gz"][
            "m01-r7-trace-capability.json"
        ]
    )
    assert all(
        p["runtimeApi"] == p["taskTime"] == "off"
        for p in capability["effective_profiles"].values()
    )
    assert not any(t["name"] in {"TASK", "CANN_API"} for t in capability["tables"])


def test_actual_native_calls_match_profiler_and_probe_edges(tmp_path):
    _, archives = evidence()
    members = archives["m01-event-identity-r2-evidence.tar.gz"]
    prefix = "m01-trace-qualification-r2/"
    audit = json.loads(members[prefix + "event-identity-audit.json"])
    capture = json.loads(members[prefix + "qualification-audit.json"])
    namespace = {}
    exec(
        compile(members["m01_event_identity_audit.py"], "archived_event_audit", "exec"),
        namespace,
    )
    total = 0
    for rank in audit["ranks"]:
        worker = json.loads(members[prefix + f"rank-{rank['rank']}.json"])
        rows = [
            json.loads(line)
            for line in members[prefix + f"events-{rank['pid']}.jsonl"].splitlines()
        ]
        assert all(row["pid"] == worker["pid"] for row in rows)
        counts = Counter(row["op"] for row in rows)
        database = next(
            r["database"] for r in capture["ranks"] if r["rank"] == rank["rank"]
        )
        name = database.removeprefix("/workspace/")
        target = tmp_path / f"rank-{rank['rank']}.db"
        target.write_bytes(members[name])
        with sqlite3.connect(target.as_uri() + "?mode=ro", uri=True) as db:
            api = dict(
                db.execute(
                    "SELECT s.value,count(*) FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name GROUP BY s.value"
                )
            )
            assert counts == dict(
                record=api["EventRecord"], wait=api["StreamWaitEvent"]
            )
            assert list(db.execute("SELECT DISTINCT globalPid FROM TASK")) == [
                (worker["pid"],)
            ]
            assert (
                db.execute("SELECT count(*) FROM COMMUNICATION_OP").fetchone()[0] == 4
            )
        edges = []
        for check in worker["checks"]:
            assert check["numeric_passed"]
            edges.extend(
                [
                    namespace["edge"](
                        rows,
                        check["produced"],
                        worker["main_stream"],
                        worker["side_stream"],
                    ),
                    namespace["edge"](
                        rows,
                        check["consumed"],
                        worker["side_stream"],
                        worker["main_stream"],
                    ),
                ]
            )
        assert edges == rank["event_edges"]
        total += len(edges)
    assert total == audit["total_known_probe_edges_verified"] == 16
    assert audit["whole_model_dependency_closure_proven"] is False


def test_both_qualifications_released_devices_and_report_does_not_claim_full_model():
    _, archives = evidence()
    for name, run in [
        ("m01-trace-qualification-r1-evidence.tar.gz", "m01-trace-qualification-r1"),
        ("m01-event-identity-r2-evidence.tar.gz", "m01-trace-qualification-r2"),
    ]:
        receipt = json.loads(archives[name][run + "/receipt.json"])
        assert receipt["exit_code"] == receipt["final_exit_code"] == 0
        assert receipt["all_devices_released"]
    text = (REPORT / "中期测试报告.md").read_text()
    assert "16条已知探针依赖通过" in text
    assert "不是35B模型性能复测" in text
    assert "本轮验收判定不变" in text
