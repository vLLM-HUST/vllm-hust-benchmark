"""Recompute supplementary ledgers against immutable r17 archive inputs."""

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path

from test_midterm_r17_device_state import RUN, SOURCE, evidence  # noqa: F401


BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/r17-norm-softmax"
)


def load(name, monkeypatch):
    path = BASE / (name + ".py.txt")
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(__import__("sys").modules, name, module)
    loader.exec_module(module)
    return module


def test_supplementary_source_hashes(evidence):  # noqa: F811
    for filename in ("r17-norm-semantics.json", "r17-softmax-semantics.json"):
        record = json.loads((BASE / filename).read_text())
        for key in ("source_sha256", "run_pinned_callsite_sha256"):
            for name, digest in record.get(key, {}).items():
                archived = evidence / name.lstrip("/")
                local = BASE / (Path(name).name + ".txt")
                source = local if local.exists() else archived
                assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_norm_inventory_and_device_cohorts(evidence, monkeypatch):  # noqa: F811
    load("audit_norm_accounting", monkeypatch)
    module = load("derive_r17_norm", monkeypatch)
    record = json.loads((BASE / "r17-norm-semantics.json").read_text())
    attention = json.loads(
        (evidence / SOURCE / "r17-request-attention.json").read_text()
    )
    assert record["cohorts"] == {
        name: module.cohort_entries(rows) for name, rows in attention["cohorts"].items()
    }
    inventories = []
    callsites = {}
    for path in sorted((evidence / RUN).glob("token-accounting-*.jsonl")):
        event = next(
            event
            for line in path.open()
            if (event := json.loads(line))["event"] == "loaded_model_inventory"
        )
        inventories.append(module.inventory_check(event))
        for role in ("target", "draft"):
            for name, digest in event[role]["source_sha256"].items():
                assert callsites.get(name, digest) == digest
                callsites[name] = digest
    assert json.loads(json.dumps(inventories)) == record["inventory_counts"]
    for name, digest in record["run_pinned_callsite_sha256"].items():
        assert callsites[name] == digest
    assert record["model_effective_flops"] is None


def test_softmax_device_cohorts(evidence, monkeypatch):  # noqa: F811
    load("derive_r12_softmax_semantics", monkeypatch)
    module = load("derive_r17_softmax", monkeypatch)
    record = json.loads((BASE / "r17-softmax-semantics.json").read_text())
    attention = json.loads(
        (evidence / SOURCE / "r17-request-attention.json").read_text()
    )
    assert record["cohorts"] == {
        name: module.count_cohort(rows) for name, rows in attention["cohorts"].items()
    }
    assert record["m01_attainment"] is None
