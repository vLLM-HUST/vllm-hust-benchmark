"""Inert source/trace fixtures; no native replay, KV writes or model calls."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("prepare_mooncake_storage_plan")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    author, traces = tmp_path / "author", tmp_path / "traces"
    layout = author / "layout/mla.py"
    layout.parent.mkdir(parents=True)
    layout.write_text(
        "MLA_MODEL_CONFIG = {'glm5': {'kv_lora_rank': 512, 'qk_rope_head_dim': 64, 'index_head_dim': 128, 'kv_precision_bytes': 2, 'num_layers': 78}}\nraise RuntimeError('fixture: never import')\n",
        encoding="utf-8",
    )
    benchmark = author / "benchmark.py"
    benchmark.write_text(
        "raise RuntimeError('fixture: never execute')\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "AUTHOR_FILES",
        {"layout/mla.py": runner.sha(layout), "benchmark.py": runner.sha(benchmark)},
    )
    traces.mkdir()
    files = []
    rows = [
        {
            "timestamp": i * 1000,
            "input_length": 512,
            "output_length": 128,
            "hash_ids": [0, 2001],
            "original_annotation": "keep on disk",
        }
        for i in range(4)
    ]
    for name in ("conversation_trace.jsonl", "toolagent_trace.jsonl"):
        path = traces / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        files.append(
            {
                "path": name,
                "role": "data",
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "sha256": runner.sha(path),
            }
        )
    source = {"source_id": "MOONCAKE", "revision": runner.REVISION, "files": files}
    return author, traces, tmp_path / "storage", source


def test_default_capacity_profile_is_not_collision_free_requirement(inputs):
    author, traces, storage, source = inputs
    before = {path.name: path.read_bytes() for path in traces.iterdir()}
    result = runner.build_plan(author, traces, storage, source=source)
    assert result["page_bytes"] == 56229888
    assert result["retained_storage_capacity_bytes"] == 224919552000
    assert len(result["jobs"]) == 2
    assert all(job["selected_original_rows"] == 3 for job in result["jobs"])
    assert all(job["selected_requests_use_modulo_mapping"] for job in result["jobs"])
    for job in result["jobs"]:
        command = job["command"]
        assert command[command.index("--max-requests") + 1] == "3"
        assert command[command.index("--max-pages") + 1] == "2000"
        assert str(traces) in command
    assert not storage.exists()
    assert not result["storage_writes_executed"]
    assert not result["performance_claim"]
    assert result["model_preset_is_layout_not_deployed_model"]
    assert before == {path.name: path.read_bytes() for path in traces.iterdir()}


def test_full_profile_does_not_limit_or_rewrite_original_trace(inputs):
    author, traces, storage, source = inputs
    result = runner.build_plan(author, traces, storage, source=source, count=None)
    assert all(job["selected_original_rows"] == 4 for job in result["jobs"])
    assert all("--max-requests" not in job["command"] for job in result["jobs"])


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "changed-file",
        "changed-code",
        "existing-storage",
        "small-sample",
        "missing-scenario",
    ],
)
def test_preflight_rejects_incomplete_or_changed_inputs(inputs, failure):
    author, traces, storage, source = inputs
    count = 3
    if failure == "source":
        source["revision"] = "changed"
    elif failure == "changed-file":
        with (traces / "conversation_trace.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("\n")
    elif failure == "changed-code":
        (author / "benchmark.py").write_text("changed", encoding="utf-8")
    elif failure == "existing-storage":
        storage.mkdir()
    elif failure == "small-sample":
        count = 2
    else:
        source["files"].pop()
    with pytest.raises(ValueError):
        runner.build_plan(author, traces, storage, source=source, count=count)


@pytest.mark.parametrize(
    "field,value",
    [("hash_ids", [-1]), ("input_length", True), ("timestamp", float("nan"))],
)
def test_field_errors_are_not_replaced_by_author_loader_defaults(inputs, field, value):
    _, traces, _, source = inputs
    spec = source["files"][0]
    path = traces / spec["path"]
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[0][field] = value
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    spec.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="trace contract"):
        runner.trace_summary(traces, spec, 3)
