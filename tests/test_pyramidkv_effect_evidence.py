"""Synthetic integrity failures must not become accepted experiment evidence."""

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports/pyramidkv-standalone-effects-20261103"
)


def module(name="evidence"):
    spec = importlib.util.spec_from_file_location(
        "effect_" + name, REPORT / f"{name}.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPORT))
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.path.remove(str(REPORT))
    return loaded


def checksums(root):
    entries = [
        p for p in sorted(root.iterdir()) if p.name != "SHA256SUMS" and p.is_file()
    ]
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in entries
        )
    )


def capsule(root, entries=None, *, parts=False):
    entries = entries or [("run/response.json", b'{"output":"7"}\n', tarfile.REGTYPE)]
    content = io.BytesIO()
    inventory = {}
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = "/tmp/never-extract-this"
            archive.addfile(info, io.BytesIO(data))
            inventory[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    blob = content.getvalue()
    if parts:
        middle = len(blob) // 2
        (root / "raw.tar.gz.part-000").write_bytes(blob[:middle])
        (root / "raw.tar.gz.part-001").write_bytes(blob[middle:])
    else:
        (root / "raw.tar.gz").write_bytes(blob)
    (root / "members.json").write_text(json.dumps(inventory))
    checksums(root)


@pytest.mark.parametrize("parts", [False, True])
def test_capsule_returns_original_bytes_without_extracting(tmp_path, parts):
    capsule(tmp_path, parts=parts)
    assert module().load_capsule(tmp_path) == {"run/response.json": b'{"output":"7"}\n'}
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "run/../escape", "run//file", "run\\file"]
)
def test_capsule_rejects_unsafe_member_even_with_matching_manifest(tmp_path, name):
    capsule(tmp_path, [(name, b"x", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="Unsafe member"):
        module().load_capsule(tmp_path)


def test_capsule_rejects_duplicate_member_even_with_matching_manifest(tmp_path):
    capsule(tmp_path, [("same", b"x", tarfile.REGTYPE)] * 2)
    with pytest.raises(ValueError, match="Duplicate member"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.DIRTYPE, tarfile.LNKTYPE])
def test_capsule_rejects_nonregular_members(tmp_path, kind):
    capsule(tmp_path, [("entry", b"", kind)])
    with pytest.raises(ValueError, match="Nonregular member"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra", "length", "digest"])
def test_capsule_rejects_member_inventory_corruption_after_outer_rehash(
    tmp_path, change
):
    capsule(tmp_path)
    path = tmp_path / "members.json"
    members = json.loads(path.read_text())
    if change == "missing":
        members.clear()
    elif change == "extra":
        members["absent"] = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    elif change == "length":
        members["run/response.json"]["bytes"] += 1
    else:
        members["run/response.json"]["sha256"] = "0" * 64
    path.write_text(json.dumps(members))
    checksums(tmp_path)
    with pytest.raises(ValueError, match="Member (inventory|length|digest)"):
        module().load_capsule(tmp_path)


def test_capsule_rejects_changed_outer_analysis(tmp_path):
    capsule(tmp_path)
    (tmp_path / "analysis.json").write_text('{"throughput": 1}')
    checksums(tmp_path)
    (tmp_path / "analysis.json").write_text('{"throughput": 999}')
    with pytest.raises(ValueError, match="File digest mismatch: analysis.json"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("change", ["duplicate", "omitted", "unsafe", "gap"])
def test_capsule_requires_complete_unique_safe_outer_inventory(tmp_path, change):
    capsule(tmp_path, parts=True)
    path = tmp_path / "SHA256SUMS"
    original = path.read_text()
    if change == "duplicate":
        path.write_text(original + original.splitlines()[0] + "\n")
    elif change == "omitted":
        path.write_text("\n".join(original.splitlines()[1:]) + "\n")
    elif change == "unsafe":
        path.write_text(original + "0" * 64 + "  ../outside\n")
    else:
        (tmp_path / "raw.tar.gz.part-001").rename(tmp_path / "raw.tar.gz.part-002")
        checksums(tmp_path)
    with pytest.raises(ValueError, match="(Checksum|File inventory|Archive part)"):
        module().load_capsule(tmp_path)


def test_capsule_rejects_duplicate_json_inventory_keys(tmp_path):
    capsule(tmp_path)
    path = tmp_path / "members.json"
    original = path.read_text()
    path.write_text(original[:-1] + "," + original[1:])
    checksums(tmp_path)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        module().load_capsule(tmp_path)


@pytest.mark.parametrize("optimized", [False, True])
def test_real_loader_integrity_checks_survive_optimized_python(tmp_path, optimized):
    capsule(tmp_path)
    (tmp_path / "members.json").write_text("{}")
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        "-c",
        (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from evidence import load_capsule; load_capsule(Path(sys.argv[2]))"
        ),
        str(REPORT),
        str(tmp_path),
    ]
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert "File digest mismatch: members.json" in result.stderr


def response_fixture():
    case = {
        "case_id": "boundary-2",
        "kind": "boundary",
        "prompt_tokens": 2,
        "prompt_sha256": hashlib.sha256(b"[11,12]").hexdigest(),
        "max_tokens": 2,
        "ignore_eos": True,
        "require_uncached": True,
        "cache_salt": "isolated-input",
    }
    payload = {
        "model": "qwen35-pyramidkv",
        "prompt": [11, 12],
        "max_tokens": 2,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 17,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": "isolated-input",
    }
    usage = {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    row = dict(
        case,
        request_id="cmpl-example",
        output="seven",
        usage=usage,
        http_status=200,
        success=True,
        finish_reason="length",
        ttft_seconds=1.0,
        e2e_seconds=3.0,
        tpot_seconds=1.0,
    )
    chunks = [
        {
            "id": "cmpl-example",
            "choices": [{"index": 0, "text": "seven", "finish_reason": None}],
        },
        {
            "id": "cmpl-example",
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
        },
        {"id": "cmpl-example", "choices": [], "usage": usage},
    ]
    return case, payload, row, chunks


def sse(chunks, done=True):
    return (
        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + ("data: [DONE]\n\n" if done else "")
    ).encode()


def test_request_checks_real_payload_sse_and_usage():
    case, payload, row, chunks = response_fixture()
    assert module().verify_response(case, row, payload, sse(chunks)) == {
        "payload_verified": True
    }


@pytest.mark.parametrize(
    "change", ["text", "id", "usage", "done", "after_done", "finish", "choice"]
)
def test_request_rejects_inconsistent_raw_stream(change):
    case, payload, row, chunks = response_fixture()
    chunks = copy.deepcopy(chunks)
    if change == "text":
        chunks[0]["choices"][0]["text"] = "different"
    elif change == "id":
        chunks[0]["id"] = "cmpl-someone-else"
    elif change == "usage":
        chunks[-1]["usage"]["completion_tokens"] = 1
    elif change == "finish":
        chunks[1]["choices"][0]["finish_reason"] = "stop"
    elif change == "choice":
        chunks[0]["choices"][0]["index"] = 1
    raw = sse(chunks, done=change != "done")
    if change == "after_done":
        raw += b'data: {"id":"cmpl-example","choices":[]}\n'
    with pytest.raises(ValueError, match="SSE"):
        module().verify_response(case, row, payload, raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", [11, 13]),
        ("seed", 18),
        ("temperature", 0.5),
        ("max_tokens", 3),
        ("ignore_eos", False),
        ("cache_salt", "other"),
        ("stream", False),
        ("stream_options", {}),
        ("model", "other-model"),
    ],
)
def test_request_rejects_changed_generation_or_prompt(field, value):
    case, payload, row, chunks = response_fixture()
    payload[field] = value
    with pytest.raises(ValueError, match="Payload"):
        module().verify_response(case, row, payload, sse(chunks))


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_tokens", 1),
        ("completion_tokens", 1),
        ("total_tokens", 999),
        ("prompt_tokens_details", None),
        ("prompt_tokens_details", {"cached_tokens": 1}),
    ],
)
def test_request_rejects_consistent_but_invalid_usage(field, value):
    case, payload, row, chunks = response_fixture()
    # Shared usage deliberately changes both SSE and the derived row.
    row["usage"][field] = value
    with pytest.raises(ValueError, match="Usage"):
        module().verify_response(case, row, payload, sse(chunks))


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", False),
        ("http_status", 503),
        ("ttft_seconds", float("nan")),
        ("e2e_seconds", -1),
        ("tpot_seconds", 9),
        ("case_id", "other"),
    ],
)
def test_request_rejects_changed_result_metadata_and_invalid_timing(field, value):
    case, payload, row, chunks = response_fixture()
    row[field] = value
    with pytest.raises(ValueError, match="Result"):
        module().verify_response(case, row, payload, sse(chunks))


def test_pressure_cached_usage_is_retained_when_policy_allows_it():
    case, payload, row, chunks = response_fixture()
    case["require_uncached"] = row["require_uncached"] = False
    row["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    assert module().verify_response(case, row, payload, sse(chunks)) == {
        "payload_verified": True
    }
    assert row["usage"]["prompt_tokens_details"]["cached_tokens"] == 1


def test_omitted_quality_payload_has_explicitly_limited_verification():
    case, _payload, row, chunks = response_fixture()
    case["kind"] = row["kind"] = "quality"
    assert module().verify_response(case, row, None, sse(chunks)) == {
        "payload_verified": False
    }


def test_omitted_synthetic_payload_is_an_error():
    case, _payload, row, chunks = response_fixture()
    with pytest.raises(ValueError, match="Payload missing"):
        module().verify_response(case, row, None, sse(chunks))


@pytest.mark.parametrize("ids", [["b", "a"], ["a"], ["a", "a"], ["a", "b", "c"]])
def test_request_membership_cannot_reorder_omit_duplicate_or_add_cases(ids):
    with pytest.raises(ValueError, match="Case order"):
        module().verify_case_order(
            [{"case_id": "a"}, {"case_id": "b"}], [{"case_id": x} for x in ids]
        )


def transaction_fixture(attempts=1):
    engine = "cmpl-example-0-abcdef12"
    rows = [{"case_id": "long", "request_id": "cmpl-example", "prompt_tokens": 8192}]
    log = "PyramidKV plugin activated\n"
    for _ in range(attempts):
        for rank in (0, 1):
            log += (
                f"(Worker_TP{rank} pid={100 + rank}) INFO worker commit acknowledged "
                f"request_id={engine} semantic_tokens=8192 physical_tokens=2048\n"
            )
        log += (
            f"INFO scheduler commit request_id={engine} semantic_tokens=8192 "
            "physical_tokens=2048 source_blocks=128 destination_blocks=32 released_blocks=128\n"
        )
    return rows, log


def test_transactions_match_each_request_and_both_tp_ranks():
    rows, log = transaction_fixture()
    result = module().verify_transactions(log, rows, True, pressure=False)
    assert len(result["transactions"]) == 1
    assert result["transactions"][0]["acknowledged_tp_ranks"] == [0, 1]
    assert result["repeated_commit_cases"] == []


def test_pressure_preserves_each_recomputed_transaction():
    rows, log = transaction_fixture(2)
    result = module().verify_transactions(log, rows, True, pressure=True)
    assert len(result["transactions"]) == 2
    assert result["repeated_commit_cases"] == [{"case_id": "long", "commits": 2}]
    assert result["requires_recomputation_review"] is True


@pytest.mark.parametrize(
    "change", ["rank", "extra_ack", "orphan", "length", "release", "repeat", "off"]
)
def test_transactions_reject_missing_extra_or_inconsistent_evidence(change):
    rows, log = transaction_fixture()
    if change == "rank":
        log = log.replace("Worker_TP1", "Worker_TP2")
    elif change == "extra_ack":
        log += log.splitlines()[1] + "\n"
    elif change == "orphan":
        log += log.splitlines()[-1].replace("cmpl-example", "cmpl-orphan") + "\n"
    elif change == "length":
        log = log.replace("physical_tokens=2048", "physical_tokens=2049")
    elif change == "release":
        log = log.replace("released_blocks=128", "released_blocks=97")
    elif change == "repeat":
        rows, log = transaction_fixture(2)
    with pytest.raises(ValueError, match="Transaction"):
        module().verify_transactions(log, rows, change != "off", pressure=False)


def test_short_and_off_requests_do_not_require_compression():
    rows = [{"case_id": "short", "request_id": "cmpl-short", "prompt_tokens": 4096}]
    assert (
        module().verify_transactions(
            "PyramidKV plugin activated", rows, True, pressure=False
        )["transactions"]
        == []
    )
    rows[0]["prompt_tokens"] = 8192
    assert (
        module().verify_transactions("native service", rows, False, pressure=False)[
            "transactions"
        ]
        == []
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_two_worker_prefixes_with_two_identical_records_are_conserved(reverse):
    rows, log = transaction_fixture()
    lines = log.splitlines()
    prefix0, body = lines[1].split("INFO ", 1)
    prefix1, body1 = lines[2].split("INFO ", 1)
    assert body == body1  # These are two indistinguishable records with explicit ranks.
    prefixes = prefix1 + prefix0 if reverse else prefix0 + prefix1
    raw = "\n".join([lines[0], prefixes + "INFO " + body, "INFO " + body, lines[3]])
    result = module().verify_transactions(raw, rows, True, pressure=False)
    assert result["transactions"][0]["acknowledged_tp_ranks"] == [0, 1]
    assert len(result["log_recoveries"]) == 1
    assert result["log_recoveries"][0]["kind"] == "two-prefix-identical-pair"


def test_two_prefixes_cannot_supply_a_missing_second_record():
    rows, log = transaction_fixture()
    lines = log.splitlines()
    prefix1 = lines[2].split("INFO ", 1)[0]
    raw = "\n".join([lines[0], prefix1 + lines[1], lines[3]])
    with pytest.raises(ValueError, match="Transaction"):
        module().verify_transactions(raw, rows, True, pressure=False)


def test_two_prefixes_with_different_records_remain_ambiguous():
    rows, log = transaction_fixture()
    lines = log.splitlines()
    prefix1, body1 = lines[2].split("INFO ", 1)
    raw = "\n".join(
        [
            lines[0],
            prefix1 + lines[1],
            "INFO " + body1.replace("8192", "8193"),
            lines[3],
        ]
    )
    with pytest.raises(ValueError, match="Transaction"):
        module().verify_transactions(raw, rows, True, pressure=False)


def test_service_summary_pools_tokens_over_wall_time_across_processes():
    first = [
        {
            "case_id": "same-id",
            "success": True,
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "ttft_seconds": 1,
            "e2e_seconds": 3,
            "tpot_seconds": 1,
        }
    ]
    second = [
        dict(
            first[0],
            usage={
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            ttft_seconds=3,
            e2e_seconds=8,
            tpot_seconds=2,
        )
    ]
    groups = [
        [{"case_ids": ["same-id"], "wall_seconds": 1}],
        [{"case_ids": ["same-id"], "wall_seconds": 8}],
    ]
    result = module("analyze").summarize_groups(
        [(groups[0], first), (groups[1], second)]
    )
    assert result["output_tokens_per_second"] == pytest.approx(10 / 9)
    assert result["cohort_output_tokens_per_second"] == [2, 1]
    assert result["cached_prompt_tokens"] == 4
    assert result["successful_requests"] == 2
    assert result["ttft_seconds"] == {"mean": 2, "p50": 2, "p95": 2.9}


def quality_fixture():
    return [
        {
            "case_id": f"{task}-{index:03d}",
            "kind": "quality",
            "task": task,
            "subset": "development" if index < 50 else "holdout",
            "qa_f1": 0.5,
        }
        for task in ("narrativeqa", "qasper", "2wikimqa")
        for index in range(100)
    ]


@pytest.mark.parametrize("failure", ["mean", "task"])
def test_quality_gate_cannot_hide_failure_in_a_task_or_subset(failure):
    off = quality_fixture()
    on = copy.deepcopy(off)
    for row in on:
        if failure == "mean":
            row["qa_f1"] = 0.54 if row["subset"] == "development" else 0.46
        elif row["subset"] == "development":
            row["qa_f1"] = 0.44 if row["task"] == "qasper" else 0.53
    result = module("analyze").quality_comparison(off, on)
    assert result["gate_passed"] is False
    subset = "holdout" if failure == "mean" else "development"
    assert result["subsets"][subset]["gate_passed"] is False
    assert result["subsets"][subset]["samples_per_task"] == {
        "narrativeqa": 50,
        "qasper": 50,
        "2wikimqa": 50,
    }


def test_quality_requires_every_original_row_and_matching_subsets():
    off, on = quality_fixture(), quality_fixture()
    on.pop()
    with pytest.raises(ValueError, match="Quality membership"):
        module("analyze").quality_comparison(off, on)


@pytest.mark.parametrize(
    "prediction,answers,want",
    [
        ("The red fox!", ["a red fox"], 1),
        ("red red fox", ["red fox"], 0.8),
        ("red fox", ["blue cat", "fox"], 2 / 3),
        ("", [""], 0),
    ],
)
def test_quality_f1_normalizes_and_uses_best_reference(prediction, answers, want):
    assert module("analyze").qa_f1(prediction, answers) == pytest.approx(want)


def telemetry_fixture():
    metrics = {
        "vllm:kv_cache_usage_perc": 0.2,
        "vllm:num_requests_running": 1,
        "vllm:num_requests_waiting": 0,
        "vllm:num_preemptions_total": 0,
        "vllm:spec_decode_num_draft_tokens_total": 2,
        "vllm:spec_decode_num_accepted_tokens_total": 1,
    }
    device = "| 1 910B2 |\n| 0 | 0 / 0 47000 / 65536 |\n| 6 910B2 |\n| 0 | 0 / 0 47002 / 65536 |\n"
    return [
        {"time_unix": 10, "phase": "warmup", "metrics": metrics, "npu_smi": device},
        {
            "time_unix": 11,
            "phase": "perf",
            "metrics": dict(
                metrics,
                **{
                    "vllm:kv_cache_usage_perc": 0.7,
                    "vllm:num_requests_running": 4,
                    "vllm:num_requests_waiting": 2,
                    "vllm:num_preemptions_total": 1,
                },
            ),
            "npu_smi": device.replace("47000", "48000").replace("47002", "48002"),
        },
    ]


def test_resource_summary_keeps_sampled_hbm_and_kv_ownership_separate():
    result = module("analyze").resource_summary(telemetry_fixture(), [1, 6])
    assert result["device_hbm_mib"] == {
        "1": {"minimum": 47000, "peak": 48000},
        "6": {"minimum": 47002, "peak": 48002},
    }
    assert result["peak_kv_usage_fraction"] == 0.7
    assert result["phases"]["perf"]["peak_running_requests"] == 4
    assert result["phases"]["perf"]["peak_waiting_requests"] == 2
    assert result["samples"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "time",
        "missing",
        "nonfinite",
        "negative",
        "error",
        "device",
        "hbm",
        "counter_reset",
        "no_hbm",
    ],
)
def test_resource_summary_rejects_incomplete_or_inconsistent_samples(change):
    rows = telemetry_fixture()
    if change == "time":
        rows[1]["time_unix"] = 9
    elif change == "missing":
        del rows[1]["metrics"]["vllm:num_preemptions_total"]
    elif change == "nonfinite":
        rows[1]["metrics"]["vllm:kv_cache_usage_perc"] = float("nan")
    elif change == "negative":
        rows[1]["metrics"]["vllm:num_requests_waiting"] = -1
    elif change == "error":
        rows[1]["error"] = "sample failed"
    elif change == "device":
        rows[1]["npu_smi"] = rows[1]["npu_smi"].replace("6 910B2", "7 910B2")
    elif change == "hbm":
        rows[1]["npu_smi"] = rows[1]["npu_smi"].replace("65536", "32768")
    elif change == "counter_reset":
        rows[1]["metrics"]["vllm:spec_decode_num_draft_tokens_total"] = 0
    else:
        for row in rows:
            row.pop("npu_smi")
    with pytest.raises(ValueError, match="Telemetry"):
        module("analyze").resource_summary(rows, [1, 6])


def metrics_fixture(preemptions, draft, accepted):
    return "\n".join(
        f'vllm:{key}_total{{model_name="model"}} {value}'
        for key, value in [
            ("num_preemptions", preemptions),
            ("spec_decode_num_draft_tokens", draft),
            ("spec_decode_num_accepted_tokens", accepted),
        ]
    )


def test_counter_deltas_use_process_boundaries():
    result = module("analyze").counter_deltas(
        metrics_fixture(0, 2, 1), metrics_fixture(3, 12, 7)
    )
    assert result == {
        "num_preemptions": 3,
        "spec_decode_num_draft_tokens": 10,
        "spec_decode_num_accepted_tokens": 6,
    }


@pytest.mark.parametrize(
    "after",
    [
        metrics_fixture(-1, 12, 7),
        metrics_fixture(0, 1, 0),
        metrics_fixture(0, 12, 20),
        metrics_fixture(0, "NaN", 7),
        "",
    ],
)
def test_counter_deltas_reject_reset_impossible_acceptance_or_missing_metrics(after):
    with pytest.raises(ValueError, match="Counter"):
        module("analyze").counter_deltas(metrics_fixture(0, 2, 1), after)


def private_fixture(root):
    (root / "data").mkdir()
    hashes = {}
    for task in ("narrativeqa", "qasper", "2wikimqa"):
        data = "".join(
            json.dumps({"_id": f"{task}-{i}", "answers": [f"answer-{i}"]}) + "\n"
            for i in range(100)
        ).encode()
        (root / "data" / f"{task}.jsonl").write_bytes(data)
        hashes[task] = hashlib.sha256(data).hexdigest()
    return {"dataset_files_sha256": hashes}


def test_private_reference_loading_checks_pinned_files_and_row_identity(tmp_path):
    manifest = private_fixture(tmp_path)
    result = module("private_inputs").load_references(tmp_path, manifest)
    assert len(result) == 300
    assert result["qasper-050"] == {"dataset_id": "qasper-50", "answers": ["answer-50"]}


@pytest.mark.parametrize("change", ["missing", "digest", "too_few", "bad_answers"])
def test_private_reference_loading_rejects_missing_wrong_or_invalid_inputs(
    tmp_path, change
):
    manifest = private_fixture(tmp_path)
    path = tmp_path / "data/qasper.jsonl"
    if change == "missing":
        path.unlink()
    elif change == "digest":
        path.write_text(path.read_text().replace("answer-50", "changed-answer"))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "too_few":
            rows.pop()
        else:
            rows[50]["answers"] = []
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["dataset_files_sha256"]["qasper"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError, match="Private"):
        module("private_inputs").load_references(tmp_path, manifest)


def test_rebuilt_private_request_matches_recorded_generation_shape():
    case, payload, _row, _chunks = response_fixture()
    case["prompt"] = [11, 12]
    result = module("private_inputs").request_bytes(case)
    assert json.loads(result) == payload
    assert result.endswith(b"\n")
    assert (
        hashlib.sha256(result).hexdigest()
        == hashlib.sha256(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
        ).hexdigest()
    )


def runtime_fixture():
    experiment = "/root/.cache/pyramidkv-issue31-20261103"
    model = "/root/workspace/Qwen3.5-35B-A3B"
    options = {
        "--host": "127.0.0.1",
        "--port": "8013",
        "--served-model-name": "qwen35-pyramidkv",
        "--tensor-parallel-size": "2",
        "--distributed-executor-backend": "mp",
        "--worker-cls": "vllm_ascend_pyramidkv.adapter.host_worker.PyramidKVHostWorker",
        "--dtype": "bfloat16",
        "--max-model-len": "32768",
        "--max-num-seqs": "4",
        "--max-num-batched-tokens": "4096",
        "--gpu-memory-utilization": "0.9",
        "--kv-cache-memory-bytes": "8589934592",
        "--seed": "17",
        "--mamba-cache-mode": "align",
        "--additional-config": '{"enable_cpu_binding":false}',
        "--limit-mm-per-prompt": '{"image":0,"video":0}',
        "--compilation-config": '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[3,6,12],"max_cudagraph_capture_size":12}',
        "--speculative-config": '{"method":"mtp","num_speculative_tokens":2}',
        "--generation-config": "vllm",
    }
    argv = ["python", "-m", "vllm.entrypoints.cli.main", "serve", model]
    for key, value in options.items():
        argv.extend([key, value])
    argv.extend(
        [
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--enable-prompt-tokens-details",
        ]
    )
    env = {
        "PYRAMIDKV_ENABLED": "1",
        "VLLM_ASCEND_PYRAMIDKV_ENABLED": "1",
        "VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA": "1",
        "VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "MODEL_PATH": model,
        "SERVE_PORT": "8013",
        "PYRAMIDKV_KV_CACHE_MEMORY_BYTES": "8589934592",
        "PYRAMIDKV_CONFIG": experiment + "/pyramidkv-aligned.json",
        "VLLM_ASCEND_PYRAMIDKV_CONFIG": experiment + "/pyramidkv-aligned.json",
        "VLLM_PLUGINS": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv",
        "PYTHONPATH": experiment
        + "/release-overlay:/root/workspace/pyramidkv-runtime/src/vllm:/root/workspace/pyramidkv-runtime/src/vllm-ascend",
    }
    return {"argv": argv, "selected_environment": env}, {
        "enabled": True,
        "kv_bytes_per_worker": 8589934592,
    }


def test_observed_runtime_qualifies_only_the_frozen_host_profile():
    snapshot, lifecycle = runtime_fixture()
    module("runtime").verify_runtime(snapshot, lifecycle)


@pytest.mark.parametrize(
    "change",
    [
        "graph",
        "mtp",
        "concurrency",
        "kv",
        "dtype",
        "duplicate",
        "extra",
        "disabled",
        "metadata",
        "overlay",
        "device",
    ],
)
def test_observed_runtime_rejects_changed_profile_or_activation(change):
    snapshot, lifecycle = runtime_fixture()
    argv = snapshot["argv"]
    field_changes = {
        "graph": ("--compilation-config", '{"cudagraph_mode":"NONE"}'),
        "mtp": ("--speculative-config", '{"method":"mtp","num_speculative_tokens":1}'),
        "concurrency": ("--max-num-seqs", "8"),
        "kv": ("--kv-cache-memory-bytes", "2147483648"),
        "dtype": ("--dtype", "float16"),
    }
    if change in field_changes:
        flag, value = field_changes[change]
        argv[argv.index(flag) + 1] = value
    elif change == "duplicate":
        argv.extend(["--seed", "18"])
    elif change == "extra":
        argv.append("--enforce-eager")
    else:
        key, value = {
            "disabled": ("VLLM_ASCEND_PYRAMIDKV_ENABLED", "0"),
            "metadata": ("VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA", "0"),
            "overlay": ("PYTHONPATH", "/some/editable/source"),
            "device": ("ASCEND_RT_VISIBLE_DEVICES", "1,6"),
        }[change]
        snapshot["selected_environment"][key] = value
    with pytest.raises(ValueError, match="Runtime"):
        module("runtime").verify_runtime(snapshot, lifecycle)


def plan_fixture():
    synthetic, _payload, _row, _chunks = response_fixture()
    synthetic["prompt"] = [11, 12]
    quality = {
        "case_id": "qasper-000",
        "kind": "quality",
        "task": "qasper",
        "dataset_id": "official-row-0",
        "subset": "development",
        "answers": ["PRIVATE REFERENCE"],
        "original_instruction_tokens": 2,
        "truncated": False,
        "prompt": [100, 101],
        "max_tokens": 128,
        "ignore_eos": False,
        "prompt_tokens": 2,
        "prompt_sha256": hashlib.sha256(b"[100,101]").hexdigest(),
    }
    return [quality, synthetic]


def test_public_plan_removes_quality_token_ids_and_references_only():
    cases = plan_fixture()
    index, synthetic = module("private_inputs").public_plan(cases)
    assert [row["case_id"] for row in index] == ["qasper-000", "boundary-2"]
    assert all("prompt" not in row and "answers" not in row for row in index)
    assert synthetic == [cases[1]]
    assert b"PRIVATE REFERENCE" not in json.dumps([index, synthetic]).encode()


@pytest.mark.parametrize("field", ["question", "context", "input", "reference_text"])
def test_public_plan_rejects_unrecognized_input_fields(field):
    cases = plan_fixture()
    cases[0][field] = "PRIVATE QUESTION"
    with pytest.raises(ValueError, match="Private plan"):
        module("private_inputs").public_plan(cases)


def test_public_plan_cannot_leak_reference_on_synthetic_case():
    cases = plan_fixture()
    cases[1]["answers"] = ["PRIVATE REFERENCE"]
    with pytest.raises(ValueError, match="Private plan"):
        module("private_inputs").public_plan(cases)


def test_private_plan_reconstruction_matches_frozen_case_bytes():
    cases = plan_fixture()
    encoded = (json.dumps(cases, ensure_ascii=False, indent=2) + "\n").encode()
    index = [
        {k: v for k, v in row.items() if k not in ("prompt", "answers")}
        for row in cases
    ]
    result = module("private_inputs").reconstruct_cases(
        index, [cases[1]], [cases[0]], hashlib.sha256(encoded).hexdigest()
    )
    assert result == cases


def test_service_quality_cache_salt_is_restored_from_public_index_and_hash_checked():
    cases = plan_fixture()
    quality = copy.deepcopy(cases[0])
    cases[0]["cache_salt"] = "net-cost-20261028-qasper-000"
    index = [
        {k: v for k, v in row.items() if k not in ("prompt", "answers")}
        for row in cases
    ]
    wanted = hashlib.sha256(
        (json.dumps(cases, ensure_ascii=False, indent=2) + "\n").encode()
    ).hexdigest()
    assert (
        module("private_inputs").reconstruct_cases(index, [cases[1]], [quality], wanted)
        == cases
    )
    index[0]["cache_salt"] = "changed"
    with pytest.raises(ValueError, match="Private plan case digest"):
        module("private_inputs").reconstruct_cases(index, [cases[1]], [quality], wanted)


@pytest.mark.parametrize(
    "change",
    ["order", "private_prompt", "synthetic_prompt", "metadata", "duplicate", "missing"],
)
def test_private_plan_reconstruction_rejects_changed_members_or_payloads(change):
    cases = plan_fixture()
    wanted = hashlib.sha256(
        (json.dumps(cases, ensure_ascii=False, indent=2) + "\n").encode()
    ).hexdigest()
    index = [
        {k: v for k, v in row.items() if k not in ("prompt", "answers")}
        for row in cases
    ]
    quality, synthetic = [copy.deepcopy(cases[0])], [copy.deepcopy(cases[1])]
    if change == "order":
        index.reverse()
    elif change == "private_prompt":
        quality[0]["prompt"] = [100, 102]
    elif change == "synthetic_prompt":
        synthetic[0]["prompt"] = [11, 13]
    elif change == "metadata":
        index[0]["subset"] = "holdout"
    elif change == "duplicate":
        synthetic.append(copy.deepcopy(synthetic[0]))
    else:
        quality.clear()
    with pytest.raises(ValueError, match="Private plan"):
        module("private_inputs").reconstruct_cases(index, synthetic, quality, wanted)


def run_fixture(root):
    folder = root / "runs/example/evaluation"
    (folder / "requests").mkdir(parents=True)
    cases = {row["case_id"]: row for row in plan_fixture()}
    rows = []
    for key, case in cases.items():
        row = {
            "case_id": key,
            "kind": case["kind"],
            "success": True,
            "output": "prediction",
        }
        rows.append(row)
        (folder / "requests" / f"{key}.json").write_text(json.dumps(row))
        (folder / "requests" / f"{key}.sse").write_bytes(b"data: [DONE]\n")
        (folder / "requests" / f"{key}.request.json").write_bytes(
            json.dumps({"prompt": case["prompt"]}).encode()
        )
    (folder / "results.json").write_text(json.dumps(rows))
    return folder, cases, rows


def test_public_export_retains_outputs_but_omits_quality_payloads(tmp_path):
    folder, cases, _rows = run_fixture(tmp_path)
    public, private, origins = module("export").collect_run(tmp_path, "example", cases)
    prefix = "runs/example/evaluation/requests/"
    key = prefix + "qasper-000.request.json"
    assert key not in public
    original = (folder / "requests/qasper-000.request.json").read_bytes()
    assert private[key]["sha256"] == hashlib.sha256(original).hexdigest()
    assert private[key]["bytes"] == len(original)
    assert public[prefix + "boundary-2.request.json"] == b'{"prompt": [11, 12]}'
    assert public[prefix + "qasper-000.sse"] == b"data: [DONE]\n"
    assert set(origins) == set(public)


@pytest.mark.parametrize("change", ["answers", "prompt", "context", "kind", "orphan"])
def test_public_export_rejects_ambiguous_or_private_result_content(tmp_path, change):
    folder, cases, rows = run_fixture(tmp_path)
    if change == "orphan":
        (folder / "requests/unknown.request.json").write_text('{"prompt": "PRIVATE"}')
    elif change == "kind":
        rows[0]["kind"] = "performance"
    else:
        rows[0][change] = "PRIVATE CONTENT"
    (folder / "results.json").write_text(json.dumps(rows))
    (folder / "requests/qasper-000.json").write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="Export"):
        module("export").collect_run(tmp_path, "example", cases)


def test_public_export_preserves_failure_receipt_and_failed_response(tmp_path):
    folder, cases, rows = run_fixture(tmp_path)
    rows[0]["success"] = False
    rows[0]["error"] = "request timeout"
    (folder / "results.json").write_text(json.dumps(rows))
    (folder / "requests/qasper-000.json").write_text(json.dumps(rows[0]))
    failure = b'{"type":"RuntimeError","error":"request failed"}'
    (folder.parent / "failure.json").write_bytes(failure)
    public, _private, _origins = module("export").collect_run(
        tmp_path, "example", cases
    )
    assert public["runs/example/failure.json"] == failure
    assert (
        json.loads(public["runs/example/evaluation/results.json"])[0]["success"]
        is False
    )


@pytest.mark.parametrize("part_bytes", [100, 480000])
def test_exported_capsule_roundtrips_through_independent_inventory_checks(
    tmp_path, part_bytes
):
    (tmp_path / "verify.py").write_bytes(b"# reviewed verifier\n")
    members = {"run/output.sse": b"data: [DONE]\n", "contract.json": b'{"frozen":true}'}
    module("export").write_capsule(tmp_path, members, {}, part_bytes=part_bytes)
    assert module().load_capsule(tmp_path) == members
    if part_bytes == 100:
        assert len(list(tmp_path.glob("raw.tar.gz.part-*"))) > 1
    else:
        assert (tmp_path / "raw.tar.gz").is_file()
    assert b"verify.py" in (tmp_path / "SHA256SUMS").read_bytes()


def test_capsule_export_refuses_to_overwrite_an_existing_original(tmp_path):
    module("export").write_capsule(tmp_path, {"record": b"original"}, {})
    before = (tmp_path / "raw.tar.gz").read_bytes()
    with pytest.raises(ValueError, match="Export.*existing"):
        module("export").write_capsule(tmp_path, {"record": b"changed"}, {})
    assert (tmp_path / "raw.tar.gz").read_bytes() == before


def test_public_export_retains_startup_failure_without_inventing_results(tmp_path):
    folder = tmp_path / "runs/startup"
    folder.mkdir(parents=True)
    failure = b'{"error":"readiness timeout"}'
    (folder / "failure.json").write_bytes(failure)
    (folder / "server.log").write_text("startup failed\n")
    public, private, _origins = module("export").collect_run(tmp_path, "startup", {})
    assert public["runs/startup/failure.json"] == failure
    assert "runs/startup/evaluation/results.json" not in public
    assert private == {}


def experiment_fixture(root):
    _folder, cases, _rows = run_fixture(root)
    plan = root / "quality-plan"
    plan.mkdir()
    encoded = (
        json.dumps(list(cases.values()), ensure_ascii=False, indent=2) + "\n"
    ).encode()
    (plan / "cases.json").write_bytes(encoded)
    (plan / "groups.json").write_text("[]\n")
    manifest = {
        "cases_sha256": hashlib.sha256(encoded).hexdigest(),
        "groups_sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "dataset_files_sha256": {},
    }
    (plan / "manifest.json").write_text(json.dumps(manifest))
    pins = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in plan.iterdir()
    }
    contract = {
        "lifecycles": [{"name": "example", "plan": "quality-plan"}],
        "files_sha256": pins,
    }
    (root / "contract.json").write_text(json.dumps(contract))
    (root / "series-state.json").write_text('{"phase":"all_requests_executed"}')
    (root / "unrelated-private.txt").write_text("PRIVATE UNRELATED CONTENT")
    return contract


def test_experiment_export_uses_frozen_files_and_explicit_privacy_inventory(tmp_path):
    experiment_fixture(tmp_path)
    public, private, origins = module("export").collect_experiment(tmp_path)
    assert "quality-plan/cases.json" not in public
    assert "quality-plan/cases.json" in private
    assert "unrelated-private.txt" not in public
    assert (
        json.loads(public["quality-plan/case-index.json"])[0]["case_id"] == "qasper-000"
    )
    assert len(json.loads(public["quality-plan/synthetic-cases.json"])) == 1
    assert json.loads(public["private-inputs.json"]) == private
    assert "quality-plan/groups.json" in origins


@pytest.mark.parametrize("change", ["active", "frozen", "extra_run"])
def test_experiment_export_rejects_live_changed_or_unclassified_input(change, tmp_path):
    experiment_fixture(tmp_path)
    if change == "active":
        (tmp_path / "series-state.json").write_text('{"phase":"running"}')
    elif change == "frozen":
        (tmp_path / "quality-plan/groups.json").write_text("changed")
    else:
        (tmp_path / "runs/unclassified").mkdir()
    with pytest.raises(ValueError, match="Export"):
        module("export").collect_experiment(tmp_path)


def test_service_plan_export_reuses_quality_dataset_identity_without_duplicated_manifest_fields(
    tmp_path,
):
    contract = experiment_fixture(tmp_path)
    service = tmp_path / "service-plan"
    service.mkdir()
    for name in ("cases.json", "groups.json", "manifest.json"):
        content = (tmp_path / "quality-plan" / name).read_bytes()
        if name == "manifest.json":
            manifest = json.loads(content)
            manifest.pop("dataset_files_sha256")
            content = json.dumps(manifest).encode()
        (service / name).write_bytes(content)
        contract["files_sha256"]["service-plan/" + name] = hashlib.sha256(
            content
        ).hexdigest()
    contract["lifecycles"].append({"name": "service", "plan": "service-plan"})
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    public, private, _ = module("export").collect_experiment(tmp_path)
    assert "service-plan/case-index.json" in public
    assert "service-plan/cases.json" in private


def group_fixture():
    expected = [
        {
            "name": "capacity",
            "kind": "capacity",
            "concurrency": 2,
            "case_ids": ["a", "b"],
        }
    ]
    rows = [
        {
            "case_id": key,
            "success": True,
            "usage": {"completion_tokens": 2},
            "ttft_seconds": 1,
            "e2e_seconds": 3,
            "tpot_seconds": 1,
        }
        for key in ("a", "b")
    ]
    actual = [
        dict(
            expected[0],
            wall_seconds=4,
            success=2,
            failures=0,
            output_tokens_per_second=1,
            mean_ttft_seconds=1,
            mean_tpot_seconds=1,
            mean_e2e_seconds=3,
        )
    ]
    return expected, actual, rows


def test_group_receipt_agrees_with_all_requests_and_frozen_membership():
    expected, actual, rows = group_fixture()
    module().verify_groups(expected, actual, rows)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "members",
        "concurrency",
        "success",
        "failures",
        "rate",
        "mean",
        "wall",
        "nonfinite",
    ],
)
def test_group_receipt_rejects_derived_values_or_membership_changes(change):
    expected, actual, rows = group_fixture()
    if change == "missing":
        actual.clear()
    elif change == "extra":
        actual.append(copy.deepcopy(actual[0]))
    else:
        field, value = {
            "members": ("case_ids", ["a"]),
            "concurrency": ("concurrency", 4),
            "success": ("success", 1),
            "failures": ("failures", 1),
            "rate": ("output_tokens_per_second", 999),
            "mean": ("mean_e2e_seconds", 1),
            "wall": ("wall_seconds", 2),
            "nonfinite": ("wall_seconds", float("nan")),
        }[change]
        actual[0][field] = value
    with pytest.raises(ValueError, match="Group"):
        module().verify_groups(expected, actual, rows)


def lifecycle_fixture():
    snapshot, lifecycle = runtime_fixture()
    lifecycle.update(name="quality8-on", request_count=2)
    snapshot.update(pid=123, at=12)
    return lifecycle, {
        "process": {
            "pid": 123,
            "pgid": 123,
            "start_ticks": "1000",
            "at": 10,
            "lifecycle": copy.deepcopy(lifecycle),
            "contract_sha256": "frozen",
        },
        "ready": {"pid": 123, "at": 20},
        "completion": {"at": 30, "requests": 2, "request_execution_complete": True},
        "shutdown": {"at": 31, "server_returncode": 0},
        "snapshot": snapshot,
    }


def test_lifecycle_receipts_prove_order_identity_and_request_completion():
    lifecycle, records = lifecycle_fixture()
    result = module("runtime").verify_lifecycle_records(lifecycle, records, "frozen", 9)
    assert result["identity"] == [123, "1000"]
    assert result["shutdown_at"] == 31
    assert result["actual_runtime_snapshot"] is True


@pytest.mark.parametrize(
    "change",
    [
        "contract",
        "lifecycle",
        "ready_pid",
        "pgid",
        "overlap",
        "ready_time",
        "completion_time",
        "shutdown_time",
        "nan",
        "incomplete",
        "count",
        "missing_snapshot",
        "snapshot_pid",
        "snapshot_time",
    ],
)
def test_lifecycle_receipts_reject_missing_mixed_or_reordered_evidence(change):
    lifecycle, records = lifecycle_fixture()
    if change == "missing_snapshot":
        records["snapshot"] = None
    else:
        section, key, value = {
            "contract": ("process", "contract_sha256", "changed"),
            "lifecycle": ("process", "lifecycle", {}),
            "ready_pid": ("ready", "pid", 99),
            "pgid": ("process", "pgid", 99),
            "overlap": ("process", "at", 8),
            "ready_time": ("ready", "at", 9),
            "completion_time": ("completion", "at", 19),
            "shutdown_time": ("shutdown", "at", 29),
            "nan": ("ready", "at", float("nan")),
            "incomplete": ("completion", "request_execution_complete", False),
            "count": ("completion", "requests", 1),
            "snapshot_pid": ("snapshot", "pid", 99),
            "snapshot_time": ("snapshot", "at", 32),
        }[change]
        records[section][key] = value
    with pytest.raises(ValueError, match="Lifecycle"):
        module("runtime").verify_lifecycle_records(lifecycle, records, "frozen", 9)


def test_missing_first_off_snapshot_is_disclosed_instead_of_fabricated():
    lifecycle, records = lifecycle_fixture()
    lifecycle.update(name="quality8-off", enabled=False)
    records["process"]["lifecycle"] = copy.deepcopy(lifecycle)
    records["snapshot"] = None
    result = module("runtime").verify_lifecycle_records(lifecycle, records, "frozen", 9)
    assert result["actual_runtime_snapshot"] is False


def test_shutdown_returncode_is_retained_separately_from_successful_requests():
    lifecycle, records = lifecycle_fixture()
    records["shutdown"]["server_returncode"] = -15
    result = module("runtime").verify_lifecycle_records(lifecycle, records, "frozen", 9)
    assert result["server_returncode"] == -15


def test_pooled_wall_time_recomputes_identically_on_python311():
    executable = shutil.which("python3.11")
    if executable is None:
        pytest.skip("Cross-version check requires Python 3.11")
    script = """
import sys,json
sys.path.insert(0,sys.argv[1])
from analyze import summarize_groups
row={"case_id":"x","success":True,"usage":{"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":0}},"ttft_seconds":0.1,"e2e_seconds":0.2,"tpot_seconds":0.1}
groups=[{"case_ids":["x"],"wall_seconds":w} for w in (1e16,1.0,1.0)]
print(json.dumps(summarize_groups([(groups,[row])])["wall_seconds"]))
"""
    result = subprocess.run(
        [executable, "-c", script, str(REPORT)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == 10000000000000002.0


def execution_fixture():
    groups, _, rows = group_fixture()
    for row in rows:
        row.update(kind="capacity", max_tokens=2)
    rows.append(
        {
            "case_id": "qasper-000",
            "kind": "quality",
            "task": "qasper",
            "qa_f1": 0.25,
            "success": True,
            "max_tokens": 2,
        }
    )
    cases = [
        {
            key: value
            for key, value in row.items()
            if key in ("case_id", "kind", "max_tokens")
        }
        for row in rows
    ]
    manifest = {"cases_sha256": "frozen-case-hash"}
    manifest_bytes = json.dumps(manifest).encode()
    plan = {"manifest_bytes": manifest_bytes, "groups": groups, "cases": cases}
    lifecycle = {
        "enabled": True,
        "group_kinds": ["capacity"],
        "skip_quality": False,
        "request_count": 3,
        "max_output_tokens": 6,
    }
    execution = {
        "plan_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "plan_manifest": manifest,
        "group_names": ["capacity"],
        "quality_case_ids": ["qasper-000"],
        "arm": "pyramidkv",
    }
    summary = {
        "arm": "pyramidkv",
        "requests": 3,
        "failures": 0,
        "quality_f1": {"qasper": 25.0},
        "plan_manifest": manifest,
    }
    return lifecycle, plan, execution, summary, rows


def test_execution_joins_selected_groups_quality_and_exact_plan_identity():
    result = module("qualification").verify_execution(*execution_fixture())
    assert result["case_ids"] == ["a", "b", "qasper-000"]
    assert [row["name"] for row in result["groups"]] == ["capacity"]


@pytest.mark.parametrize(
    "change",
    [
        "manifest_sha",
        "manifest",
        "arm",
        "groups",
        "quality",
        "counts",
        "budget",
        "failed",
        "aggregate",
        "summary_manifest",
        "summary_arm",
        "case_order",
        "group_member",
        "private_field",
    ],
)
def test_execution_rejects_partial_changed_or_mixed_plan_receipts(change):
    lifecycle, plan, execution, summary, rows = execution_fixture()
    if change == "manifest_sha":
        execution["plan_manifest_sha256"] = "changed"
    elif change == "manifest":
        execution["plan_manifest"] = {}
    elif change == "arm":
        execution["arm"] = "baseline"
    elif change == "groups":
        execution["group_names"] = []
    elif change == "quality":
        execution["quality_case_ids"] = []
    elif change == "counts":
        summary["requests"] = 2
    elif change == "budget":
        lifecycle["max_output_tokens"] = 5
    elif change == "failed":
        summary["failures"] = 1
    elif change == "aggregate":
        summary["quality_f1"]["qasper"] = 100
    elif change == "summary_manifest":
        summary["plan_manifest"] = {}
    elif change == "summary_arm":
        summary["arm"] = "baseline"
    elif change == "case_order":
        rows.reverse()
    elif change == "group_member":
        plan["groups"][0]["case_ids"].append("qasper-000")
    elif change == "private_field":
        plan["cases"][0]["answers"] = ["PRIVATE"]
    with pytest.raises(ValueError):
        module("qualification").verify_execution(
            lifecycle, plan, execution, summary, rows
        )


def test_capacity_only_execution_excludes_quality_in_both_receipts_and_results():
    lifecycle, plan, execution, summary, rows = execution_fixture()
    lifecycle.update(skip_quality=True, request_count=2, max_output_tokens=4)
    execution["quality_case_ids"] = []
    summary.update(requests=2, quality_f1={})
    result = module("qualification").verify_execution(
        lifecycle, plan, execution, summary, rows[:2]
    )
    assert result["case_ids"] == ["a", "b"]


def test_contract_qualification_rejects_absent_or_unpinned_experiment():
    with pytest.raises(ValueError, match="Contract"):
        module("qualification").verify_contracts({})


def service_fixture():
    runs = {}
    for budget in (8, 2):
        for arm in ("off", "on"):
            for repeat in (1, 2):
                rows, groups = [], []
                for tokens in (1024, 8192, 24576):
                    for concurrency in (1, 4):
                        for cohort in range(3):
                            name = f"perf-{tokens}-c{concurrency}-r{cohort}"
                            ids = [
                                f"{name}-{i:02d}"
                                for i in range(3 if concurrency == 1 else 8)
                            ]
                            groups.append(
                                {
                                    "name": name,
                                    "kind": "performance",
                                    "concurrency": concurrency,
                                    "case_ids": ids,
                                    "wall_seconds": 1.0
                                    if arm == "on"
                                    else (2.0 if budget == 8 else 4.0),
                                }
                            )
                            rows.extend(
                                {
                                    "case_id": key,
                                    "kind": "performance",
                                    "prompt_tokens": tokens,
                                    "success": True,
                                    "usage": {
                                        "completion_tokens": 128,
                                        "prompt_tokens_details": {"cached_tokens": 0},
                                    },
                                    "ttft_seconds": 0.2,
                                    "e2e_seconds": 0.8,
                                    "tpot_seconds": 0.004,
                                }
                                for key in ids
                            )
                groups.append(
                    {
                        "name": "excluded-warmup",
                        "kind": "warmup",
                        "concurrency": 1,
                        "case_ids": ["warmup"],
                        "wall_seconds": 1000000,
                    }
                )
                rows.append(
                    {
                        "case_id": "warmup",
                        "kind": "warmup",
                        "success": True,
                        "usage": {"completion_tokens": 99999999},
                    }
                )
                runs[f"service{budget}-{arm}-{repeat}"] = {
                    "groups": groups,
                    "rows": rows,
                }
    return runs


def test_service_comparison_keeps_budgets_pairs_cells_and_exclusions_distinct():
    result = module("analyze").service_comparison(service_fixture())
    eight, two = result["8gib"], result["2gib"]
    assert eight["all_formal"]["off"]["successful_requests"] == 198
    assert eight["all_formal"]["off"]["output_tokens_per_second"] == 352
    assert eight["all_formal"]["on"]["output_tokens_per_second"] == 704
    assert two["all_formal"]["off"]["output_tokens_per_second"] == 176
    assert len(eight["cells"]) == 6
    cell = eight["cells"]["24576-c4"]
    assert cell["off"]["successful_requests"] == 48
    assert cell["on"]["output_tokens_per_second"] == 1024
    assert cell["throughput_ratio_on_over_off"] == 2
    assert [pair["throughput_ratio_on_over_off"] for pair in cell["pairs"]] == [2, 2]
    assert two["cells"]["24576-c4"]["throughput_ratio_on_over_off"] == 4
    assert eight["processes"]["service8-off-1"]["successful_requests"] == 99


@pytest.mark.parametrize(
    "change",
    [
        "missing_process",
        "missing_cell",
        "duplicate_cohort",
        "changed_concurrency",
        "missing_request",
    ],
)
def test_service_comparison_rejects_incomplete_or_changed_formal_design(change):
    runs = service_fixture()
    selected = runs["service8-off-1"]
    if change == "missing_process":
        del runs["service2-on-2"]
    elif change == "missing_cell":
        selected["groups"] = selected["groups"][3:]
    elif change == "duplicate_cohort":
        selected["groups"].append(copy.deepcopy(selected["groups"][0]))
    elif change == "changed_concurrency":
        selected["groups"][0]["concurrency"] = 4
    elif change == "missing_request":
        selected["rows"].pop(0)
    with pytest.raises(ValueError, match="Service"):
        module("analyze").service_comparison(runs)


def capacity_fixture():
    runs = {}
    for budget, stem in ((8, "quality8"), (2, "capacity2")):
        for arm in ("off", "on"):
            groups, rows = [], []
            for concurrency in (1, 2, 4):
                name = f"capacity-{concurrency}"
                ids = [f"{name}-{i}" for i in range(concurrency)]
                groups.append(
                    {
                        "name": name,
                        "kind": "capacity",
                        "concurrency": concurrency,
                        "case_ids": ids,
                        "wall_seconds": 10,
                    }
                )
                rows.extend(
                    {
                        "case_id": key,
                        "kind": "capacity",
                        "prompt_tokens": 32704,
                        "max_tokens": 64,
                        "success": True,
                        "usage": {
                            "completion_tokens": 64,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                        "ttft_seconds": 2,
                        "e2e_seconds": 8,
                        "tpot_seconds": 0.09,
                    }
                    for key in ids
                )
            runs[f"{stem}-{arm}"] = {"groups": groups, "rows": rows}
    return runs


def test_capacity_report_preserves_only_the_three_measured_operating_points():
    result = module("analyze").capacity_comparison(capacity_fixture())
    assert set(result) == {"8gib", "2gib"}
    assert set(result["2gib"]) == {"c1", "c2", "c4"}
    assert result["2gib"]["c4"]["on"]["successful_requests"] == 4
    assert result["8gib"]["c2"]["off"]["output_tokens_per_second"] == 12.8


@pytest.mark.parametrize(
    "change", ["missing", "length", "output_budget", "concurrency"]
)
def test_capacity_report_rejects_missing_or_changed_operating_points(change):
    runs = capacity_fixture()
    selected = runs["capacity2-on"]
    if change == "missing":
        selected["groups"].pop()
    elif change == "length":
        selected["rows"][0]["prompt_tokens"] = 8192
    elif change == "output_budget":
        selected["rows"][0]["max_tokens"] = 128
    elif change == "concurrency":
        selected["groups"][0]["concurrency"] = 8
    with pytest.raises(ValueError, match="Capacity"):
        module("analyze").capacity_comparison(runs)


def request_set_fixture():
    case, payload, row, chunks = response_fixture()
    case["prompt"] = payload["prompt"]
    qa = copy.deepcopy(case)
    qa.update(
        case_id="qasper-000",
        kind="quality",
        task="qasper",
        subset="development",
        dataset_id="official-0",
        answers=["seven"],
    )
    qrow = {key: value for key, value in qa.items() if key not in ("prompt", "answers")}
    qrow.update({key: value for key, value in row.items() if key not in case})
    qrow.update(request_id="cmpl-quality", qa_f1=1.0)
    qchunks = json.loads(json.dumps(chunks).replace("cmpl-example", "cmpl-quality"))
    prefix = "runs/example/evaluation/requests/"
    raw = {}
    for item, original, stream in ((case, row, chunks), (qa, qrow, qchunks)):
        raw[prefix + item["case_id"] + ".json"] = json.dumps(original).encode()
        raw[prefix + item["case_id"] + ".sse"] = sse(stream)
    encoded = module("private_inputs").request_bytes(case)
    raw[prefix + case["case_id"] + ".request.json"] = encoded
    private = {
        prefix + qa["case_id"] + ".request.json": {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    }
    references = {"qasper-000": {"dataset_id": "official-0", "answers": ["seven"]}}
    return raw, prefix, [case, qa], [row, qrow], private, references


def test_public_request_verification_never_claims_private_payload_or_official_f1_coverage():
    raw, prefix, cases, rows, private, _references = request_set_fixture()
    cases[1] = {
        key: value
        for key, value in cases[1].items()
        if key not in ("prompt", "answers")
    }
    result = module("qualification").verify_requests(raw, prefix, cases, rows, private)
    assert result["synthetic_payloads_verified"] == 1
    assert result["private_payloads_verified"] == 0
    assert result["official_f1_recomputed"] == 0
    assert result["requests_verified"] == 2


def test_private_request_verification_reconstructs_and_scores_from_pinned_references():
    result = module("qualification").verify_requests(*request_set_fixture())
    assert result["private_payloads_verified"] == 1
    assert result["official_f1_recomputed"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "missing",
        "result",
        "private_leak",
        "private_missing",
        "private_hash",
        "private_length",
        "private_answers",
        "official_id",
        "qa_score",
        "synthetic_payload",
    ],
)
def test_request_set_rejects_orphans_leaks_and_unverified_private_reconstruction(
    change,
):
    raw, prefix, cases, rows, private, references = request_set_fixture()
    qa_key = prefix + "qasper-000"
    if change == "extra":
        raw[prefix + "unexpected.sse"] = b"extra"
    elif change == "missing":
        del raw[qa_key + ".sse"]
    elif change == "result":
        raw[qa_key + ".json"] = b"{}"
    elif change == "private_leak":
        raw[qa_key + ".request.json"] = b"PRIVATE"
    elif change == "private_missing":
        private.clear()
    elif change == "private_hash":
        private[qa_key + ".request.json"]["sha256"] = "0" * 64
    elif change == "private_length":
        private[qa_key + ".request.json"]["bytes"] += 1
    elif change == "private_answers":
        cases[1]["answers"] = ["changed"]
    elif change == "official_id":
        references["qasper-000"]["dataset_id"] = "another-row"
    elif change == "qa_score":
        rows[1]["qa_f1"] = 0.0
        raw[qa_key + ".json"] = json.dumps(rows[1]).encode()
    elif change == "synthetic_payload":
        raw[prefix + "boundary-2.request.json"] = b"{}"
    with pytest.raises(ValueError):
        module("qualification").verify_requests(
            raw, prefix, cases, rows, private, references
        )


def test_http_and_engine_completion_counts_exclude_hidden_retries():
    after = 'vllm:request_success_total{finished_reason="length"} 2\n'
    log = '"POST /v1/completions HTTP/1.1" 200 OK\n' * 2
    module("qualification").verify_request_counts("", after, log, 2)


@pytest.mark.parametrize(
    "change",
    [
        "extra_post",
        "error_post",
        "missing_post",
        "extra_engine",
        "reset",
        "nan",
        "duplicate_metric",
    ],
)
def test_http_and_engine_counts_reject_unaccounted_activity(change):
    before = ""
    after = 'vllm:request_success_total{finished_reason="length"} 2\n'
    log = '"POST /v1/completions HTTP/1.1" 200 OK\n' * 2
    if change == "extra_post":
        log += '"POST /v1/completions HTTP/1.1" 200 OK\n'
    elif change == "error_post":
        log = log.replace("200", "500", 1)
    elif change == "missing_post":
        log = ""
    elif change == "extra_engine":
        after = after.replace(" 2", " 3")
    elif change == "reset":
        before = after.replace(" 2", " 4")
    elif change == "nan":
        after = after.replace(" 2", " nan")
    elif change == "duplicate_metric":
        after = 'vllm:request_success_total{finished_reason="length"} 1\n' * 2
    with pytest.raises(ValueError, match="Request count"):
        module("qualification").verify_request_counts(before, after, log, 2)


def pressure_fixture():
    cases = [
        {"case_id": "a", "cache_salt": "unique-a"},
        {"case_id": "b", "cache_salt": "unique-b"},
    ]
    rows = [
        {"case_id": "a", "usage": {"prompt_tokens_details": {"cached_tokens": 8192}}},
        {"case_id": "b", "usage": {"prompt_tokens_details": {"cached_tokens": 0}}},
    ]
    groups = [{"name": "perf-24576-c4-r0", "case_ids": ["a", "b"]}]
    resource = {
        "phases": {
            "perf-24576-c4-r0": {
                "first_counters": {"vllm:num_preemptions_total": 0},
                "last_counters": {"vllm:num_preemptions_total": 8},
            }
        }
    }
    return cases, rows, groups, resource


def test_pressure_cache_review_keeps_unique_salt_and_phase_preemption_evidence():
    result = module("qualification").pressure_cache_review(*pressure_fixture())
    assert result["unique_request_salts"] == 2
    assert result["cached_tokens"] == 8192
    assert result["cached_cases"][0]["phase_preemptions"] == 8
    assert result["individual_request_preemption_trace_available"] is False


@pytest.mark.parametrize(
    "change",
    ["shared_salt", "absent_salt", "no_preemption", "missing_phase", "unassigned_case"],
)
def test_pressure_cache_review_rejects_unexplained_or_cross_request_reuse(change):
    cases, rows, groups, resource = pressure_fixture()
    if change == "shared_salt":
        cases[1]["cache_salt"] = "unique-a"
    elif change == "absent_salt":
        cases[1].pop("cache_salt")
    elif change == "no_preemption":
        resource["phases"][groups[0]["name"]]["last_counters"][
            "vllm:num_preemptions_total"
        ] = 0
    elif change == "missing_phase":
        resource["phases"].clear()
    elif change == "unassigned_case":
        groups[0]["case_ids"] = ["b"]
    with pytest.raises(ValueError, match="Pressure"):
        module("qualification").pressure_cache_review(cases, rows, groups, resource)


def test_only_private_case_plans_can_be_hash_only_frozen_inputs():
    raw = {"launcher.py": b"public original"}
    private = {"quality-plan/cases.json": {"sha256": "frozen-private-hash"}}
    expected = {
        "launcher.py": hashlib.sha256(raw["launcher.py"]).hexdigest(),
        "quality-plan/cases.json": "frozen-private-hash",
    }
    module("qualification").verify_frozen_files(raw, private, expected)
    private["launcher.py"] = {"sha256": expected["launcher.py"]}
    del raw["launcher.py"]
    with pytest.raises(ValueError, match="frozen file"):
        module("qualification").verify_frozen_files(raw, private, expected)


def telemetry_context_fixture():
    return (
        telemetry_fixture(),
        [{"name": "warmup"}, {"name": "perf"}],
        [],
        {"ready_at": 9, "completed_at": 12},
        metrics_fixture(0, 0, 0),
        metrics_fixture(1, 2, 1),
    )


def test_telemetry_is_bounded_by_process_metrics_time_and_selected_phases():
    result = module("qualification").verify_telemetry(*telemetry_context_fixture())
    assert result["counter_deltas"]["num_preemptions"] == 1
    assert result["resources"]["samples"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "unknown_phase",
        "reversed_phase",
        "before_ready",
        "after_completion",
        "counter_above_after",
        "counter_below_before",
    ],
)
def test_telemetry_rejects_cross_process_or_unaccounted_samples(change):
    samples, groups, rows, times, before, after = telemetry_context_fixture()
    if change == "unknown_phase":
        samples[0]["phase"] = "unregistered"
    elif change == "reversed_phase":
        groups.reverse()
    elif change == "before_ready":
        times["ready_at"] = 10.5
    elif change == "after_completion":
        times["completed_at"] = 10.5
    elif change == "counter_above_after":
        after = metrics_fixture(0, 2, 1)
    elif change == "counter_below_before":
        before = metrics_fixture(1, 0, 0)
    with pytest.raises(ValueError, match="Telemetry"):
        module("qualification").verify_telemetry(
            samples, groups, rows, times, before, after
        )


def startup_log_fixture():
    args = {
        "model": "/root/workspace/Qwen3.5-35B-A3B",
        "dtype": "bfloat16",
        "seed": 17,
        "host": "127.0.0.1",
        "port": 8013,
        "max_model_len": 32768,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 4096,
        "tensor_parallel_size": 2,
        "distributed_executor_backend": "mp",
        "worker_cls": "vllm_ascend_pyramidkv.adapter.host_worker.PyramidKVHostWorker",
        "kv_cache_memory_bytes": 8589934592,
        "gpu_memory_utilization": 0.9,
        "served_model_name": ["qwen35-pyramidkv"],
        "generation_config": "vllm",
        "enable_prefix_caching": True,
        "mamba_cache_mode": "align",
        "enable_chunked_prefill": True,
        "async_scheduling": True,
        "enable_prompt_tokens_details": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "additional_config": {"enable_cpu_binding": False},
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "compilation_config": {
            "cudagraph_mode": "CUDAGraphMode.FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [3, 6, 12],
            "max_cudagraph_capture_size": 12,
        },
    }
    return args, {"enabled": False, "kv_bytes_per_worker": 8589934592}


def startup_log(args):
    encoded = repr(args).replace(
        "'CUDAGraphMode.FULL_AND_PIECEWISE'",
        "<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>",
    )
    return (
        "Installed plugin: /root/.cache/pyramidkv-issue31-20261103/release-overlay/vllm_ascend_pyramidkv/__init__.py 0.4.0rc1\n"
        + "(APIServer pid=123) INFO non-default args: "
        + encoded
        + "\n"
    )


def test_startup_log_retains_import_identity_and_parsed_host_profile():
    args, lifecycle = startup_log_fixture()
    module("runtime").verify_startup_log(startup_log(args), lifecycle)


@pytest.mark.parametrize(
    "change",
    [
        "dtype",
        "model",
        "kv",
        "tp",
        "graph",
        "mtp",
        "prefix",
        "import",
        "missing",
        "duplicate",
    ],
)
def test_startup_log_rejects_missing_or_changed_fallback_evidence(change):
    args, lifecycle = startup_log_fixture()
    if change == "dtype":
        args["dtype"] = "float16"
    elif change == "model":
        args["model"] = "/different/model"
    elif change == "kv":
        args["kv_cache_memory_bytes"] = 2147483648
    elif change == "tp":
        args["tensor_parallel_size"] = 1
    elif change == "graph":
        args["compilation_config"]["cudagraph_capture_sizes"] = [1, 2, 4]
    elif change == "mtp":
        args["speculative_config"]["num_speculative_tokens"] = 1
    elif change == "prefix":
        args["enable_prefix_caching"] = False
    log = startup_log(args)
    if change == "import":
        log = log.replace("release-overlay", "editable-source")
    elif change == "missing":
        log = log.splitlines()[0]
    elif change == "duplicate":
        log += log.splitlines()[-1] + "\n"
    with pytest.raises(ValueError, match="Startup"):
        module("runtime").verify_startup_log(log, lifecycle)


def test_large_member_inventory_remains_below_repository_file_limit(tmp_path):
    members = {
        f"runs/service2-on-2/evaluation/requests/perf-24576-c4-r{i:04d}-07.request.json": b"original"
        for i in range(5100)
    }
    module("export").write_capsule(tmp_path, members, {})
    assert all(path.stat().st_size <= 1000 * 1024 for path in tmp_path.iterdir())
    assert (tmp_path / "members.json.gz").is_file()
    assert module().load_capsule(tmp_path) == members


@pytest.mark.parametrize("change", ["both", "corrupt"])
def test_compressed_member_inventory_rejects_ambiguity_and_corruption(change, tmp_path):
    capsule(tmp_path)
    encoded = (tmp_path / "members.json").read_bytes()
    (tmp_path / "members.json.gz").write_bytes(
        gzip.compress(encoded) if change == "both" else b"invalid gzip"
    )
    if change == "corrupt":
        (tmp_path / "members.json").unlink()
    checksums(tmp_path)
    with pytest.raises(ValueError, match="[Ii]nventory"):
        module().load_capsule(tmp_path)


@pytest.fixture(scope="module")
def measured_capsule():
    return module().load_capsule(REPORT)


@pytest.fixture(scope="module")
def measured_verification(measured_capsule):
    return module("qualification").verify_evidence(measured_capsule)


def test_delivered_experiment_verifies_all_twelve_lifecycles_without_private_claims(
    measured_verification,
):
    result = measured_verification
    assert result["coverage"] == {
        "requests_verified": 1834,
        "synthetic_payloads_verified": 1138,
        "private_payloads_verified": 0,
        "official_f1_recomputed": 0,
    }
    assert len(result["lifecycles"]) == 12
    assert result["runtime_snapshot_gaps"] == ["quality8-off"]
    assert result["quality_gate_passed"] is True


@pytest.mark.parametrize(
    "change",
    [
        "unfinished",
        "missing_run",
        "unknown_run",
        "changed_contract",
        "private_public_file",
        "wrong_run_contract",
        "overlapping_run",
        "private_inventory",
        "changed_client",
        "unfree_device",
    ],
)
def test_delivered_evidence_rejects_semantic_changes_beyond_archive_hashes(
    change, measured_capsule
):
    raw = dict(measured_capsule)

    def alter(name, update):
        value = json.loads(raw[name])
        update(value)
        raw[name] = json.dumps(value).encode()

    if change == "unfinished":
        alter("series-state.json", lambda value: value.update(phase="running"))
    elif change == "missing_run":
        raw = {
            key: value
            for key, value in raw.items()
            if not key.startswith("runs/capacity2-on/")
        }
    elif change == "unknown_run":
        raw["runs/not-preregistered/process.json"] = b"{}"
    elif change == "changed_contract":
        alter("contract.json", lambda value: value.update(total_request_budget=1833))
    elif change == "private_public_file":
        original = raw.pop("serve.sh")
        alter(
            "private-inputs.json",
            lambda value: value.update(
                {
                    "serve.sh": {
                        "sha256": hashlib.sha256(original).hexdigest(),
                        "bytes": len(original),
                    }
                }
            ),
        )
    elif change == "wrong_run_contract":
        alter(
            "runs/quality8-off/process.json",
            lambda value: value.update(contract_sha256="0" * 64),
        )
    elif change == "overlapping_run":
        alter("runs/quality8-on/process.json", lambda value: value.update(at=1))
    elif change == "private_inventory":
        alter(
            "private-inputs.json",
            lambda value: value.update(
                {"unknown/private.json": {"bytes": 1, "sha256": "0" * 64}}
            ),
        )
    elif change == "changed_client":
        raw["runs/quality8-off/client-command.json"] = b"[]"
    elif change == "unfree_device":
        raw["runs/quality8-off/npu-before.txt"] = raw[
            "runs/quality8-off/npu-before.txt"
        ].replace(b"No running processes found", b"A process was running")
    with pytest.raises(ValueError):
        module("qualification").verify_evidence(raw)


def test_committed_analysis_is_recomputed_from_delivered_originals(
    measured_capsule, measured_verification
):
    actual = module("analyze").analyze(measured_capsule, measured_verification)
    assert actual == json.loads((REPORT / "analysis.json").read_text())
    assert actual["quality"]["gate_passed"] is True
    assert set(actual["service"]) == {"8gib", "2gib"}
    assert set(actual["capacity"]["2gib"]) == {"c1", "c2", "c4"}


@pytest.mark.parametrize("optimized", [False, True])
def test_cli_rejects_changed_analysis_even_with_refreshed_checksums(
    tmp_path, optimized
):
    for source in REPORT.iterdir():
        if source.is_file():
            shutil.copy2(source, tmp_path / source.name)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    analysis["quality"]["gate_passed"] = False
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    checksums(tmp_path)
    command = (
        [sys.executable] + (["-O"] if optimized else []) + [str(tmp_path / "verify.py")]
    )
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=180
    )
    assert result.returncode != 0
    assert "Analysis differs" in result.stderr


def test_cli_verifies_delivered_public_capsule_without_private_coverage():
    result = subprocess.run(
        [sys.executable, str(REPORT / "verify.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "verified"
    assert receipt["coverage"]["requests_verified"] == 1834
    assert receipt["coverage"]["official_f1_recomputed"] == 0
