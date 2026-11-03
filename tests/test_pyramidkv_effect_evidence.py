"""Synthetic integrity failures must not become accepted experiment evidence."""

import copy
import hashlib
import importlib.util
import io
import json
import os
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
