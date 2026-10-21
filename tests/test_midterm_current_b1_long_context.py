import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
RUN = "m01-recovery-model-long-context-r2"


def evidence():
    folder = REPORT / "long-context-current-b1-20261021"
    index = json.loads((folder / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        raw = (folder / part["name"]).read_bytes()
        assert len(raw) == part["bytes"]
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        chunks.append(raw)
    raw = b"".join(chunks)
    assert len(raw) == index["archive_bytes"]
    assert hashlib.sha256(raw).hexdigest() == index["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    assert set(data) == {m["name"] for m in index["members"]}
    for member in index["members"]:
        assert hashlib.sha256(data[member["name"]]).hexdigest() == member["sha256"]
    return index, data


def test_full_sse_reconstructs_six_long_input_results():
    index, data = evidence()
    inputs = json.loads(data["long-context-inputs-r1.json"])["cases"]
    summary = json.loads(data[f"{RUN}/long-context.json"])
    assert len(inputs) == len(summary["cases"]) == 6
    assert {(c["target_min_tokens"], c["filler_position"]) for c in inputs} == {
        (length, position) for length in (16384, 32768) for position in (0.1, 0.5, 0.9)
    }
    for i, (case, recorded) in enumerate(zip(inputs, summary["cases"], strict=True)):
        output, reasoning, usage, finish, done = "", "", None, None, False
        for line in gzip.decompress(data[f"{RUN}/long-sse-{i:03d}.gz"]).splitlines():
            if not line.startswith(b"data:"):
                continue
            content = line[5:].strip()
            if content == b"[DONE]":
                done = True
                continue
            event = json.loads(content)
            assert not event.get("error")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                output += choice.get("delta", {}).get("content") or ""
                reasoning += choice.get("delta", {}).get("reasoning_content") or ""
                finish = choice.get("finish_reason") or finish
        assert done and recorded["success"]
        assert (output, reasoning, usage, finish) == (
            recorded["output"],
            recorded["reasoning"],
            recorded["usage"],
            recorded["finish_reason"],
        )
        assert (
            usage["prompt_tokens"] == case["input_tokens"] >= case["target_min_tokens"]
        )
        assert recorded["correct"] == (output.strip() == case["expected"])
    assert summary["correct"] == sum(row["correct"] for row in summary["cases"])
    assert index["result"] == {k: summary[k] for k in index["result"]}
    assert summary["output_throughput"] is None
    supplement = json.loads(data["long-context-final-answer-supplement-r1.json"])
    assert supplement["primary_whole_content_correct"] == summary["correct"] == 0
    assert (
        supplement["source_sha256"]
        == hashlib.sha256(data[f"{RUN}/long-context.json"]).hexdigest()
    )
    for original, diagnosis in zip(summary["cases"], supplement["cases"], strict=True):
        assert original["output"].count("</think>") == 1
        assert original["reasoning"] == ""
        final = original["output"].split("</think>", 1)[1].strip()
        assert diagnosis["final_segment"] == final
        assert diagnosis["final_segment_exact"] == (final == original["expected"])
    assert supplement["supplementary_final_segment_correct"] == sum(
        c["final_segment_exact"] for c in supplement["cases"]
    )


def test_same_launcher_empty_cache_and_real_salt_admission():
    _, data = evidence()
    receipt = json.loads(data[f"{RUN}/receipt.json"])
    assert receipt["all_devices_released"] and receipt["bench_exit_code"] == 0
    assert receipt["status"] == "measured"
    actual = json.loads(data[f"{RUN}/launch-command.json"])
    original = json.loads(data["mmlu-b1-microflow-fa-on-r10/launch-command.json"])
    assert actual[2:] == original[2:]
    cache = json.loads(data[f"{RUN}/compiler-cache-initialization.json"])
    assert cache["initial_entries"] == []
    assert RUN in cache["destination"]
    imports = json.loads(data[f"{RUN}/long-context-runtime-imports.json"])
    assert imports["observer_enabled"] is False
    policy = [
        json.loads(line) for line in data[f"{RUN}/kvmat-policy.jsonl"].splitlines()
    ]
    events = [
        json.loads(line) for line in data[f"{RUN}/kvmat-engine.jsonl"].splitlines()
    ]
    admitted = [
        e for e in events if e.get("event") == "school028_request_salt_admitted"
    ]
    assert len(policy) == len(admitted) == 6
    salts = {
        hashlib.sha256(p["runtime_cache_salt"].encode()).hexdigest() for p in policy
    }
    assert len(salts) == 6 and salts == {e["cache_salt_sha256"] for e in admitted}
    assert all(
        p["runtime_control_path"] == "request_scoped_prefix_cache_bypass"
        and p["reused_tokens"] == 0
        for p in policy
    )


def test_prior_controller_failure_is_retained_without_model_requests():
    _, data = evidence()
    failed = "m01-recovery-model-long-context-r1"
    receipt = json.loads(data[f"{failed}/receipt.json"])
    assert receipt["status"] == "failed" and receipt["all_devices_released"]
    assert "bench_started_wall" not in receipt
    assert not any(name.startswith(f"{failed}/long-sse-") for name in data)
    assert b"ready = response.status == 6" in data["measure_long_context_r1.py"]
    assert b"ready = response.status == 200" in data["measure_long_context_r2.py"]
