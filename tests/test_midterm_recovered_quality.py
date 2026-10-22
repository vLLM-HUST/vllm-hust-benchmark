import collections
import gzip
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from vllm_hust_benchmark.mmlu_pro_pyramidkv import extract_answer

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
RUN = "m01-recovery-model-quality-r1"


def read_archive(folder):
    index = json.loads((folder / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        raw = (folder / part.get("name", part.get("file"))).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        if "bytes" in part:
            assert len(raw) == part["bytes"]
        chunks.append(raw)
    raw = b"".join(chunks)
    if "archive_sha256" in index:
        assert hashlib.sha256(raw).hexdigest() == index["archive_sha256"]
    expected = index["members"]
    if isinstance(expected, list):
        expected = {m["name"]: m["sha256"] for m in expected}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    assert data.keys() == expected.keys()
    for name, content in data.items():
        assert hashlib.sha256(content).hexdigest() == expected[name]
    return index, data


@pytest.mark.parametrize(
    "folder,run",
    [
        ("recovered-20261021", RUN),
        ("native-recovered-20261021", "m01-recovery-model-quality-b0-r3"),
    ],
)
def test_full_sse_and_original_scorer_reproduce_every_case(folder, run):
    index, data = read_archive(REPORT / "quality-regression" / folder)
    result = json.loads(data[f"{run}/mmlu-pro.json"])
    labels = json.loads(data["verified-mmlu-quality-inputs.json"])["cases"]
    plan = json.loads(data[f"{run}/quality-request-plan.json"])
    scorer = ROOT / "src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py"
    assert hashlib.sha256(scorer.read_bytes()).hexdigest() == plan["scorer_sha256"]
    assert len(result["cases"]) == len(labels) == len(plan["requests"]) == 200
    for i, (case, label) in enumerate(zip(result["cases"], labels, strict=True)):
        assert case == json.loads(data[f"{run}/quality-case-{i:03d}.json"])
        text, reasoning, finish, usage, done = "", "", None, None, False
        for line in (
            gzip.decompress(data[f"{run}/quality-sse-{i:03d}.gz"]).decode().splitlines()
        ):
            if not line.startswith("data:"):
                continue
            content = line[5:].strip()
            if content == "[DONE]":
                done = True
                continue
            event = json.loads(content)
            assert not event.get("error")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text += choice.get("delta", {}).get("content") or ""
                reasoning += choice.get("delta", {}).get("reasoning_content") or ""
                finish = choice.get("finish_reason") or finish
        assert done and case["success"]
        assert text == case["output"] and reasoning == case["reasoning"]
        assert finish == case["finish_reason"] and usage == case["usage"]
        assert case["case_id"] == label["case_id"] and case["gold"] == label["answer"]
        assert case["answer"] == extract_answer(text)
        assert case["correct"] == (extract_answer(text) == label["answer"])
    assert result["correct"] == sum(c["correct"] for c in result["cases"])
    assert result["accuracy"] == result["correct"] / 200
    assert result["truncated"] == sum(
        c["finish_reason"] == "length" for c in result["cases"]
    )
    assert index["result"] == {k: result[k] for k in index["result"]}


def test_paired_inputs_cache_isolation_and_terminal_receipts():
    _, data = read_archive(REPORT / "quality-regression/recovered-20261021")
    _, prior = read_archive(REPORT / "quality-regression/raw")
    result = json.loads(data[f"{RUN}/mmlu-pro.json"])
    for old, audit_name in (
        ("mmlu-quality-b0-r4", "b0"),
        ("mmlu-quality-microflow-r1", "microflow"),
    ):
        for filename in (
            "quality-tokenize-preflight.json",
            "quality-request-plan.json",
        ):
            assert json.loads(data[f"{RUN}/{filename}"]) == json.loads(
                prior[f"{old}/{filename}"]
            )
        baseline = json.loads(prior[f"{old}/mmlu-pro.json"])
        transitions = collections.Counter()
        for a, b in zip(baseline["cases"], result["cases"], strict=True):
            assert a["case_id"] == b["case_id"]
            transitions[
                ("correct" if a["correct"] else "incorrect")
                + "_to_"
                + ("correct" if b["correct"] else "incorrect")
            ] += 1
        audit = json.loads(data[f"quality-recovery-r1-pair-{audit_name}.json"])
        assert audit["paired_transitions"] == dict(transitions)
        assert (
            audit["delta_percentage_points"]
            == (result["correct"] - baseline["correct"]) / 2
        )
        assert audit["noninferiority_claim"] is None
    receipt = json.loads(data[f"{RUN}/receipt.json"])
    assert receipt["status"] == "measured" and receipt["bench_exit_code"] == 0
    assert receipt["all_devices_released"]
    cache = json.loads(data[f"{RUN}/compiler-cache-initialization.json"])
    assert cache["initial_entries"] == []
    assert "m01-recovery-model-quality-r1-cache" in cache["destination"]
    imports = json.loads(data[f"{RUN}/quality-runtime-imports.json"])
    assert imports["observer_enabled"] is False
    assert not any("/token-accounting-" in name for name in data)


def test_report_quality_table_is_derived_from_raw_results():
    _, data = read_archive(REPORT / "quality-regression/recovered-20261021")
    spec = importlib.util.spec_from_file_location(
        "renderer", REPORT / "render_recovered_quality.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    prior = json.loads(
        (REPORT / "quality-regression/finish-breakdown.json").read_text()
    )
    rendered = renderer.render(data, prior)
    spec = importlib.util.spec_from_file_location(
        "builder", ROOT / "scripts/build_midterm_publication.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    report = (REPORT / "中期测试报告.md").read_text()
    assert builder.tables(rendered) == builder.tables(
        builder.section(report, "### 5.8 恢复环境完整回答质量复核")
    )
    assert "-0.50" in rendered and "-1.50" in rendered


def test_recovered_native_pair_and_source_isolation():
    _, native = read_archive(REPORT / "quality-regression/native-recovered-20261021")
    _, candidate = read_archive(REPORT / "quality-regression/recovered-20261021")
    run = "m01-recovery-model-quality-b0-r3"
    for filename in ("quality-tokenize-preflight.json", "quality-request-plan.json"):
        assert json.loads(native[f"{run}/{filename}"]) == json.loads(
            candidate[f"{RUN}/{filename}"]
        )
    b0 = json.loads(native[f"{run}/mmlu-pro.json"])
    b1 = json.loads(candidate[f"{RUN}/mmlu-pro.json"])
    transitions = collections.Counter()
    for a, b in zip(b0["cases"], b1["cases"], strict=True):
        assert a["case_id"] == b["case_id"] and a["gold"] == b["gold"]
        transitions[
            ("correct" if a["correct"] else "incorrect")
            + "_to_"
            + ("correct" if b["correct"] else "incorrect")
        ] += 1
    pair = json.loads(native["quality-recovery-native-pair-r3.json"])
    assert pair["paired_transitions"] == dict(transitions)
    assert pair["delta_percentage_points"] == (b1["correct"] - b0["correct"]) / 2
    assert pair["noninferiority_claim"] is None
    receipt = json.loads(native[f"{run}/receipt.json"])
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    cache = json.loads(native[f"{run}/compiler-cache-initialization.json"])
    assert cache["initial_entries"] == [] and run in cache["destination"]
    identity = json.loads(native[f"{run}/quality-runtime-imports.json"])
    assert (
        identity["plugins"]
        == "ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
    )
    assert identity["observer_enabled"] is False
    manifest_bytes = native["native-quality-runtime-r1.json"]
    manifest = json.loads(manifest_bytes)
    assert (
        hashlib.sha256(manifest_bytes).hexdigest()
        == identity["native_runtime"]["manifest_sha256"]
    )
    assert len(manifest["files"]) == identity["native_runtime"]["files_checked"] == 1551
    for name, item in manifest["restored"].items():
        content = native[
            "workspace/native-quality-recovery-runtime-r1/vllm_ascend/" + name
        ]
        assert hashlib.sha256(content).hexdigest() == item["native_sha256"]
        assert item["native_sha256"] != item["candidate_sha256"]
    assert (
        b"betterscale"
        not in native[
            "workspace/native-quality-recovery-runtime-r1/vllm_ascend/worker/model_runner_v1.py"
        ]
    )


def test_native_pair_table_is_derived_from_raw_results():
    _, native = read_archive(REPORT / "quality-regression/native-recovered-20261021")
    _, candidate = read_archive(REPORT / "quality-regression/recovered-20261021")
    spec = importlib.util.spec_from_file_location(
        "native_renderer", REPORT / "render_native_quality_pair.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    spec = importlib.util.spec_from_file_location(
        "builder", ROOT / "scripts/build_midterm_publication.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    report = (REPORT / "中期测试报告.md").read_text()
    assert builder.tables(renderer.render(native, candidate)) == builder.tables(
        builder.section(report, "### 5.9 恢复环境 Native 与优化方案配对质量")
    )
