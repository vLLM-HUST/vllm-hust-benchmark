import ast
import gzip
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
RUN = "m01-recovery-model-m05-parser-r1"
OLD = "m01-recovery-model-long-context-r2"


def evidence():
    folder = REPORT / "m05-parser-confirmation-20261021"
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
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        data = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
    assert set(data) == {m["name"] for m in index["members"]}
    for member in index["members"]:
        assert hashlib.sha256(data[member["name"]]).hexdigest() == member["sha256"]
    return index, data


def stream(raw):
    content, reasoning, usage, finish, done = "", "", None, None, False
    for line in gzip.decompress(raw).splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            continue
        event = json.loads(payload)
        assert not event.get("error")
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            fields = [
                delta[k]
                for k in ("reasoning", "reasoning_content")
                if delta.get(k) is not None
            ]
            if fields:
                assert all(isinstance(v, str) and v == fields[0] for v in fields)
                reasoning += fields[0]
            finish = choice.get("finish_reason") or finish
    assert done and finish and usage
    return content, reasoning, usage, finish


def test_original_score_inputs_and_scorer_remain_unchanged():
    index, data = evidence()
    old = json.loads(data[f"{OLD}/long-context.json"])
    assert old["correct"] == 0
    assert index["audit"]["original_whole_content_correct_unchanged"] == 0
    assert data[f"{OLD}/request-plan.json"] == data[f"{RUN}/request-plan.json"]
    assert json.loads(data[f"{OLD}/tokenize-preflight.json"]) == json.loads(
        data[f"{RUN}/tokenize-preflight.json"]
    )
    scorers = []
    for name in ("long_context_client_r1.py", "long_context_client_m05_parser_r1.py"):
        node = next(
            n
            for n in ast.parse(data[name]).body
            if isinstance(n, ast.FunctionDef) and n.name == "exact_correct"
        )
        scorers.append(ast.dump(node))
    assert scorers[0] == scorers[1]


def test_raw_api_fields_and_strict_scores_are_reproducible():
    index, data = evidence()
    summary = json.loads(data[f"{RUN}/long-context.json"])
    original = json.loads(data["long-context-inputs-r1.json"])
    assert summary["total"] == summary["completed"] == 6 and summary["failed"] == 0
    for i, (case, row) in enumerate(
        zip(original["cases"], summary["cases"], strict=True)
    ):
        content, reasoning, usage, finish = stream(data[f"{RUN}/long-sse-{i:03d}.gz"])
        assert (content, reasoning, usage, finish) == (
            row["output"],
            row["reasoning"],
            row["usage"],
            row["finish_reason"],
        )
        assert (
            reasoning.strip() and "</think>" not in content and "<think>" not in content
        )
        assert row["correct"] == (content.strip() == case["expected"])
        assert usage["prompt_tokens"] == case["input_tokens"] >= 16384
    assert summary["correct"] == sum(c["correct"] for c in summary["cases"])
    assert index["result"] == {k: summary[k] for k in index["result"]}
    assert summary["output_throughput"] is None
    content, reasoning, _, finish = stream(data[f"{RUN}/parser-qualification-sse.gz"])
    assert content.strip() == "4" and reasoning.strip() and finish == "stop"
    assert json.loads(data[f"{RUN}/parser-qualification.json"])["passed"]


def test_launch_cache_isolation_and_terminal_receipt():
    _, data = evidence()
    before = json.loads(data[f"{OLD}/launch-command.json"])
    after = json.loads(data[f"{RUN}/launch-command.json"])
    assert after[:-2] == before and after[-2:] == ["--reasoning-parser", "qwen3"]
    receipt = json.loads(data[f"{RUN}/receipt.json"])
    assert receipt["all_devices_released"] and receipt["child_exit_codes"] == [0, 130]
    assert receipt["status"] == "measured" and receipt["bench_exit_code"] == 0
    assert (
        json.loads(data[f"{RUN}/compiler-cache-initialization.json"])["initial_entries"]
        == []
    )
    salts = []
    for run, count in ((OLD, 6), (RUN, 7)):
        policy = [
            json.loads(line) for line in data[f"{run}/kvmat-policy.jsonl"].splitlines()
        ]
        engine = [
            json.loads(line) for line in data[f"{run}/kvmat-engine.jsonl"].splitlines()
        ]
        actual = {
            hashlib.sha256(p["runtime_cache_salt"].encode()).hexdigest() for p in policy
        }
        assert len(policy) == len(engine) == len(actual) == count
        assert actual == {p["cache_salt_sha256"] for p in engine}
        assert all(
            p["runtime_effective_decision"] == "recompute" and p["reused_tokens"] == 0
            for p in policy
        )
        salts.append(actual)
    assert not salts[0] & salts[1]


def test_report_table_is_generated_from_audited_results():
    index, _ = evidence()
    modules = []
    for path in (
        REPORT / "render_m05_parser.py",
        ROOT / "scripts/build_midterm_publication.py",
    ):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    renderer, builder = modules
    actual = builder.section(
        (REPORT / "中期测试报告.md").read_text(), "### 7.3 思考字段分离后的独立确认"
    )
    assert builder.tables(actual) == builder.tables(renderer.render(index))
    assert "正文严格匹配6/6" in actual
    assert "原运行的正文严格匹配0/6" in actual
