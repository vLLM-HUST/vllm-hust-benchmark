import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


REPORT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"
DATA = REPORT / "m06-paired-20261021"


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_recompute_from_immutable_raw_records():
    result = module(REPORT / "render_m06_pair.py").load(DATA)
    audit = module(DATA / "audit_m06_prechecked_pair_r1.py")
    inputs = []
    for arm in ("b0", "b1"):
        run = f"m01-recovery-model-m06-prechecked-{arm}-r1"
        with tarfile.open(DATA / f"m06-prechecked-{arm}-r1-evidence.tar.gz") as tar:
            contents = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
        for line in contents[f"{run}/SHA256SUMS"].decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(contents[f"{run}/{name}"]).hexdigest() == digest
        receipt = json.loads(contents[f"{run}/receipt.json"])
        bench = json.loads(contents[f"{run}/mmlu-pro.json"])
        for key, value in audit.metrics(receipt, bench).items():
            assert value == pytest.approx(result["arms"][arm][key])
        inputs.append(bench["input_lens"])
        client = contents[f"{run}/bench.log"].decode()
        assert client.count("Initial test run completed.") == 1
        assert "Skipping endpoint ready check." not in client
        service = contents[f"{run}/service.log"].decode()
        assert service.count('POST /v1/chat/completions HTTP/1.1" 200 OK') == 201
        assert "timeout=60s" in service
        assert receipt["child_exit_codes"] == ([0, 130] if arm == "b1" else [0, 0])
    assert inputs[0] == inputs[1]
    b0, b1 = (result["arms"][a] for a in ("b0", "b1"))
    change = 100 * (1 - b1["lifecycle_s"] / b0["lifecycle_s"])
    assert result["lifecycle_cost_reduction_percent"] == pytest.approx(change)
    assert result["m06_lifecycle_gate_pass"] == (change >= 10)


def test_report_render_and_protocol_deviation_preserved():
    renderer = module(REPORT / "render_m06_pair.py")
    result = renderer.load(DATA)
    builder = module(REPORT.parents[1] / "scripts/build_midterm_publication.py")
    report = (REPORT / "中期测试报告.md").read_text()
    # Preserve the sealed renderer, raw metrics and explicit display migration.
    assert builder.tables(
        renderer.render(result).replace("B0", "开发栈对照")
    ) == builder.tables(
        builder.section(report, "### 5.10 恢复环境配对性能与全流程成本")
    )
    prior = json.loads((DATA / "m06-pair-result-r2.json").read_text())
    assert prior["original_prechecked_protocol_confirmation"] is False
    assert result["original_prechecked_protocol_confirmation"] is True
    assert result["quality_scored"] is False
    index = json.loads((DATA / "archive.json").read_text())
    assert index["os_page_cache_globally_flushed"] is False


def test_damaged_evidence_is_rejected(tmp_path):
    (tmp_path / "archive.json").write_bytes((DATA / "archive.json").read_bytes())
    first = json.loads((DATA / "archive.json").read_text())["files"][0]
    (tmp_path / first["name"]).write_text("{}")
    with pytest.raises(AssertionError):
        module(REPORT / "render_m06_pair.py").load(tmp_path)
