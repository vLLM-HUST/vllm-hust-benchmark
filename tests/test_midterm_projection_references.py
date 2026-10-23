import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"


def evidence():
    folder = REPORT / "m01-recovery/projection-references-20261021"
    index = json.loads((folder / "archive.json").read_text())
    packed = folder / index["archive"]
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == index["archive_sha256"]
    contents = {}
    with tarfile.open(packed) as archive:
        for member in archive:
            assert member.isfile()
            raw = archive.extractfile(member).read()
            assert (
                hashlib.sha256(raw).hexdigest()
                == index["members"][member.name]["sha256"]
            )
            assert len(raw) == index["members"][member.name]["bytes"]
            contents[member.name] = raw
    assert set(contents) == set(index["members"])
    return index, contents, json.loads(contents["summaries.json"])


def test_projection_archive_custody_and_scope():
    index, contents, summaries = evidence()
    assert set(summaries) == {"dense", "moe", "vocab"}
    assert index["model_effective_flops"] is None and index["m01_attainment"] is None
    for family, summary in summaries.items():
        assert (
            summary["model_effective_flops"] is None
            and summary["m01_attainment"] is None
        )
        receipt = json.loads(
            contents[f"{family}/m01-{family}-reference-recovery-r1/receipt.json"]
        )
        assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
        assert (
            receipt["all_four_idle_after"] and receipt["prior_cache_imported"] is False
        )
        assert index["raw_archives"][family]["url"].startswith(
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
        )
        for rank in summary["ranks"]:
            name = (
                f"m01-{family}-reference-recovery-r1/results/rank-{rank['rank']}.json"
            )
            raw = contents[f"{family}/{name}"]
            assert hashlib.sha256(raw).hexdigest() == rank["raw_sha256"]
    assert sum(r["timings"] for r in summaries["moe"]["ranks"]) == 440496
    assert [r["cases"] for r in summaries["dense"]["ranks"]] == [121, 121]
    assert [r["shapes"] for r in summaries["vocab"]["ranks"]] == [45, 45]


def test_report_table_generated_from_measured_summaries():
    _, _, summaries = evidence()
    spec = importlib.util.spec_from_file_location(
        "projection_renderer", REPORT / "render_projection_references.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render(summaries)
    report = (REPORT / "中期测试报告.md").read_text()
    builder_spec = importlib.util.spec_from_file_location(
        "publication", ROOT / "scripts/build_midterm_publication.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    assert builder.tables(rendered) == builder.tables(
        builder.section(report, "#### 恢复环境投影算子参考标定")
    )
    assert "6372.136" in rendered and "5389.571" in rendered
    assert "完整有效 F 和 M01 达成率仍不报告" in rendered
    assert "报告版本：V0.74" in (REPORT / "中期测试报告.md").read_text()
