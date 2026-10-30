"""Projection contracts with synthetic scalars, never substitute experiment data."""

import importlib
import json
from pathlib import Path

from test_full_answer_joint_math import synthetic_data

ROOT = Path(__file__).resolve().parents[1]


def test_new_pair_is_current_without_replacing_old_compute_or_failures(monkeypatch):
    data = synthetic_data()
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    math = importlib.import_module("recompute_full_answer_joint")
    builder = importlib.import_module("build_midterm_v103")
    metrics = math.paired_metrics(data)
    for arm, calculated in zip(data["arms"], metrics["arms"], strict=True):
        arm.update(calculated)
        arm["cost"]["cny_per_million_output_tokens"] = calculated[
            "cny_per_million_output_tokens"
        ]
    data.update({k: v for k, v in metrics.items() if k != "arms"})
    continuous = {
        key: dict(
            phases=[
                dict(
                    name=name,
                    failures=0,
                    requests=8,
                    cohort_output_tokens_per_second=100,
                    timing={
                        kind: dict(p95=10, p99=20) for kind in ("ttft_ms", "tpot_ms")
                    },
                )
                for name in (
                    "4096-warmup",
                    "4096-formal",
                    "16384-warmup",
                    "16384-formal",
                )
            ],
            rows=[dict(private_fixture_only=True)],
        )
        for key in ("b0", "b1")
    }
    old = json.loads((ROOT / "website-publication/data/midterm-v102.json").read_text())
    records = {
        builder.NEW + "/joint-summary.json": data,
        builder.NEW + "/continuous-summary.json": continuous,
        "website-publication/data/midterm-v102.json": old,
    }

    def read(path):
        return (
            b"synthetic report"
            if path == builder.REPORT
            else json.dumps(records[path]).encode()
        )

    result = builder.build("a" * 40, read)
    assert result["version"] == "V1.03"
    assert result["sections"][0]["id"] == "full-answer-joint"
    assert result["current_summary"]["quality"]["b0"]["correct"] == 150
    assert result["historical_current_summary_v102"] == old["current_summary"]
    for key in (
        "wholechain_reference",
        "paired_own_compute",
        "corrected_quality",
        "acceptance_completion",
    ):
        assert result[key] == old[key]
    assert "rows" not in result["full_answer_joint"]["arms"][0]
    assert "rows" not in result["packed_continuous"]["b1"]
    assert (
        result["full_answer_joint"]["quality_observation"][
            "statistical_noninferiority_established"
        ]
        is False
    )
    assert result["sections"][0]["tables_en"]
    assert all(
        e["path"].startswith(builder.NEW + "/")
        for e in result["sections"][0]["evidence"]
    )


def test_report_writer_only_replaces_current_outline():
    source = (ROOT / "scripts/build_midterm_v103.py").read_text()
    assert (
        'current_table, remainder = remainder.split("## 三、测试范围与环境", 1)'
        in source
    )
    assert "中期测试报告-v102.md" in source


def test_report_writer_preserves_history_and_unique_section_numbers(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    builder = importlib.import_module("build_midterm_v103")
    math = importlib.import_module("recompute_full_answer_joint")
    pair = synthetic_data()
    derived = math.paired_metrics(pair)
    for arm, calculated in zip(pair["arms"], derived["arms"], strict=True):
        arm.update(calculated)
        arm["cost"]["cny_per_million_output_tokens"] = calculated[
            "cny_per_million_output_tokens"
        ]
        arm["utilization"] = dict(mean_percent=30)
    pair.update({k: v for k, v in derived.items() if k != "arms"})
    pair["latency_reduction_percent"] = dict(ttft_ms=0, tpot_ms=0)
    continuous = {
        key: dict(
            phases=[
                dict(
                    name=name,
                    requests=8,
                    failures=0,
                    cohort_output_tokens_per_second=100,
                    timing={
                        kind: dict(p95=10, p99=20) for kind in ("ttft_ms", "tpot_ms")
                    },
                )
                for name in (
                    "4096-warmup",
                    "4096-formal",
                    "16384-warmup",
                    "16384-formal",
                )
            ]
        )
        for key in ("b0", "b1")
    }
    archived = ROOT / builder.BASE / "中期测试报告-v102.md"
    original = (archived if archived.exists() else ROOT / builder.REPORT).read_bytes()
    target = tmp_path / builder.REPORT
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    (tmp_path / builder.NEW).mkdir(parents=True)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "measurements", lambda read: (pair, continuous))
    monkeypatch.setattr(
        builder,
        "evidence_readme",
        lambda pair, continuous: "Synthetic report-writing fixture only.",
    )
    builder.prepare()
    report = target.read_text()
    assert (target.parent / "中期测试报告-v102.md").read_bytes() == original
    assert report.count("### 5.12 ") == report.count("### 5.13 ") == 1
    for marker in ("### 7.5 ", "### 8.4 ", "### 当前证据与缺口清单"):
        assert report.count(marker) == 1
    start, end = "<!-- BEGIN ALIGNED_MMLU -->", "<!-- END ALIGNED_MMLU -->"
    assert (
        report.split(start)[1].split(end)[0]
        == original.decode().split(start)[1].split(end)[0]
    )
    assert "报告版本：V1.03" in report
    assert "本轮固定200题完整回答：150/200对150/200" in report
    assert "此前纠正组合原配置的质量结果为156/200" in report
