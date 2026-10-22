"""Render the captured-window result directly from the immutable assessment."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render(data):
    terms = data["resource_terms"]["resource_mean_seconds"]
    rows = [
        ("有效行基础运算", f"{data['basic_operations']:,} 次"),
        ("设备同步窗口", f"{data['measured_window_seconds']:.9f} 秒"),
        ("计算参考项", f"{terms['compute']:.6f} 秒"),
        ("通信参考项", f"{terms['communication']:.6f} 秒"),
        ("HBM分方向参考项", f"{terms['hbm_directional']:.6f} 秒"),
        ("HBM合计参考项", f"{terms['hbm_pooled']:.6f} 秒"),
        ("资源参考时间（取最大值）", f"{data['empirical_reference_seconds']:.6f} 秒"),
        ("本次设备窗口经验达成率", f"{data['empirical_attainment_percent']:.6f}%"),
        ("内部阈值", f"{data['internal_threshold_percent']}%"),
        ("本次设备窗口判定", "达到阈值" if data["threshold_met"] else "未达到阈值"),
    ]
    return "\n".join(
        [
            "| 指标 | 结果 |",
            "| --- | ---: |",
            *(f"| {name} | {value} |" for name, value in rows),
        ]
    )


if __name__ == "__main__":
    print(
        render(
            json.loads(
                (
                    ROOT / "m01-recovery/accounting-assessment-20261022/assessment.json"
                ).read_text()
            )
        )
    )
