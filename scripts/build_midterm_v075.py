"""Versioned projection; V0.74 and all experimental originals remain unchanged."""

import hashlib
import json
import subprocess
from pathlib import Path

from build_midterm_publication import section as report_section, tables
from build_v075_review import AUDIT, section

ROOT = Path(__file__).resolve().parents[1]
REVISION = "706b08d5e022adac27a9d69add8b10844688299a"  # pragma: allowlist secret
REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告.md"
OUTPUT = ROOT / "website-publication/data/midterm-v075.json"


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def build():
    raw = pinned(REPORT)
    assert "报告版本：V0.75" in raw.decode()
    data = json.loads(pinned("website-publication/data/midterm-v074.json"))
    audit = json.loads(pinned(AUDIT))
    data.update(
        version="V0.75",
        source_commit=REVISION,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}",
        report_sha256=hashlib.sha256(raw).hexdigest(),
    )
    data["sections"].insert(1, section(audit))
    verdict = next(s for s in data["sections"] if s["id"] == "m01-verdict")
    verdict["tables"] = tables(report_section(raw.decode(), "#### M01 本轮验收判定"))
    verdict["tables_en"][0]["rows"][0] = [
        "Effective F",
        "333,265 target positions; 1,730,876,356,085,005 listed target FLOPs verified for the timed run",
        "Not full F: actual GDN partitions and useful draft contexts remain open; no old-run counts or uniform acceptance scaling",
    ]
    verdict["evidence"].append(section(audit)["evidence"][1])
    data["m01_method"]["additional_existing_evidence"].append(
        audit["m01"]["source_path"]
    )
    return data


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
