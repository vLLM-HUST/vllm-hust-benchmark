"""Publish the compute/reference identity without changing measured results."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f716a10ba8c2042e201de00b05ba75beb95579b0"  # pragma: allowlist secret
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
OUTPUT = ROOT / "website-publication/data/midterm-v076.json"


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def build():
    raw = pinned(REPORT)
    assert "报告版本：V0.76" in raw.decode()
    data = json.loads(pinned("website-publication/data/midterm-v075.json"))
    method = json.loads(pinned(REPORT_DIR + "/milestone-v15/method-status.json"))
    data.update(
        version="V0.76",
        source_commit=REVISION,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}",
        report_sha256=hashlib.sha256(raw).hexdigest(),
        m01_method=method,
    )
    data["milestone_contract"]["method"] = method
    sections = {section["id"]: section for section in data["sections"]}
    sections["m01-verdict"]["scope"] += (
        " C参=F/(k×T参×10¹²)，η=C/C参×100%=T参/T×100%。"
        "每次计算中实测与参考采用同一F和k；C与C参单位均为TFLOPS。"
        "达标条件为C≥0.5×C参；参考值不是已证明的理论上限或芯片标称峰值。"
    )
    sections["m01-verdict"]["scope_en"] += (
        " C_reference=F/(k*T_reference*10^12); eta=C/C_reference*100%"
        "=T_reference/T*100%. Measured and reference values use the same F and k"
        " within each calculation; both compute values are TFLOPS per chip."
        " The gate is C>=0.5*C_reference. The reference is not a proven theoretical"
        " upper bound or vendor peak."
    )
    sections["milestone-v15"]["tables"][0]["rows"][0][1] = (
        "C=F/(k×T×10¹²)，C参=F/(k×T参×10¹²)，η=C/C参×100%=T参/T×100%；同F、同k，η≥50%"
    )
    sections["milestone-v15"]["tables_en"][0]["rows"][0][1] = (
        "C=F/(k*T*10^12); C_reference=F/(k*T_reference*10^12); "
        "eta=C/C_reference*100%=T_reference/T*100%; same F and k; >=50%"
    )
    return data


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
