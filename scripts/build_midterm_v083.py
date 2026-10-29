"""Project current method identity into every current consumer; preserve history."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
METHOD = REPORT_DIR + "/current-method.json"
OUTPUT = ROOT / "website-publication/data/midterm-v083.json"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    # Historical builders must not re-label a later method as their old result.
    if json.loads(read(METHOD)).get("method_version") == "reference-revision-20261029":
        original_read = read

        def read(path):
            if path == METHOD:
                path = REPORT_DIR + "/current-method-v099.json"
            elif path == REPORT_DIR + "/method-history-index.json":
                path = REPORT_DIR + "/method-history-index-v099.json"
            return original_read(path)

    raw = read(REPORT)
    assert "报告版本：V0.83" in raw.decode()
    data = json.loads(read("website-publication/data/midterm-v082.json"))
    method = json.loads(read(METHOD))
    data.update(
        version="V0.83",
        source_commit=revision,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        m01_method=method,
    )
    contract = data["milestone_contract"]
    contract["historical_method_at_v082"] = contract["method"]
    contract["method"] = copy.deepcopy(method)
    contract["method_source"] = {
        "path": METHOD,
        "sha256": hashlib.sha256(read(METHOD)).hexdigest(),
    }
    contract["decision_relevant_gaps"]["m01"] = []
    data["method_history"] = json.loads(read(REPORT_DIR + "/method-history-index.json"))
    sections = {s["id"]: s for s in data["sections"]}
    zh = "原V1.5保留50%、独立推导及无插桩要求。90%是2026年10月24日冻结、25日完成核验的本轮实施判据，不是主责单位或第三方批准的正式指标变更。本轮保留同轮计数成本，不属严格无插桩；99.9397%不是芯片理论峰值利用率。"
    en = "Original V1.5 retains its 50% gate, independently derived reference and uninstrumented requirement. The 90% gate belongs to the round implementation method frozen on 2026-10-24 and verified on 2026-10-25, not an externally approved formal indicator amendment. Same-run accounting costs remain; timing is not strictly uninstrumented. 99.9397% is not chip theoretical-peak utilization."
    method_section = sections["milestone-v15"]
    method_section["title"] = (
        "原V1.5依据与本轮实施方法 / Original V1.5 and current round method"
    )
    for key, note in (("scope", zh), ("scope_en", en)):
        method_section[key] = note
        sections["m01-verdict"][key] += " " + note
    method_section["tables"][0]["rows"][0][1] = (
        "本轮独立整链C参取三次能力中位数，η≥90%；原V1.5的50%作为历史依据单列"
    )
    method_section["tables_en"][0]["rows"][0][1] = (
        "Current round: median of three calibrated wholechain capacities; eta>=90%. Original V1.5 50% remains historical."
    )
    method_section["source_heading"] = (
        "原V1.5依据与本轮实施方法 / Original V1.5 and current round method"
    )
    method_section["tables"][0]["headers"][1] = "依据及本轮方法"
    method_section["tables_en"][0]["headers"][1] = "Basis and current round method"
    method_section["evidence"].insert(
        0,
        {
            "path": METHOD,
            "kind": "blob",
            "label": "本轮方法状态 / Current round method",
        },
    )
    method_section["evidence"].append(
        {
            "path": REPORT_DIR + "/method-history-index.json",
            "kind": "blob",
            "label": "历史方法身份索引 / Historical method identities",
        }
    )
    historical = {
        "reference-closure",
        "worker-model",
        "fenced-model",
        "trace-qualification",
        "m01",
        "runtime-recovery",
        "state-commit",
        "model-runtime",
        "preprocess-publication",
        "projection-references",
        "logits-collective-reference",
    }
    for s in data["sections"]:
        if s["id"] in historical:
            original_scope = s.get("scope", "")
            s["historical"] = True
            s["as_of"] = "2026-10-24"
            s["scope"] = (
                "历史记录（截至2026年10月24日；组件或旧参考方法）："
                + s.get("scope", "")
                + " 当时状态不代表当前整链参考判定。"
            )
            s["scope_en"] = (
                "Historical record through 2026-10-24 (component or prior-reference method): "
                + s.get("scope_en", original_scope)
                + " These stage-local statuses do not describe current wholechain attainment."
            )
        for e in s.get("evidence", []):
            if e.get("path", "").endswith("milestone-v15/reuse-audit.json"):
                e["label"] = (
                    "整链方法完成前的历史复用快照 / Historical pre-wholechain reuse snapshot"
                )
            if e.get("path", "").endswith("milestone-v15/method-status.json"):
                e.update(path=METHOD, label="本轮方法状态 / Current round method")
    resources = sections["resources"]
    resources["scope"] += (
        "开发栈内部资源系数与同期利用率观测；不构成固定0.18/0.18 B0的M06货币成本通过。综合月费、同一成功请求集合及同期时长加权利用率适用证据仍待核验；生命周期结果保留为历史补充。"
    )
    resources["scope_en"] += (
        "Development-stack resource coefficients and same-window utilization only; not M06 monetary-cost acceptance against fixed 0.18/0.18 B0. All-in monthly fees, matched successful requests and time-weighted same-window utilization evidence remain to be verified. Lifecycle observations remain historical supplements."
    )
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    OUTPUT.write_text(
        json.dumps(
            build(
                subprocess.check_output(
                    ["git", "rev-parse", a.revision], cwd=ROOT, text=True
                ).strip()
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
