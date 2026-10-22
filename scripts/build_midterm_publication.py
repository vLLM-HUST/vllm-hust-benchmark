"""Build a report-pinned website view; audit archived KV controls without inference."""

import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "76ecccb99558a084ae59b0024856e54c09b3da62"  # pragma: allowlist secret
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
OUTPUT = ROOT / "website-publication/data/midterm-v067.json"

# Repository-relative locators are resolved against the website's exact benchmark
# publication commit, not a moving main branch or a local acquisition path.
EVIDENCE = {
    "baseline018": ["baseline018-sweprefix", "BASELINE-IDENTITY-AUDIT-20261022.md"],
    "dataset-quality-inventory": [
        "../../website-publication/data/hle-verified-20261015/report.json",
        "../../website-publication/data/frontierscience-20261013/report.json",
    ],
    "reference-closure": ["m01-recovery/reference-closure-20261022"],
    "worker-model": ["m01-recovery/worker-model-20261022"],
    "fenced-model": ["m01-recovery/fenced-model-20261022"],
    "trace-qualification": ["m01-recovery/trace-qualification-20261021"],
    "long-context-parser": ["m05-parser-confirmation-20261021"],
    "paired-lifecycle": ["m06-paired-20261021"],
    "candidates": [
        "aligned-mmlu-b0.tar.gz",
        "aligned-mmlu-b1.tar.gz",
        "b1-e32-screen-evidence.tar.gz",
        "b1-e24-screen-evidence.tar.gz",
        "b1-kvmat-screen-evidence.tar.gz",
        "b1-kvmat-e24-screen-evidence.tar.gz",
        "microflow-school028",
        "vspec-school028",
    ],
    "quality": ["quality-regression/raw", "quality-regression/finish-breakdown.json"],
    "recovered-quality": ["quality-regression/recovered-20261021"],
    "native-paired-quality": [
        "quality-regression/native-recovered-20261021",
        "quality-regression/recovered-20261021",
    ],
    "long-context": ["long-context-originals.tar.gz", "long-context-originals.json"],
    "current-b1-long-context": ["long-context-current-b1-20261021"],
    "long-diagnostic": [
        ".",
        "synthetic-b0-evidence.tar.gz",
        "synthetic-b1-evidence.tar.gz",
    ],
    "resources": [
        ".",
        "aligned-mmlu-b0.tar.gz",
        "aligned-mmlu-b1.tar.gz",
        "aligned-mmlu-common.tar.gz",
    ],
    "m01": ["m01-recovery/r21-device-maps"],
    "runtime-recovery": ["m01-recovery/runtime-recovery"],
    "state-commit": ["m01-recovery/state-commit-recovery"],
    "model-runtime": ["m01-recovery/model-runtime-20261021"],
    "preprocess-publication": ["m01-recovery/preprocess-publication-20261021"],
    "projection-references": ["m01-recovery/projection-references-20261021"],
    "logits-collective-reference": [
        "m01-recovery/logits-collective-reference-20261021"
    ],
    "m01-verdict": [
        "m01-recovery/request-assessment-20261022",
        "m01-recovery/accounting-assessment-20261022",
        "m01-recovery/reference-closure-20261022",
        "m01-recovery/worker-model-20261022",
        "m01-recovery/model-runtime-20261021",
        "m01-recovery/logits-collective-reference-20261021",
    ],
}


def evidence_locator(relative):
    path = (ROOT / REPORT_DIR / relative).resolve()
    assert path.exists(), relative
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "kind": "tree" if path.is_dir() else "blob",
        "label": "B0 + B1 原始归档 / original archives"
        if relative == "."
        else relative,
    }


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def section(text, heading):
    start = text.index(heading)
    level = len(heading.split(" ", 1)[0])
    tail = text[start + len(heading) :]
    boundary = re.search(r"(?m)^#{1," + str(level) + r"} ", tail)
    return tail[: boundary.start()] if boundary else tail


def tables(text):
    found = []
    for block in re.findall(r"(?m)(?:^\|.*\|[ \t]*\n)+", text):
        rows = [
            [c.strip() for c in row.strip().strip("|").split("|")]
            for row in block.strip().splitlines()
        ]
        if len(rows) > 2 and all(
            re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in rows[1]
        ):
            found.append({"headers": rows[0], "rows": rows[2:]})
    return found


def kv_audit():
    results = []
    all_salts = set()
    for name in ("b1-kvmat-screen", "b1-kvmat-e24-screen"):
        summary = json.loads((ROOT / REPORT_DIR / f"{name}.json").read_text())
        archive = ROOT / REPORT_DIR / summary["archive"]["file"]
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == summary["archive"]["sha256"]
        )
        with tarfile.open(archive) as tar:
            contents = {}
            for member in tar.getmembers():
                if member.isfile():
                    relative = member.name.split("/", 1)[1]
                    raw = tar.extractfile(member).read()
                    assert (
                        hashlib.sha256(raw).hexdigest()
                        == summary["archive"]["files"][relative]
                    )
                    contents[relative] = raw
        run = summary["run_id"]
        policy = [
            json.loads(line)
            for line in contents[f"{run}/kvmat-policy.jsonl"].splitlines()
        ]
        engine = [
            json.loads(line)
            for line in contents[f"{run}/kvmat-engine.jsonl"].splitlines()
        ]
        salts = {
            hashlib.sha256(row["runtime_cache_salt"].encode()).hexdigest()
            for row in policy
        }
        assert len(policy) == len(engine) == len(salts) == 200
        assert salts == {row["cache_salt_sha256"] for row in engine}
        assert not salts & all_salts
        all_salts.update(salts)
        assert all(
            row["runtime_effective_decision"] == "recompute"
            and row["runtime_target_reuse_tokens"] == 0
            for row in policy
        )
        rates = re.findall(
            r"Prefix cache hit rate: ([0-9.]+)%",
            contents[f"{run}/service.log"].decode(),
        )
        assert rates and all(float(rate) == 0 for rate in rates)
        receipt = json.loads(contents[f"{run}/receipt.json"])
        assert receipt["all_devices_released"] and receipt["bench_exit_code"] == 0
        before = contents[f"{run}/npu-before.txt"].decode()
        assert all(
            f"No running processes found in NPU {card}" in before
            for card in range(4, 8)
        )
        results.append(
            {
                "run_id": run,
                "requests": len(policy),
                "unique_salts": len(salts),
                "prefix_hit_samples": len(rates),
                "max_logged_prefix_hit_percent": max(map(float, rates)),
                "service_pid": receipt["service_pid"],
                "devices_released": True,
                "archive_sha256": summary["archive"]["sha256"],
            }
        )
    return results


def build():
    raw = pinned(REPORT)
    text = raw.decode()
    assert "报告版本：V0.67" in text
    url = f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}"
    specs = [
        (
            "baseline018",
            "0.18系列基线 / 0.18 baseline evidence",
            "### 3.3 0.18系列基线证据",
            "已有Qwen3.5 SWE-prefix-reuse五个900秒正式窗口，零失败，保留0.18/0.18及text-only兼容补丁身份。第五章数据集B0/B1为0.28.1 / Ascend 0.25.1rc2开发栈对照，其收益不代表相对0.18的收益；不同工况不互相代替。",
        ),
        (
            "dataset-quality-inventory",
            "既有任务质量与配对缺项 / Existing task quality and pairing gaps",
            "### 5.11 HLE-Verified与FrontierScience既有任务质量",
            "HLE与FrontierScience已有任务质量观测，不仅是性能遥测。Qwen替代裁判、单次运行，未决项保留完整分母；不是官方评分复现。当前B1组合配对为空，未决及截断评分仍需复核，不宣称质量改善。HLE公开归档未包含全部问答载荷，不能独立重评。",
        ),
        (
            "reference-closure",
            "同工况参考补测汇总 / Matched reference measurements",
            "#### 同工况参考补测汇总",
            "GDN递归、WY/KKT、卷积、HBM及同API通信参考已完成；设备映射已核验。HBM物理读写比例偏差约3.6%，参考分量时间不等于模型关键路径。请求窗口核算见M01本轮判定，不替换正式B1。",
        ),
        (
            "worker-model",
            "双worker采集与参考补测 / Dual-worker capture and references",
            "#### 双worker采集修复与实际形状参考补测",
            "同工况200请求全部完成，双worker任务轨迹已取得，四卡释放。实际75种通信载荷各有三次参考；跨接口264个不一致样本保留，同API通信参考已完成。请求窗口核算见M01本轮判定，不替换正式B1。",
        ),
        (
            "fenced-model",
            "完整模型同步边界补采 / Full-model device fences",
            "#### 完整模型设备同步边界补采",
            "此前fenced-r1记录：同工况35B模型200请求全部完成，两卡同步回执通过；rank 0缺设备任务轨迹，服务管理进程exit 130。此采集问题已在上方worker-r1补测修复，旧证据保留。完整有效F与匹配参考仍未闭合，M01未通过本轮验收，不替换正式B1。",
        ),
        (
            "trace-qualification",
            "M01采集资格 / M01 collection qualification",
            "#### 应用级采集与原生事件身份验证",
            "两卡BF16合成资格，非35B模型性能复测。原生日志与profiler计数一致，16条已知探针依赖配对通过；整模型事件代次、图内事件与HCCL通知尚未全覆盖。插桩计时不代替未插桩性能，M01仍未通过本轮验收。",
        ),
        (
            "long-context-parser",
            "长输入接口修复确认 / Long-input API repair confirmation",
            "### 7.3 思考字段分离后的独立确认",
            "同一恢复运行栈及原六题，启用原生qwen3思考解析器，正文严格匹配6/6、零失败；独立思考字段另存。固定资格请求不计入六题，输出token包含思考生成。新服务、空编译缓存和独立请求盐，OS文件页缓存未全局清空。原0/6记录保留，不代表主数据集质量或性能增益。",
        ),
        (
            "paired-lifecycle",
            "恢复环境配对性能与成本 / Recovered paired performance and lifecycle cost",
            "### 5.10 恢复环境配对性能与全流程成本",
            "同一200条输入、out256、每臂一次预检查，独立服务和空编译缓存。操作系统文件页缓存未全局清空。请求阶段与启动至释放的全流程分别统计；保留P99退化、协议偏差及退出日志，不推断长期摊销、统计显著性或任务准确率。",
        ),
        (
            "candidates",
            "同工况性能候选 / Performance candidates",
            "### 5.4 E24/R36与KV materialization同工况补测",
            "200请求、out256。各指标分别选择B1，不将不同配置的最优指标合并为单一部署结果。",
        ),
        (
            "quality",
            "质量回归 / Quality regression",
            "### 5.7 MMLU-Pro性能输入的任务质量回归",
            "同一200题，输出上限8192；固定评分器、完整分母。达到输出上限仍可被提取器判对，不等于自然完成；不是全集评分或统计非劣性结论。",
        ),
        (
            "recovered-quality",
            "恢复环境质量复核 / Recovered-runtime quality",
            "### 5.8 恢复环境完整回答质量复核",
            "同一200题、原评分器、out8192，新服务与空编译缓存，计量观察器关闭。保存全部SSE和截断，按固定分母评分；与既有B0及Microflow结果分别比较，不冒充同环境新增性能配对或统计非劣性结论。",
        ),
        (
            "native-paired-quality",
            "恢复环境配对质量 / Recovered native-paired quality",
            "### 5.9 恢复环境 Native 与优化方案配对质量",
            "同一恢复基础运行栈、200题、原评分器、out8192，独立服务和空缓存。Native恢复五个原生源码模块，禁用候选插件；两臂请求与分词逐题一致。完整分母包含截断，逐题差异见中文报告，不宣称全集质量或统计非劣性，不替换正式吞吐。",
        ),
        (
            "long-context",
            "长上下文 / Long context",
            "### 7.1 既有配置长上下文探针",
            "四组共160条探针，其中72条实际输入≥16K均回答正确；长度32,781、131,085、261,901 token，每条仅输出7 token。探针没有完整流式时序，不能推导TTFT/TPOT分位数。",
        ),
        (
            "current-b1-long-context",
            "当前B1长输入检索 / Current B1 long-input retrieval",
            "### 7.2 当前B1恢复环境长输入检索",
            "保持E24/R36、KV materialization与Microflow配置，独立空缓存和逐请求隔离。6题完成、零传输失败，实际输入16,457与32,843 token。正文严格匹配0/6；唯一思考结束标记之后的答案段补充解析6/6，不替换原始判分，不计入数据集准确率或性能排名。",
        ),
        (
            "long-diagnostic",
            "长输入连续运行诊断 / Sustained long-input diagnostic",
            "### 10.2 异配置候选诊断（不参与原方案验收）",
            "实际16,384 token、C8、300秒测量，固定输入温缓存工况，无语义评分。此处B1是异运行栈的诊断候选，不是中期MMLU-Pro最优B1；吞吐下降27.5086%，窗口资源成本增加37.9474%。资格预热及排空另计，不能将此组和数据集配对拼接为同一结论。",
        ),
        (
            "resources",
            "资源成本与尾时延 / Resource cost and tail latency",
            "### 5.2 MMLU-Pro同机配对补测",
            "此表为原始E16/R36配对，不是当前Microflow最优配置。主成本包含4张预约卡，另列2卡服务口径；请求区间成本下降14.63%，完整生命周期成本增加2.68%，P99 TPOT增加57.07%。没有人民币计费依据。",
        ),
        (
            "m01",
            "M01 有效算力进展 / Effective compute progress",
            "#### r21 设备映射与同轮计量",
            "V0.57：恢复环境投影及词表通信参考已完成；M01本轮结论为证据受限、未通过验收，不等于实测低于50%。具体同窗口、运算量与匹配参考限制见独立判定表。保留既有原始计量与全部失败，不采纳原始证据不可用的r23结果，正式B0/B1性能不变。",
        ),
        (
            "runtime-recovery",
            "运行时恢复与FIA验证 / Runtime recovery and FIA validation",
            "#### 运行时恢复与 FIA 数值资格验证",
            "33项锁定源码哈希一致；按CANN 9.1.0重建产物不同于原准入产物，未放宽原哈希准入规则。每轮新进程、新输出、独立可写缓存，不导入上一轮推理状态。双卡各18个用例；固定元数据图重放采用query倍率1、-1、2，每次输出填NaN并重算CPU参考，每卡36项变输入检查通过，保存数组独立复算。该组仅覆盖采样输出和固定元数据图重放，不覆盖动态长度、块表更新、双缓冲生命周期，不能据此排除完整模型跨轮状态污染。完整模型恢复计量见下方独立记录。",
        ),
        (
            "state-commit",
            "状态提交隔离验证 / Isolated state-commit validation",
            "#### 状态提交内核恢复验证",
            "锁定state_commit.publish源码；r1/r2导入失败原样保留。r3使用独立虚拟环境、禁用系统site-packages，两卡各24个用例（6空、18非空）及同卡跨流事件检查通过；保存数组独立CPU重算完全相等，变输入产生不同accepted输出。三次尝试均新进程、新输出目录、全新编译缓存，不导入前次缓存或推理状态，四卡释放。恢复候选未证明等同原运行时；仅证明合成用例下eager状态提交和同卡跨流行为，不证明完整模型质量、图执行、跨卡时钟对齐或全模型执行流闭合，不能排除完整模型跨轮状态污染。M01仍未判定，B0/B1数值不变。",
        ),
        (
            "model-runtime",
            "恢复环境同工况计量 / Recovered same-workload measurements",
            "#### 恢复环境同工况计量",
            "两轮均为同一MMLU-Pro 200请求、out256、新服务及空编译缓存；双卡FULL图捕获及Microflow分区检查通过。360秒采样完整覆盖请求、末次前向事件和全部导出同步三个区间。表内吞吐包含计量插桩开销，不替换正式B1；十类基础运算小计不是完整有效F，M01仍未判定。完整原始记录、逐文件SHA256和Release定位随证据入库。",
        ),
    ]
    specs.append(
        (
            "preprocess-publication",
            "GDN 发布与预处理图验证 / GDN publication and preprocessing graphs",
            "#### GDN 元数据发布与预处理图重放验证",
            "两卡、两个元数据bank、80/1024/2048/4096容量，64次变元数据图重放及560项独立CPU数值复算通过。前两次探针准入失败保留；成功轮新进程、空编译缓存，四卡释放。仅验证实际MTPFrame发布和preprocess局部路径，未覆盖完整模型并发批次、全执行流依赖或历史设备映射恢复。两轮out256性能记录缺少结束原因与原始生成token，不能据此判完整质量。完整有效F和M01仍未判定，正式B0/B1不变。",
        )
    )
    sections = []
    specs.append(
        (
            "projection-references",
            "投影算子参考标定 / Projection reference calibration",
            "#### 恢复环境投影算子参考标定",
            "恢复环境两卡覆盖121种密集形状、36,708个实际专家分布及45种词表形状；按r7调用次数加权，保留全部重复与两卡差异。每组新进程、空编译缓存，四卡释放。独立热重放组件参考不等于整模型关键路径；图填充运算不冒充有效F，M01仍未判定，正式B0/B1不变。",
        )
    )
    specs.extend(
        [
            (
                "logits-collective-reference",
                "词表通信参考 / Logits collective reference",
                "#### 恢复环境词表通信参考标定",
                "76种TP2载荷，每种3组、每组512次；数值、边界与对端计数器校验通过，无重置，四卡释放。保留flit原单位与全部重复，不叠加收发流量。仅为源码推导的词表相关通信参考，不是整模型HCCL覆盖或M01通过结论。",
            ),
            (
                "m01-verdict",
                "M01 本轮判定 / M01 acceptance verdict",
                "#### M01 本轮验收判定",
                "测试方案为vLLM-HUST E24/R36 + KV materialization请求重计算 + Prefill Microflow FA；Qwen3.5-35B-A3B BF16/TP2、FULL、MTP2、每卡20 GiB KV。恢复环境worker-r1单独计量，与第五章824.894609 token/s吞吐中位数不是同一轮联合成绩。本次插桩请求窗口135.340126192秒，基础运算口径经验参考达成率30.2443%–30.2505%，低于内部50%阈值。计数范围反映采样边界，不是置信区间。两卡任务边界、同窗口HBM和通信计数已核验；不外推为未插桩服务结果，不替换正式B0/B1。186.80秒设备包络的21.92%仅为诊断值。",
            ),
        ]
    )
    for ident, title, heading, note in specs:
        selected = tables(section(text, heading))
        assert selected, heading
        sections.append(
            {
                "id": ident,
                "title": title,
                "scope": note,
                "tables": selected,
                "source_heading": heading.lstrip("# "),
                "evidence": [evidence_locator(path) for path in EVIDENCE[ident]],
            }
        )
    return {
        "schema": "midterm-publication/v1",
        "version": "V0.67",
        "report_url": url,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": REVISION,
        "sections": sections,
        "kv_audit": {
            "status": "historical-controls-verified-not-a-bug-exclusion",
            "scope": "仅核对E16/E24逐请求重算归档；每轮独立服务、运行前四卡无进程、请求盐唯一且跨两轮不重复、已记录前缀命中率均0%。这些证据不支持常规前缀缓存跨轮命中的解释，但不能排除Mamba/GDN状态残留或底层错误。未进行新的设备复测，未将隔离性或收益因果归因判为通过。Microflow与其他配置不自动继承本审计结论。",
            "runs": kv_audit(),
        },
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
