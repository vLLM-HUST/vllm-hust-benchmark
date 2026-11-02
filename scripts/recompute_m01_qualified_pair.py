"""Recompute 3.4.1 from work, timing and independently audited reference certificates."""

from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import io
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m01-qualified-pair-20261103"
E = REPORT / "evidence"


def read(name):
    return json.loads((E / name).read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_path(name, ref):
    total = Decimal(0)
    last = Decimal(0)
    count = 0
    source = E / name
    if source.exists():
        compressed = source.read_bytes()
        artifact = name
    else:
        artifact = name + ".archive.json"
        descriptor = read(artifact)
        assert descriptor["encoding"] == "gzip" and descriptor["filename"] == name
        compressed = b"".join((E / x["path"]).read_bytes() for x in descriptor["parts"])
        assert hashlib.sha256(compressed).hexdigest() == descriptor["compressed_sha256"]
    with io.TextIOWrapper(
        gzip.GzipFile(fileobj=io.BytesIO(compressed)), encoding="utf-8"
    ) as f:
        for line in f:
            n = json.loads(line)
            seconds = Decimal(str(n["seconds"]))
            start = Decimal(str(n["start"]))
            end = Decimal(str(n["end"]))
            assert (
                seconds >= 0
                and abs(start - last) < Decimal("1e-8")
                and abs(end - start - seconds) < Decimal("1e-8")
            )
            total += seconds
            last = end
            count += 1
    assert abs(total - Decimal(ref)) < Decimal("1e-8")
    return dict(
        nodes=count,
        sum_component_seconds=str(total),
        certified_lower_seconds=str(ref),
        artifact=artifact,
        sha256=digest(E / artifact),
    )


def aggregate(repetitions, components):
    st = sum((Decimal(r["T_seconds"]) for r in repetitions), Decimal(0))
    sr = sum((Decimal(r["Tref_lower_seconds"]) for r in repetitions), Decimal(0))
    sf = [sum(r["F_interval"][j] for r in repetitions) for j in [0, 1]]
    assert all(sum(c["F_interval"][j] for c in components) == sf[j] for j in [0, 1])
    return dict(
        sum_T_seconds=str(st),
        sum_Tref_lower_seconds=str(sr),
        sum_F_interval=sf,
        C_interval=[str(Decimal(f) / (2 * st * 10**12)) for f in sf],
        eta_percent_lower=str(100 * sr / st),
    )


def calculate():
    manifest = read("evidence-manifest.json")
    for entry in manifest["files"]:
        p = E / entry["path"]
        assert (
            p.is_relative_to(E)
            and p.stat().st_size == entry["bytes"]
            and digest(p) == entry["sha256"]
        ), entry["path"]
    domain = read("b0-work-domain-plan.json")
    assert domain["aliases"] == {"uninstrumented-3": "uninstrumented-2"}
    assert (
        domain["canonical_domain_sha256"]["uninstrumented-2"]
        == domain["canonical_domain_sha256"]["uninstrumented-3"]
    )
    timing = read("b0-timing-audit.json")
    assert (
        timing["status"] == "native_timing_collection_independently_audited"
        and len(timing["cases"]) == 3
    )
    work = {n: read(f"b0-r{n}-work-audit.json") for n in [1, 2]}
    refs = {n: read(f"b0-r{n}-reference-audit.json") for n in [1, 2]}
    traces = {
        n: check_path(
            f"b0-r{n}-reference-critical-path.jsonl.gz",
            refs[n]["reference_lower_seconds"],
        )
        for n in [1, 2]
    }
    for n in [1, 2]:
        assert (
            refs[n]["status"]
            == "all_lower_graph_source_leaves_and_dependencies_independently_audited"
        )
        assert (
            refs[n]["source_operation_sequence_complete"]
            and refs[n]["all_TP_readiness_and_completion_checked"]
        )
        assert read(f"b0-r{n}-components-audit.json")["remaining_gaps"] == 0
        assert (
            work[n]["status"]
            == "timed_native_masks_and_component_sums_independently_audited"
            and work[n]["mask_subset_checks"] > 0
        )
    repetitions = []
    family = {}
    for i, c in enumerate(timing["cases"]):
        assert c["name"] == f"uninstrumented-{i + 1}" and c["strict_uninstrumented"]
        assert (c["requests"], c["input_tokens"], c["output_tokens"]) == (
            200,
            282265,
            51200,
        )
        n = 1 if i == 0 else 2
        w = work[n]
        bounds = w["F_interval"]
        assert 0 < bounds[0] <= bounds[1]
        for j, b in enumerate(w["bounds"]):
            assert (
                b["basic_F"]
                == bounds[j]
                == sum(b["families"].values())
                == sum(b["basic_parts"].values())
            )
            for k, v in b["families"].items():
                family.setdefault(k, [0, 0])[j] += v
        repetitions.append(
            dict(
                name=c["name"],
                F_interval=bounds,
                T_seconds=str(c["synchronized_seconds"]),
                Tref_lower_seconds=refs[n]["reference_lower_seconds"],
                requests=c["requests"],
                input_tokens=c["input_tokens"],
                output_tokens=c["output_tokens"],
                uninstrumented=True,
                synchronized_ranks=2,
                source_sha256=c["source_file_sha256"],
                work_domain=n,
            )
        )
    labels = {
        "target:linear_attention:prefill": (
            "目标模型 GDN 预填充",
            "Target GDN prefill",
        ),
        "target:linear_attention:spec": (
            "目标模型 GDN 解码与验证",
            "Target GDN decode and verification",
        ),
        "target:full_attention:": (
            "目标模型全注意力层",
            "Target full-attention layers",
        ),
        "target_final:final_norm:": (
            "目标模型最终归一化与词表",
            "Target final normalization and vocabulary",
        ),
        "draft:full_attention:": ("必要草稿依赖", "Necessary draft dependencies"),
    }
    components = [
        dict(id=k, label_zh=labels[k][0], label_en=labels[k][1], F_interval=v)
        for k, v in family.items()
    ]
    b0 = dict(
        role="B0",
        repetitions=repetitions,
        work_components=components,
        aggregate=aggregate(repetitions, components),
        F_correspondence_qualified=True,
        reference_lower_qualified=True,
        reference_certificate_path="reports/m01-qualified-pair-20261103/evidence/b0-r1-reference-certificate.json",
        reference_certificates=[
            f"evidence/b0-r{n}-reference-certificate.json" for n in [1, 2]
        ],
        reference_paths=list(traces.values()),
    )
    qualification = read("b1-qualification.json")
    assert (
        qualification["status"]
        == "B1_timed_F_and_reference_lower_bound_correspondence_qualified"
    )
    data = read("b1-work-components.json")
    target = data["target_parts"]
    draft = data["draft_interval"]
    assert (
        sum(target.values()) == qualification["target_basic_F"]
        and data["prefill_segments"] == qualification["prefill_segments"]
    )
    labels = {
        "target_projection": ("目标模型投影", "Target projections"),
        "target_qk_av": ("目标模型 QK / AV", "Target QK / AV"),
        "target_vocabulary": ("目标模型输出词表", "Target vocabulary"),
        "target_scalar": (
            "目标模型基础标量与状态更新",
            "Target scalar and state updates",
        ),
        "target_prefill": ("GDN 预填充", "GDN prefill"),
    }
    components = [
        dict(
            id=k,
            label_zh=labels[k][0],
            label_en=labels[k][1],
            F_interval=[3 * v, 3 * v],
        )
        for k, v in target.items()
    ]
    components.append(
        dict(
            id="necessary_draft",
            label_zh="必要草稿依赖",
            label_en="Necessary draft dependencies",
            F_interval=[3 * draft[k] for k in ["lower", "upper"]],
        )
    )
    reps = []
    assert len(qualification["repetitions"]) == 3
    for i, r in enumerate(qualification["repetitions"]):
        assert r["name"] == f"uninstrumented-{i + 1}"
        bounds = [sum(target.values()) + draft[k] for k in ["lower", "upper"]]
        assert bounds == r["effective_basic_F_interval"]
        reps.append(
            dict(
                name=r["name"],
                F_interval=bounds,
                T_seconds=r["synchronized_seconds"],
                Tref_lower_seconds=r["Tref_lower_seconds"],
                requests=200,
                input_tokens=282265,
                output_tokens=51200,
                uninstrumented=True,
                synchronized_ranks=2,
                source_sha256=r["source_sha256"],
            )
        )
    trace = check_path(
        "b1-reference-critical-path.jsonl.gz", qualification["Tref_lower_seconds"]
    )
    b1 = dict(
        role="B1",
        repetitions=reps,
        work_components=components,
        aggregate=aggregate(reps, components),
        F_correspondence_qualified=True,
        reference_lower_qualified=True,
        reference_certificate_path="reports/m01-qualified-pair-20261103/evidence/b1-reference-certificate.json",
        reference_certificates=["evidence/b1-reference-certificate.json"],
        reference_paths=[trace],
    )
    for got, expected in zip(
        b1["aggregate"]["C_interval"],
        qualification["aggregate"]["C_TFLOPS_per_chip_interval"],
    ):
        assert abs(Decimal(got) - Decimal(expected)) < Decimal("1e-30")
    assert abs(
        Decimal(b1["aggregate"]["eta_percent_lower"])
        - Decimal(qualification["aggregate"]["eta_percent_lower"])
    ) < Decimal("1e-30")
    gain = [
        100
        * (
            Decimal(b1["aggregate"]["C_interval"][0])
            / Decimal(b0["aggregate"]["C_interval"][1])
            - 1
        ),
        100
        * (
            Decimal(b1["aggregate"]["C_interval"][1])
            / Decimal(b0["aggregate"]["C_interval"][0])
            - 1
        ),
    ]
    return dict(
        schema="m01-qualified-pair/v1",
        chips=2,
        milestone_percent=50,
        final_target_percent=90,
        evidence_qualified=True,
        reference_kind="certified_lower_bound_of_fixed_full_reference",
        current_milestone_met=Decimal(b1["aggregate"]["eta_percent_lower"]) >= 50,
        final_target_supported=Decimal(b1["aggregate"]["eta_percent_lower"]) >= 90,
        arms=[b0, b1],
        C_gain_percent_interval=[str(v) for v in gain],
        report_path="reports/m01-qualified-pair-20261103/README.md",
        evidence_manifest_sha256=digest(E / "evidence-manifest.json"),
        formulas=dict(
            C="sum(F)/(k*sum(T)*10^12)",
            eta_lower="100*sum(Tref_lower)/sum(T)",
            gain="100*(C1/C0-1)",
            reference="finish(v)=d(v)+max(finish(u):u->v);Tref=max(finish(v))",
        ),
        workload=dict(
            model="Qwen3.5-35B-A3B",
            precision="BF16",
            chips="2x910B2",
            tensor_parallel=2,
            dataset="MMLU-Pro fixed200",
            max_completion_tokens=256,
            input_tokens_per_repetition=282265,
            output_tokens_per_repetition=51200,
        ),
        quality_scope="Fixed-budget service workload; all outputs length-terminated. Full-answer quality and throughput belong to the separately identified max8192 batch. No current answer-quality score is borrowed from an older256-token run.",
    )


def report(d):
    b0, b1 = d["arms"]
    lines = [
        "# 3.4.1 单芯片有效算力：当前配对验收",
        "",
        "**本次 50% 里程碑：已达标。**"
        if d["current_milestone_met"]
        else "**本次 50% 里程碑：未达标。**",
        "",
        "Qwen3.5-35B-A3B，BF16、TP2、两片910B2；MMLU-Pro固定200请求，每题上限256 token。每轮输入282265、输出51200 token。两侧各3轮全部计入。",
        "",
        "## 计算口径",
        "",
        "Cᵢ = ΣⱼFᵢⱼ / (2 × ΣⱼTᵢⱼ × 10¹²)；ηᵢ ≥ 100 × ΣⱼT参,ᵢⱼ,下界 / ΣⱼTᵢⱼ。同一轮 C参=F/(kT参10¹²)，所以 η=100T参/T。F上下界用于C，参考时间下界用于η保守判定。",
        "",
        "T采用加载、预热后无插桩同步墙钟，开始与结束均覆盖两张卡。提交、执行、搬运、通信、同步和区间内真实空闲均保留；HTTP与分词在两侧都位于区间外。另一次路径采集用于核对F；未从计时中扣除估计开销。",
        "",
        "## 每轮原始量",
        "",
        "| 运行 | F区间（FLOPs） | 同步T（秒） | T参下界（秒） |",
        "| --- | --- | --- | --- |",
    ]
    for a in d["arms"]:
        for r in a["repetitions"]:
            lines.append(
                f"| {a['role']} / {r['name']} | [{r['F_interval'][0]}, {r['F_interval'][1]}] | {r['T_seconds']} | {r['Tref_lower_seconds']} |"
            )
    lines += [
        "",
        "## 算量逐项求和",
        "",
        "矩阵乘加按2次基础运算，基础标量按原算术目录计数；特殊函数调用另列。由保留输出反向闭包得到必要依赖，包含为保留输出提供状态的草稿工作。填充、重复和无依赖的未采纳工作不进入有效F。",
        "",
    ]
    for a in d["arms"]:
        lines += [
            "### " + a["role"],
            "",
            "| 运算项 | 三轮F下界 | 三轮F上界 |",
            "| --- | ---: | ---: |",
        ]
        for c in a["work_components"]:
            lines.append(
                f"| {c['label_zh']} | {c['F_interval'][0]} | {c['F_interval'][1]} |"
            )
        g = a["aggregate"]
        lines += [
            "",
            "ΣT = "
            + " + ".join(r["T_seconds"] for r in a["repetitions"])
            + " = "
            + g["sum_T_seconds"]
            + " s",
            "",
            f"C = [{g['sum_F_interval'][0]}, {g['sum_F_interval'][1]}] / (2 × {g['sum_T_seconds']} × 10¹²)",
            "",
            "= [" + ", ".join(g["C_interval"]) + "] TFLOPS/芯片",
            "",
            "η ≥ 100 × ("
            + " + ".join(r["Tref_lower_seconds"] for r in a["repetitions"])
            + ") / "
            + g["sum_T_seconds"]
            + " = "
            + g["eta_percent_lower"]
            + "%",
            "",
        ]
    lines += [
        "## 独立参考时间",
        "",
        "使用正式测量前固定的独立组件标定与资源依赖方法。组件按原数值检查、计时统计和适用域规则复用；缺失形状按同一协议补测。参考输入中不含B0/B1整机耗时和目标分数。模型路径映射与规范化算子耗时分别核验；参考不是硬件标注的理论FLOPS。",
        "",
        "finish(v)=d(v)+max{finish(u):u→v}，T参=max finish(v)。串行项累加，双卡并行与通信按依赖及资源顺序求最长路径，不能把所有设备时间直接相加。",
        "",
        "无插桩记录不能唯一确定的末次状态采用源码支持的所有分支。F求必要依赖上下界；参考图保留对所有分支均成立的节点和顺序，求完整参考时间的保守下界。删去非负工作或约束只能降低完成时间；完整参考方法保持不变。两组使用同一参考原则，各自工作量、图和证据范围分别核算。",
        "",
        "B0按预定结构规则保留629个批次、移除236个不确定批次，并在下界图中省略主机/元数据/复制控制节点；完整参考的这些项没有被赋零。保留节点全部完成组件标定与源码映射。B1按已验证的请求局部末次分支求下界，保留350468节点。两者都报告下界，不报告完整参考的精确值。",
        "",
        "B0第2/3轮仅在全部算量输入字节身份一致时共用复算，计时分别保留；第1轮独立计算。共享缓存/状态抽象通过单路径等价、分支包含及逐掩码检查。B1通过无插桩/采集源码对应、公开调度几何与容量证明。",
        "",
        "草稿参考使用原独立来源、固定donor规则与适用域门槛。B0完整负载的step1端点组满足原最低样本数和误差要求；结构选出子集的n=9记录保留。样本不足的step0端点组不在预定保留图中；不据此宣布完整参考点估计合格。保留图的小形状生产组全部使用精确来源。",
        "",
        "## 判定与比较",
        "",
        f"B1 η下界 = {b1['aggregate']['eta_percent_lower']}%，按原本次门槛50%判定。后续90%用同一公式单独考核，本批下界未证明90%。",
        "",
        "B0作为固定对照基线，列出完整C和η计算；项目里程碑判定作用于B1。B1相对B0的有效算力提升区间：["
        + ", ".join(d["C_gain_percent_interval"])
        + "]%。提升率与参考效率分别报告。",
        "",
        "本批均在256-token长度上限结束，属于固定输出预算工况。完整回答质量/吞吐采用网站3.4.2标明的独立批次，不借用旧256-token正确数。Request Throttling未纳入本次收益。",
        "",
        "## 复算与证据",
        "",
        "    python scripts/recompute_m01_qualified_pair.py --check",
        "",
        "复算器校验公开证据SHA256，逐项求F、核对3轮计时、累加完整关键路径、按Decimal计算C和η；网页再从原始量计算展示值。",
        "",
        "[证据清单](evidence/evidence-manifest.json)提供校验和与专用存储原件路径。原始失败、域不足记录和大型追踪保持原样。协议、两侧证明、补测、计时与关键路径在evidence目录。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with localcontext() as ctx:
        ctx.prec = 40
        data = calculate()
        raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        outputs = {
            REPORT / "summary.json": raw,
            ROOT / "website-publication/data/m01-qualified-pair-20261103.json": raw,
        }
    for p, raw in outputs.items():
        if args.check:
            assert p.read_text(encoding="utf-8") == raw, str(p)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(raw, encoding="utf-8", newline="\n")
    if not args.check:
        (REPORT / "README.md").write_text(report(data), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            dict(
                check=args.check,
                current_milestone_met=data["current_milestone_met"],
                arms=[dict(role=a["role"], **a["aggregate"]) for a in data["arms"]],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
