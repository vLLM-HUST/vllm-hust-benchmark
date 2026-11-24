"""Export the frozen, payload-free Tradeoff v3 projection; never recompute native metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ENGINE = "vLLM-HUST/vllm-hust-engine"
ENGINE_COMMIT = "a039ae0ea16296bc3367d31ea8630413b6ddd7c6"  # pragma: allowlist secret
ASSET_SHA = "2ddfaa0280fd8be0b415b166ac41ff06950ecc32b7b6a6424f464297d489d9d6"  # pragma: allowlist secret
FAMILY = "optimization-exploration-20261115-v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def export(root, out):
    root, out = root.resolve(), out.resolve()
    evidence = json.loads((root / "tradeoff-catalog-v3/tradeoff.json").read_bytes())
    if (
        evidence["schema"] != "vllm-hust-tradeoff-evidence/v1"
        or (
            evidence["physical_trials"],
            evidence["admitted_trials"],
            evidence["source_row_references"],
        )
        != (24, 22, 34)
        or evidence["candidate_promoted"]
        or evidence["quality_qualified"]
        or evidence["stability_qualified"]
    ):
        raise ValueError("Exact retained Tradeoff-v3 exploratory projection required")
    asset = json.loads((root / "tradeoff-mod-asset-v0.1.2.json").read_bytes())
    if asset["sha256"] != ASSET_SHA or not asset["redownload_verified"]:
        raise ValueError("Published source archive identity/readback mismatch")
    for spec in evidence["sources"]:
        if sha((root / spec["path"]).read_bytes()) != spec["sha256"]:
            raise ValueError("Source native analysis changed")
    mapping = json.loads(
        (
            root
            / "E19-E26-layout-overlap-off-v23/runs/E19-01-E26-NZ-linearND-native-off/device-mapping.json"
        ).read_bytes()
    )
    if mapping["npu_smi_mapping"].count("Ascend 910B2") != 2:
        raise ValueError("Hardware claim must come from the original device inventory")
    payloads = {
        name: (root / "tradeoff-catalog-v3" / name).read_bytes()
        for name in (
            "tradeoff.json",
            "tradeoff.svg",
            "all-trials.svg",
            "all-trials.csv",
        )
    }
    # This is an already normalized metric projection, not raw model/judge traffic.
    forbidden = re.compile(
        r"/root/|/workspace/|/data/shared_upload/|sk-[A-Za-z0-9]|gh[pousr]_[A-Za-z0-9]|\x00"
    )
    for name, raw in payloads.items():
        if forbidden.search(raw.decode()):
            raise ValueError(
                "Host path/credential marker in public projection: " + name
            )
    provenance = dict(
        schema="optimization-exploration-provenance/v1",
        report_date="2026-11-15",
        source_repository=ENGINE,
        source_publication_commit=ENGINE_COMMIT,
        source_archive=asset,
        payloads={
            name: dict(sha256=sha(raw), bytes=len(raw))
            for name, raw in payloads.items()
        },
        reference_identity="OriginalND E09 is a historical B1-r6 internal control, not the official B0 arm",
        source_analyses=evidence["sources"],
        catalog_sha256=evidence["catalog_sha256"],
        hardware_inventory_sha256=sha(
            (
                root
                / "E19-E26-layout-overlap-off-v23/runs/E19-01-E26-NZ-linearND-native-off/device-mapping.json"
            ).read_bytes()
        ),
        scope="Historical MMLU200 B1-internal exploration only. Each original native analyzer owns raw/input/clock/activation/compiler/cleanup admission. This export verifies projection/source identity and does not reexecute inference or revalidate private raw archives.",
        raw_access="Native raw/source archives remain in the private Engine Release and require repository access; public files contain no request prompts, completions, gold answers or host paths.",
        quality_qualified=False,
        stability_qualified=False,
        formal_acceptance=False,
        candidate_promoted=False,
    )
    payloads["provenance.json"] = (
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    target = out / "website-publication/data" / FAMILY
    target.mkdir(parents=True, exist_ok=True)
    for name, raw in payloads.items():
        path = target / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Never replace a frozen projection; use a new version")
        path.write_bytes(raw)
    report = out / "reports/performance-exploration-20261115-r1/README.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("""# B1 内部优化探索 · 2026-11-15

Tradeoff v3 收录24次不同的模型试验：22次完整MMLU200、2次启动失败；34处来源引用按实测ID/指标/seal去重，真正重复全部保留。范围为双Ascend910B2、Qwen3.5-35B-A3B BF16、固定200/max8192/EOS/temp0/seed0/burst200、MTP2/FULL。原始映射文件核验两张910B2，不能由目录名推断硬件。

这些是B1-r6内部配置观测，**原ND E09不是正式B0**；不替换固定五数据集B0/B1或四目标结论。容量、State策略、原生多流、decode padding、布局/router与组合的全部正负结果见[来源绑定指标](../../website-publication/data/optimization-exploration-20261115-v1/tradeoff.json)、[完整试验CSV](../../website-publication/data/optimization-exploration-20261115-v1/all-trials.csv)、[取舍图](../../website-publication/data/optimization-exploration-20261115-v1/tradeoff.svg)、[完整试验表](../../website-publication/data/optimization-exploration-20261115-v1/all-trials.svg)和[provenance](../../website-publication/data/optimization-exploration-20261115-v1/provenance.json)。失败没有零吞吐/零分；不挑最好一轮、不拟合跨配置曲线。菱形仅表示吞吐/平均TPOT/平均TTFT的观测非支配点，不考虑质量。

E19关闭态相对原ND吞吐+5.629%、meanTTFT−8.573%、meanTPOT−0.553%，正确149→143/200，截断18→20；E19Off→E18On吞吐−0.299%、meanTPOT−0.490%，正确143→154/200。只作带范围的历史观测，不将组合约5%收益归给多流；分数/序列变化不能证明模型或引擎单方根因。所有原生比较与配对对错转换保留在指标JSON。E20的E28/必需GDN96已完成：吞吐1102.424 tok/s，相对原ND吞吐+3.272%、meanTTFT−11.558%、meanTPOT+8.720%、P95TPOT+9.045%，正确149→137/200、截断18→19，性能初筛不通过；不能由E26/E32端点插值或隐去本次负结果。

native原clock不含启动/加载/预热/评分/报告；TTFT含burst排队，生成量会变化。每轮冷成本见Engine的对应报告。TorchDynamo/Triton封口范围不包含CANN，未知格式/cache/临时内存保留未知。没有独立质量等价、稳定性、MFU或等预算Autotune贡献证明；MOD默认关闭，不自动部署。

来源为[Engine固定发布提交](https://github.com/vLLM-HUST/vllm-hust-engine/commit/a039ae0ea16296bc3367d31ea8630413b6ddd7c6)的payload-free投影，原始字节/来源摘要保持不变。[MOD0.1.2离线源包](https://github.com/vLLM-HUST/vllm-hust-engine/releases/download/performance-exploration-20261115-r1/tradeoff-tuning-mod-v0.1.2.zip) SHA256为`2ddfaa0280fd8be0b415b166ac41ff06950ecc32b7b6a6424f464297d489d9d6`，已回读验证；Engine私有原件需仓库授权，不把仅导入投影视为重新执行或私有raw逐项复验。

复现导出：恢复对应Engine实验目录后执行`scripts/export_optimization_exploration.py --engine-experiment-root /absolute/path/to/optimization-20261115 --output /absolute/path/to/new-benchmark-staging`。网站仍按已合并固定benchmark commit装配，网站仓库只提交展示代码。
""")
    print(
        json.dumps(
            dict(
                publication=str(target),
                files=len(payloads),
                physical_trials=24,
                admitted_trials=22,
                inference_requests=0,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.engine_experiment_root, args.output)
