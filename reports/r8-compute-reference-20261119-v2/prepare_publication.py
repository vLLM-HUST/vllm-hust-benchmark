"""Build the website projection from the immutable public r8 certificate."""

from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import json
import zipfile
import shutil
import re

SOURCE_MANIFEST_SHA256 = (  # pragma: allowlist secret -- immutable public evidence identity
    "32f6ac06dd4a6be03e26591c2e2e1aca437a01dbace9a1757e74e1e1a15fe5f7"  # pragma: allowlist secret -- public manifest hash
)
SOURCE_COMMIT = "72a677b591f9c5c0d911e9a280759278bc0f4a74"  # pragma: allowlist secret -- immutable public evidence identity


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def near(actual, expected):
    if abs(Decimal(str(actual)) - expected) > abs(expected) * Decimal("1e-36"):
        raise ValueError("Certified arithmetic differs")


def verify_source(source):
    if sha(source / "publication-checksums.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Public source inventory differs")
    inventory = read(source / "publication-checksums.json")
    for name, digest in inventory.items():
        if sha(source / name) != digest:
            raise ValueError("Original source bytes differ: " + name)
    pair = read(source / "compute-reference/matched-reference-pair.json")
    stats = read(source / "compute-reference/qualified-full-reference-statistics.json")
    inputs = read(source / "recomputation-inputs.json")
    summary = read(source / "summary.json")["compute_reference"]
    graph = read(source / "compute-reference/reference-graph-audits.json")
    if (
        pair["certificate_kind"] != "qualified_full_reference_denominator_reuse/v1"
        or pair["status"] != "measured"
        or any(
            pair[k] is not True
            for k in [
                "full_model_certificate",
                "clean_native_clock_source_qualified",
                "full_reference_denominator_qualified",
                "original_reference_work_and_time_retained",
            ]
        )
        or pair["shared_work_cancellation_used"] is not False
        or pair["original_efficiency_used"] is not False
    ):
        raise ValueError("Unqualified source certificate")
    if summary["binding"]["engine_release"] != "full-answer-b1-20261117-r8" or summary[
        "capture_profile"
    ] != dict(
        id="submission-v2",
        max_tokens=256,
        formal_repetitions=3,
        ignore_eos=False,
        requests_per_repetition=200,
    ):
        raise ValueError("Wrong engine/capture protocol")
    if (
        pair["qualified_full_reference_statistics_sha256"]
        != sha(source / "compute-reference/qualified-full-reference-statistics.json")
        or graph["source_and_workload_applicability_qualified"] is not True
        or graph["original_graph_recomputed"] is not True
    ):
        raise ValueError("Reference provenance differs")
    scope = stats["scope"]
    if (
        (
            scope["requests"],
            scope["input_tokens"],
            scope["output_tokens"],
            scope["max_completion_tokens"],
            scope["dtype"],
            scope["tp"],
            scope["accelerator"],
            scope["frontend"],
            scope["timing"],
        )
        != (
            200,
            282265,
            51200,
            256,
            "BF16",
            2,
            "Ascend 910B2",
            "native_inproc",
            "dual_rank_synchronized_after_warmup",
        )
        or stats["original_efficiency_used"] is not False
        or stats["target_times_used"] is not False
    ):
        raise ValueError("Foreign reference domain")
    if (
        sha(source / "current-work-and-timing.zip") != inputs["zip_sha256"]
        or sha(source / "compute-reference/matched-reference-pair.json")
        != inputs["certificate_source_sha256"]
    ):
        raise ValueError("Clock/work archive binding differs")
    with (
        zipfile.ZipFile(source / "current-work-and-timing.zip") as archive,
        localcontext() as ctx,
    ):
        ctx.prec = 60
        if (
            set(archive.namelist()) != set(inputs["members"]) | {"SHA256.json"}
            or len(archive.namelist()) != len(set(archive.namelist()))
            or json.loads(archive.read("SHA256.json")) != inputs["members"]
        ):
            raise ValueError("Raw inventory differs")
        for name, digest in inputs["members"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Raw member differs")
        for role, arm in pair["arms"].items():
            rows = arm["repetitions"]
            ref = stats["arms"][role]
            refrows = ref["reference_repetitions"]
            if (
                len(rows) != 3
                or len(refrows) != 3
                or len({x["timing_sha256"] for x in rows}) != 3
            ):
                raise ValueError("Three distinct clocks required")
            total = [0, 0]
            elapsed = Decimal(0)
            for i, row in enumerate(rows, 1):
                prefix = role if i == 1 else f"{role}-r{i:03d}"
                work = json.loads(archive.read(prefix + "-necessary-work.json"))
                timing = json.loads(archive.read(prefix + "-timing.json"))
                if (
                    row["repetition"] != i
                    or timing["repetition"] != i
                    or timing["output_tokens"] != 51200
                    or work["requests"] != 200
                    or work["F_interval"] != row["F_interval"]
                    or inputs["members"][prefix + "-necessary-work.json"]
                    != row["work_sha256"]
                    or inputs["members"][prefix + "-timing.json"]
                    != row["timing_sha256"]
                ):
                    raise ValueError("Current repetition population differs")
                time = Decimal(timing["end_ns"] - timing["start_ns"]) / 10**9
                if time != Decimal(row["T_seconds"]):
                    raise ValueError("Raw clock arithmetic differs")
                for j, endpoint in enumerate(("lower", "upper")):
                    value = work["endpoints"][endpoint]
                    if (
                        sum(value["components"].values()) != value["total_basic"]
                        or value["total_basic"] != row["F_interval"][j]
                    ):
                        raise ValueError("Necessary-work component sum differs")
                    total[j] += row["F_interval"][j]
                elapsed += time
            if total != arm["F_interval"] or elapsed != Decimal(arm["T_seconds"]):
                raise ValueError("Current aggregate differs")
            refF = [sum(x["F_interval"][j] for x in refrows) for j in range(2)]
            refT = sum((Decimal(x["Tref_lower_seconds"]) for x in refrows), Decimal(0))
            if (
                refF != ref["effective_flops_interval"]
                or refF != arm["reference_F_interval"]
                or refT != Decimal(ref["sum_reference_seconds_lower"])
                or refT != Decimal(arm["reference_seconds_lower"])
                or refrows != arm["original_reference_repetitions"]
            ):
                raise ValueError("Reference denominator originals differ")
            upper = Decimal(refF[1]) / (2 * refT * 10**12)
            lower = Decimal(total[0]) / (2 * elapsed * 10**12)
            near(ref["reference_tflops_per_card_upper"], upper)
            near(arm["reference_tflops_per_card_upper"], upper)
            near(arm["achieved_tflops_per_card_interval"][0], lower)
            near(
                arm["achieved_tflops_per_card_interval"][1],
                Decimal(total[1]) / (2 * elapsed * 10**12),
            )
            near(arm["efficiency_lower_percent"], 100 * lower / upper)
            if arm["shared_work_cancellation_used"] is not False:
                raise ValueError("Different work must not cancel")
        eta = Decimal(pair["arms"]["B1"]["efficiency_lower_percent"])
        if pair["milestone_met"] != (eta >= 50) or pair["final_numeric_met"] != (
            eta >= 90
        ):
            raise ValueError("Unrounded threshold differs")
    return pair, stats, summary, graph


def build(benchmark, source, raw_asset_path, receipt_path=None):
    SOURCE = source
    pair, stats, summary, graph = verify_source(SOURCE)
    raw_asset = read(raw_asset_path)
    if (
        raw_asset["sha256"] != sha(SOURCE / "current-work-and-timing.zip")
        or raw_asset["bytes"] != (SOURCE / "current-work-and-timing.zip").stat().st_size
        or raw_asset["redownload_verified"] is not True
    ):
        raise ValueError("Raw asset differs from current clocks")
    prefix = Path("data/r8-compute-reference-20261119-v2")
    data = benchmark / "website-publication" / prefix
    report = benchmark / "reports/r8-compute-reference-20261119-v2"
    report.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    for name in read(SOURCE / "publication-checksums.json"):
        # The public portal uses immutable original charts/report and evidence JSON.
        if name in ["README.md", "recompute.py", "checksums.json"]:
            continue
        p = data / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, p)

    def asset(name):
        p = data / name
        return dict(path=str(prefix / name), sha256=sha(p), bytes=p.stat().st_size)

    a = dict(
        schema="engine-current-compute-reference/v1",
        engine_release=summary["binding"]["engine_release"],
        baseline_id="original018-cann8.5.1",
        status="measured",
        full_model_certificate=True,
        accepted_scope="topic4_submission_v2_fixed200_max256_three_formal_runs",
        certificate_kind=pair["certificate_kind"],
        efficiency_exact_percent=None,
        hardware_peak_utilization_percent=None,
        model="Qwen3.5-35B-A3B",
        precision="BF16",
        tensor_parallel_size=2,
        chip_model="Ascend 910B2",
        protocol=dict(
            requests=200,
            max_completion_tokens=256,
            formal_repetitions=3,
            temperature=0,
            seed=0,
            eos="enabled",
            quality_plan_sha256=summary["binding"]["quality_plan_sha256"],
        ),
        source_status_sha256=sha(SOURCE / "summary.json"),
        source_pair_sha256=sha(
            SOURCE / "compute-reference/matched-reference-pair.json"
        ),
        source_repository="DeepLinkNext/DeepLinkEngine",
        source_revision=SOURCE_COMMIT,
        source_manifest_sha256=sha(SOURCE / "publication-checksums.json"),
        raw_asset=raw_asset,
        arms={},
        milestone_met=pair["milestone_met"],
        final_numeric_met=pair["final_numeric_met"],
        chart=asset(Path("current-reference-bounds.svg")),
        report=asset(Path("report.html")),
        certificate=asset(Path("compute-reference/matched-reference-pair.json")),
        reference_statistics=asset(
            Path("compute-reference/qualified-full-reference-statistics.json")
        ),
        recomputation_url=f"https://github.com/DeepLinkNext/DeepLinkEngine/blob/{SOURCE_COMMIT}/benchmark/reports/r8-compute-reference-20261119-v1/README.md",
        original_reference_work_and_time_retained=True,
        original_efficiency_used=False,
        shared_work_cancellation_used=False,
        source_and_workload_applicability_qualified=True,
        clean_native_clock_source_qualified=True,
        full_reference_denominator_qualified=True,
    )
    for role, arm in pair["arms"].items():
        a["arms"][role] = {
            **arm,
            "F_interval": list(map(str, arm["F_interval"])),
            "reference_F_interval": list(map(str, arm["reference_F_interval"])),
            "native_checksums_sha256": graph["arms"][role]["native_checksums_sha256"],
            "necessary_work_sha256": sha(
                SOURCE / "compute-reference/necessary-work-audit.json"
            ),
            "reference_graph_sha256": sha(
                SOURCE / "compute-reference/reference-graph-audits.json"
            ),
        }
    p = benchmark / "website-publication/data/engine-release-history-v1.json"
    c = read(p)
    entry = next(x for x in c["entries"] if x["id"] == "r8")
    assert "compute_reference" not in entry
    if (
        entry["release"] != a["engine_release"]
        or entry["baseline_id"] != a["baseline_id"]
    ):
        raise ValueError("Catalog runtime differs")
    entry["compute_reference"] = a
    p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n")
    policy = benchmark / "scripts/build_public_site.py"
    s = policy.read_text()
    s, n = re.subn(
        r'^(PUBLIC_ENGINE_HISTORY_SHA256 = ")[a-f0-9]{64}(".*)$',
        lambda m: m[1] + sha(p) + m[2],
        s,
        flags=re.M,
    )
    assert n == 1
    policy.write_text(s)
    (data / "summary-public-v1.json").write_text(
        json.dumps(a, ensure_ascii=False, indent=2) + "\n"
    )
    (report / "README.md").write_text("""# r8 有效算力参考发布订正（2026-11-19）

此前完整结果已归档到 DeepLinkEngine，benchmark Release 目录没有接入，网站展示代码仍在草稿 PR714。本次补接已有封存认证，不新增推理、不更改冻结引擎与原始测量。

B0效率下界22.2269749464%，B1效率下界89.7838253325%。B1的50%里程碑达成，90%最终目标未达成。MMLU200/out256/EOS，每臂三轮，BF16/TP2/双910B2；完整回答五项成绩、上下文与四租户数据保持原协议和身份。

每臂所有轮次先求和，再计算：

- C_current_lower = ΣF_current_lower / (2 × ΣT_current × 10^12)
- C_ref_upper = ΣF_ref_upper / (2 × ΣT_ref_lower × 10^12)
- η_lower = 100 × C_current_lower / C_ref_upper

参考时间下界由独立组件标定和保留资源依赖图最长路径求得：finish(v)=duration(v)+max(finish(parent))。本轮与原参考算量不同，保留各自F/T，不相消、不复用旧效率百分比。B0原参考上界26.8182576764809057、B1原参考上界19.3403833468717505 TFLOP/s/卡。

[配对认证及全部六轮来源](../../website-publication/data/r8-compute-reference-20261119-v2/compute-reference/matched-reference-pair.json) · [完整参考分母统计](../../website-publication/data/r8-compute-reference-20261119-v2/compute-reference/qualified-full-reference-statistics.json) · [网页报告](../../website-publication/data/r8-compute-reference-20261119-v2/report.html) · [公开原件与复算入口](https://github.com/DeepLinkNext/DeepLinkEngine/blob/72a677b591f9c5c0d911e9a280759278bc0f4a74/benchmark/reports/r8-compute-reference-20261119-v1/README.md)。

数据生成与CPU核验使用本目录的prepare_publication.py；其中固定公开来源文件清单并逐项核对12份工作/时钟原件、全部三轮和参考统计。原证据保持原字节，订正仅补充目录和发布投影。
""")
    shutil.copyfile(Path(__file__), report / "prepare_publication.py")
    guide = benchmark / "website-publication/ENGINE_RELEASE_HISTORY.md"
    guide.write_text(
        guide.read_text()
        + "\n## r8 有效算力参考（2026-11-19）\n\nr8.compute_reference 接入已封存的固定MMLU200/out256三轮完整参考分母复用认证。B1下界89.7838%，50%里程碑达成、90%最终目标未达成；不消去不同F。[统计、原件与复算](../reports/r8-compute-reference-20261119-v2/README.md)。\n"
    )
    if receipt_path:
        receipt_path.write_text(json.dumps(a, ensure_ascii=False, indent=2) + "\n")
    print("Verified and projected", a["arms"]["B1"]["efficiency_lower_percent"], sha(p))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-asset", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    build(args.benchmark, args.source, args.raw_asset, args.receipt)
