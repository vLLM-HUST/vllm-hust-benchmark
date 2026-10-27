#!/usr/bin/env python3
"""Reproduce this compact package; writes only alongside this script.

Source payloads are pinned in package-inputs.json. Once copied, generation also
works from the package alone. Optional archive receipts remain distinct evidence;
no timing, model fit, source-evidence modification, commit or upload occurs.
"""

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = Path(
    "/root/vllm-hust-migration-20261023/storage-reference-recheck-20261024-r5"
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def pointer(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive-manifest",
        type=Path,
        default=Path(str(DEFAULT_ARCHIVE) + "-manifest.json"),
    )
    ap.add_argument(
        "--archive-readback",
        type=Path,
        default=Path(str(DEFAULT_ARCHIVE) + "-readback.json"),
    )
    public_base = DEFAULT_ARCHIVE.parent / "storage-reference-public-20261024-r5"
    ap.add_argument(
        "--public-manifest",
        type=Path,
        default=Path(str(public_base) + "-manifest.json"),
    )
    ap.add_argument(
        "--public-readback",
        type=Path,
        default=Path(str(public_base) + "-readback.json"),
    )
    ap.add_argument("--public-release-readback", type=Path)
    args = ap.parse_args()
    inputs = json.loads((HERE / "package-inputs.json").read_text())
    copied = []
    for record in inputs["files"]:
        assert Path(record["name"]).name == record["name"]
        target = HERE / record["name"]
        source = Path(record["source"])
        if source.exists():
            assert sha(source) == record["sha256"], str(source)
            payload = source.read_bytes()
        else:
            assert target.exists() and sha(target) == record["sha256"]
            payload = target.read_bytes()
        assert len(payload) == record["bytes"]
        target.write_bytes(payload)
        copied.append(dict(record, package_path=record["name"]))
    original = json.loads((HERE / "publication-summary-source.json").read_text())
    assert (
        sha(HERE / "publication-summary-source.json")
        == inputs["source_publication_summary_sha256"]
    )
    review = json.loads(
        (HERE / "storage-independent-final-reference-graph-review-r5.json").read_text()
    )
    external = json.loads((HERE / "profiling-final-dag-audit-r5.json").read_text())
    closure_sha = original["closure"]["sha256"]
    assert review["closure_sha256"] == external["closure_sha256"] == closure_sha
    assert (
        review["status"] == "complete_independent_graph_query_and_formula_review_passed"
    )
    assert external["status"] == "independent_final_dag_arithmetic_passed"
    assert len(original["scenarios"]) == 3 and all(
        s["all_duration_nodes_qualified"] for s in original["scenarios"]
    )
    assert original["attainment"] is False
    for s in original["scenarios"]:
        assert math.isclose(
            s["efficiency_percent"],
            100
            * s["reference_seconds"]
            / original["formal_result_unchanged"]["T_seconds"],
            rel_tol=1e-12,
        )
    archive = dict(
        schema="storage-reference-archive-index/v1",
        status="private_archive_awaiting_readback",
        visibility="private_storage_full_evidence",
        public_upload_allowed=False,
        privacy_scope="Contains full r6 worker/route payload and accounting evidence that may expose reconstructable token sequences or request/answer content. Do not publish this archive. A separately reviewed explicit allowlist public archive is required.",
        manifest=None,
        readback=None,
        published_download_url=None,
        url_note="This full archive is private evidence, not an upload candidate. Local paths and private-original hashes provide provenance only.",
        excluded_from_git=[
            "reference-recheck-query-bindings-r5-final.json",
            "reference-recheck-closure-r5-final.json",
            "reference-recheck-dag-r5-final-seed*.jsonl.gz",
            "raw timings, route captures and model archives",
        ],
    )
    if args.archive_manifest.exists():
        manifest = json.loads(args.archive_manifest.read_text())
        assert manifest["closure_sha256"] == closure_sha
        archive["manifest"] = dict(
            **pointer(args.archive_manifest),
            schema=manifest.get("schema"),
            data_members=manifest.get("data_members"),
            total_uncompressed_bytes=manifest.get("total_uncompressed_bytes"),
            member_count=len(manifest["members"]),
            coverage=manifest.get("coverage"),
        )
        wanted = {
            "reference-recheck-query-bindings-r5-final.json",
            "reference-recheck-closure-r5-final.json",
        } | {f"reference-recheck-dag-r5-final-seed{i}.jsonl.gz" for i in range(3)}
        archive["main_payloads"] = [
            m for m in manifest["members"] if Path(m["name"]).name in wanted
        ]
        assert len(archive["main_payloads"]) == 5
    if args.archive_readback.exists():
        receipt = json.loads(args.archive_readback.read_text())
        archive["readback"] = dict(**pointer(args.archive_readback), receipt=receipt)
        assert (
            archive["manifest"]
            and receipt["manifest_sha256"] == archive["manifest"]["sha256"]
        )
        assert (
            receipt["all_members_read_back"]
            and receipt["source_hashes_rechecked_after_packaging"]
        )
        assert receipt["data_members_verified"] == archive["manifest"]["data_members"]
        assert receipt["members_verified"] == receipt["data_members_verified"] + 1
        tarball = Path(receipt["archive"])
        if tarball.exists():
            assert (
                tarball.stat().st_size == receipt["archive_bytes"]
                and sha(tarball) == receipt["archive_sha256"]
            )
        archive["status"] = "private_storage_only_pending_public_allowlist_review"
        archive["verification_basis"] = (
            "Complete member readback from named producer receipt; archive SHA and size additionally checked when local tarball is accessible."
        )
    if (
        not args.archive_manifest.exists()
        and not args.archive_readback.exists()
        and (HERE / "archive-index.json").exists()
    ):
        prior = json.loads((HERE / "summary.json").read_text())
        assert sha(HERE / "archive-index.json") == prior["archive_index"]["sha256"]
        archive = json.loads((HERE / "archive-index.json").read_text())
    if args.public_manifest.exists() and args.public_readback.exists():
        pm = json.loads(args.public_manifest.read_text())
        pr = json.loads(args.public_readback.read_text())
        assert (
            pm["visibility"] == "public_allowlist_projection"
            and pm["closure_sha256"] == closure_sha
        )
        assert (
            pr["manifest_sha256"] == sha(args.public_manifest)
            and pr["all_members_read_back"]
        )
        assert (
            pr["private_full_archive_not_included"]
            and pr["public_recompute_status"]
            == "all_public_graphs_and_formulas_recomputed_without_private_payload"
        )
        assert (
            pr["data_members_verified"] == pm["member_count"]
            and pr["members_verified"] == pm["member_count"] + 1
        )
        public_tar = Path(pr["archive"])
        if public_tar.exists():
            assert (
                public_tar.stat().st_size == pr["archive_bytes"]
                and sha(public_tar) == pr["archive_sha256"]
            )
        archive["public_archive"] = dict(
            status="public_allowlist_readback_verified_pending_upload",
            visibility="public_allowlist_projection",
            manifest=dict(
                **pointer(args.public_manifest), member_count=pm["member_count"]
            ),
            readback=dict(**pointer(args.public_readback), receipt=pr),
            published_download_url=None,
            publication_note="Separate public package, with complete public-query/DAG recomputation; no upload has occurred. The private full archive remains forbidden from public upload.",
        )
    elif "public_archive" not in archive:
        archive["public_archive"] = dict(
            status="public_allowlist_package_pending", published_download_url=None
        )
    release_path = HERE / "public-release-readback.json"
    if args.public_release_readback:
        shutil.copyfile(args.public_release_readback, release_path)
    if release_path.exists():
        release = json.loads(release_path.read_text())
        public = archive["public_archive"]
        assert release["state"] == "uploaded"
        assert (
            release["remote_readback_sha256"]
            == public["readback"]["receipt"]["archive_sha256"]
        )
        assert (
            release["remote_readback_bytes"]
            == public["readback"]["receipt"]["archive_bytes"]
            == release["size"]
        )
        assert (
            release["browser_download_url"]
            == "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/storage-reference-public-20261024-r5.tar.gz"
        )
        public.update(
            status="uploaded_remote_bytes_verified",
            published_download_url=release["browser_download_url"],
            remote_readback=release,
            publication_note="Reviewed public archive uploaded; complete remote download SHA256 and byte count match. Private archive remains private.",
        )
    dump("archive-index.json", archive)
    summary = copy.deepcopy(original)
    summary["status"] = "reference_closed_independently_reviewed"
    summary["independent_final_graph_audits"] = [
        dict(
            path="profiling-final-dag-audit-r5.json",
            sha256=sha(HERE / "profiling-final-dag-audit-r5.json"),
            reviewer="profiling",
            scope=external["scope"],
        ),
        dict(
            path="storage-independent-final-reference-graph-review-r5.json",
            sha256=sha(
                HERE / "storage-independent-final-reference-graph-review-r5.json"
            ),
            reviewer="storage",
            scope=review["qualification"],
        ),
    ]
    summary["package_provenance"] = dict(
        source_summary="publication-summary-source.json",
        source_path=next(
            x["source"]
            for x in inputs["files"]
            if x["name"] == "publication-summary-source.json"
        ),
        source_sha256=inputs["source_publication_summary_sha256"],
        derived_changes=[
            dict(field="status", before=original["status"], after=summary["status"]),
            dict(
                field="independent_final_graph_audits",
                change="Added actual profiling and storage completed independent reviews with SHA256.",
            ),
            dict(
                field="package_provenance",
                change="Added source identity and exact derivation record.",
            ),
            dict(
                field="archive_index",
                change="Added local archive/readback index; no invented public URL.",
            ),
            dict(
                field="B0_followup",
                change="Discloses separately authorized B0 completion work; not completed by this B1 package.",
            ),
        ],
        unchanged="All existing numerical results, scenarios, F/T identity, thresholds, measured failures and limitations remain exactly as in source summary.",
    )
    summary["archive_index"] = dict(
        path="archive-index.json", sha256=sha(HERE / "archive-index.json")
    )
    summary["B0_followup"] = dict(
        status="separately_authorized_pending_completion",
        identity="Official vLLM 0.18 + vLLM-Ascend 0.18 B0",
        scope="Separate newly authorized B0 work. This package closes B1 reference accounting only and does not claim B0 completion.",
    )
    for key, value in original.items():
        if key != "status":
            assert summary[key] == value, key
    dump("summary.json", summary)
    dump(
        "evidence-index.json",
        dict(
            schema="storage-reference-compact-evidence-index/v1",
            files=copied,
            large_evidence="See archive-index.json and the hash-linked full archive manifest; no large closure/query/raw payload is embedded in this Git package.",
            review_identity="profiling and storage are distinct actual reviewers; copied reviews are byte-identical to named sources.",
        ),
    )
    rows = "\n".join(
        f"| {s['seed']} | {s['reference_seconds']:.8f} | {s['reference_tflops_per_chip']:.8f} | {s['efficiency_percent']:.8f}% | 未达标 |"
        for s in summary["scenarios"]
    )
    archive_text = "私有完整归档回读尚待完成；不得直接公开上传。"
    if archive.get("readback"):
        receipt = archive["readback"]["receipt"]
        archive_text = f"当前私有完整归档已完成 {receipt['members_verified']} 个成员的 SHA 回读，归档大小 {receipt['archive_bytes']:,} 字节，SHA256 `{receipt['archive_sha256']}`。它包含完整 r6 worker/route payload 和 accounting 原件，可能含可还原 token 序列及请求/答案内容，仅限私有存储，不得直接公开上传。"
    public_text = "另行制作的公开白名单包尚待完成。"
    if archive.get("public_archive", {}).get("readback"):
        pr = archive["public_archive"]["readback"]["receipt"]
        public_text = f"独立公开白名单包已完成 {pr['members_verified']} 个成员的 SHA 回读，大小 {pr['archive_bytes']:,} 字节，SHA256 `{pr['archive_sha256']}`；仅凭公开查询与 DAG 即可复算三个结果。该公开包尚未上传，下载 URL 为空。"
    readme = f"""# B1 参考算力闭合证据（2026-10-24）

B1 的有效算力、参考算力及效率已形成可验收数值，并通过 storage 与 profiling 各自独立的完整依赖图复算。冻结主场景 seed0：有效算力 **7.10422076 TFLOPS/卡**，参考算力 **19.08186676 TFLOPS/卡**，参考时间 **47.50334896 秒**，效率 **37.23021890%**。三个场景均未达到原定 **50%** 门槛；数值闭合与门槛达标分别报告。

| seed | T参（秒） | 参考算力（TFLOPS/卡） | η | 50% 门槛 |
| --- | ---: | ---: | ---: | --- |
{rows}

F = 1,812,905,151,088,034，原 B1 T = 127.59352582134306 秒，参与芯片数为 2。计算式为 `C=F/(2*T*1e12)`、`C参=F/(2*T参*1e12)`、`η=100*T参/T`。seed0 在完整结果产生前已固定为展示主值；三个 seed 全部保留，范围只表示这三个调度场景的敏感性，不是置信区间。

每个场景包含 359,368 个必要节点，均已获得合格时延。storage 独立审计核对了 5,394,622 次组件引用、942,210 个实际使用查询、依赖与资源顺序、TP2 集合通信，以及三个最长路径。profiling 独立审计另行复算全部 1,078,104 个节点。完整图没有用零值替代未知时延。

## 方法与适用范围

参考时间采用独立组件计量与源码依赖图聚合，保留原 B1 的 F、T、必要工作量和三个冻结工作组。target 使用明确的 r6 canonical 参考轨迹；draft 使用固定同请求、同 draft step 的统计路由参考，并经过距离支持、实际组大小及预测占比的验证。它们不声称恢复了 B1 实际运行的逐位专家 ID、hidden 或 KV 状态。

组件模型的验证门槛和支持域是经验条件，不构成整模型误差上界。CPU 新测采用预触碰普通内存的内在操作计量，与旧 pinned/DMA 事务上下文存在差别；两 rank 锚点新/旧中位比约 0.741、0.687，仅披露而未校正。原失败计量完整保留；仅 rank0 的 4 行 concat 点按事前冻结的更长 graph 批次修复精度，其余 11 个域模型不变。

本包身份是实际 B1 开发运行栈。官方 vLLM 0.18 + vLLM-Ascend 0.18 的 B0 另有新授权，仍待补齐；本包不表示 B0 已完成，也不表示报告、网站发布已经完成。

## 文件与追溯

- [summary.json](summary.json)：保持原摘要字段，增加真实独立复核及归档索引。
- [publication-summary-source.json](publication-summary-source.json)：原摘要逐字副本，SHA256 `{inputs["source_publication_summary_sha256"]}`。状态与新增字段的逐项变化记录在 `summary.json.package_provenance`；原有数字及限制保持不变。
- [storage 最终图审计](storage-independent-final-reference-graph-review-r5.json) 与 [profiling 最终图审计](profiling-final-dag-audit-r5.json)：两个独立审查身份及全部场景结果。
- [evidence-index.json](evidence-index.json)：本目录方法和审查原件的原始路径、大小及 SHA256。
- [archive-index.json](archive-index.json)：大体积查询、closure、DAG、原始计量的归档清单和真实回读回执索引。{archive_text}原始绝对路径用于来源追溯，不是公开下载链接。
- 公开与私有双索引：{public_text}专家路由 ID 属计算几何，不等于词表 token ID；公开包保留必要几何和时延，私有原始输入通过 SHA 追溯。
- [SHA256SUMS](SHA256SUMS)：本目录除清单自身之外的所有文件校验值。

## 重建

运行 `python package.py`，从 `package-inputs.json` 固定的来源复制紧凑证据；原路径不可访问时使用本目录中 SHA 一致的副本。生成器仅写本目录，不运行计量、不拟合、不上传、不提交。

归档完成后可运行 `python package.py --archive-manifest /实际路径/manifest.json --archive-readback /实际路径/readback.json`，原样记录私有完整包回执并刷新摘要索引及校验清单。该回执不授予公开权限；公开包须另外按明确的文件和字段白名单制作并审查，不得把本索引替换成私有包的下载链接。`sha256sum -c SHA256SUMS` 可验证本目录。

公开包完成后使用 `--public-manifest /实际路径/public-manifest.json --public-readback /实际路径/public-readback.json` 更新单独的公开索引。生成器核验公开包来源身份、成员回读与公开 DAG 复算结果，再更新索引；不会把私有完整包标记为公开包。

约 1.7 GB 的查询文件、约 13 MB 的 closure 及完整原始数据留在归档中，不复制进入 Git。
"""
    (HERE / "README.md").write_text(readme)
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (HERE / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
    print(
        json.dumps(
            dict(
                status=summary["status"],
                files=len(files) + 1,
                bytes=sum(p.stat().st_size for p in HERE.iterdir() if p.is_file()),
                archive_status=archive["status"],
            )
        )
    )


if __name__ == "__main__":
    main()
