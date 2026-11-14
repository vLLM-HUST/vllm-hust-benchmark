"""Create a public website projection while retaining immutable research sources.

Run only on an assembled, validated website checkout. The output is a new
directory; this tool never deletes or rewrites the input research evidence.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import zipfile
from urllib.parse import unquote, urlsplit
from pathlib import Path

PROJECT = re.compile(
    r"betterscale|traceloom|gennova|latchmoe|tricard|"
    r"prefix-cache-routing-reliability|prefix[ _-]routing|pr173-dev4|"
    r"ascend-adaptive-quantized|adaptive[ _-]+quantized[ _-]+kv|"
    r"vllm-hust-perf-analyzer|vllm-hust-cuda-(?:qk-rope|hybrid-apc|async-tp|fullgraph-matrix)",
    re.I,
)
IDENTITY_KEYS = {
    "id",
    "name",
    "name_en",
    "name_zh",
    "label",
    "title",
    "title_en",
    "title_zh",
    "engine",
    "engine_name",
    "solution",
    "component_id",
    "plugin_id",
    "repository",
    "repository_url",
    "component_ids",
    "mods",
    "mod",
    "series_id",
    "series_label",
    "implementation",
    "runtime_name",
    "worker_class",
    "worker_cls",
    "family",
}
DROP = object()

# The owner explicitly restored this historical C16 comparison on 2026-10-31.
C16_EVIDENCE_PATH = "data/leaderboard_frontier_swe_evidence.json"
PUBLIC_C16_RUN_IDS = frozenset(
    {
        "dab8ed7c4ee74f2d958dab5528514294",  # pragma: allowlist secret - public B0 run ID
        "fa0e01a32eef456bab4bebe51e4aa7dd",  # pragma: allowlist secret - public B1 run ID
    }
)

# The MOD card may show this historical service comparison. The separate
# project page, raw artifacts, and all other BetterScale records stay filtered.
BETTERSCALE_ID = "betterscale"
BETTERSCALE_CARD = {
    "id": BETTERSCALE_ID,
    "name": "BetterScale",
    "name_en": "BetterScale",
    "name_zh": "BetterScale",
    "artifact_type": "runtime_component",
    "system_role": "graph_execution_policy",
    "integration_contracts": [],
    "execution_planes": ["worker", "native", "device"],
    "deployment_topology": "in_process",
    "delivery_model": "python_distribution",
    "ownership": "hust_owned_subsystem",
    "repository_relationship": "organization_native",
    "repository_visibility": "private",
    "maturity": "experimental",
    "maintainers": ["CubeLander"],
    "maintainer_profiles": [{"login": "CubeLander", "name": "田景远"}],
    "summary_en": "Multi-device inference execution optimization for Ascend.",
    "summary_zh": "面向昇腾的多卡推理执行优化。",
    "public_effect_status": "measured",
    "public_effect_en": "Historical DeepSeek-V4-Flash HTTP test: output throughput rose 39.63% at DP8/C16 and 35.17% at TP8/C4 against their respective baselines.",
    "public_effect_zh": "DeepSeek-V4-Flash 历史 HTTP 实测中，DP8／并发 16 和 TP8／并发 4 的输出吞吐分别较各自基线提高 39.63% 和 35.17%。",
    "public_effect_detail_en": [
        "2026-09-14 · DeepSeek-V4-Flash-0731 W8A8 on 8 × Ascend 910B2 (HCCS), vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1, K5.",
        "DP8, concurrency 16, 8 GiB KV/rank: 268.89 → 375.45 output tok/s (+39.63%), runs 158 → 155. The candidate included the DP startup preparation carried into 0.3.1.",
        "TP8, concurrency 4, 12 GiB KV/rank: 155.13 → 209.68 output tok/s (+35.17%), runs 160 → 161. The baseline included the LCM startup compatibility fix.",
        "Each comparison divides total output tokens by total timed duration across all nine synthetic HTTP rounds (three workloads, three repeats). Startup, the separately logged prelude and quality scoring were excluded. Each arm passed 32/32 retrieval questions. TP decode P99 chunk-gap worsened. These measurements do not establish maximum concurrency, SLO capacity, or performance of a newer wheel.",
    ],
    "public_effect_detail_zh": [
        "2026-09-14 · DeepSeek-V4-Flash-0731 W8A8，8 张昇腾 910B2（HCCS），vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1，K5。",
        "DP8，并发 16，每 rank 8 GiB KV。原生基线 268.89 → 优化后 375.45 输出 token/s（+39.63%，run 158 → 155）。候选包含后续收录于 0.3.1 的 DP 启动预热。",
        "TP8，并发 4，每 rank 12 GiB KV。含 LCM 启动兼容修复的基线 155.13 → 优化后 209.68 输出 token/s（+35.17%，run 160 → 161）。",
        "各自按九组固定合成 HTTP 负载（三种负载，各三次）的输出 token 总数除以计时总长。启动、单独记录的预备阶段和质量评分不计时。各组检索 32/32，TP 解码 P99 输出块间隔变差。结果仅适用于上述实测工况，不能用于推断最大并发、SLO 容量或新版软件包性能。",
    ],
    "evidence_level": "hardware_verified",
}
BETTERSCALE_CATALOG_ENTRIES = {
    "data/mod-taxonomy.json": {
        "kind": "runtime_mod",
        "capability": "execution_optimization",
        "phases": ["prefill", "decode"],
        "lifecycle": "implemented",
        "evidence": "measured_beneficial",
    },
    "data/plugin-workload-navigation.json": ["distributed_pipeline"],
}


def add_betterscale_card(path, projected, original):
    if path == "data/ecosystem.json":
        if any(item["id"] == BETTERSCALE_ID for item in original["components"]):
            projected["components"].append(copy.deepcopy(BETTERSCALE_CARD))
    elif path in BETTERSCALE_CATALOG_ENTRIES:
        key = (
            "plugins"
            if path == "data/plugin-workload-navigation.json"
            else "components"
        )
        if BETTERSCALE_ID in original[key]:
            projected[key][BETTERSCALE_ID] = copy.deepcopy(
                BETTERSCALE_CATALOG_ENTRIES[path]
            )


def validate_betterscale_card(path, data):
    if path == "data/ecosystem.json":
        cards = [item for item in data["components"] if item["id"] == BETTERSCALE_ID]
        assert not cards or cards == [BETTERSCALE_CARD], path
        if cards:
            data["components"].remove(cards[0])
    elif path in BETTERSCALE_CATALOG_ENTRIES:
        key = (
            "plugins"
            if path == "data/plugin-workload-navigation.json"
            else "components"
        )
        if BETTERSCALE_ID in data[key]:
            assert data[key].pop(BETTERSCALE_ID) == BETTERSCALE_CATALOG_ENTRIES[path], (
                path
            )


def matches(value):
    return bool(PROJECT.search(json.dumps(value, ensure_ascii=False)))


def target_record(value):
    if not isinstance(value, dict):
        return False
    # Withhold the whole measured comparison if its evidence belongs to a
    # withdrawn work. Removing just the evidence URL corrupts its provenance.
    if ("metrics" in value or "configuration" in value) and matches(value):
        return True
    if any(matches(v) for k, v in value.items() if k in IDENTITY_KEYS):
        return True
    for key in ("configuration", "runtime", "provenance", "source_provenance"):
        nested = value.get(key)
        if isinstance(nested, dict) and target_record(nested):
            return True
    return False


def project_json(value, removed_ids, record=False, retained_runs=()):
    if isinstance(value, dict):
        run_id = value.get("run_id")
        if isinstance(run_id, str) and run_id in retained_runs:
            return copy.deepcopy(value)
        if record and (
            target_record(value)
            or (isinstance(value.get("id"), str) and value["id"] in removed_ids)
        ):
            if isinstance(value.get("id"), str):
                removed_ids.add(value["id"])
            return DROP
        result = {}
        for key, item in value.items():
            if PROJECT.search(key) or key in removed_ids:
                continue
            child = project_json(
                item,
                removed_ids,
                record=isinstance(item, dict),
                retained_runs=retained_runs,
            )
            if child is not DROP:
                result[key] = child
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            child = project_json(
                item, removed_ids, record=True, retained_runs=retained_runs
            )
            if child is not DROP:
                result.append(child)
        return result
    if isinstance(value, str) and value in removed_ids:
        return DROP
    if isinstance(value, str) and PROJECT.search(value):
        if value.startswith(("https://", "http://", "./", "data/", "docs/")):
            return DROP
        sentences = re.split(r"(?<=[.!?。！？;；])\s*", value)
        retained = " ".join(
            sentence for sentence in sentences if not PROJECT.search(sentence)
        ).strip()
        return retained if retained else DROP
    return value


def prune_frontier(data, removed_ids):
    """Keep declared plot selections consistent with the remaining evidence."""
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != "leaderboard-frontier/v1"
    ):
        return data
    for _ in range(len(data["cohorts"]) + 1):
        previous = len(data["cohorts"])
        point_ids = {p["id"] for p in data["points"]}
        cohorts = []
        for cohort in data["cohorts"]:
            points = [p for p in data["points"] if p["cohort_id"] == cohort["id"]]
            contract = cohort["workload"]["contract"]
            retain = bool(points)
            if "display_series_ids" in contract:
                series = {p.get("load", {}).get("concurrency_series") for p in points}
                contract["display_series_ids"] = [
                    s for s in contract["display_series_ids"] if s in series
                ]
                retain = retain and bool(contract["display_series_ids"])
            if "comparison_point_ids" in contract:
                contract["comparison_point_ids"] = [
                    p for p in contract["comparison_point_ids"] if p in point_ids
                ]
                retain = retain and bool(contract["comparison_point_ids"])
            if contract.get("default_groups") == []:
                contract.pop("default_groups")
            if retain:
                cohorts.append(cohort)
            else:
                removed_ids.add(cohort["id"])
        data["cohorts"] = cohorts
        cohort_ids = {c["id"] for c in cohorts}
        for point in data["points"]:
            if point["cohort_id"] not in cohort_ids:
                removed_ids.add(point["id"])
        data["points"] = [p for p in data["points"] if p["cohort_id"] in cohort_ids]
        if len(cohorts) == previous:
            break
    return data


def clean_html(text):
    # Remove links/images whose destination belongs to a withdrawn project.
    text = re.sub(
        r"<a\b[^>]*>.*?</a>", lambda m: "" if matches(m[0]) else m[0], text, flags=re.S
    )
    text = re.sub(r"<img\b[^>]*>", lambda m: "" if matches(m[0]) else m[0], text)
    # This shared card also describes PegaFlow; preserve that unrelated content.
    text = re.sub(r"LatchMoE is not applicable.*?(?=PegaFlow is)", "", text, flags=re.S)
    text = re.sub(r"LatchMoE 不适用于.*?(?=PegaFlow 是)", "", text, flags=re.S)
    text = re.sub(r"'atlas-latchmoe-role': '[^']*',\s*", "", text)
    # Remove obsolete project-specific metadata from the shared route table.
    return text


def clean_javascript(path, text):
    if path == "assets/plugins-page.js":
        for start, end in [
            ("    traceloom: {", '    "clm-lifecycle": {'),
            ('    "tricard-clm-lifecycle": {', "    betterscale: {"),
            ("    betterscale: {", "    bidkv: {"),
            ("    latchmoe: {", '    "pegaflow-vllm-connectors": {'),
            ('    "ascend-adaptive-quantized-kv": {', '    "kvcompress-ascend": {'),
        ]:
            if start in text:
                a = text.index(start)
                b = text.index(end, a)
                text = text[:a] + text[b:]
    if path == "assets/dataset-matrix-v1.js":
        text = text.replace(
            "(r.solution === 'betterscale' ? '2026-10-15' : r.dataset === 'mmlu-pro' ? '2026-10-12' : r.dataset === 'frontierscience' ? '2026-10-13' : '2026-10-14/15')",
            "(r.dataset === 'mmlu-pro' ? '2026-10-12' : r.dataset === 'frontierscience' ? '2026-10-13' : '2026-10-14/15')",
        )
        text = text.replace(
            "BetterScale 的未决判分分别为 20 / 27 条，保留在总分母中。", ""
        )
        text = text.replace(
            "BetterScale has 20 / 27 unresolved judges, retained in the overall denominators.",
            "",
        )
        text = text.replace(
            " · <a href=\"${data.betterscale_publication.url}\">${t('BetterScale 公开实测报告', 'BetterScale public report')}</a>",
            "",
        )
    if path == "assets/leaderboard-frontier.js":
        text = text.replace(
            "Only Pareto-frontier points are shown, connected separately for BetterScale and Native. BetterScale spans workload-tuned configurations, not a fixed-capacity sweep.",
            "Only currently published Pareto-frontier points are shown; each configuration retains its measurement scope.",
        )
        text = text.replace(
            "BetterScale lines connect best trade-offs across workload-tuned configurations; other lines follow measured concurrency series",
            "Lines follow the published measured concurrency series",
        )
        text = text.replace(
            "BetterScale 连线连接不同 workload 调优配置的最佳权衡点；其他连线按实测并发序列连接",
            "连线按当前公开的实测并发序列连接",
        )
        start = "            ${point.configuration.mods.includes('betterscale')?"
        end = "            <p class=\"frontier-popup-load\">${t('warmup')}"
        if start in text:
            a = text.index(start)
            text = text[:a] + text[text.index(end, a) :]
        text = re.sub(
            r"        // BetterScale checkbox.*?(?=        const mods=)",
            "",
            text,
            flags=re.S,
        )
        text = text.replace(
            "仅展示并分别连接 BetterScale 与 Native 的 Pareto 前沿点；BetterScale 跨 workload 调优配置，不代表固定容量扫描。",
            "展示当前公开方案的 Pareto 前沿点；不同配置不代表固定容量扫描。",
        )
        text = text.replace(
            "Only show and connect BetterScale and Native Pareto frontier points; BetterScale is tuned across workloads, not a fixed-capacity sweep.",
            "Show the Pareto frontier of currently published configurations; different settings are not a fixed-capacity sweep.",
        )
    if path == "assets/site.js":
        start, end = "        'betterscale.html': {", "        'index.html': {"
        if start in text:
            a = text.index(start)
            text = text[:a] + text[text.index(end, a) :]
    if path == "assets/midterm-report.js":
        text = (
            "\n".join(
                line
                for line in text.splitlines()
                if "website-publication/data/betterscale-" not in line
            )
            + "\n"
        )
    return text


def archive_matches(path):
    if path.suffix.lower() not in (".pptx", ".docx", ".xlsx", ".zip"):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if PROJECT.search(member.filename):
                    return True
                if (
                    member.filename.endswith((".xml", ".json", ".md", ".txt"))
                    and member.file_size < 5_000_000
                ):
                    if PROJECT.search(
                        archive.read(member).decode("utf-8", errors="replace")
                    ):
                        return True
    except zipfile.BadZipFile:
        pass
    return False


def prune_removed_artifact_links(value, removed):
    """Optional curve downloads must refer to files in the public artifact."""
    if isinstance(value, dict):
        url = value.get("concurrency_curves_url")
        if isinstance(url, str):
            parsed = urlsplit(url)
            local = unquote(parsed.path).removeprefix("./").lstrip("/")
            if not parsed.scheme and not parsed.netloc and local in removed:
                del value["concurrency_curves_url"]
        for child in value.values():
            prune_removed_artifact_links(child, removed)
    elif isinstance(value, list):
        for child in value:
            prune_removed_artifact_links(child, removed)


def build(source, destination, policy_revision):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Output must be a new directory outside the input checkout")
    destination.mkdir(parents=True)
    removed, changed, documents, removed_ids = [], [], {}, set()
    excluded = {
        ".git",
        ".github",
        ".agents",
        "tests",
        "scripts",
        "hooks",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "screenshots",
        "output",
        "deploy",
    }
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(source)
        if any(part in excluded for part in rel.parts) or any(
            part.startswith(".") for part in rel.parts
        ):
            continue
        name = rel.as_posix()
        if PROJECT.search(name):
            removed.append(name)
            continue
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        if archive_matches(path):
            removed.append(name)
            continue
        if suffix in (".svg", ".txt", ".log", ".csv", ".jsonl"):
            if PROJECT.search(raw.decode("utf-8", errors="replace")):
                removed.append(name)
                continue
        if suffix == ".md" and PROJECT.search(raw.decode("utf-8", errors="replace")):
            text = raw.decode("utf-8")
            first_heading = next(
                (line for line in text.splitlines() if line.startswith("# ")), ""
            )
            if PROJECT.search(first_heading) or re.search(
                r"FRONTIER-(?:DSV4|SMALL-FISH|REPEAT-SELECTION|QWEN35-(?:CONCURRENCY|SWE-PREFIX))",
                name,
            ):
                removed.append(name)
                continue
            raw = "\n".join(
                line for line in text.splitlines() if not PROJECT.search(line)
            ).encode("utf-8")
        if suffix == ".png" and re.search(
            r"sharegpt_story|multi_workload|current.research|当前研究成果|agent_research_online_candidate",
            name,
            re.I,
        ):
            removed.append(name)
            continue
        if suffix == ".json" and name != "data/benchmark-source-manifest.json":
            data = json.loads(raw)
            projected = project_json(
                data,
                removed_ids,
                retained_runs=PUBLIC_C16_RUN_IDS if name == C16_EVIDENCE_PATH else (),
            )
            prune_frontier(projected, removed_ids)
            documents[name] = (data, projected, raw)
            continue
        if suffix == ".html":
            raw = clean_html(raw.decode("utf-8")).encode("utf-8")
        elif suffix == ".js":
            raw = clean_javascript(name, raw.decode("utf-8")).encode("utf-8")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if raw != path.read_bytes():
            changed.append(name)
    # Repeat with the full removed-id set so cross-file references cannot expose orphan records.
    for name, (original, _, raw) in documents.items():
        projected = project_json(
            original,
            removed_ids,
            retained_runs=PUBLIC_C16_RUN_IDS if name == C16_EVIDENCE_PATH else (),
        )
        prune_frontier(projected, removed_ids)
        prune_removed_artifact_links(projected, set(removed))
        if projected is DROP:
            removed.append(name)
            continue
        add_betterscale_card(name, projected, original)
        out = (
            raw
            if projected == original
            else (json.dumps(projected, ensure_ascii=False, indent=2) + "\n").encode()
        )
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out)
        if out != raw:
            changed.append(name)
    manifest_path = destination / "data/benchmark-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = []
    for entry in manifest["files"]:
        target = destination / entry["website_path"]
        if not target.is_file():
            continue
        raw = target.read_bytes()
        item = copy.deepcopy(entry)
        item["unprojected_sha256"] = item["sha256"]
        item["sha256"] = hashlib.sha256(raw).hexdigest()
        item["bytes"] = len(raw)
        entries.append(item)
    manifest["files"] = entries
    manifest["projection"] = {
        "policy": "selected-research-withdrawal-20261030",
        "public_run_ids": sorted(PUBLIC_C16_RUN_IDS),
        "revision": policy_revision,
        "scope": "Public website projection; original research records are not rewritten.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Remove rendered links to files omitted from this public artifact.
    omitted = set(removed)
    for path in destination.glob("*.html"):
        text = path.read_text()

        def link(match):
            href = re.search(r'href=["\']([^"\']+)', match[0])
            if href and unquote(urlsplit(href[1]).path).removeprefix("./") in omitted:
                return ""
            return match[0]

        path.write_text(re.sub(r"<a\b[^>]*>.*?</a>", link, text, flags=re.S))
    audit = {
        "removed_files": sorted(set(removed)),
        "projected_files": sorted(set(changed)),
        "removed_record_ids": sorted(removed_ids),
        "policy_revision": policy_revision,
    }
    return audit


def validate(destination):
    manifest = json.loads(
        (destination / "data/benchmark-source-manifest.json").read_text()
    )
    for entry in manifest["files"]:
        raw = (destination / entry["website_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], entry["website_path"]
        assert len(raw) == entry["bytes"], entry["website_path"]
    for path in (destination / "data").rglob("*.json"):
        if path.name == "benchmark-source-manifest.json":
            continue
        data = json.loads(path.read_text())
        name = path.relative_to(destination).as_posix()
        validate_betterscale_card(name, data)
        if name == C16_EVIDENCE_PATH:
            for run in data.get("runs", []):
                if run.get("run_id") in PUBLIC_C16_RUN_IDS:
                    run.clear()
        assert not matches(data), path
    for name in [
        "betterscale.html",
        "traceloom.html",
        "gennova.html",
        "assets/betterscale.js",
        "assets/gennova.js",
    ]:
        assert not (destination / name).exists(), name
    for name in [
        "index.html",
        "plugins.html",
        "dataset-introductions.html",
        "leaderboard-runs.html",
        "assets/site.js",
    ]:
        assert (destination / name).is_file(), name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--website", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    audit = build(args.website, args.output, args.policy_revision)
    validate(args.output)
    # Exercise the same schema gate used by the rendered chart, including
    # references between configurations, cohorts and comparison series.
    subprocess.run(
        [
            "node",
            "-e",
            "const fs=require('fs');const M=require('./assets/leaderboard-frontier-model.js');M.validate(JSON.parse(fs.readFileSync('data/leaderboard_frontier.json')));",
        ],
        cwd=args.output,
        check=True,
    )
    for javascript in (args.output / "assets").glob("*.js"):
        subprocess.run(["node", "--check", str(javascript.resolve())], check=True)
    if args.audit:
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "removed_files": len(audit["removed_files"]),
                "projected_files": len(audit["projected_files"]),
                "removed_record_ids": len(audit["removed_record_ids"]),
                "validated": True,
            }
        )
    )
