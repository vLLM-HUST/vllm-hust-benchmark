# vllm-hust-website (Website Demo Kit)

**数据提交规则：本仓库不接受数据 PR。** 实验结果、曲线、矩阵、基线、价格依据、元数据与证据 必须 PR 到
[vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark)。 网站只接收展示和代码变更。详见
[贡献规则](CONTRIBUTING.md)。

<!-- BEGIN:VERSION_META -->

**vllm-hust 项目概览** vllm-hust 围绕上游 vLLM 展开，聚焦国产算力适配、AGI4S 服务场景和 benchmark 驱动验证。

## Project Highlights

- ✅ **上游兼容入口**：围绕 vLLM 兼容接口组织 CLI、服务与 benchmark 链路
- ✅ **国产算力使能**：面向 Ascend 等国产硬件扩展后端与部署能力
- ✅ **AGI4S 服务优化**：关注长上下文、工具调用与结构化输出场景
- ✅ **OpenAI 兼容 API**：完整支持 `/v1/chat/completions` 和流式响应
- ✅ **安装与依赖治理增强**：发布链路与版本一致性检查更稳健
- ✅ **Benchmark 驱动优化**：通过对比数据持续验证服务性能与稳定性

## Quick Start

请勿用下方 PyPI 包安装当前服务栈，它属于不同的版本组合；先查阅维护中的部署指南，选择经过验证的 core/plugin/镜像组合。

```bash
# 获取部署工具（并不直接安装运行时）
git clone https://github.com/vLLM-HUST/vllm-hust-dev-hub.git
```

[部署运维指南](https://github.com/vLLM-HUST/vllm-hust-dev-hub/blob/main/docs/sage-mate-production-runtime.md)

_该区块由 `data/version_meta.json` 驱动，运行 `python scripts/sync_version_meta.py` 自动更新。_

<!-- END:VERSION_META -->

## Version Metadata Maintenance

- Source of truth: `data/version_meta.json`
- Refresh public GitHub/PyPI facts: `python scripts/refresh_site_status.py --refresh`
- Render README from reviewed metadata: `python scripts/sync_version_meta.py`
- Read-only daily drift/age check: `.github/workflows/site-status-check.yml`
- README consistency workflow: `.github/workflows/sync-version-meta.yml` (does not refresh source
  SHAs)
- Consistency/stale check: `bash scripts/check_stale_versions.sh`
- 维护说明：`docs/VERSION_METADATA.md`

## Website Workstation Embed

首页现在支持把远端 `vllm-hust-workstation` 作为一个可配置面板展示出来，适合把部署在远端算力节点上的 workstation 直接呈现到 website 界面。

- 配置文件：`data/workstation_embed.json`
- 加载脚本：`assets/workstation-embed.js`
- 展示方式：`embed`（iframe 内嵌）或 `link`（仅提供打开入口）

示例：

```json
{
	"enabled": true,
	"mode": "embed",
	"workstation_url": "https://ws.sage.org.ai",
	"backend_url": "https://api.sage.org.ai",
	"docs_url": "./docs/CLOUDFLARE_SETUP.md",
	"label_zh": "vLLM-HUST 工作站",
	"label_en": "vLLM-HUST Workstation"
}
```

注意：

- `website` 是静态站点，不代理 workstation API；要展示远端控制台，需要 workstation 自己对外可访问。
- 若 website 走 HTTPS，则 workstation 也必须走 HTTPS，否则浏览器会阻止 iframe 混合内容。
- 如果生产环境要限制可嵌入来源，请在 workstation 侧配置 `APP_FRAME_ANCESTORS`。

推荐的 `sage.org.ai` 域名拆分：

- `vllm-hust.sage.org.ai`：GitHub Pages 静态官网
- `ws.sage.org.ai`：远端节点上的 `vllm-hust-workstation`
- `api.sage.org.ai`：远端节点上的 `vllm-hust` OpenAI 兼容接口

## Architecture And Planning Notes

- vLLM 实际运行时架构与 9 个子任务映射：`docs/ARCHITECTURE_TASK_MAPPING.md`
- 首阶段实施优先级与验证清单：`docs/IMPLEMENTATION_PRIORITIES.md`

## Benchmark Data Chain

website does not ingest raw compare directories and does not hand-edit benchmark rows.

The dataset validation results surface is a separate, read-only path. The independent fixed-machine
validation service runs after a PR, then publishes a versioned result artifact for the website to
display at [`dataset-validation.html`](dataset-validation.html). The website does not trigger that
service and does not perform evaluation, admission, publication, or merge-gate work. The frontend
contract and empty local fixture are documented in
[`data/DATASET_VALIDATION_RESULTS.md`](data/DATASET_VALIDATION_RESULTS.md). The existing leaderboard
snapshot chain below remains a legacy compatibility surface until its replacement data contract is
explicitly migrated; it must not be used as the dataset validation service contract.

The only supported leaderboard data chain is:

1. Any benchmark pipeline that exports standard leaderboard artifacts (`leaderboard_manifest.json` +
   `*_leaderboard.json`)
1. Snapshot publish in `vllm-hust-benchmark/leaderboard-data/snapshots`
1. HF publish from the same benchmark snapshot files to `intellistream/vllm-hust-benchmark-results`
1. Website sync workflow publishes a privacy-sanitized projection of those benchmark snapshots for
   offline/static fallback; machine-specific absolute paths never cross the public boundary:
   - `data/leaderboard_single.json`
   - `data/leaderboard_multi.json`
   - `data/leaderboard_historical.json` (recovered trend-only records)
   - `data/leaderboard_compare.json`
   - `data/last_updated.json`

Leaderboard rendering consumes benchmark GitHub raw snapshots first, then HF, then the checked-in
website projection. `leaderboard_compare.json` provides neutral engine-vs-engine head-to-head views.

Leaderboard version rendering follows a split UI contract:

- The main table stays compact for scanability.
- Expanded details stay richer for provenance and reproduction.
- Design rule and regression anchor: `docs/VERSION_METADATA.md#leaderboard-version-display-contract`

`leaderboard_compare.json` now also carries a mandatory hard-constraint snapshot derived from
validated entries:

- single-chip effective utilization >= 90%
- typical scene throughput >= 2x baseline and TTFT/TPOT reduction > 20%
- long-context (>=32K) throughput + TTFT/TPOT P95/P99 stability
- single-business token-cost reduction >= 30% with high multi-tenant utilization

All hard-constraint records must declare `constraints.scenario_source = vllm-benchmark`, using the
vLLM benchmark inheritance path as the accountable scenario and dataset source.

## Repository Quickstart

本仓库提供统一的 `quickstart.sh` 模式入口：

- `--standard`：标准模式，依赖优先从 PyPI 安装（稳定/发布导向）
- `--dev`：开发模式，在 standard 基础上尝试本地 editable 覆盖（使用 `--no-deps`）
- `requirements-dev.txt`：本地校验依赖单一来源，供 quickstart、CI 和 `scripts/validate-local.sh` 统一复用
- 安装前会执行同前缀包动态清理（默认前缀：`vllm-hust-website`）

```bash
# 查看帮助
./quickstart.sh --help

# 标准模式
./quickstart.sh --standard

# 开发模式（默认）
./quickstart.sh --dev

# 可选：跳过清理或 hooks
./quickstart.sh --dev --skip-cleanup --skip-hooks
```

> 注意：脚本不会创建 `venv/.venv`，请使用当前已配置的非-venv Python 环境。

## Local Validation

本地校验依赖以 `requirements-dev.txt` 为唯一声明源。首次进入仓库后，执行下面任一方式即可获得和 CI 一致的基础 Python 工具链：

```bash
# 推荐：同时安装校验依赖并配置 hooks
./quickstart.sh --dev

# 或者仅安装校验依赖
python3.11 -m pip install -r requirements-dev.txt
```

在本地提交前，如果你想尽量一次性对齐 CI 的 `validate` job，直接运行：

```bash
./scripts/validate-local.sh
```

这个脚本会依次执行：

- `pre-commit run --all-files`
- `pytest tests/ -v`
- hook 模板与 `quickstart.sh` 安装语句校验

要求：

- 需要 `python3.11`
- 优先使用当前环境里的 `pre-commit` / `pytest`
- 如果当前环境缺少这些工具，但本机有 `uv`，脚本会自动使用 `uv run --python 3.11 --with-requirements requirements-dev.txt ...`
  兜底，继续复用同一份依赖声明

## systemd --user 本地常驻

如果要把静态站点在本机以 `systemd --user` 常驻方式挂起来，而不是手动执行 `python3 -m http.server 8000`，仓库现在内置了最小管理脚本：

```bash
# 一次性安装 / 更新 service
./scripts/deploy_website_service.sh install-service

# 安装并立即重启到最新配置
./scripts/deploy_website_service.sh deploy

# 查看状态 / 日志
./scripts/deploy_website_service.sh status
./scripts/deploy_website_service.sh logs
```

默认行为：

- 绑定地址：`127.0.0.1`
- 端口：`8000`
- 根目录：当前仓库根目录
- systemd 服务名：`vllm-hust-website`

可选环境变量：

```bash
export WEBSITE_HOST=127.0.0.1
export WEBSITE_PORT=8000
export WEBSITE_ROOT_DIR=/path/to/vllm-hust-website
export WEBSITE_SYSTEMD_SERVICE_NAME=vllm-hust-website
```
