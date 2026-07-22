# vLLM-HUST 完整 Benchmark 工作流

> 本文档总结了从 benchmark 执行 → 导出制品 → 发布到 website 的完整链路，
> 以及在执行过程中遇到的典型问题和解决方案。

---

## 一、工作流总览

```
第 1 阶段: Benchmark 执行 (本地容器)
  run <scenario> --model <model> --execute
  -> 原始 benchmark JSON

第 2 阶段: 导出 Leaderboard 制品 (本地容器)
  export-leaderboard-artifact --benchmark-result-file ...
  -> run_leaderboard.json + leaderboard_manifest.json

第 3 阶段: 本地预览 (可选)
  publish-website -> leaderboard_single.json
  python3 -m http.server 8000 -> ?dataSource=local

第 4 阶段: 提交到 GitHub + CI 自动发布
  方式 A: 本地 git push (需网络可连 GitHub)
  方式 B: GitHub API (HTTPS 被拦截时)
  方式 C: fork + PR (无直接 push 权限时)
  -> CI 自动同步到 HuggingFace -> 公开网站可见

第 5 阶段: 等待 maintainer 审批 CI
  PR 需要 maintainer 点击 "Approve and run"
  CI 通过后数据自动出现在公开网站
```

---

## 二、完整步骤

### 2.1 Benchmark 执行

```bash
export ASCEND_RT_VISIBLE_DEVICES=1

MODEL="/workspace/data/models/Llama-3.1-8B-Instruct"
DATASET="/workspace/data/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"

# 方案 A: 手动三步法 (推荐)
nohup vllm-hust serve "$MODEL" \
    --dtype bfloat16 --host 127.0.0.1 --port 18000 \
    > /tmp/server.log 2>&1 &
SERVER_PID=$!

# 等 /v1/models 返回 200 (不要用 /health)
for i in $(seq 1 300); do
    code=$(curl -s -o /dev/null -w "%{http_code}" \
        http://127.0.0.1:18000/v1/models 2>/dev/null || echo "000")
    [ "$code" = "200" ] && echo "Ready after ${i}s" && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "Server died!"; exit 1; }
    sleep 2
done

vllm bench serve --model "$MODEL" \
    --backend vllm --endpoint /v1/completions \
    --dataset-name sharegpt --dataset-path "$DATASET" \
    --num-prompts 200 --host 127.0.0.1 --port 18000 \
    --save-result --result-dir /workspace/data/results

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true

# 方案 B: run 命令 (需传 serve 参数)
python -m vllm_hust_benchmark.cli run sharegpt-online \
    --model "$MODEL" \
    --set dataset_path="$DATASET" \
    --set dtype=bfloat16 \
    --set save_result=true \
    --set result_dir=/workspace/data/results \
    --execute
```

### 2.2 导出 & 本地发布

```bash
cd /workspace/vllm-hust-benchmark

# 先配置 git 用户信息 (影响网页上显示的 @用户名)
git config --global user.name "你的名字"
git config --global user.email "your@email.com"

# 导出制品
python -m vllm_hust_benchmark.cli export-leaderboard-artifact sharegpt-online \
    --benchmark-result-file /workspace/data/results/<结果.json> \
    --constraints-file docs/examples/constraints_metrics.sample.json \
    --peak-mem-mb 10240 \
    --output-dir .benchmarks/exports/run-xxx \
    --run-id run-xxx \
    --engine vllm-hust \
    --engine-version manual \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --model-parameters 8B --model-precision BF16 \
    --hardware-vendor Huawei --hardware-chip-model Ascend-910B \
    --chip-count 1 --node-count 1 \
    --submitter 你的名字 \
    --input-length 1024 --output-length 256 \
    --execute

# 本地预览 (可选)
python -m vllm_hust_benchmark.cli publish-website \
    --source-dir .benchmarks/exports/run-xxx \
    --output-dir /workspace/vllm-hust-website/data \
    --execute

cd /workspace/vllm-hust-website
python3 -m http.server 8000
# 访问 http://127.0.0.1:8000/?dataSource=local
```

### 2.3 提交到 GitHub (三种方式)

#### 方式 A: 本地 git push (网络正常时)

```bash
cd /workspace/vllm-hust-benchmark

# 创建分支
git checkout -b submissions/你的名字-场景-模型

# 复制 artifact 到 submissions/
mkdir -p submissions/run-你的名字
cp .benchmarks/exports/run-xxx/run_leaderboard.json submissions/run-你的名字/
cp .benchmarks/exports/run-xxx/leaderboard_manifest.json submissions/run-你的名字/

# 提交
git add submissions/run-你的名字/
git commit -m "chore: add benchmark result for 你的名字"

# 推送 (需要有 push 权限)
git push origin submissions/你的名字-场景-模型

# 创建 PR (需要 gh CLI)
gh pr create --title "chore: add benchmark result" --body "description"
```

#### 方式 B: 通过 GitHub API (HTTPS 被拦截时)

当 github.com 无法直接连接时，使用 API:

```bash
# 1. 本地 clone 仓库 (通过 ghproxy 等镜像)
git clone https://ghproxy.net/https://github.com/vLLM-HUST/vllm-hust-benchmark.git
cd vllm-hust-benchmark

# 2. 应用 patch 或手动复制文件
git am /path/to/patch

# 3. 通过 API 创建分支 + 上传文件 + 创建 PR
# 需要 GitHub Personal Access Token (repo 权限)
TOKEN="你的token"

# 创建分支
SHA=$(curl -s -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/git/refs/heads/main \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['object']['sha'])")

curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/git/refs \
  -d "{\"ref\":\"refs/heads/分支名\",\"sha\":\"$SHA\"}"

# 上传文件 (创建 blob -> tree -> commit -> update ref)
# ... (详见下方完整脚本)
```

#### 方式 C: fork + PR (无 push 权限时)

```bash
# 1. 创建 fork
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/forks

# 2. 在 fork 上创建分支 + 上传文件 (同方式 B)
FORK_REPO="你的用户名/vllm-hust-benchmark"
# ... 创建分支、上传文件 ...

# 3. 从 fork 创建 PR → 原仓库 main
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/pulls \
  -d '{
    "title":"chore: add benchmark result",
    "head":"用户名:分支名",
    "base":"main"
  }'
```

### 2.4 等待审批

PR 创建后，由于是 fork PR，CI 需要 **maintainer 在 PR 页面 Approve** 才能运行。
CI 通过后，`push-to-hf` 工作流会自动同步到 HuggingFace，公开网站可见。

---

## 三、GitHub API 完整提交流程脚本

以下脚本封装了从本地 artifact 到 GitHub PR 的全流程：

```bash
#!/bin/bash
# 用法: ./submit-to-github.sh <token> <分支名> <submission目录>

TOKEN=$1
BRANCH=$2
SUBMIT_DIR=$3

REPO="vLLM-HUST/vllm-hust-benchmark"
FORK_REPO="你的用户名/vllm-hust-benchmark"

# 1. 创建 fork
echo "Creating fork..."
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/$REPO/forks > /dev/null

# 2. 获取 main SHA
MAIN_SHA=$(curl -s -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/$FORK_REPO/git/refs/heads/main \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['object']['sha'])")

# 3. 创建分支
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/$FORK_REPO/git/refs \
  -d "{\"ref\":\"refs/heads/$BRANCH\",\"sha\":\"$MAIN_SHA\"}"

# 4. 上传 submission 文件
for file in "$SUBMIT_DIR"/*.json; do
    filename=$(basename "$file")
    content=$(base64 -w0 "$file")
    blob_sha=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      https://api.github.com/repos/$FORK_REPO/git/blobs \
      -d "{\"content\":\"$content\",\"encoding\":\"base64\"}" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
    echo "  $filename: $blob_sha"
done

# 5. 创建 tree + commit
TREE_SHA=$(curl -s -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/$FORK_REPO/git/commits/$MAIN_SHA \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['tree']['sha'])")

# ... 创建带有新文件的 tree ...

# 6. 创建 PR
PR_URL=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/$REPO/pulls \
  -d "{\"title\":\"chore: add benchmark result\",\"head\":\"用户名:$BRANCH\",\"base\":\"main\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('html_url','error'))")
echo "PR: $PR_URL"
```

---

## 四、关键问题记录

### 4.1 Server 启动条件

`run` 命令的 `serve_parameters` 必须非空才会启动 vLLM Server。
必须至少传一个 serve 参数如 `--set dtype=bfloat16`。

### 4.2 NPU 设备选择

```bash
npu-smi info                      # 查看各卡 HBM 使用情况
export ASCEND_RT_VISIBLE_DEVICES=1  # 选空闲卡
```

### 4.3 健康检查陷阱

必须等 `/v1/models` 返回 200，不要用 `/health`（模型加载前就返回 200）。

### 4.4 publish-website 只更新本地文件

不影响公开网站。公开网站数据通过 CI 自动同步到 HuggingFace。

### 4.5 网站加载慢

URL 加 `?dataSource=local` 强制使用本地数据。

### 4.6 Git 用户名影响网页显示

```bash
git config --global user.name "Raing5Days"
git config --global user.email "M202574102@hust.edu.cn"
```

网页显示为 `@用户名` (自动加 @ 前缀)。

### 4.7 PR 需要 maintainer 审批

Fork PR 的 CI 需要仓库 maintainer 在 PR 页面点 Approve 才能运行。

---

## 五、排查指南

| 症状 | 原因 | 解决 |
|------|------|------|
| 全 404, duration < 1s | Server 没启动 | 加 `--set dtype=bfloat16` |
| Server 崩溃: Free mem X/Y < util | NPU 被占用 | `ASCEND_RT_VISIBLE_DEVICES=N` |
| Server 在跑但 bench 全 404 | 健康检查假阳性 | 等 `/v1/models` 返回 200 |
| 数据写到 JSON 但网页没显示 | 只跑了 publish-website | 提 PR -> CI 自动发布 |
| 网页一直转圈加载 | GitHub/HF 请求超时 | URL 加 `?dataSource=local` |
| git push 连不上 github.com | 网络拦截 | 换 API 方式提 PR |
| PR CI 一直不跑 | 需要 maintainer 审批 | 在 PR 评论区 @maintainer |
| 网页显示 @my 不是自己名字 | git user.name 未设置 | 修改后重新导出 |

---

## 六、成果物

```
/workspace/data/results/
  result_<run-id>.json                     # 原始 benchmark 结果

/workspace/vllm-hust-benchmark/
  .benchmarks/exports/<run-id>/
    run_leaderboard.json                  # Leaderboard 制品
    leaderboard_manifest.json             # 导出清单

/workspace/vllm-hust-website/data/
  leaderboard_single.json                 # 本地排行榜

GitHub PR (#24) -> CI -> HuggingFace:
  https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/24
  https://vllm-hust.sage.org.ai/          # 公开网站
```
