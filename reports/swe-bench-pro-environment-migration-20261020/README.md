# SWE-bench Pro V1 环境迁移：单真实 Agent 验收

本仓库提交 `742b5d516b193b5d01b0a04a5ae2044378e4e1e4` 的通用入口已独立执行一个真实官方 Agent。 从本仓库
`scripts/run_swe_bench_pro.py run` 启动；仅同步 benchmark 包、入口和锁定清单， 运行进程没有导入 Controller 模块。新的独立 ARM
容器真实读取和修改了任务仓库。 这是环境/入口验收，**修题完成尚未达成**，不是性能对照或榜单成绩。

## 配置与来源

- 仅一个 NodeBB V1 program，时间 0 到达，不设置客户端 slot，不追加任务。
- 固定官方 Agent `d74716a3c8104a113f77cc9ab94cf407ecdcf1e9` 与官方 V1
  `66f92766bba642462d4bbe5479e83f91f9211862`；准备入口重新核对 parquet、完整任务行、 官方源码及不可变 ARM image ID。
- 保留 Qwen3.5-35B-A3B、BF16、TP2、两张 Ascend 910B2，服务上下文 32768。 本次不部署、重启或修改模型。入口直接调用现有模型 API。
- 100 步、1200 秒、输出上限 4096 tokens、工具超时 60 秒；HTTP read=null、其余各 180 秒，
  不自动重试；temperature=0、seed=0、thinking=false；任务容器 2 CPU/4 GiB。
- 当前 CPU 平台用具名 Ray actor 管理入口进程，复用原客户端依赖，没有安装到模型环境。 通用入口本身没有 Ray、RIDE、NPU 管理或 Controller 依赖。

复用命令形式（实际私有设置/输入路径未公开）：

```bash
PYTHONPATH=/absolute/benchmark/src python /absolute/benchmark/scripts/run_swe_bench_pro.py prepare \
  --settings /absolute/settings-private.json --reuse-inputs /absolute/verified-old-inputs \
  --output /absolute/new-inputs
PYTHONPATH=/absolute/benchmark/src python /absolute/benchmark/scripts/run_swe_bench_pro.py run \
  --settings /absolute/settings-private.json --inputs /absolute/new-inputs --output /absolute/new-run
```

独立源码准备脚本也实际下载了固定版本，179 个 Agent 文件和 174 个官方 V1 文件校验通过； 回执见
[source-bootstrap-verification.json](source-bootstrap-verification.json)。

## 真实结果

`swe-pro-v1-benchmark-migration-20261020-r1` 于 2026-10-20 12:15:03 UTC（20:15:03 CST）开始， Agent 有效执行
`89.542657` 秒，正常收到 53 次模型回复， 第 54 次请求出现 `ContextWindowExceededError`。官方成功调用计数为 53； 计时事件记录 54 次开始、53
次返回、1 次异常，不能混用两种计数。

Agent 修改了 6 个文件，保存 4537 字节私有补丁，执行了搜索/读取/修改工具命令， 没有执行测试命令，没有主动
`Submitted`。未运行官方评分，`quality_score=null`。 拒绝信息指出输入至少 28673 加预留输出 4096 超过 32768；这是拒绝阈值下界， 本次没有额外调用
tokenizer 复算完整输入，不能把它冒充精确 prompt 长度。

实际终态字段摘录：

```json
{
  "status": "ContextWindowExceededError",
  "model_calls": 53,
  "agent_submitted": false,
  "patch_bytes": 4537,
  "controller_modules_loaded": [],
  "container_cleanup_exit_code": 0
}
```

入口进程 exit=0 只说明编排与结果保存完成，不表示 Agent 提交或修题成功。 模型开始/终态的 running、waiting、KV 使用均为 0；任务容器已移除。
没有扩大批次，没有调大上下文，没有启动第二个 Agent。

## 可审计原件与边界

- 完整私有原件包：148012 字节、50 个文件，SHA256：
  `c22e3dc940aaa2ec416f8bed62e2a7f8520cdab84a3247463f288e59f620828f`。整包与逐文件独立校验 PASS。
- 包含启动命令/设置、benchmark 入口源码及 hash、输入核验、模型 API 与 metrics 前后快照、
  客户端依赖清单、完整官方轨迹、逐请求/工具原始日志、私有补丁、终态及容器清理凭据。
- 私有材料留在运行者本地和原工作区，不把原题、答案、补丁正文、服务地址或凭据上传到本仓库。
- [acceptance.json](acceptance.json) 公开配置、计数、源码绑定及原件哈希。
- benchmark 定向测试 18 项、Controller 接入及既有入口/计时回归测试 27 项通过。 pre-commit 通过；不据此声称远端 CI 已通过。

已证明：用户可以从 benchmark 仓库入口准备输入并独立启动、观察、保存和清理真实 Agent。 仍待解决：长轨迹上下文管理和一个完整的修改→测试→提交流程；本次保持旧约束，未同时修改这部分。

使用说明见 [用户指南](../../docs/SWE_BENCH_PRO.zh-CN.md)。
