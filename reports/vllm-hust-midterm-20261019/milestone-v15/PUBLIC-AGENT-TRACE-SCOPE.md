> 本页为本轮大纲执行补充；脚本路径相对 vLLM-HUST Engine 仓库根目录。本目录机器清单与 Engine 固定版本逐字节一致。

# SWE / Terminal 公开轨迹固定抽样执行补充

本轮按用户要求，将 SWE-bench Pro、Terminal-Bench 2.1 的重复执行改为公开真实请求轨迹的推理性能回放，
并固定抽样以控制单轮耗时。原测试大纲摘录、历史任务成绩及失败原件保持原样。 这是新版本的执行范围，不称官方统一的 serving trace 标准，也不产生任务解决率。

## 来源与版本

- SWE：Scale 官方 `claude-sonnet-4-paper` 运行的 `.traj`，
  [官方来源说明](https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/traj/README.md)。
  使用轨迹中的逐轮
  `query`，按对应SWE-Agent1.1.0的公开转换规则恢复API消息与工具返回ID；不从任务描述拼接对话，不读取保存的回答或评分来选择请求。工具schema由固定官方1.1.0
  bundle恢复，生成的command_docs摘要与全部32份轨迹完全一致；这部分为有来源的格式恢复，不称原始wire请求抓包。固定schema、上游提交/文件摘要与MIT许可见`swe-agent-v110-tools.json`。
  按冻结的 SWE-bench Pro **v1 731题**核对任务ID；该公开运行有637个可用任务，94个源轨迹缺失项逐个列明。
  另一个250-turn公开运行有730份轨迹，但本配置不混用不同运行的轨迹补齐来源。
- Terminal：作者公开的
  [pi Agent provider 请求记录](https://huggingface.co/datasets/shadowpa0327/pi_agents_terminal_bench)，
  固定 revision `ccc3fad7ed2cf81f98094f8fdd001bdb480e096e`。 保存的是实际 OpenAI
  请求体、完整消息、工具定义、工具调用/返回及时间戳；作者许可标注 research-only。 与任务仓库 `harbor-framework/terminal-bench-2-1` 的固定提交
  `7131e4375048a0e408a8fb404b5f499d726b695b` 核对；89题中88题有对应公开请求， `pytorch-model-recovery`
  无对应公开记录。所选任务还逐项核验原 instruction 为请求中的完全一致子串。 同任务多次运行时按公开 corpus 路径字典序取首份，不按成功/失败或耗时挑选。

所有原文件均绑定 URL、字节数及 SHA256；Terminal 同时核对固定 revision 的 Git blob或Git LFS内容SHA256。
工具包及公开清单只保存ID、来源和摘要，不附带原始对话全文；运行时下载到用户缓存， 已有完整缓存可离线复核。Hugging Face 不可达时可从镜像取得相同固定哈希的字节，不能换版本或数据。
S3使用匿名HTTPS下载，无需AWS凭据；私有Engine Release凭据仍按原`.env`规则读取。

## 固定抽样与预算

机器清单为 `public-agent-trace-fixed32-v1.json`；构建输入为 `public-agent-trace-source-selection-v1.json`，构建算法在
`scripts/freeze_agent_traces.py`。

1. 每项固定选择32个任务。SWE按可用637任务的仓库比例用最大余数法分配名额； 各仓库内按
   `SHA256(UTF8('public-agent-trace-fixed32-v1\0swe\0'+task_id))` 升序取题。
   Terminal在88个可用任务中按相同公式、dataset=`terminal`升序取32题。平局按任务ID字典序。
1. 每任务最多取两个不同的真实请求：原顺序中首个、末个合格请求。 合格范围为文本请求、目标模型完整模板输入不超过16,384 token；不截断、不补齐、不改写历史消息。
   工具定义保留；SWE只去掉`agent`等非API元数据。图像、过长输入、重复请求以及公开源中已脱敏/缺失消息或非法工具参数的快照明确记录排除原因；缺失内容不恢复、不补写。
   只有一个合格请求就只执行一个，无合格请求的任务单列；不复制请求凑数，不替换任务。
1. B0/B1同请求、同序、同完整输入和工具定义。每项**最多64请求/臂**，准确数量以冻结清单为准。
   `max_tokens=2048`、`temperature=0`、`seed=17`、`enable_thinking=false`，EOS保留，
   `tool_choice=none`关闭结构化工具响应解析；两侧固定服务均保留工具定义入prompt，输出按原始文本流测量，
   并发8闭环、有限队列，socket超时600秒，不自动重试。模型仍为同源Qwen3.5-35B-A3B BF16。 这是本次明确的新性能负载预算，不沿用公开运行的原模型、长输出预算或到达时间。
1. 使用目标权重自带 tokenizer 与 chat template，构建时固定 Transformers5.14.1；
   清单保存输入token数和元数据文件哈希，实际响应usage须一致。每个请求使用独立cache salt， 两臂均要求零跨请求KV复用。B1
   r4组合、4GiB/rank主机状态、B0原0.18和图模式约束不变。

选择发生在任何B0/B1新测量之前，不按历史或本轮成功率、输出长度、速度、先返回顺序挑选。 这是固定、可复算的公开可用轨迹子集，不宣称随机代表731/89题全体，也不把范围外任务计为已执行。
每条请求快照独立实际生成；后续快照继续使用公开原始历史，不把本轮新回答接回Agent，也不执行历史工具命令。

## 一键入口、报告与验收边界

Engine根 `one_click.py` 默认冻结并执行：MMLU固定200、HLE分层200、Frontier160、SWE trace、Terminal trace。
完整默认清单为15阶段：两侧C8、MMLU配对、Frontier四阶段、HLE四阶段、SWE两臂、Terminal两臂。
首次第三方执行仍重新取得本机C8资格；`--datasets swe,terminal`只选择两项trace，不能当五项完整运行。
`--prepare-only`只冻结、校验并出准备报告，不发推理请求。原有正在执行的HLE任务不修改、不抢卡。

报告单列公开轨迹的来源、抽样分母、上下文分布、真实输出量、吞吐、TTFT、TPOT、样本P95/P99、 长度终止、失败、未返回和实际达到16K的请求数。TPOT仍按原大纲的请求完成边界计算，
吞吐采用实际请求提交至响应全部结束的墙钟窗口；不把重叠请求耗时相加。 原时间戳仅作来源记录；本次并发8负载不冒充原在线流量。两臂未完成时不计算正式改善率。

trace不依赖任务Docker环境，CPU工作区已就绪也不再是性能回放的阻塞。
本轮不执行SWE修复验证器或Terminal任务验证器，报告`quality_measured=false`，不套用公开作者的成绩。
MMLU/HLE/Frontier的原质量协议和预算保持原样。四目标逐项显示适用范围与卡点； trace回放成功不自动证明任务质量、五项端到端质量验收或16K连续窗口稳定性。
原吞吐≥10%、平均TTFT/TPOT改善各≥5%、效率≥50%、成本下降≥10%及连续窗口P95/P99要求不降低； 有限抽样分位数单列，不替代连续窗口指标。

首次恢复原件的网络时间和SDK/模型启动时间单独记录，后续复用已校验缓存； 每项每臂输出上界为实际冻结请求数×2048，耗时以新配对实测为准，不用公开作者速度充当本机预计结果。

## 本版本实际冻结数量

SWE：32任务、64请求，输入1519–16350 token；Terminal：32任务、48请求，输入2065–16294
token。Terminal有16个所选任务只有一条合格的公开请求，不复制或替换来凑满64。每臂输出上界分别131072和98304
token。清单SHA256为`eacf3628cda8d7accc2e39b20fce687b1deb1cb0b0f470e8b11c8dd4517f71f8`。

本版本已通过83项CPU回归、Ruff、64份原件校验、真实公开数据准备CLI及112条独立模板渲染/编码复核；准备不等于推理验收。本机新trace配对尚未开始，当前HLE固定200运行保持其原封存版本与设备所有权。

正式trace阶段在发出首条生成请求前，先对全部冻结输入调用实际服务的`/tokenize`，保存原始token ID及其摘要并核对输入长度。B1还须读取已封存的同范围B0回执并逐请求核对token
ID完全一致；失败立即停止，不发生成请求。该CPU预检时段与实际推理性能窗口分开。
