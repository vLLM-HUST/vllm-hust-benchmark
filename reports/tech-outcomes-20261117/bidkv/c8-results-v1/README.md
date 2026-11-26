# 81 双卡 C8 资格数值证据

本包是显式CPU绑定false的旧自定义配置资格记录。新MOD要求恢复上游默认true，所以本包不被复用为新true配置资格；数值和受控原件全部保留。

两臂四阶段全部完成并释放设备。已从每条受控原始 SSE 独立复算 Native 342 / BidKV 343 条请求，逐条校验结束原因、固定 256-token
usage、输入长度、唯一盐和时间戳；所有请求零失败，prefix / external-prefix 命中增量均为零。usage 的缓存详情全部为
null，保留为未知，由唯一盐与引擎计数器共同佐证零复用。

| 臂     | 阶段         | 请求数（含排空） | 失败 | 平均首生成事件延迟 s | 平均完成 TPOT ms/token |
| ------ | ------------ | ---------------: | ---: | -------------------: | ---------------------: |
| native | 4096-warmup  |               16 |    0 |            22.944540 |              33.854221 |
| native | 4096-formal  |              230 |    0 |             1.399592 |              35.869720 |
| native | 16384-warmup |               16 |    0 |            12.791702 |              78.570012 |
| native | 16384-formal |               80 |    0 |             6.845640 |              93.418393 |
| bidkv  | 4096-warmup  |               16 |    0 |            22.571065 |              33.585978 |
| bidkv  | 4096-formal  |              232 |    0 |             1.540488 |              34.735040 |
| bidkv  | 16384-warmup |               16 |    0 |            12.638047 |              78.236883 |
| bidkv  | 16384-formal |               79 |    0 |             6.742841 |              95.082776 |

四个阶段分别为 4K/16K 的 60 秒预热与 300 秒正式提交窗口，8 路闭环、全部排空，真实输入长度为 4115/16407 tokens。原单独合成 C8 配方使用
`ignore_eos=true/max_tokens=256`，与五数据集正式推理全部保留 EOS 的规则分开。各阶段输出及分位数见两臂 `phases.json`；逐请求时钟、token
计数和原件 SHA 见 `requests.jsonl`；独立重算见 `independent-verification.json`。

本表是原 C8 合成资格观察，正式阶段数量包含排空；它不把排空输出除以 300 秒作为吞吐，也不替代完整五数据集、固定到达率服务 SLO 或 32K
稳定性验收。`first_generated_event` 含 content/reasoning；正式五项 TTFT 使用 `first_content_event`。原始两个时钟均保留。

Native 关闭 BidKV，Candidate 已启用，但本 C8 两臂调度决策与策略调用均为 0；不能据此宣称 BidKV victim selection 收益。无策略失败或非法选择，见
`policy-summary.json`。

原题、对话、生成内容、token ID 数组及完整原始 SSE 不在此分发；受控文件清单、大小、SHA 和 NAS 位置见各臂
`controlled-originals-index.json`。需已有授权的 Coder/NAS 访问，无公开下载链接。独立校验器来自此前公开的
`finite-driver-v1/verify_c8.py.source`；本目录另附自行编写的数值导出器与 Native 控制器源码，不含私有 Engine 源码。

C8 执行源固定公开提交 b8a1c0c0062abce4b08e367ff003b1416df0afa2，Candidate 启动器
dbd3e0627e8ba57850ada6424a90f4de1a925227，独立复核器 dba199007aa62c78bcc4fbe54bad4d54a2bd5998。执行身份与实际源码
SHA 详见两臂 `execution-identity.json`。
