# 独立的小规模容量压力验证

本探针已在Native启动准入失败，未提交任何生成或计时窗口。实际HMA
padding后262144上下文至少需3.07GiB，给定3.0GiB不足；原估算过低。API父进程未退出，旧health控制未即时识别已见fatal，最终触发1200秒timeout。完整[失败摘要](failure-01/failure-summary.json)、[fatal/准备日志摘录](failure-01/fatal-and-preparation-extract.txt)、[受控原件索引](failure-01/controlled-originals-index.json)保留；吞吐未测为null。旧detached
launcher未捕获实际wait exit code，明确保留未知，不推填1。原计划/源码不改写，不追加旧false4GiB重跑；先按用户新MOD要求保留默认CPU绑定true另做资格与五项。

本计划在取得结果前公开，不能改写已完成的 KV20GiB/rank 五档负结果，也不替代冻结五数据集。原模型、运行时、TP2、prepared 输入和全部声明的 context262144
保持一致；仅两臂 KV20GiB/rank 改为3GiB/rank，继续启用当前 HMA 安全逻辑，Candidate host pool30GiB。

先在两臂检查两个长前缀的32-token冷/热/真实恢复逐 token 一致及零失败，再分别做 C16/180s 的探索窗口。仅当两臂均零失败、Native 实际发生抢占、Candidate
吞吐正向时，才启动全新 Candidate→Native 各900s的同配置确认。否则不扩大此profile，转入冻结五数据集。所有正、零、负和失败结果保留；不做参数网格搜索或换题。

配置、触发条件和执行源码 SHA 见 [plan.json](plan.json)；`.py.source` 去掉 `.source` 后是执行原字节。本合成多轮 SWE
复用负载沿用原固定输出预算，五项实际标准请求全部保留 EOS，两者分开报告。2GiB 草案在推理前因无法满足原声明的262144上下文容量被否决，没有相应结果。
