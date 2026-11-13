# 五数据集固定评测标准 · five-datasets-20261113-v1

本标准由本轮已经确认的样本和协议冻结，供两个项目共用。它是项目评测标准， 不是数据集作者发布的官方统一 serving trace。机器入口为
[standard.json](standard.json)。

| 数据集             |                      固定任务 | 每臂 solver 请求 | 用途                            |
| ------------------ | ----------------------------: | ---------------: | ------------------------------- |
| MMLU-Pro           |                   200 / 12032 |              200 | 子集质量、推理性能              |
| HLE-Verified       |                    200 / 1811 |              200 | 分层子集质量、推理性能          |
| FrontierScience    |    Olympiad 100 + Research 60 |              160 | 固定版本全 160 题质量、推理性能 |
| SWE-bench Pro      | 32 / 637 可用轨迹（原集 731） |               64 | trace 推理性能                  |
| Terminal-Bench 2.1 |   32 / 88 可用轨迹（原集 89） |               48 | trace 推理性能                  |

MMLU 复用既有 200 题，不推测其最初抽样 seed。HLE 为 Gold 文本 64、Gold 图像 10、 Revision 文本 113、Revision 图像
13；比例最大余数分配后层内 floor(i\*N/quota) 系统抽样。 Frontier 保留固定来源全部 160 题。SWE 按仓库比例分层，再按任务哈希排序；Terminal 匹配 2.1
任务后按哈希排序。trace 取每任务首尾不同的有效公开请求，缺一条就保留一条，不补造。 具体 ID、顺序、来源 revision、逐文件/逐请求哈希和排除清单均在协议 JSON 中。

MMLU 输出上限 8192，temp=0、seed=0、200 同时到达；HLE/Frontier solver 16384、 judge 8192、并发 16、temp=0、seed=17、关闭
thinking。trace 上限 2048、并发 8、temp=0、 seed=17、关闭 thinking；保留 EOS、原始 history 和
tools，tool_choice=none，不执行工具。 trace 输入上限 16384，冻结 Qwen3.5-35B-A3B tokenizer 与模板；两臂 /tokenize 必须逐项
与冻结 token ID 哈希一致，使用新 cache_salt、零 KV 复用。模型/模板变化另立 profile。

HLE/Frontier 两臂使用同一固定 judge，并公开替代 judge 身份；judge 请求不计 solver 吞吐/费用。 失败、未返回、非法答案、length
stop、未决评分均保留计划分母。按 HLE 四层及 Frontier 两轨 单列结果。trace 不给任务成功率；有限批次分位数不替代连续 4K/16K C8 长稳与四目标验收。

不分发题目、答案或原始对话。Terminal 作者抓取受 research-only 条款约束；只保存原始定位与哈希。 SWE 工具恢复保留官方 SWE-agent MIT
来源。第三方须有来源访问权及私有参考引擎读取权； 公开标准不是软件授权或数据再分发授权。参考实现与 bootstrap 固定提交见 standard.json。
每轮另记模型、硬件、B0/B1、实际执行提交和评分身份，不把本标准当作运行成绩。

v1 不覆盖修改：样本、顺序、来源、生成预算、评分、tokenizer 或统计口径变化须发布新版本； 旧实验保留原身份，不自动升级为本标准。两个 benchmark 中的本目录应逐字节相同。

在本目录运行 `python verify.py` 校验全部附件的大小、SHA256、样本数与顺序。
