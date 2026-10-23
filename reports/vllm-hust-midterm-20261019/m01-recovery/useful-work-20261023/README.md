# M01 有效工作量条件性复算

## 对象与方法

Qwen3.5-35B-A3B，MMLU-Pro 200 请求，既有 worker-r1 插桩轮。 输入 282265 tokens，输出 51200 tokens；B1 为 E24/R36、KV
materialization 与 FA Microflow 组合。依据 V5.27 大纲沿用的 V5.26 方法，排除填充、重复、 未采纳草稿和缓存省略工作。此目录不新增性能成绩或达标结论。

根据输入输出长度及无前缀复用记录，必要主模型位置候选为 `sum(input + output - 1) = 333265`，因果注意力对为 287475848。
以下均为源代码语义及轨迹派生值，不是硬件指令计数。

## 分项结果

| 主模型分项                          |     基础运算候选 |
| ----------------------------------- | ---------------: |
| 投影与 QK/AV                        | 1671240625848320 |
| 归一化、残差及输出 gate             |     542333132245 |
| RoPE、MoE、router 及 attention gate |     561044962200 |
| 规范 attention softmax              |     137935084640 |
| 必要输出词表投影                    |   52076478464000 |
| 因果卷积                            |     572732620800 |
| GDN prefill                         |   37780650750720 |
| GDN 必要 decode 递推                |    5621391360000 |
| GDN Q/K L2 与 gate preprocessing    |     123814612800 |
| 已列主模型分项合计                  | 1768657006835725 |

必要 decode 位置为 51000；设备记录 active verify 为 57303，额外 6303 位置单列排除。prefill 复用实际 chunk 算法与初始状态账本，其算法冗余仍需
按有效工作定义复核。两卡分片按全模型计一次，特殊函数不指定任意 FLOP 权重。

## 草稿计数纠正

早期派生文件 `acceptance-epoch-reconstruction-r1.json` 中 `useful_accepted_draft_tokens=31899`
的名称不准确：它是逐轮已提交输出减一， 不是精确的有效采纳草稿数。末轮预算截断可能保留草稿并丢弃最后的 bonus 或纠正 token。旧文件保留，以本目录
`terminal-draft-acceptance-bounds-r1.json` 的解释为准。

在两草稿、统计范围一致、额外输出仅由末轮预算截断造成的条件下，保留草稿 范围为 32021 至 32056；对应输出词表投影为 32569158533120 至 32604757688320
FLOPs。此范围不包含首步上下文准备，也不代表完整草稿算量。 已绑定的 proposer 源码表明，首步先处理上下文 hidden states，再选择位置进行
词表投影，因此不能用总采纳率缩放全部草稿算子。

## 判定与证据

当前判定为计量未闭合。完整有效 F、参考时间、效率和达标判定均未生成。 尚需核对草稿上下文与保留输出依赖、末轮 sampler 身份、有效工作与无插桩
同步计时的对应关系，以及独立参考时间适用性。不得将本轮候选算量除以其他 轮次的最佳耗时。原始记录未改写。

- `target-useful-candidate-r1.json`：必要位置、投影及 QK/AV。
- `target-nonmatrix-candidate-r2.json`：归一化、标量及 softmax。
- `target-vocab-conv-candidate-r1.json`：词表与卷积。
- `target-gdn-candidate-r1.json`：GDN 与已列分项合计。
- `terminal-draft-acceptance-bounds-r1.json`：末轮计数纠正及条件范围。
- `useful-draft-scope-audit-r1.json`：草稿输入范围与源码绑定。

各 JSON 保留输入及源码 SHA256；绝对路径是证据来源标识，不是部署配置。 脚本依赖既有 M01 轨迹和辅助源码，不能仅凭本目录独立复现全量实验。 本批离线复算完整测试 64
项通过，不代替设备补采或 M01 验收。

原始复算脚本及本批相关测试保存在 `derivation-sources.tar.gz`，SHA256：
`955196081af1a0ad14914523712cc774fea5065c44d52cb9211d25423956be11`。 归档保留源码字节，不将格式化后的副本冒充 JSON
中记录的原始脚本。
