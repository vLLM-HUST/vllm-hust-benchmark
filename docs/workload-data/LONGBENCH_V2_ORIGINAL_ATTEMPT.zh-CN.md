# 长上下文：LongBench v2 原始任务接入与当前限制

状态：已有原始任务执行及官方评分尝试，当前不勾来源验证完成或模型质量合格；不删除0分，也不按答案换题。

## 原始来源与执行

固定数据THUDM/LongBench-v2 revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`，503条；固定THUDM/LongBench源码
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`。完整数据和源码归档均核对SHA。

使用上游`pred.py`的`get_pred`、0shot模板、`extract_answer`及`pred == answer`判定，非CoT、无RAG替代、无无上下文改题。只绑定已有本地tokenizer、128K部署配置与模型API路由。模型仍为双卡910B2上的Qwen3.5-27B，不能等同其他参考模型表现。

上游`query_llm`默认允许头尾截断。本接入先预检完整官方提示和128输出预算，且API路由核对实际prompt SHA与服务端prompt
token；有截断或模板变化即拒绝调用。因此本批没有截断上下文。128输出预算、温度0.1保留上游非CoT设置，未为跑通加预算。

## 容量预检与采样边界

规则在模型生成前固定：按原始顺序，取完整提示加128输出能落入131072窗口的前三条；不检查答案或模型结果。[预检记录](validation/longbench-v2-capacity-selection-20261022.json)保留所有被检查但未选中的任务。

| 原行号 | 完整提示token | 状态                  |
| ------ | ------------: | --------------------- |
| 0      |        295877 | 资源/窗口受限，未截断 |
| 1      |         38444 | 选中，原始Short标签   |
| 2      |        530991 | 资源/窗口受限，未截断 |
| 3      |        149662 | 资源/窗口受限，未截断 |
| 4      |         24590 | 选中，原始Short标签   |
| 5      |        138653 | 资源/窗口受限，未截断 |
| 6      |         46971 | 选中，原始Short标签   |

“Short”是发布方相对长度标签，不是短问答。本批仅覆盖约2.5万–4.7万token，未覆盖该来源Medium/Long。更长范围需合适的原生上下文模型、有效部署配置与足够NPU资源，不能仅靠增加卡数假定当前模型可支持。

## 真实结果与勾选

[执行尝试回执](validation/longbench-v2-original-attempt-npu-20261022.json)：三条原题官方评分0/3；两条正常stop，一条达到原版128输出预算而length终止。三条实际prompt
token均与全输入预检一致。模型答错不是执行器错误；预算终止也不冒充正常输出完成。当前仅证明具名任务/配置的真实执行与评分过程，不宣称模型质量通过、长上下文稳定性/性能通过或完整套件完成。

- [x] 原始数据取得与SHA检查、固定官方模板/评分器核对。
- [x] 完整输入容量预检、超限任务保留。
- [x] 原三条实际执行与原始评分，错误/预算结果保留。
- [ ] 三条全部正常输出终态；保留原始结果，不为满足这一勾静默改题/改预算。
- [x] 服务器数据组件回查同三条原题，完整context/question一致，answer隔离。[回执](validation/longbench-v2-provider-roundtrip-20261022.json)，该步骤模型调用0。
- [x] 最终公共执行脚本的整体交接复验：恢复服务器后，以同三条原题完成公共脚本NPU执行及官方评分，进程退出0；完整输入token一致，仍为0/3、两条stop及一条length。[公共入口回执](validation/longbench-v2-public-handoff-npu-20261022.json)。复验不增加唯一题数，不把预算终态升级为正常完成。
- [ ] 全503条及Medium/Long范围；不预先承诺当前双卡可运行。

## 复现入口

`scripts/run_longbench_v2_original_samples.py`需原始完整文件、固定官方归档、上述预检JSON、匹配tokenizer目录和本地模型服务；输出必须为新的私有目录。它调用官方函数而非重写评分，路由层只改变部署模型字段，禁止改变原始prompt。

```sh
python scripts/run_longbench_v2_original_samples.py \
  --data-file /authorized/LongBench-v2/data.json \
  --official-archive /authorized/longbench-official.zip \
  --capacity-selection docs/workload-data/validation/longbench-v2-capacity-selection-20261022.json \
  --tokenizer /authorized/Qwen3.5-27B \
  --output /private/fresh-run --endpoint http://127.0.0.1:19466/v1
```

公开回执不含问题、答案、上下文或模型回答。上游私有预测文件会截短其保存的context，但不代表模型输入被截断；完整原输入仍由固定文件与SHA引用。此处是执行接入与来源边界，不迁科研观测/机制分析。CI只验证软件参数和事实口径，不跑模型或性能。

服务器恢复后核对了两张910B2、保留的数据目录及无旧验证进程状态，重新启动原模型双卡graph服务并完成上述公共入口复验。任务私有的持续验证队列已部署；它不属于本数据包，也不作为此来源或完整benchmark通过的依据。此前SSH中断和固定批次队列缺陷保留为运维历史，不能据此推断实例回收原因。
