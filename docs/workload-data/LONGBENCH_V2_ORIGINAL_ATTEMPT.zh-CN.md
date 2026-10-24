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

“Short”是发布方相对长度标签，不是短问答。本批仅覆盖约2.5万–4.7万token，未完成该来源Medium/Long原生执行。更长范围需合适的原生上下文模型、有效部署配置与足够NPU资源，不能仅靠增加卡数假定当前模型可支持。

2026-10-25按发布方原顺序事前固定Medium前三题（原行3、5、8）与Long前三题（原行0、2、7），不先筛选可运行题。直接执行未改的官方提示构造及完整tokenizer预检：Medium完整输入149662/138653/68762
tokens，前两题超过当前131072窗口；Long为295877/530991/815430
tokens，三题均超限。另保留原128-token输出预算，两组因此均未进入推理。见[长度组容量回执](validation/longbench-v2-length-group-capacity-20261025.json)。旧Short三题不重跑，阻塞题不截断或换成更短题，不将容量检查当原生完成。

## 真实结果与勾选

[执行尝试回执](validation/longbench-v2-original-attempt-npu-20261022.json)：三条原题官方评分0/3；两条正常stop，一条达到原版128输出预算而length终止。三条实际prompt
token均与全输入预检一致。模型答错不是执行器错误；预算终止也不冒充正常输出完成。当前仅证明具名任务/配置的真实执行与评分过程，不宣称模型质量通过、长上下文稳定性/性能通过或完整套件完成。

- [x] 原始数据取得与SHA检查、固定官方模板/评分器核对。
- [x] 完整输入容量预检、超限任务保留。
- [x] 原三条实际执行与原始评分，错误/预算结果保留。
- [x] 三题按原协议执行并进入官方评分；保留两条stop及一条原128预算length终态。length不等于执行器故障，不要求为取得stop重跑或改题/改预算；这一勾不表示模型质量通过。
- [x] 服务器数据组件回查同三条原题，完整context/question一致，answer隔离。[回执](validation/longbench-v2-provider-roundtrip-20261022.json)，该步骤模型调用0。
- [x] 最终公共执行脚本的整体交接复验：恢复服务器后，以同三条原题完成公共脚本NPU执行及官方评分，进程退出0；完整输入token一致，仍为0/3、两条stop及一条length。[公共入口回执](validation/longbench-v2-public-handoff-npu-20261022.json)。复验不增加唯一题数，不把预算终态升级为正常完成。
- [ ] 全503条及Medium/Long范围；不预先承诺当前双卡可运行。

## 复现入口

`scripts/run_longbench_v2_original_samples.py`需原始完整文件、固定官方归档、上述预检JSON、匹配tokenizer目录和本地模型服务；输出必须为新的私有目录。它调用官方函数而非重写评分，路由层只改变部署模型字段，禁止改变原始prompt。

当前入口不再把题数硬编码为三题：容量绑定JSON的`selected`可显式声明1至503条原题，必须原始行号递增、无重复、ID及完整prompt SHA明确，且每条的正整数prompt
tokens加原128输出预算均可容纳于绑定的131072窗口。原公开预检文件仍固定原行1、4、6，不改历史抽样。超限、缺字段或混淆身份会整批拒绝，不自动删除、换题或缩短上下文；一次调试少于三题不构成来源级验证通过。该扩展数量校验只经CPU软件测试，不代表全503题或更新后入口已经过NPU验证。

当前入口仅接受本机 HTTP 服务的 `/v1` 基地址（末尾斜杠可选），拒绝缺少 `/v1`、完整的 `/chat/completions`
路径、凭据和查询参数；参数错误在加载数据和发起模型请求前退出。2026-10-24 的这一参数校验修复仅做服务器 CPU 软件回归，未新增原生题目验证。上述历史交接回执中的 helper SHA
保持为当时实际执行的版本，不回写为当前脚本的 SHA。

### 容量绑定准备：不需要运行模型

`scripts/prepare_longbench_v2_original_capacity.py`直接调用固定版官方`get_pred`构造原始non-CoT提示，在进入`query_llm`之前以内部信号退出；不返回假答案，不进入截断、生成或评分逻辑。再用本地模型的原chat
template计算完整输入token数。选定的原行号在容量检查前固定，必须递增且不重复；没有“自动挑能放下的题”。

在**独立的Python3.12客户端环境**安装，勿混入正在运行的NPU引擎环境：

```sh
python3.12 -m venv /private/longbench-v2-client
/private/longbench-v2-client/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu
/private/longbench-v2-client/bin/python -m pip install \
  -r configs/workload-native/longbench-v2-qwen35-client-py312.txt
/private/longbench-v2-client/bin/python -m pip check
/private/longbench-v2-client/bin/python scripts/prepare_longbench_v2_original_capacity.py \
  --data-file /authorized/LongBench-v2/data.json \
  --official-archive /authorized/longbench-official.zip \
  --tokenizer /authorized/Qwen3.5-27B \
  --row-index 1 --row-index 4 --row-index 6 \
  --output /private/fresh-capacity
```

这里复用历史三题固定行号，不重新选择样本；命令只做CPU准备，不部署或运行CPU模型。需要全部原题的预检时，用`--full`替代`--row-index`参数；它保留所有503题的容量状态，不等于全套原生验证完成。输出目录须不存在，结果为其中的`capacity-selection.private.json`。即使准备进程退出0，也必须查看`ready_for_execution`：超限题仍在`selected`内且标为资源限制，原生执行入口会整批拒绝，不静默裁剪或换题。

服务器已用原文件完成两组CPU预检：原行1/4/6的完整提示SHA及38444/24590/46971
token与历史执行记录完全一致；反例原行0/1/2中0及2超限，三题均保留，原生执行器校验明确拒绝。见[原题容量准备回执](validation/longbench-v2-original-capacity-cpu-20261024.json)。该步骤模型调用和评分均为0，不升级历史三题或完整套件的NPU状态。

已验证客户端为Linux
ARM64、Python3.12.13、PyTorch2.10.0+cpu、Transformers5.5.4、Tokenizers0.22.2、OpenAI3.1.0、Datasets4.3.0、PyArrow22.0.0、NumPy1.26.4；完整67包版本见上述生成快照，独立环境`pip check`通过。该栈用于兼容现有Qwen3.5部署，不宣称复现[上游旧软件栈](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/requirements.txt)或原作者模型基线。准备过程修复了两处环境兼容问题：显式固定PyArrow22以兼容NumPy1.26；显式`return_dict=False`仅固定tokenizer返回表示，不改输入。宿主环境已有的无关OpenCV/Prometheus依赖冲突没有被隐藏为“全局环境干净”，也未为本客户端去修改引擎环境。

### 原生执行和官方评分

```sh
python scripts/run_longbench_v2_original_samples.py \
  --data-file /authorized/LongBench-v2/data.json \
  --official-archive /authorized/longbench-official.zip \
  --capacity-selection docs/workload-data/validation/longbench-v2-capacity-selection-20261022.json \
  --tokenizer /authorized/Qwen3.5-27B \
  --output /private/fresh-run --endpoint http://127.0.0.1:19466/v1
```

上例可直接复用公开的历史三题绑定；使用新准备结果时，将`--capacity-selection`替换为`/private/fresh-capacity/capacity-selection.private.json`。此步骤才会向已部署的本地NPU模型发起真实请求。

公开回执不含问题、答案、上下文或模型回答。上游私有预测文件会截短其保存的context，但不代表模型输入被截断；完整原输入仍由固定文件与SHA引用。此处是执行接入与来源边界，不迁科研观测/机制分析。CI只验证软件参数和事实口径，不跑模型或性能。

服务器恢复后核对了两张910B2、保留的数据目录及无旧验证进程状态，重新启动原模型双卡graph服务并完成上述公共入口复验。任务私有的持续验证队列已部署；它不属于本数据包，也不作为此来源或完整benchmark通过的依据。此前SSH中断和固定批次队列缺陷保留为运维历史，不能据此推断实例回收原因。
