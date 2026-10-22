# BBH 原始任务读取与执行边界（下一批候选）

本能力处于下一批本地开发分支，不改变已经通过CI的首批和文本扩展批次，不单独创建BBH PR。

## 原始来源与可用资产

发布方为[suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)，固定提交`9ee07bd481feebf959a6b59d61ea57bdcf30964d`。
发布论文描述23项任务；原仓库27个任务JSON文件共6511条实例含不同配置，不能据文件数量新增27类workload。
当前固定准备和读取范围只有`boolean_expressions`文件250条，以及单独的原始CoT提示文件；其他任务尚未登记成可用读取资产。
原归档和两个成员校验和固定，不改原始题目或示例；原题`target`只保留在参考侧，不进入模型输入或来源索引。保留发布方MIT仓库声明和上游归属；本仓不重新分发原始题目、提示示例、答案或私有响应。

## 使用链路

安装本批匹配的根CLI和数据组件，按发布方条款自行取得固定原归档：

```sh
workload-data prepare-bbh-boolean --data-archive /data/bbh-original.zip --output /data/bbh-original
```

输出目录必须不存在。准备接口验证原归档、唯一成员和SHA，原样复制任务JSON与提示文件，0模型调用、0评分，不解压无关文件。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

for record in iter_workload("BBH", data_root=Path("/data/bbh-original"), limit=3):
    original_input = record.task["input"]  # 只有原始input
    scorer_reference = record.outcome_record()  # target，不传给模型
```

支持通用bind、pipeline和resolve_input。原任务没有独立ID，使用文件校验和与原行序号组成身份，不编造到达时间；完整记录比较时应使用同一绑定和相同limit等选择契约，不能删除出处或选择字段来制造相等。

## 分层验证记录

- CPU夹具覆盖原字段、非空文本检查、参考隔离、错误归档/成员拒绝、原字节复制、已有目录拒绝覆盖，以及显式绑定的索引回查。这些人工夹具不是BBH任务证据。
- 服务器已独立安装本地候选数据wheel，SHA256
  `55c77277f73c1e3f757bfc66ac08be84966cbfb904f03ada4aa2f4a1b99b3ce8`。全部250行逐条对照原始input/target，前三条bind→pipeline→resolve回查通过，0模型调用。候选不是CI工件，未与研究侧联调。
- 另一次模型探测在已有lm-eval
  0.4.11环境中，使用发布方原始三示例CoT提示与原任务索引0、1、2，已有Qwen3.5-27B、双910B2、BF16、TP2、graph、非思考配置。模型原生chat传输不是论文原模型设置。
- 模型探测三条均为完整输入，实际/预检均532 tokens；每条输出48
  tokens。三条均`finish_reason=stop`，实际`stop_reason`均为双换行。固定harness的双换行停止规则使模型在第一段说明后结束，没有最终答案；其原regex与exact_match指标均为0。正常停止不等于已完成作答。
- 私有执行回执SHA256
  `284e07c8f6398be081b84598e14a6cccf73cc17c76c6b3d44fc58393bd4c72a9`；原提示和任务身份不变。核验脚本首次使用不存在的record_id属性、第二次混比全文件读取与limit=3索引记录，两次失败脚本保留；统一绑定和limit后通过，没有放宽原字段检查。

lm-eval提供参考评测实现，不是发布方原始评分源码；其BBH说明也未确认论文方法等价性。因此不称本次为官方全套成绩、论文复现、全量模型验证或性能提升。保持0分结果，不删题、不改停止词或提取规则来获取通过。若后续验证另一种停止/传输设置，必须单独登记为不同运行协议，不能替换本次结果。

## 可复用执行入口

在固定`lm-eval==0.4.11`环境中使用本批脚本；脚本验证原任务/提示、harness配方和评分模块校验和，不修改停止词或提取规则：

```sh
python scripts/run_bbh_boolean_original.py --data-root /data/bbh-original --output /data/bbh-run-private --endpoint http://127.0.0.1:19466/v1 --model MODEL_ALIAS
```

只允许无凭据的本机HTTP接口，输出目录必须不存在；原始请求、响应及参考答案仅写入此私有目录。默认固定前三题，不隐式启动模型、下载数据或执行生成代码。复用现有NPU模型服务，未迁入观测钩子或研究特征分析。

根候选wheel与数据候选wheel安装后的`prepare-bbh-boolean`已在服务器通过；公开脚本同三题复验仍为0/3、输入token一致、双换行提前停止，重复运行不增加唯一题数。见[脱敏记录](validation/bbh-boolean-original-npu-20261023.json)。脚本正确性夹具加入CPU
smoke，不在CI跑真实模型。

尚缺：研究消费联调、最终CI工件复验、其他BBH任务，以及完整250题模型执行。上述缺项补齐前不勾选来源级完整接入；几类相关workload合批提交后再以固定CI工件核验。
