# QReCC 原始测试数据准备与原生检索缺口

## 来源与任务范围

官方来源：[apple-aiml-research/ml-qrecc](https://github.com/apple-aiml-research/ml-qrecc)，固定提交
`1d174464290790c0198b474ad022952ecd83b76b`，原成员 `dataset/qrecc_data.zip`。 归档 SHA-256
`6ed52a0b0e495f752424b15742119a5f8f33ca7a029c07122be1b849de19a308`；test 成员 `qrecc_test.json`
有16451条、17685054字节，SHA-256
`a4d8ef686827feb00c90997e0d6ff135928bf6f4bcb26fb523b5fac025d58f84`。train原成员有63501条，本次准备接口不输出train。

发布方分别列出问题改写、段落检索、阅读理解子任务。`Context` 是发布的会话历史，不是已检索证据库；仅有任务JSON不能宣称原版检索问答执行通过。固定问题和历史必须原样保留，不使用
Rewrite/Answer/Answer_URL 作为待预测任务的输入。

发布方 README 声明数据为 CC BY-SA 3.0，代码为 Apache-2.0，两者不混同。本仓仅提供元数据、准备代码和使用入口，不公开复制原始数据或答案。使用者自行按许可取得归档。

## 准备和读取

安装本批根包和数据组件后：

```sh
workload-data prepare-qrecc-test --data-archive /data/qrecc_data.zip --output /data/qrecc-original-test
```

输出目录必须不存在；准备前校验原始归档与成员哈希，原样复制test成员，不构造题目、不改写JSON、不执行模型、检索或评分。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = iter_workload("QRECC", data_root=Path("/data/qrecc-original-test"), limit=3)
for record in records:
    inputs = record.task["input"]  # 原始 Context、Question
    historical_references = record.outcome_record()  # 不传入模型
```

通用 `workload-data pipeline` 可按同一目录准备来源引用，并用 `resolve_input` 回查原行；准备数据库不包含正文和答案。Context
是历史数据，不伪装为实时工具执行或真实到达时间。

## 当前证据与缺项

### 已保存答案的原作者评分入口（不是端到端运行）

固定版本的 `utils/evaluate_qa.py` 是函数库，没有命令行主程序。新增入口直接调用其原始 `compute_exact` 和 `compute_f1`，不重写指标；要求该文件
SHA-256 为 `f8fe4c1c163004953791983fe141363566b741851f42d7f3fa473646ed672165`。

```bash
python scripts/score_qrecc_original_qa.py \
  --data-file /data/qrecc-original-test/qrecc_test.json \
  --scorer-file /path/to/pinned-ml-qrecc/utils/evaluate_qa.py \
  --predictions-file /private/qrecc-original-predictions.json \
  --output /private/fresh-qrecc-qa-score
```

预测文件是 JSON 数组，每项只能含 `original_row_index`、`Conversation_no`、`Turn_no`、`prediction`； 行号为原 test
中从零开始的位置，其他任务身份必须与原行一致，预测为模型原始文本，不能填入原参考答案充数。 原题在运行前固定抽样，评分入口要求至少三项、原始顺序、无重复，不丢弃错误答案；空预测允许按作者指标计分。
该工具不验证预测生成过程，因此不能凭评分成功宣称原始检索或模型执行完成。原题、作者函数及预测文件身份在评分前后校验；已有输出目录拒绝覆盖。
输出仅含身份、哈希和指标，不复制正文与参考答案。原始检索资产不足仍是端到端阻塞，不使用 Context 或答案URL临时拼接替代检索库。
当前封装与准备回归11项通过；另核验原作者固定函数的三个人工评分夹具（完全匹配、部分匹配、空答案），均不是原始QReCC任务验证。

### 原有数据准备与运行证据

- 原始归档已由官方 Git blob API 取得，并复制至授权服务器，双方归档SHA一致。
- 本地真实test前三行已完成原样读取、参考字段隔离、准备索引及输入回查；没有模型调用。
- 单元测试覆盖未知归档拒绝、成员原样复制、其他成员不输出和已有输出目录拒绝覆盖；这些是人工夹具，不是原始任务执行。
- 服务器独立 Python 3.12 环境已安装候选数据wheel（SHA-256
  `43832634c5678f10a9a9a8e65ec47b091209d8466f8f4f645d3e2281d276bc6d`），`-I`
  确认site-packages导入；原归档准备、原test前三条读取、pipeline索引和resolve_input逐条一致，参考字段保持分离。首次私有核验脚本定位归档目录错误，0模型调用，纠正脚本路径后通过，未放宽数据身份检查。
- 发布方固定README说明端到端任务使用包含1000万网页、5400万 passages
  的集合，并分别提供检索和阅读评估脚本：[`evaluate_retrieval.py`](https://github.com/apple-aiml-research/ml-qrecc/blob/1d174464290790c0198b474ad022952ecd83b76b/utils/evaluate_retrieval.py)、[`evaluate_qa.py`](https://github.com/apple-aiml-research/ml-qrecc/blob/1d174464290790c0198b474ad022952ecd83b76b/utils/evaluate_qa.py)。当前取得的16451行test成员不是该检索集合；仅凭Context、Rewrite或答案字段不能宣称检索/端到端通过。
- 原始检索集合尚未取得，检索运行、阅读链路和发布方评分均未执行，模型调用0次。现有本地与服务器读取/准备回执可复用；不得为填空改用另一套语料或用QReCC的Context伪造检索结果。
- 该能力已在叠加草稿PR #426 的分支中；PR尚未合并。当前可以复用原归档准备、安装读取和三条消费记录，端到端检索与阅读缺口继续保持未完成，不另开单数据集PR。
