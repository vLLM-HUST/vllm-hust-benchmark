# QuALITY 原始问题级读取（后续合批候选）

## 原始来源与计数单位

来源为[nyu-mll/quality](https://github.com/nyu-mll/quality)，固定提交`f84977c40dbfef70c9cab48037b7becfc8e45f73`，当前范围仅发布方`data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev`。
这是发布方提供的HTML-stripped版本，不是本仓清洗出的改题版本。文件8804985字节，SHA256
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`。

原始JSONL一行是一篇文章、一个撰题者的题组，不是一个问题：230行题组包含115篇不同文章、2086个问题；同篇文章的两个题组不算两个独立文档。
每条可用记录对应原始`question_unique_id`，保留完整文章、原问题和四个选项。顺序为原题组顺序及组内问题顺序；另外保存`source_parent_row_index`、`source_question_index`以回查原件。

文件条目的`rows=2086`指展开后的原始问题数，`parent_rows=230`指原JSONL行数，来源单位为`original_questions`。不会把一组所有问题拼成单个虚构模型请求。

## 输入与参考隔离

- 输入仅含`article`、`question`、`options`，不缩短原文、精选证据或改写选项。
- `gold_label`、`writer_label`、`validation`、`speed_validation`在独立参考侧；分组时不把其他题的答案或验证意见复制进输入。
- 原`difficult`、文章ID、题组ID、许可和原行/问题位置是来源元数据，不是实际请求到达顺序、模型耗时、缓存命中或系统性能。
- 按文章保留原许可声明，包含CC BY 4.0、OANC和Project Gutenberg等条款，不宣称统一许可，不在本仓发布文章、原题、参考答案或完整索引数据库。

## 使用

按发布方条款自行取得上述固定文件并放入显式数据目录。安装匹配的数据组件与根CLI：

```sh
workload-data pipeline bind QUALITY --data-root /data/quality --files QuALITY.v1.0.1.htmlstripped.dev --revision f84977c40dbfef70c9cab48037b7becfc8e45f73 --output /data/quality-binding.json
```

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

for record in iter_workload("QUALITY", data_root=Path("/data/quality"), limit=3):
    original_fields = record.task["input"]
    scorer_reference = record.outcome_record()  # 不传给模型
```

读取、绑定计数、pipeline索引和resolve_input使用同一问题视图，不增加生产依赖。索引输出目录必须位于原数据目录之外；完整记录比较需使用相同绑定和limit，不能移除出处字段来掩盖不一致。

## 已验证与未验证

- 原发布方Git blob API取得文件，本地和授权服务器SHA一致。所有2086个问题的ID唯一，每题四选项、gold为1至4的整数，原文非空。
- CPU人工夹具验证题组展开、父子字段碰撞拒绝、位置/ID保留、参考隔离，以及绑定/索引/回查；这些夹具不是原始任务或模型证据。
- 服务器独立安装候选数据wheel（SHA256
  `e7afd01dc8d8d900d663dd9a3109ab1994b2c7ba24bded5bb8d145fb554b0489`），全部2086个原问题逐条与源文章、问题、选项及gold对照并建立索引；原顺序前三篇不同文章的首题（平坦问题索引0、6、15）完整输入/参考回查通过，0模型调用。
- 私有核验脚本先因索引目录位于数据根下被正确拒绝，再因误以为prepared_rows返回task_id而失败。修正核验脚本目录与公开引用接口用法后通过，未修改生产隔离规则或原始数据，两次失败脚本保留。

以上是读取阶段的独立证据，0模型调用不被后续执行追溯改写。

## 原题多选似然执行

当前[原NYU多选集成交付已收口](validation/quality-native-delivery-20261024.json)，不是发布方训练基线复现。新增末尾2083–2085与历史0/6/15共六题、24个原候选似然已完成[原文及评分审计](validation/quality-native-audit-20261024.json)：完整文章/问题/选项重新分词后与保存请求token逐项一致，有限似然经原harness解析及评分结果一致；固定CI组件全2086题索引、六题prepare/resolve通过。审计0新增模型调用。源文件取得和原问题覆盖与第三方harness协议分别声明，不能称为SCROLLS数据或官方完整榜单成绩。

新运行客户端已核对lm-eval[api]0.4.11、Transformers5.5.4、tokenizers0.22.2、numpy1.26.4、datasets4.3.0、pyarrow22.0.0、huggingface-hub1.33.0、tenacity9.2.1、openai3.1.0；32个已安装数据包文件与固定3f397e69
CI
wheel逐字节一致。服务模型/配置见[双NPU环境](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)。下文较早的待验记录保留原阶段身份，不覆盖本次具名回执。

2026-10-24已使用固定3f397e69
CI数据组件重新完成全部2086原题字段核对及建索引、原0/6/15三题完整输入/参考回查，见[固定CI组件回执](validation/quality-fixed-ci-reader-20261024.json)。这是0模型调用的读取/交接证据，不冒充新增原生执行。

原入口默认仍选历史三题；新增可重复`--question-index ORIGINAL_FLAT_INDEX`显式选择原问题，或用`--full`选择整个原dev。保留完整文章、原问题、原选项和固定harness多选似然/评分，不加入聊天模板或截断；重复、越界或full与显式题号并用均拒绝。原生结果记录实际原题号，不能用选项调用数当题数。新增入口已完成末尾三原题NPU验证及审计；历史回执继续固定旧脚本摘要，不改写为新入口执行证据。

```sh
python scripts/run_quality_original_mc.py --data-root /data/quality --output /data/quality-mc-private --tokenizer /models/matching-tokenizer --model MODEL_ALIAS
```

需要固定`lm-eval==0.4.11`环境和匹配现有NPU模型的本地tokenizer。脚本核对原文件、harness源码及预先选定的0、6、15号原问题，不执行SCROLLS数据下载、去重或长度剪枝。复用该harness的QuALITY提示、多选请求和评分方法，不修改上游源码；原文章和问题保持完整，选项仅使用该实现原有的空白归一化。使用raw
completions，不转换chat。

每题对四个原选项分别计算完整续接似然，gold只在评分侧，不参与请求文档。保留原始acc和按选项字符数归一化的acc_norm，不只挑较高指标。该实现的em等于acc_norm乘100，不是自由生成文本精确匹配。

12个选项请求均检查完整输入token数、有限似然和返回条数。接口按原harness使用`echo=True`、`logprobs=1`、`max_tokens=1`，其`length`终态来自附加单token预算，不是将任务答案截成一个token；判据是四个完整选项的似然已取得，而非生成答案是否stop。没有以此协议声称验证自由生成或长输出。

授权双910B2、Qwen3.5-27B、BF16、TP2、graph服务已执行原三题；原始acc为1/3、acc_norm为2/3，原输入约6.1K至6.9K
tokens，未截断。最终公开脚本同三题复验一致，仍是三题、合计24次选项模型调用，不是六个唯一问题。见[原题执行回执](validation/quality-original-mc-npu-20261023.json)。

这是原NYU dev题目与固定第三方harness方法的集成验证，**不是SCROLLS数据集、发布方训练基线复现、官方完整榜单成绩或性能提升证明**。替代模型、选择范围及输入呈现差异均保留。

**剩余范围**：研究端正式新工件消费锁及完整2086题模型执行；固定CI数据组件安装/读取及六题MC集成交付已完成。不将全部读取称作全部模型执行。与其他workload一起合批，未单独新建QuALITY
PR。
