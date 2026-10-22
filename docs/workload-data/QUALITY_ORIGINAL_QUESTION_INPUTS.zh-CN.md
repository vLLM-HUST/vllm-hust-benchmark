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

**未完成**：模型执行、适用评分、公开执行入口、研究消费和最终CI工件复验。当前不勾选来源完成，不将2086行读取或3条回查称作2086/3题执行。将与其他workload一起合批，未单独新建QuALITY
PR。
