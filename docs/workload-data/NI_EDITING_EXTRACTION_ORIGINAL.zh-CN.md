# NI 原始改写与实体抽取任务（下一批候选）

与BBH独立推理放入同一后续批次，覆盖改写和实体抽取；不为每个任务文件单独建PR。

## 来源选择与任务含义

来源为[allenai/natural-instructions](https://github.com/allenai/natural-instructions)，沿用固定提交`55a365637381ce7f3748fa2eac7aef1a113bbb82`：

| 文件                                              | 原始实例数 | 对应任务                                       | 公开声明的实例许可 |
| ------------------------------------------------- | ---------: | ---------------------------------------------- | ------------------ |
| task275_enhanced_wsc_paraphrase_generation.json   |       1178 | 根据原题指定维度改变句子的时态、单复数、语态等 | CC BY 4.0          |
| task1453_person_entity_extraction_btc_corpus.json |        261 | 从原句识别人名，不把Twitter用户名当人名        | CC BY 4.0          |

这是发布方的NI指令任务变体，不是原版WSC/BTC整套benchmark。保持原Definition、input、output和原始行序；output只在参考侧。指令和实例许可分别处理，不由NI指令许可替底层语料重新授权；本仓不发布任何原始实例或答案。

筛选时发现task1288文件名带paraphrasing，但Definition实际要求Yes/No同义判断，不将其登记成改写生成。本批也未选取Instance
License为Unknown的task045或task179；不是凭文件名或凑场景挑选任务。

## 使用与复用

自行按许可取得目录中固定URL和SHA的两个文件。已有NI读取器保留根Definition并分离实例output，支持iter_workload、bind、pipeline及resolve_input；无需构造替代prompt或引入观测逻辑。

```sh
python scripts/run_ni_editing_extraction_original.py --data-root /data/ni-original-tasks --collator-source /data/ni_collator_original.py --scorer-root /data/ni-official-scorer --output /data/ni-editing-run-private
```

沿用[原NI环境与取得说明](NATURAL_INSTRUCTIONS_ORIGINAL_INPUTS.zh-CN.md)：Python3.9、Transformers4.18.0、固定原版DataCollatorForNI与compute_metrics及原GPT2评分资源，模型仅通过本机NPU服务调用。
新入口只配置两个固定任务资产，复用并校验已有运行脚本字节，不改变原脚本默认三任务、预算、官方collator或评分器；选择配置在退出时恢复。默认每个任务前三条；full-small仍严格限制250条，不能静默把这两个文件当全量。

## 证据与未完成项

- 每个文件原始行0、1、2已在Qwen3.5-27B、双910B2、BF16、TP2、graph、非思考配置生成并正常stop，原输入未截断。替代模型及model-native
  chat传输不是论文原模型复现。
- 改写原版函数EM33.3333、ROUGE-L96.9987；实体抽取EM100、ROUGE-L100。指标0–100，ROUGE-L不是正确率，也不证明改写语义全面正确。
- 公开入口最终代码同六条复验结果一致；合计六个唯一NI实例、12次模型调用，不能算12个独立任务。
- 服务器独立安装候选数据wheel，两个文件各三条原输入、参考、索引及resolve回查一致，0模型调用；不是完整1178/261条模型执行。
- [脱敏回执](validation/ni-editing-extraction-original-npu-20261023.json)与此前分类/翻译/摘要回执分开，历史数据不改分母。仍缺最终CI工件复验、研究侧消费和剩余实例，来源级不勾全。

企业原文、真实数据载荷、答案、模型响应、评分资源原件都不进入Git或CI；CI只运行代码与台账的CPU正确性检查。
