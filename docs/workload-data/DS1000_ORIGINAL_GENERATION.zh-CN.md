# DS-1000 原始补全生成探测

## 范围与实际结果

固定
[xlangai/DS-1000 的4416080版本](https://huggingface.co/datasets/xlangai/DS-1000/tree/4416080ac5cb80bdf7576aefb8f9a0b4d5426a44)
test.jsonl，按来源目录核验字节摘要。预选原始行0、1、2，problem_id为0、1、2，均属于Pandas。 这不是1000题或全部库的代表性评测，也不把同三条数据回查再算成新增任务。

2026-10-23，已有双卡Ascend910B2 Qwen3.5-27B服务执行原始prompt补全，未添加聊天包装或缩短题目。

| 原始行/problem_id | prompt tokens | completion tokens | 原始生成终止 | 生成后检测到结束标记 |
| ----------------- | ------------: | ----------------: | ------------ | -------------------- |
| 0                 |           476 |               723 | stop         | 是                   |
| 1                 |           513 |              1024 | length       | 是                   |
| 2                 |           556 |               936 | stop         | 是                   |

三条完整prompt预检均与实际usage一致；原版顺序后处理得到非空候选代码。 非空、存在标记及正常HTTP响应均不证明代码语法/功能正确。
第二题保留length；虽然预算内已有结束标记，也不能在未评分时断言题目已正确完成。

## 原版协议及差异

审阅固定发布方
[b39aab71版本](https://github.com/xlang-ai/DS-1000/tree/b39aab71da6d23ef8d3cac59a7c5f834516ab334)
run_inference.py。原文件摘要核验后，仅通过AST提取静态generation_config和stop_words，
不导入或执行其中HF模型加载、trust_remote_code或CUDA代码。

保留其greedy、max_new_tokens=1024、num_beams=1；结束标记为 `</code>` 和 `# SOLUTION END`。
发布方在完成生成及解码后才按这些标记分割，因此本次API不设置提前stop， 不把“更早停下”改造成更好结果。既有NPU服务以raw completions替代本地HF CUDA batch，
显式add_special_tokens=true、skip_special_tokens=true，并记录这一传输/模型替换。

code_context包含参考和测试逻辑，未送入模型、未被执行；原始文件保持不变。 本次并非完整原版环境或论文结果复现。请求、原输出和后处理候选仅存服务器私有目录。

## 可复用入口

`scripts/run_ds1000_original_generation.py` 仅依赖Python标准库，无需导入或安装发布方CUDA推理栈。
使用者私下取得固定test.jsonl及固定run_inference.py；入口先核验两者摘要，再解析后者的 静态配置，绝不执行该脚本。目录和文件须保留原名，输出目录须不存在且不与数据目录互相包含。

```bash
python scripts/run_ds1000_original_generation.py \
  --data-root /private/ds1000 \
  --publisher-inference /private/ds1000-code/run_inference.py \
  --output /private/ds1000-generation-new \
  --model workload-qwen35-27b-128k-nonthinking \
  --origin http://127.0.0.1:19466 --context-capacity 131072
```

只允许本机HTTP origin，不自动下载、改题或重试。完整原prompt经tokenize预检， 与实际usage核对；code_context从不进入请求。
退出码0仅表示三条生成均正常stop；2单独表示出现原始预算length；1为请求/协议失败。 退出码2不能通过增加预算或给API提前stop来“修好”。
即使退出码0或后处理代码非空，也不证明功能正确；真实评分仍须原版环境和安全隔离。 raw响应与后处理候选只在私有输出目录，不在终端打印代码或参考。

## 清单

2026-10-24[当前服务器隔离预检](validation/code-execution-isolation-preflight-20261024.json)仍不能建立可验证的文件系统隔离，功能执行受阻。user/network
namespace成功不等于代码沙箱可用；未裸执行原code_context或生成代码，也未改预算和结束标记。待具备隔离环境后复用原补全，接发布方test_execution/test_string评分。

- [x] 原始固定文件服务器读取/prepare/resolve三条输入与参考分离回查。
- [x] 原始前三题NPU补全及真实终止记录；原始预算和后处理顺序保留。
- [x] 可复用公开入口、CPU协议测试和服务器同三题交接复跑；仍为stop/length/stop，未增预算或改停止方式。
- [ ] 原版执行环境、隔离、test_execution/test_string及功能评分。
- [ ] 其他库、全量和最终消费端交接。

不裸执行生成代码或code_context，不以当前无可用沙箱为由伪造评分。 本记录是生成阶段探测，不能将三条均写成正常stop、完整任务通过或性能提升。
