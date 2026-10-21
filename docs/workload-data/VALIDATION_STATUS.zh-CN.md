# 首批接入验证状态

更新：2026-10-22。逐来源进度见[可打勾清单](DATASET_CHECKLIST.zh-CN.md)。

## 已验证

- 软件专项 105 项通过：读取、输入/答案隔离、失败不发布完整索引、模型显式绑定、流式终止、数据-only 边界。软件夹具不是数据集执行证据。
- 安装包对 24 来源各 3 条完成本地读取、准备及原文回查：16 个公开来源、全部 8 份企业数据，共 72 条；此步骤没有模型调用或评分。
- GSM8K 原始 test:0–2 已在双卡 910B2 / Qwen3.5-27B / graph 模式完成固定 lm-eval 0.4.11 生成与评分，3
  条均答对。保留上游提示、五示例和评分；本地同字节文件、chat 传输、2048
  输出预算、禁思考及三条范围分别披露。[脱敏记录](validation/gsm8k-npu-20261022.json)不是全量质量或性能结论。
- pre-commit 格式、私钥及 secret 检查通过。17 个基线新增条目逐项确认是公开来源 SHA/版本号，未放宽扫描范围。

## 未完成与边界

- AgentBench、GAIA、tau2 官方执行器待逐环境接入；QBI 753/165/solo114 仅为历史派生登记，不提供默认回退，本次未重跑。
- 研究端消费接线在独立线程验证；固定提交安装验收和配套 PR 仍须补齐。
- 企业 v1 三来源各前 3 条已在服务器尝试：代码判断 3 条、JSON 第 1 条获得流式终态，但均耗尽原始输出预算，不能证明任务质量；其余 JSON 两条及会话三条因原始输出预算超出当前
  16384
  上下文余量被拒绝。[脱敏尝试记录](validation/enterprise-v1-attempts-20261022.json)不含载荷。没有改消息、流式标志或输出预算来制造通过，也不勾选这些来源。
- 整库 Windows 回归曾得到 1836 passed、258 failed、60 errors、11 skipped；新增清单在并行格式化中出现的匹配问题已修正并专项复测。其余错误包含
  fcntl 缺失、平台脚本和编码问题，未逐项归因，不能声称整库全绿。Linux CI 结果需另核验。
- CPU smoke 不取企业/gated 数据、不加载模型、不做性能测试。目录候选不等于数据已取得、已适配或已完成服务器验证。

## 复现入口

数据和依赖另行取得，输出须使用新建私有目录，不提交原题、答案和模型响应。

```sh
python -m pip install -e . './workload-preparation[data]'
python -m pip install 'lm-eval[api]==0.4.11' 'datasets==3.6.0' 'pyarrow==19.0.1' 'huggingface-hub==1.31.0'
python scripts/run_gsm8k_original_samples.py --data-root /data/gsm8k --endpoint http://127.0.0.1:19466/v1/chat/completions --model workload-qwen35-27b --output /private/gsm8k-new-run
python scripts/run_source_request_samples.py --help
```

GSM8K 原文件名、SHA 固定在脚本；其他来源按各自原生任务契约，不套用其提示或评分。
