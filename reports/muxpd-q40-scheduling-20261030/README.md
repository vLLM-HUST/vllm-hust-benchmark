将 `layers_per_slice` 从 4 改为 40 后，07338 的前 208 步完整 logits 与原版全部逐字节相同，七条 peer
请求的保留前缀也与原版相同。采集与无采集器两次运行的全部八条保留前缀一致，并已通过独立复核。

**这是短窗口调度诊断，不是完整答案修复。原 full200 答案不退化门仍 FAIL，运行时 PR #27 保持 Draft。**

- [结果、调度证据与适用边界](REPORT-v1.md)
- [主分析](analysis-final-v1.json.gz)与[独立复核](independent-pair-v1.json)
- [冻结源码包](source-capsule-v1.tar.gz)与[发布映射](publication-map-v1.json)

原始 NPY、完整调度和采样 JSONL 仍在服务器。下一步是原批次八题跑到自然结束或实际 8192 长度上限，检查完整答案。
