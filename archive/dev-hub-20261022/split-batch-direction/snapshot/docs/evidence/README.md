# Evidence custody

本目录的实测产物（JSON 捕获、L0/L1/L3 结果、日志、窗口回执、hazard 归档）
已按 custody 政策迁往冻结归档仓，本目录仅保留 reviewed source 脚本
（探针/矩阵 .py/.sh 与 hazard 源码 vintage，哈希同时被 custody manifest 覆盖）。

- 冻结原件: https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/411e14c5a67dac52b3bba51ac3f25e4c593fcc28/archive/dev-hub-20261022/split-batch-direction/snapshot/docs/evidence
- 内层 manifest（逐文件 bytes/sha256/git_blob，source_commit 03b0fa51627b6f0d580abc991421efdc5070439b）:
  https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/411e14c5a67dac52b3bba51ac3f25e4c593fcc28/archive/dev-hub-20261022/split-batch-direction/manifest.json
- 取件: `python tools/benchmark_evidence.py` 指向同一冻结 commit 时可校验拉取
  （本 MOD 的外层指针仍指向历史批次 67505eee；方向批次的逐文件校验用上链 hash 手工核对）

后续新窗口证据按窗口收尾批量追加到同一归档分支，并在本文件登记批次链接。
