# TerminalBench2.1 cancel-async-tasks ARM64评分控制

原任务 revision `7131e4375048a0e408a8fb404b5f499d726b695b`，Harbor 0.24.0。 仅更改 Dockerfile 固定 ARM64
基镜像引用及任务镜像名；原指令、测试、oracle、六项时间阈值保持不变。 资源为 1 CPU、2 GiB、CPU 24–31、MTU1400，原评分预算900秒。

| 控制      | 测试通过 | reward | 模型调用 |
| --------- | -------: | -----: | -------: |
| 原 nop    |      0/6 |      0 |        0 |
| 原 oracle |      6/6 |      1 |        0 |

各使用不同新容器、各一次，均已释放。固定官方 uv0.9.5 安装器及 ARM64 包只通过只读传输缓存提供；原测试未修改。 原 shell env 文件缺失警告保留，随后 uvx
和原六项测试实际执行。 首次隔离配置缺少 Docker Compose 插件，在容器及测试启动前失败，原 reward=null 保留；不计任务负结果。

本结果只证明该题环境与评分控制通过，模型任务数为0，不替代既有 OpenSSL 的1/89模型实测，也不是全量或B0/B1配对验收。

## 证据

`summary.json` 与 `independent-review.json` 为公开聚合；68项原件封存的受控索引和哈希见 `restricted-evidence-index.json`。
未公开任务答案、oracle脚本或原始运行内容。离线运行 `python3 verify.py` 可复核公开聚合及绑定，不代替受控原件审计。
