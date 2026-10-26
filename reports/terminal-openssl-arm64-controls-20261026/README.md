# TerminalBench2.1 OpenSSL ARM64环境与评分控制

原任务为 openssl-selfsigned-cert，任务revision 7131e4375048a0e408a8fb404b5f499d726b695b，Harbor 0.24.0 /
b53b8134e1241686dca7759af188f987ecc48e8b。原指令、测试与oracle不变，仅修改task.toml镜像引用和Dockerfile固定FROM。

| 控制     | 测试通过 / 分母 | 奖励 | 模型调用 | 资源释放 |
| -------- | --------------: | ---: | -------: | -------- |
| 原nop    |             0/6 |    0 |        0 | 已释放   |
| 原oracle |             6/6 |    1 |        0 | 已释放   |

每项各一次、无重试，使用不同全新容器，1CPU、2GiB、cpuset 24-31、MTU1400。原固定python:3.13-slim-bookworm ARM64
manifest为608b27b2db85abf38d81685085cd1166a5ed7ada66549f0dfe53a75bb830cd5d；派生镜像为940a014b319f79b1f83bd91912bea97ba2035b4333fd19a5dd49a83a5b89ec53。原Dockerfile安装步骤实际安装OpenSSL
3.0.22-1~deb12u2。

结论：该ARM64环境上的原评分器能够区分未求解状态与原oracle解。模型调用及模型任务数均为0，不能作为89题模型覆盖、模型准确率或原AMD64镜像等价证明。

## 证据

summary.json为公开聚合结果。independent-review.json为独立复核投影，移除实际容器标识；provenance.json保留原文件摘要。复核覆盖26份封存文件、8份原任务文件、6份实际Harbor源码pin，以及91项锁定安装链与已安装direct_url归档摘要；未声称重新下载并独立复核全部wheel。

受限原始Harbor任务输出、CTRF、奖励、实际镜像及容器资源回读、源文件与安装记录保存在授权CPU持久卷swe-pro-shuhao-20261025/terminal-openssl-arm64-r1。summary中的结果SHA与evidence_manifest_sha256绑定原件；公开材料不包含任务解答或凭据。

```bash
python3 reports/terminal-openssl-arm64-controls-20261026/verify.py
```

此入口离线核验公开摘要与独立复核的一致性，不重建镜像或重新执行控制。
