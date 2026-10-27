# TerminalBench 2.1 break-filter 环境控制

本目录仅包含原 `break-filter-js-from-html` 的 ARM64 派生环境控制，不是模型任务成绩，不声称原 AMD64 镜像等价。

原 r1 构建成功，但挂载检查误拒原 Harbor 三个日志挂载和一个依赖挂载。nop 退出 130、reward=null、无 CTRF，oracle 未运行；原件保留。

r2 复用同一已构建镜像，仅修精确挂载准入。两个独立新鲜容器的结果：

| 控制   | 原测试          | reward | 判定           |
| ------ | --------------- | -----: | -------------- |
| nop    | 0 通过 / 1 失败 |      0 | 负控制符合预期 |
| oracle | 0 通过 / 1 失败 |      0 | 正控制未通过   |

oracle 测试确实执行，在原 5 秒 alert 等待后断言失败；不是挂载问题或测试未执行。原任务、测试、oracle、1200 秒 verifier 预算和阈值不变。原临时过滤 HTML
与浏览器 DOM 未保留，因此现有证据不能判定 parser、浏览器、时序或 ARM64 原因。原 CPU preflight 路径读取失败也保留，不称其成功。

两个容器均为 1 CPU / 2 GiB、cpuset 24-31、MTU1400，三个日志 RW 挂载及一个依赖 RO 挂载身份已核，现均释放。模型调用与新增 Agent 为 0；Terminal
模型覆盖仍为 3/89，86 题未执行，不能把本次正控制失败记作模型 0/1。

`summary.json` 绑定原 77 文件封存；`independent-review.json` 为独立审计的受限字段投影。`storage-summary.json` 与
`storage-whitelist.json` 保留 storage 原公开白名单字节。公开文件不包含题答、oracle 正文、过滤输入、原始日志或请求标识。

离线复核：

```sh
python3 reports/terminal-break-filter-controls-20261027/verify.py
```

复核只读取已归档聚合，不重新执行控制或诊断。
