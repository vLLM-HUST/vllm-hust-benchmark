# Terminal Cython 单项诊断

## 对象与方法

对原 `build-cython-ext` 正控制失败所涉及的唯一内部测试补采完整错误栈。复用原源码提交、144 个 Git blob 和文件模式、原 8 条 oracle 修正及从原日志核定的 25
个依赖版本。资源为 ARM64、1 CPU、2 GiB、MTU1400；准备上限 600 秒，单项测试上限 900 秒。

原容器和已安装 wheel 字节不可恢复。本轮使用新容器和显式 `no-build-isolation` 重建，归档源码的本地基线提交不冒充原 Git 提交对象。原 Git HTTP2 输送失败退出
128 的准备记录保留，其后仅更换为逐 blob 校验的本地源码输送。

## 结果

重建准备退出 0；唯一测试退出 1。调用 `planarity.networkx_graph(pg)` 后，`representation.py:338` 读取节点属性 `pos` 时触发
`KeyError`；实际属性为 `vertex_position`、`vertex_start`、`vertex_end`。

这确定了本轮重建中的属性接口不匹配，不等同于恢复原已终止容器的完整错误栈，也不证明某次历史升级或 ARM64 架构是原因。原 nop 1/11、oracle 10/11、reward 0
不变；本任务正控制仍未通过。新增 Agent、模型调用和控制成绩均为 0，容器已释放。

## 证据

`summary.json` 保存结果、配置和受限原件哈希；`independent-review.json` 为独立复核回执。题目、oracle 和完整测试正文不公开。

`python3 verify.py` 离线检查公开文件哈希与语义一致性。原始诊断的重新派生入口是 `derive.py`，需要受控 CPU 工作区原件，不属于无私有证据的复现实验。
