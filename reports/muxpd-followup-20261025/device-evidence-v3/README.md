# MuxPD TP4 终态证据追加 v3

本批固定接纳截至 source98 Manager group 结束的 **9 个新终态实验**，并增加一个显式 support 包。 旧 v1/v2 与质量协议原字节保持不变。22-block
取消实验及性能计时、profiler 结果均不在本批范围。

[完整报告](REPORT-v3.md)（另保留[执行端原字节](originals/PRELIMINARY-REPORT-v3.md.source)）记录结论和限制：

- f7 普通 worker、source98 普通 worker，以及此前带观察器的对应结果，在冻结输入上的六项对比均为 30/30 条八 token 序列一致；这批普通 worker
  运行不保存完整 logits。
- raw/group 两个 Manager source bridge 均完成启动、HTTP 128/8 请求和有限关闭检查，实际退出码 **130** 原样保留。原生
  SIGTERM、abort/force-kill 和资源跟踪警告仍在原始日志中；不宣称完全优雅关闭或 wheel 资格。
- 2048 输出预算的原生尝试保持失败：终态仅 39/200 条输出，其中 3 条截断。8192 预算的原生控制1 完成 200 条，165 条正确、191 条 stop、9 条
  length；截断判错，严格无截断条件仍失败。 仅有一个原生控制，**没有插件成对质量验收结论**。
- raw 压力套件整体保持 **FAIL**。阶段1观察到 64 次分配失败和一次 P 持有块期间的 D 抢占； 阶段2没有遇到规定的双 peer P8 边界，未发起 abort，阶段3未运行。

这些证据支持 #11 在明确配置和冻结输入范围内的调查处置，不证明任意输入等价。 #23 的成对质量检查和 #3 / PR #10 的当前性能验证仍未完成。**本批没有吞吐、goodput 或分组 P
加速结论。**

## 原件与复验

[归档索引](archive-index-v3.json)包含 10 个包、786 个外层原件成员，压缩包共 65,501,327 bytes。 源码98的固定提交树另有嵌套成员清单；原始源码
tar、完整运行文件及编译缓存位于 capsule，不复制进 Git。
[明确终态选择](terminal-selection-v3.json)、[support 选择](support-selection-v3.json)和
[源码选择](source-selection-v3.json)定义本批边界。

[输入独立复验](analysis/device-input-reverification-v3.json)重新核验七个普通终态包、逐字节重现六项 token 对比，
并重新计算单轮原生质量分数。[Manager 归档复验](analysis/manager-archive-reverification-v3.json) 独立检查两个专用类型包及当前完整成员集合。
[Manager builder 冻结身份](analysis/manager-capsule-builder-freeze-v1.json)和
[无写入验证](analysis/manager-validation-no-write-v1.json)保留专用验证器证据。

早期 Manager 验证器曾生成两个新的 bytecode cache，已按
[副作用回执](analysis/manager-validator-bytecode-side-effect-v1.json)保留于 support 并移出运行目录；
原有文件没有改变。最终验证器在校验前后保持完整成员集合一致。不能据此描述成“终态后目录从未被触碰”。

[单轮原生描述](analysis/native-control-01-description-v1.json)与
[压力失败诊断原文](originals/PRESSURE-SUITE-DIAGNOSIS.md.source)可分别复核质量与生命周期限制。 早期 stop-decision、23:25
的补充协议快照和终态结果有不同的时间与计数，均保持各自身份。

## 发布状态

**16 个新增资产已于 2026-10-26 00:47:19 UTC 完成独立匿名下载回验**：大小与 SHA256 一致，10 个 capsule 的 786 个外层成员及 source98
嵌套源码的 99 个文件均通过核验。 [公开回验说明](PUBLIC-READBACK-v3.md)保留两次失败尝试与第三次完成回执；39 资产 API 清单前后稳定，旧 23 资产仅比较 API
身份，本次未重新下载其全部字节。

[资产准备清单](release-assets-v3.json)、[上传选择](upload-selection-v3.json)和原报告仍保留上传前的身份与状态。已有21资产 v2
回验和后续2资产协议回验保持原时间快照；本次事实通过新回执追加。

```sh
python reports/muxpd-followup-20261025/device-evidence-v3/verify_device_v3.py
python reports/muxpd-followup-20261025/tools/build_capsule.py verify \
  --archive /downloaded/capsule.tar.gz \
  --manifest reports/muxpd-followup-20261025/device-evidence-v3/custody/CAPSULE.manifest.json \
  --archive-sha256 EXPECTED_SHA256
```

上述工具只验证已有字节、清单和有限结论，不运行模型或归档代码，不访问网络，不等同于硬件、质量或性能通过。 所有执行端绝对路径和 producer SHA
保留在[复制清单](canonical-copy-map.json)中。
