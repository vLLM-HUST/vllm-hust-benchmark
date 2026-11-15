# State Scheduler 研究归档说明

State Scheduler 已撤下性能 MOD 定位。当前动作集停止，代码和负结果归入 Compilation Control 研究材料，不作为当前 OSDI 投稿主线。没有已验证的推理提速，
不建议把它作为 serving 加速组件启用。LatchMoE 和 MoE Control 未作变更。

## 停止依据

- FULL/PIECEWISE 执行偏好：调优捕获预算后，混合长 prefill 和 held-out 到达轨迹 未出现稳定的相反赢家，停止该动作集。
- P1 同步替换：独立输出正确性参考未通过，保留原生同步与负结果。
- P2 准备复用：真实 FX、post-fusion、输入绑定与 AOT 状态审计后，已实现最小 进程内原型；严格同输入数值门槛未通过。一次原生/原生控制也在有限值输出上失败，
  不能归因为缓存错误，也不能据此断言准备复用必然无效。未运行性能 AB/BA， 未通过图重放正确性验收，未获得可采信的加速结果。

仅在获得独立、可信的正确性参考后才考虑重启，不重复同类试验或放宽容差。

## 保留内容与使用边界

[规范源码仓库](https://github.com/Qixin-Gaoke/state-scheduler-plugin) 保留代码、实验脚本、
冻结门槛、反例和完整证据说明；仓库访问仍受原有权限控制。原始 trace 和日志保留在 实验载体的 WPS 目录之外，备份进入 `历史`，没有作为公开下载发布。

[原 0.2.0 发布制品](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/state-scheduler-v0.2.0)
继续保留，wheel、源包、归档和 SHA256SUMS 不变。它们仅包含当时的离线工具和旧共享状态组件， 不包含后续 P2 原型。78 项历史测试、安装和 ECPA 静态发现检查不等于性能验收。
不要把旧包的 M0 状态理解为仍在推进的性能承诺。

代码和负结果归档不等于删除 MOD、删除 GitHub 仓库或更改已安装服务；本次没有执行这些操作。

## English summary

Performance MOD positioning is withdrawn. Current action sets are stopped; code and negative results
are retained as Compilation Control research materials, not an active OSDI submission mainline. No
serving speedup is qualified. P2's strict reference failed for both native/cached and native/native
finite outputs, so cache-specific fault remains unresolved and performance AB/BA was not run.
Reopening requires an independent justified correctness reference. The original 0.2.0 assets remain
unchanged for historical offline inspection; they do not include the later P2 prototype. Do not
activate as a serving optimization.
