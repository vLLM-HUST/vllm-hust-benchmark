# 首次准入分叉：服务端事件诊断

本轮首次准入差异可以由实际进入顺序解释：把各自记录到的准入顺序和完整请求体送到两条原版入口，均精确复现真机首轮活跃成员和剩余预算。未发现本条件下相同顺序导致的准入语义分叉。历史7.92%完整差距及两小时后半程缓存下降的因果仍未查清；本轮不能替代性能等价验收。

## 协议与原件

三组 Native、独立 ThunderAgent 原版、Controller 原版；每组128program、首轮一次调用、time0无客户端slots、每调用64输出、request保护900秒、resume/transport7200秒及0retry。无候选、aging或yield。带对称观察器的诊断，独立冷EMA和每组前相同prefix reset；观察开销未校准。输入为已归档两轮数据的首轮投影，SHA c264d084b5291f48d3341aac298ed1590ef315fe9f5fc261ac0a9e72c5c32a8e。

唯一原包 first-observation-private.tar.gz：9596701字节、113文件、SHA7792eb170cc375af9ca1b5f7215fb0da3d6931b5907b1a85063a726113a607e5；整包及逐文件SHA通过。三个服务各128请求的进入、准入结束、后端body回调结束、响应头/体发送、请求结束及program释放均齐，0drop/0snapshot_error。事件是应用边界，不是引擎逐块缓存事件。

## Controller层：先到达，再按容量准入

| 项目 | ThunderAgent 原版 | Controller 原版 |
|---|---:|---:|
| 首个收到的program | task-004 | task-009 |
| 首个进入准入函数的program | task-004 | task-009 |
| 首次直接准入program数 | 48 | 47 |
| 首次排队时剩余预算（token） | 12642 | 25020 |
| 首个周期检查相对首请求（秒） | 0.5110080242156982 | 3.9544196128845215 |

两组初始活跃成员交集27，对称差41。不同program的请求长度不同，相同容量下按实际顺序装入，会得到不同工作集。本轮预算和成员均能由同顺序CPU语义对照精确复现：两条入口在每一种观测顺序下的完整受控fixture状态轨迹一致。CPU阶段保留完整program_id及请求参数，因为原版估计使用完整payload的JSON字符长度；未将program_id缩成task名称。

## Engine、Program、Request层与限制

本轮三个组各128正常返回、8192输出token、0错误，均释放128program。原模型Ray Job保持RUNNING，终态即时running/waiting/KV均0；新增三个服务和三个GET采集job均已精确STOPPED。模型、两卡、共享设施和旧实验未停止。

| 组 | 带观察器完整耗时（秒） | 正常返回 |
|---|---:|---:|
| native | 238.219784 | 128 |
| thunderagent_original | 237.241871 | 128 |
| controller_original | 235.558012 | 128 |

这些耗时只描述本次首轮观测，不是原两轮工作负载的复跑，也不构成性能等价验收。请求到达顺序、周期检查相位不同且观察开销未校准；不能将Controller本次略快写成“7.92%问题已修复”。

真实事件证明顺序差异在调度之前已经存在；但仍没有把客户端线程调度、网络接收、读取/解析请求体及HTTP包装各自贡献拆开。首个周期检查相位也不同，尚未量化它对整体耗时的作用。此次CPU复算覆盖首轮准入和受控fixture返回，不回放真实模型返回时序、周期timer或缓存。

缺逐请求cached_tokens及逐块evict/touch，因此仍不能把历史后半程缓存miss精确归因给暂停或某个缓存块驱逐。暂停停止条件缺陷及先前候选0.34%的负结果保持。下一步在已采数据上对进入→准入→恢复→返回逐段量化，不新增推理或修改算法。

## 复算入口

`PYTHONPATH=src:. <Python> -m benchmarks.audit_server_events --root <terminal-originals> --output <new-output.json>`

输出 initial-events-audit-r2-private.json；脚本先核对归档成员SHA，再验证事件身份/序号/时钟/请求释放覆盖，最后用相同实际准入顺序在独立原版和Controller原版上进行CPU语义对照。首次临时脚本因缺少仓库根路径而未找到benchmarks的启动失败日志保留；后续明确PYTHONPATH=src:.，未改变产品或远端运行。
