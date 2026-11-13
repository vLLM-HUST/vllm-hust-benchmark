# BidKV：同基础引擎关闭/启用 MOD 的 True 新配置

本v2仅完成Native03的0生成健康/128 token
IDs准入，实际两rank默认绑定均warning/skip，未执行C8或正式五项。已经定位官方稀疏物理/逻辑卡ID映射bug，原件保留；后续使用两臂共同隔离兼容修复的新v3，不用True参数存在冒充优化生效。

本目录冻结 MOD 专用提示词 SHA `43bc925a6ad749971a64e5244b86451f5d4cf3361e87423b1eda8ea0b0d04461`
下的两臂定义、实际启动模板、C8 与五项执行源。B0/B1 都是相同官方 0.23 SDK、同一隔离 prepared scheduler、相同 Qwen3.5-35B-A3B BF16
和双卡配置，仅关闭/启用 BidKV 策略；不是把参考 Engine 当作 B1。代码接入是 Draft PR28 的实际实验载体，不能宣称已发布通用兼容。

在实际安装源核实 `enable_cpu_binding` 官方默认 True 后，两臂明确恢复 True；此前 False
是共同人为配置，未找到必须关闭的依据。其它已冻结模型/预算/任务/seed/MTP/APC/graph/KV16GiB/image1 不变。旧 False C8 与中止01原件保留且不混作新资格。新
True 的模型、图编译/初始化、真实绑定尝试及每 worker/线程 CPU affinity、低频主机/NPU采样在生成前封存。发生 warning/skip 如实保留并定位，不预先关闭默认优化。

先执行当前 Native True 0-generation 预检，再分别新 C8，完整复核 C8 原始 SSE/时钟/usage/零复用后执行 Native 全五、BidKV 全五、全新
Native 共同 judge。两臂各672 solver，另720共同 judge计划输入，原样本/顺序/预算/EOS/原评分器不变。原 C8 固定长度合成资格仍与保留 EOS 的五正式数据集分开。

生命周期命令为
`python -B run_mod_v2.py --out /data/shared_upload/cybber695/bidkv-five-datasets-20261113-01/results/five-datasets-mod-02`。所有子实验使用新输出、临时目录与缓存，父
wrapper wait 各实际子进程记录真实返回码。已先行 detached 的 Native True 零生成预检控制进程无法事后 wait，OS退出码标未捕获；其实际 owned API
退出码仍由预检 wait 记录，不为补包装重载模型。

`mod-binding.json` 单列共享测量桥接、额外参数、源码与实际加载/调用/触发证据层次。此 KV16GiB/seq16 配置可能无抢占；0 calls 不能宣称 BidKV
机制收益。任何内存压力探针单独预注册，不能替代完整五项。用原任务金标准和原评分器判断质量，不把 token 相同当作答案正确，也不把 token 不同当作质量下降。

初始化/编译/数据准备、主机和设备内存均单列，judge 流量不计 solver 吞吐或成本；原问题、答案、对话、完整 SSE 和私有 Engine
源码保持受控，仅公开可公开配置、数值和原件哈希索引。全部 `.py.source` 为项目自行编写，使用时去掉末尾 `.source`；执行仍需合法访问原冻结输入与评分工具。
