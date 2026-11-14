# BidKV MOD v3：保留默认 CPU 优化并修复共同拓扑适配

**失败源归档，已被v4替代。**
Native04实际模型初始化因copy漏CANN算子资源失败，0生成、未执行C8或正式五项；见[原字节失败receipt](../native04-failure-v1/README.md)。以下是当时冻结的前瞻计划与源码，不代表计划已经完成；helper469不再用于新运行。

B0/B1 都使用同一官方 vLLM0.23 隔离 measurement host、相同 Ascend0.23 package copy、Qwen3.5-35B-A3B BF16
和双卡参数，关闭/启用目标 BidKV 策略。原固定五项样本、顺序、预算、EOS、原评分和统计不变；reference Engine 只是可选输入/评分参考，不定义 B1。

实际官方 CPU binding 默认 True。此前 False C8 和中止01保留；随后 True 0-generation Native03 虽健康且128token
IDs匹配，但两rank都因稀疏ID误报亲和冲突而skip：topo用physical6/7，运行列表已经logical0/1，所需CPU96-119/48-71实际均在allowed0-191内。v3在独立copied
Ascend包只修这一查找，原官方树不动，两臂相同共同补丁、默认True。多chip歧义/冲突安全拒绝，未知未分配行不identity猜测；不改CPU分配、thread
binding、migration或IRQ算法，不手工操作host服务/IRQ。

最终prepare
helper来自[BidKV PR28](https://github.com/vLLM-HUST/vllm-hust-bidkv/pull/28)提交dc3a6a4ddd3cc750393dbaa97d1379565f3f1329，SHA469b4cd103c88f7b1ba9f7bad646ea3a553035cb0700a653c756bb2314d2496d；原CPU源71719e397ab003b31ba5e46e1a6da4da658ae7221eeead04311b65ad4d12d533，prepared源b7a84571c349c22935414b2c5a4c6cf91223414f80c0954f797340ad0f1f55f2。453源/库（8个.so）逐字节核验、原树未改、fresh实际import与真实topo/pool预览通过，7项mapping加14Bridge
CPU检查通过。实际NPU worker/thread affinity、警告与skip仍须由新Native04/Candidate03预检记录，不能把True参数写成有效绑定。

先Native04同配置0-generation准入与128token IDs，再两臂新True
C8；独立复算原始SSE/计数后完整Native五项、BidKV五项、全新Native共同judge。两臂各672
solver，另720计划judge输入。所有输出/cache/TMP/RPC使用新namespace，旧False或未修复True资格不混用。C8原256固定长度合成检查与五项EOS正式推理分开。

生命周期命令：`python -B run_mod_v3.py --out /data/shared_upload/cybber695/bidkv-five-datasets-20261113-01/results/five-datasets-mod-03`。Native04零生成预检由独立supervisor拥有并真实wait；生命周期父wrapper依次wait
C8/正式child，client/API也保存实际returncode。初始化/模型加载/编译阶段10s主机/NPU采样和ready时间单列，正式solver与judge流量分别统计，不把准备、预热或内存当作免费收益。

见`mod-binding.json`、`execution-plan.json`与精确`source-manifest.json`。本16GiB/seq16
profile可能没有抢占；初始化enabled、HTTP成功或0
calls都不证明BidKV机制收益。补充内存压力诊断须单独事前冻结，不代替full五项、不挑最好一轮。原问题、答案、对话、tokenID数组、完整SSE及私有参考代码继续受控保存，公开数值和原件hash索引；`.py.source`均为项目自编代码。
