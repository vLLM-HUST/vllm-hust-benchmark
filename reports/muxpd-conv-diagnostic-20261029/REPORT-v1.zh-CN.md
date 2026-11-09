# 首个混合批 GDN 卷积：诊断协议与失败沿革

本轮只考察 native 首次P7+D1混合admission中，首个GDN层的seed
D卷积。它承接[07338晚期窗口诊断](../muxpd-new-loss-batch7-20261028/REPORT-v1.zh-CN.md)：目标第207
token发生非平票分叉，但此前seed和其他peer已分叉，不能唯一归因某个算子。

**两次v2模型运行均通过历史native前缀复现及observed/plain桥；四rank的A/B/C卷积输出和更新后state逐字节一致。** 在本次固定首层、首个seed
D边界，未观察到混合几何或mode0/mode1切换造成的差异。原full200仍为native165/200、旧raw166/200、prefix167/200，prefix恢复旧3个loss但新增07338
loss、11次length，严格答案不退化门FAIL。局部算子对照不能替代完整质量复测或产生新的吞吐/Goodput声明。

## 已完成的失败与加载诊断

| 项目           | 实际状态                     | 证据及边界                                                                                                |
| -------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| v1首个模型诊断 | FAIL；controller终态，exit−9 | 四rank预热后初始snapshot报库身份错误，observer尚未安装、请求尚未入队；root取消后强制清理，最终无owned进程 |
| 首次辅助smoke  | FAIL                         | 导入不存在的`is_custom_op_enabled`失败，发生在设备算子调用前                                              |
| 修订辅助smoke  | PASS                         | 单卡、一次零BF16输入mode0卷积；输出及state为零、返回原output对象；完整maps保存                            |
| v2 CPU准备     | 60项PASS，ruff PASS          | 验证诊断控制与身份检查；不证明NPU数值一致或模型质量                                                       |

v1的四rank在`worker_snapshot → operator_identity`报`Loaded operator library identity differs: libcust_opapi.so`。旧检查将所有相同basename的已加载库合成一个SHA集合，再与唯一预期库比较，未区分不同vendor完整路径。旧进程没有保留下实际maps，因此不能用后续smoke冒充那次进程的完整加载现场。

后续独立smoke实际同时加载：

| 实际库                                                       | SHA256                                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------------ |
| custom_transformer/op_api/lib/libcust_opapi.so               | `08c7121f70c17a293dc7fe887dc3c8d8997e165c37e8b897e1cbe6e171c8fd7b` |
| CANN opp/vendors/batch_invariant/op_api/lib/libcust_opapi.so | `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069` |
| vllm_ascend_C.cpython-312-aarch64-linux-gnu.so               | `ffaee2556caa173ace80ba25b35a0aa5cdd11b6f3985292ff50716bf238449e7` |

同一smoke进程调用原v1身份函数，准确复现相同错误。smoke结果SHA为`9ce08d7e30eb97648d39fd7d3883272b265e70eca9f7346b9f00992bfef9563d`，原始maps
SHA为`287fcbfa9d7ee25bce48803b3ac55feebec8ac7deeb22efdcfb8de0e823f7fa2`。这是身份检查缺陷的直接复现，不是07338退化根因证明。

当前C++
loader先搜索`ASCEND_CUSTOM_OPP_PATH`，再搜索vendors/config.ini配置的默认vendor，最后搜索libopapi；辅助符号查询可能打开不含目标conv符号的默认vendor。当前`load_priority=batch_invariant`，但smoke的`VLLM_BATCH_INVARIANT`为null。出现该库的mapping不代表开启了batch
invariance。源码查找策略也不等于实测了C++缓存函数指针的归属。

v1失败清理随后对未安装的observer调用`conv_finish`，引入第二轮异常；父进程在shutdown期间未及时完成，controller最终强制结束。这些事实保留在v1失败原件中，不能标为正常卸载。v2采用完整四rank普通回执，避免诊断Python异常穿透executor响应队列；对本次shutdown等待的唯一原因不作未经证明的判断。

## v2修复与固定实验边界

v2按规范化完整路径及SHA校验必需库，额外默认vendor、vendor配置和实际loader头分别冻结。完整raw
maps和全部同名候选均保存，未知路径、缺失必需库、字节变化仍失败；未观测实际缓存函数指针。

snapshot/install/finish均返回rank/phase/ok/data/error。父进程收齐四rank后先写checkpoint，再验证。未尝试安装时不调用finish；部分安装按rank清理其拥有的slots；传输异常后停止追加RPC和engine操作，并在shutdown前保存错误。

冻结清单SHA为`b500a57e4d5d64a8e40e2fe206b679020480218c7d51d3e8fa30333069fffb30`。数值observer SHA
`24fa004b01a5f4bc7130bc9d8af5d6723dd616d56e90d5571a1e161d0d04c43f`，plan文件SHA
`d98d808a8aea0fa9e0d8e227762b6f4903551866b364c0780334f4f42e0e845c`，均与v1相同。源码仍为core `d0f22d2b`、Ascend
`03766ac6`、observer插件`6c606f78`；Ascend既有build脚本dirty状态继续严格绑定。

两组固定profile为native-conv-observed与native-conv-plain，Qwen3.5-35B-A3B、BF16、TP4/EP4、同步、FULL_DECODE_ONLY图[1,2,4,8]、HCCL=true、原native混合调度。batch_id7保持07151
seed先产生1个输出，再释放原7个peer；目标07338到208个输出时supervisor
abort，原生成上限仍8192。此次物理分配0,1,2,7，root已核对logical0..3映射；它不是历史的相同物理四卡分配。

观测组只保留首个GDN层seed D的一行输入、完整物理conv
state行、权重和实际输出/更新后state。A为实际native混合调用；排空并恢复hooks后，从相同独立输入/state进行B单序列mode0与C单序列mode1各两次重放。B/C使用算子要求的合法history参数；A↔B先验证几何变化，B↔C再比较模式。A=B=C是有效负结果，不要求实验产生差异。原prefix
graph内部实际输入仍未被此探针观察。

## 两组终态与数值结果

v2观测组于2026-10-29 14:39:30 UTC终态，plain组于14:42:48
UTC终态；两份controller均PASS、exit0、owned进程为空、四卡idle，run均PASS且errors为空。两组各自完整八请求保留前缀均匹配两份历史native对照，observed/plain八请求完整保留前缀也一致。四rank各记录一次capture、四次shadow调用并恢复三个hooks。[较早的观测生命周期快照](observed-v2/lifecycle-summary-v1.json)创建于plain结果可用之前，其中`interpretation_allowed:false`保留创建时语义；最终结论见[完整分析](analysis/analysis-v2.json)。

| 比较                       | 每rank范围                   | 四rank实际结果                        |
| -------------------------- | ---------------------------- | ------------------------------------- |
| A实际mixed与B独立mode0输出 | 2048个BF16元素               | 逐字节相同，差异元素0，max abs/RMS均0 |
| B独立mode0与C独立mode1输出 | 2048个BF16元素               | 逐字节相同，差异元素0，max abs/RMS均0 |
| A↔B、B↔C更新后state        | 完整物理行，3×2048个BF16元素 | 逐字节相同，差异元素0，max abs/RMS均0 |
| 同mode的两次私有重放       | 输出和完整更新后state        | mode0及mode1均逐字节相同              |

实际mixed的x形状为[9533,2048]，stride[3072,1]，无bias，weight形状[4,2048]，activation_mode=1，run_mode=0；四rank
seed物理cache
slot均为3。state来自[3418,3,2048]缓存中的完整一行，未克隆整个缓存。A↔B几何桥和B↔C合法模式对照均通过。分析SHA为`bd4dbbfbc166e66112df846d2f25f9f342fa0f84a8f43b15775de014d8699aa1`，`passed`与`interpretation_allowed`均为true。

这个结果收窄了排查方向：不能再用本次首层、首个D卷积的mode差异解释已观察到的退化。它没有证明后续层/后续token、其他GDN算子或通信路径等价，也没有实际捕获prefix
FULL_DECODE_ONLY图内的conv输入。独立私有mode1重放不能冒充实际prefix图执行。

[独立审计](independent-audit/summary-v2.json)直接读取64对raw/NPY、32对历史完整前缀、8对observed/plain前缀、原始step
history及16份maps/RPC/helper回执，全部通过；没有导入producer或analyzer复用其结论。四rank各保存的180,832字节估计峰值低于16MiB限制。两次process.log均记录executor在grace期后发送SIGTERM，无SIGKILL；controller无`cleanup_required`且exit0。这里的清理门PASS不等于worker全部自然退出。审计摘要SHA为`5fc276d159e2e4acc0815adb91c53c44dfd30a7dbbfaa507e91a164621df4dc3`。

v1失败作为独立旧协议沿革保留，未从v1矩阵挑选成功重试；v2有单独冻结的两次固定运行。原始包的本地构建、独立审计与后续公开可用性分别记录，不能把本地分析或API上传摘要当成匿名下载验证。
