MuxPD：有序归约与首次混合批次实验合并报告（2026-10-29）

**结论：本轮没有修复目标题 mmlu-pro-07338 的输出退化，不应把这些实验作为运行时修复合入。质量门仍为 FAIL；PR #27 保持 Draft。** 已证实在指定 GDN0
解码位置执行 BF16 有序归约能够改变实际模型输出，并推迟 seed 题的一处早期分岔；它没有恢复完整答案，也没有恢复目标题的第207/208个
token。本轮没有性能测量，不能据此声称吞吐量、goodput、时延或显存收益。后续仍需真正的运行时修正、旧退化题及完整200题逐题评测。

本报告汇总六次实际设备尝试及其原始失败、成功和后处理记录。每次尝试均已终态，所属进程集合为空；现有控制器记录未显示强制终止或控制器代为清理。包内保留每次尝试的
controller、process.log、producer、输入/计划/配置、run.json、全部 checkpoints、RPC 记录以及实际导出的
raw/NPY；没有按成败筛掉任何一次尝试。报告中的 PASS 均指相应实验协议或证据完整性通过，不等同于答案质量通过。

| 阶段                  | 唯一实际尝试                    | 结果与边界                                                                                                                                                   |
| --------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ordered-private v1    | ordered-decode-private-v1-01    | FAIL：第9项私有图 capture_begin 在进入算子体之前失败。前8项 eager 已返回，但最后统一保存未执行，没有可用于数值结论的导出张量。第10–14项未执行。              |
| ordered-private v2    | ordered-decode-private-v2-01    | PASS：统一私有 RPC 的 inference_mode 作用域后，原固定14项完整执行。180对 raw/NPY、304次逐位比较及生命周期独审通过；capture本身不作为数值输出。               |
| B v1 plain            | first-mixed-plain-v1-01         | FAIL：构造兼容层仍绑定旧诊断 worker SHA，加载模型前即拒绝。没有请求、推理、capture或数值输出；后两臂未启动。                                                 |
| B v2 plain            | first-mixed-plain-v1-01         | PASS：仅修正上述身份常量。完整8条保留前缀与历史 prefix 对照一致。                                                                                            |
| B v2 matched-false    | first-mixed-matched-false-v1-01 | 设备 PASS；原主分析 FAIL；严格修正分析 schema 后重新分析 PASS。模型 flag 始终 false，保留前缀与 fresh plain 和历史对照一致；排空后固定私有6项通过。          |
| B v2 first-mixed-true | first-mixed-true-v1-01          | 协议/数值 PASS，目标修复失败。仅首次混合批次指定 GDN0 解码输出选择有序结果，随后恢复 false。通过明确的后处理续门补充启动，原模型与38个 child producer 未改。 |

所有设备尝试均发生在 TP=4、物理卡2/3/4/6上。本包没有新增 TP=8 或性能实验。

**归约证据。** 同一组四rank输入、原TP通信组和解码stream上的对照表明，BF16固定顺序 `(r0+r1)+r2+r3` 的逐步舍入结果为 A；另一个实际归约输出为 B。A 与 B
的4096字节输出SHA分别为：

- A：`224637e71cbb1d40810013da48c7ec9d0ad6f70e00a66d43d1b56732a62f4d57`
- B：`8a918306108fa1529a1ed021460bc2792d25c952ce60f910e7207839140ad38a`

ordered-private v2 的固定矩阵保留 eager B/C、0123/0132各两次，以及两个私有图各一次capture、两次replay，共14项。独立整数 BF16
round-to-nearest-even 参考逐步核验中间值，未用一次高精度总和代替中间舍入。B/0132得到B，C/0123得到A。这个事实没有识别 HCCL 内部具体采用哪一种算法。

B 的 shared combined 函数每次先执行原 native all-reduce，再 all-gather 原始本地行、物化三个 BF16 中间和，并由设备bool选择 native
或0123结果。warmup、模型和排空后私有测试使用独立storage；warmup和模型引用保留至进程退出。matched-false 的私有6项是 eager
false/true、capture false、replay
false/true/false。capture输出不用于数值判定，重放采用固定地址并验证哨兵覆盖；清理在同步之后，失败同步或reset为不可重试的粘滞失败。

B三臂独立审计共核验2258份证据、256对 raw/NPY（false 220对、true 36对）、149个checkpoint和56份私有RPC模式证明。四rank true
的实际bool均为true；local/native/gathered及三个逐步和与false对照一致，selected/output均逐位为A，native仍为B。这证明指定位置的干预实际发生，不能解释为flag未生效。

**对实际token的影响。** seed07151保留218个token，其余七题各保留208个token。这些不是完整最终答案。下表的token位置从1开始；历史native参照是两份已有
native-quality-v2-hccl-true 结果。

| 题目              | true相对本轮plain/false                             | true相对旧native                                                                             |
| ----------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 07151             | 第56个token首次变化：4526→426；218个位置中160处不同 | 前151个token一致，第152个再次分岔：native28051、true6505。早期分岔被推迟，没有恢复完整序列。 |
| 07213             | 全208个token一致                                    | 全208个token一致                                                                             |
| 07275             | 全208个token一致                                    | 全208个token一致                                                                             |
| **07338（目标）** | **全208个token一致**                                | **前206个一致；第207/208仍为 `[198,262]`，两份native均为 `[318,2777]`。原分岔未修复。**      |
| 07401             | 全208个token一致                                    | 全208个token一致                                                                             |
| 07461             | 全208个token一致                                    | 第203个起的6处差异保留                                                                       |
| 07524             | 全208个token一致                                    | 全208个token一致                                                                             |
| 07585             | 全208个token一致                                    | 全208个token一致                                                                             |

直接结论是：只改变首次混合批次中seed的这一层归约，不足以消除07338的目标分岔。它也不能证明对所有层或所有解码步推广该方法就能修复答案。原始完整逐位置比较见 B v2 的
`independent-result-audit-v1/prefix-native-comparison-v2.json`。

旧native数值参考最初取自物理卡0/1/2/7，本轮为2/3/4/6。已有固定输入、字节身份及新同组对照支持本报告的局部数值比较；跨分配的历史比较不能单独证明硬件或某个通信算法是唯一原因。

**失败和分析修正没有被抹去。** ordered-private v1的原始失败和零导出状态完整保留。v2仅修正私有RPC推理模式上下文，原数学runtime保持原字节。B v1构造失败完整保留，B
v2仅将兼容层硬编码worker SHA改为实际worker SHA，并加入实际磁盘绑定回归检查。

B v2 matched-false设备已成功后，原分析器在比较函数身份时漏写期望值中的 `unwrapped:false` 与完整
`wrapper_source`，把实际有效的七字段身份误判为失败。原 `analysis-matched-false-v1.json` 和独立
`summary-matched-false-v1.json` 的 FAIL 均原样保留；没有生成v1 false
release，也没有重跑已成功的模型。新外部分析器只增加5行严格期望字段，完整检查原函数对象、路径与SHA，生成新的v2分析和独立release。

true启动使用外部 `mixed_continuation_control_v2.py` 和冻结的
`continuation-manifest-v2.json`。相对原矩阵父进程argv，仅索引1、20、22改变：父控制器路径，以及false主分析、独立release两条路径。原child命令、38个producer、输入、计划、卡分配及原freeze保持不变。原先的主分析与独审成功双门继续有效。**这是事后明确记录的协议补充，不是原矩阵父argv的精确执行，也不是事前已修正的矩阵。**
原root launcher、矩阵和失败分析全部保留。

关键证据身份：

| 文件                                         | SHA-256                                                            |
| -------------------------------------------- | ------------------------------------------------------------------ |
| ordered-private v2独审 `summary-v2.json`     | `b1b276499b389d9eaa6d6802162b359df18c8f1d7e65f90ba3d386877de4a2bb` |
| B v2原freeze                                 | `784f87f59486ff9aae8dfea66d4f79b8f996cb53664f207e085312dd974a085c` |
| B v2 false新独立release                      | `179077065fecf4768b1f81532a60484f79db25380406ca5c11da78006c5bfdec` |
| true续门manifest                             | `2bc7f108030b9851fc9968fd30fb581d8526ff752929b5a238ea590e19a7e070` |
| B三臂独审 `summary-first-mixed-true-v2.json` | `7339d3f67b44ef54255970b9cbfb5c7d692d540e700a3bdae8d99cfb70dfe74a` |
| 完整8题token比较                             | `598c23774935aaf0b2264b8b53444a2d28d3ee3941db9c36eca219ca65287d33` |

**证据包范围和复现。** `source-manifest-v1.json`逐文件记录归档路径、原路径、字节数和SHA，包内同字节副本为
`MANIFEST.json`。六个来源目录包括ordered-private v1/v2、B v1/v2、后处理补充和B设计。全部实际attempt一并纳入；只排除
`compiler-cache`、`__pycache__`、`.pytest_cache`、`.ruff_cache` 和
`.git`，符号链接或非普通文件会使打包失败。报告、脚本、测试、红绿日志和外部引用索引随包保存。

更早的graph/native/full-quality历史数据通过原始reference文档和
`external-reference-index-v1.json`保留原路径及声明SHA。本包没有声称包含所有更早历史大文件、模型权重、运行环境或编译缓存；外部索引也没有把历史声明重新包装成这次读取验证。归档内完整性可只用本包独立核验；要原样重跑带绝对路径的历史分析脚本，仍需准备其明确列出的外部依赖和路径。这份包是可验证、可重建的证据归档，不是可直接重新启动设备实验的模型分发包。

构建与核验命令见 `README-v1.md`。实际压缩体积和归档SHA见随附
`package-receipt-v1.json`、`archive-verification-v1.json`。本次仅在本地新目录准备，没有上传，没有修改benchmark工作树、冻结producer或任何原始结果。
