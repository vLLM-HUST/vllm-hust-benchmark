# GenNova 独立引擎验证报告

## 测试对象与配置

测试对象为 GenNova 独立推理引擎，不依赖外部 vLLM-HUST MOD 挂载。 完整模型性能轮次源码为 `ea13161d09446b8e8db0fc903b2356e75d1b4002`。

| 项目         | 配置                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| 模型         | Qwen3.5-35B-A3B                                                        |
| 精度与设备   | BF16，Ascend 910B2，TP2 / PP1 / DP1                                    |
| 执行容量     | execution seats 24，resident seats 36，token pages 2048                |
| 调度与上下文 | mixed tokens 4096，max position 65536，Native MTP 2                    |
| 运行栈       | Torch 2.13.0+cpu，torch-npu 2.13.0.rc1，私有构建的 GenNova Ascend 算子 |
| 负载         | 固定 MMLU-Pro 200 请求，282,265 个输入 token，每请求 256 个输出 token  |
| 请求条件     | greedy，thinking 模板保留，全部请求突发提交，ignore_eos=true           |

模型身份按同一挂载 checkpoint 的索引 SHA256 及 14 个权重分片的文件属性核验， 不将其表述为新计算的全量权重内容哈希。 输入保持既有测试的模板组合，完整 token IDs
及输入构造脚本随证据封存。 每轮独立进程和编译缓存目录；权重文件及已核验的只读原生算子二进制共享。 GenNova 此入口不提供 APC 或主机状态 offload，不等同于既有 HTTP 服务入口。

## 测试方法

先执行原生算子与图重放探针，再验证真实权重的短文本、中文输出和 6,041-token 长输入。 完整性能轮次在加载权重、编译及图捕获结束后计时，从 tokenized 请求提交至 CPU
可见完整输出。 启动、HTTP 和分词时间不计入该吞吐指标。两 rank 输出在计时外核对。 四张可见 NPU 全部预留，实际使用两张；保存运行中设备状态、退出状态与资源释放回执。

## 结果

| 指标                        |                                 GenNova 实测结果 |
| --------------------------- | -----------------------------------------------: |
| 完成请求                    |                                        200 / 200 |
| 输出 token                  |                                           51,200 |
| 耗时（取两 rank 较大值）    |                                     61.789713 秒 |
| 输出吞吐                    |                                 828.616888 tok/s |
| 平均 CPU 交付首 token 延迟  |                                     26.332747 秒 |
| 平均 CPU 交付 TPOT          |                                   39.338815 毫秒 |
| 两 rank 输出 token 序列一致 |                                        200 / 200 |
| 输出包含 EOS 的请求         |                                          0 / 200 |
| Torch 设备峰值分配          |                     每 rank 42,384,885,248 bytes |
| 冷启动时间                  | rank 0：1,510.498488 秒；rank 1：1,512.095675 秒 |
| 进程退出与设备释放          |                         exit 0，全部预留设备释放 |

吞吐由 `51200 / max(rank0.seconds, rank1.seconds)` 复算。 原生算子资格探针 6 项通过；真实模型资格的 3 项检查在两 rank 均通过。
上述结果证明该模型和请求范围内的独立推理路径有效，不代表完整 MMLU-Pro 准确率评分。

## 修复与回归

1. [GenNova #171](https://github.com/vLLM-HUST/GenNova/pull/171)：修复干净目录中原生 ACLNN 构建遗漏相邻 CMake
   辅助文件的问题。已合入 main；相关 CPU 测试 34 项通过，原生资格探针通过。
1. [GenNova #172](https://github.com/vLLM-HUST/GenNova/pull/172)：仅合并相邻的 construction state snapshot
   物理区间，保留稀疏间隙、原目标地址和非零状态。已合入 main；CPU 回归 74 项通过，NPU 对照 8 项通过，设备峰值额外分配为 0。

第二项修复的 2,048-block 连续页探针中，拷贝区间从 2,049 个减少为 1 个； snapshot 从 0.383729 秒降至 0.007121 秒，restore 从
0.102063 秒降至 0.001737 秒。 这是有界状态拷贝探针，不是完整模型启动时间复测。上表性能结果未包含第二项修复。 该探针的原始 receipt
记录提交前基座，候选源码字节已随证据保存，对应后续提交 `ca46d7941bd041706d395f8031ea961af6431ea2`。 GenNova 仓库未配置远端
CI；以上为实际执行的本地 CPU 和 NPU 检查。

## 比较边界

既有完整 B1 为 E24/R36 + KV materialization recompute + Prefill Microflow， HTTP 吞吐中位数为 824.894609
tok/s。该入口不同于本报告 GenNova 引擎计时，不能据此认定胜负。

配对入口探索中，`b1-r1` 因 offload 锁页主机内存分配失败终止； `b1-r2` 完成了请求，但直接引擎入口绕过 HTTP 阶段的 KV materialization 请求控制，
不具备完整 B1 比较资格。其原始数据与被否决的派生比值均保留，未用来替换 B1。 适配器已补充原组件控制函数调用及真实事件断言；该修正未执行新的 NPU 复测。

## 结论与证据

GenNova 已通过独立原生算子、真实 checkpoint 和完整 200 请求运行验证。 两项引擎修复均已合入 main，合并后提交为
`fcaea20b481965d9b62b41a5b0f84736079748ca`。 本报告不作超过当前 B1 或完整数据集质量达标的结论。

`manifest.json` 记录各证据压缩包和包内文件 SHA256。 `native-r1`、`checkpoint-r2`、`engine-r1`、`snapshot-r1` 保存成功记录；
`checkpoint-r1` 保存隔离环境缺少 Triton 的首次失败，`b1-r1` 与 `b1-r2` 保存比较入口探索记录。 `protocol-and-tools.tar.gz`
保存输入、执行脚本、算子构建回执及 `adjudication.json`。 原始请求、输出 token、交付时间、服务日志和 NPU 状态均在对应包内，未包含模型权重或编译缓存。
