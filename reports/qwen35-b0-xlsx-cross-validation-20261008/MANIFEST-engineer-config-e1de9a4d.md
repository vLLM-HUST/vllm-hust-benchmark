# 35B B0 baseline 证据清单

生成时间: 2026-10-08。范围: Qwen3.5-35B-A3B, 22 个计划数据集(其中 20 个有结果文件)。

## 0. 结果表
- 文件: `35B B0 baseline.xlsx`
- SHA-256(修改后): `ddb9d8dab05dbccb2adb094b0598c812f2c2e69976128337ce23343fb8c16506`
- 修改前原件: `_evidence/35B B0 baseline.ORIG.xlsx`,SHA-256 `82e81a8dbe574cf7593c3d6f74eb3a835351c3279dc12e9f3581d6f91157b743`(2026-10-08 按用户指示以结果 JSON 为准,修正了 JSONSCHEMABENCH、LONGBENCH 共 10 个单元格)
- xlsx 数值已与 21 份结果 JSON 核对并修正,修正后 105 个单元格全部一致,见第 7 节。

## 1. 证据等级(必读)
| 等级 | 含义 |
|---|---|
| D 文档记录 | 来自各数据集 `准备操作.md` 的手写命令,**不是运行时收据** |
| R 结果实测 | 远程 `/root/dataset` 下的压测结果 JSON / materialized 文件,有生成时间与哈希 |
| S 当前快照 | 2026-10-08 在工作区 peifangrui-dev 查到的环境,**不保证等于 10-04/05 运行时** |
| U 用户口述 | 用户确认但无文件佐证 |
| X 缺失 | 没有证据 |

## 2. 服务端配置(D;22 份文档的 serve 命令块逐字相同)
```
export ASCEND_RT_VISIBLE_DEVICES=0,1   # GSM8K 文档未写,用户口述确认同为 0,1 (U)
vllm serve /models/Qwen3.5-35B-A3B --served-model-name qwen3.5-35b \
  --host 127.0.0.1 --port 18180 --dtype bfloat16 --kv-cache-dtype auto --block-size 128 \
  --tensor-parallel-size 2 --enable-expert-parallel --pipeline-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-enforce-eager \
  --seed 0 --scheduling-policy fcfs --distributed-executor-backend mp --disable-custom-all-reduce \
  --no-trust-remote-code --load-format auto --no-enable-log-requests --uvicorn-log-level info \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --cudagraph-capture-sizes 1 2 4 8 16
```
压测: `vllm bench serve --backend openai-chat --dataset-name custom --custom-output-len 256 --num-prompts 200 (HUMANEVAL 164) --request-rate inf --temperature 0 --seed 0`

| 项 | 值 | 等级 |
|---|---|---|
| dtype | bfloat16 | D |
| TP / PP / DP | 2 / 1 / 1 | D |
| EP | 开 (--enable-expert-parallel) | D |
| DCP | 未设置 | D |
| disable_custom_all_reduce | 是 | D |
| gpu_memory_utilization | 0.85 | D |
| max_model_len | 32768 | D |
| max_num_seqs | 16 | D |
| max_num_batched_tokens | 8192 | D |
| prefix caching | 关 | D |
| chunked prefill | 开 | D |
| speculative | 未设置 | D |
| eager | 否 | D |
| compile / cudagraph | mode=3, FULL_DECODE_ONLY | D |
| capture sizes | 1 2 4 8 16 | D |
| 实际生效配置 / 启动日志 | 无 | X |
| 镜像名称 | vLLM-HUST 验证环境(CANN 9.1.0 + Python 3.12) | U |
| 镜像 digest | 无 | X |
| vLLM 版本 | 0.28.1.post1.dev143+gf18cf803c.empty,源码在 /vllm-workspace/vllm(非 git 仓库,无法查 commit);机器上只有这一个 vllm(which -a / pip list 均仅此一份) | S + U(用户确认运行时即此版本,无其他 vLLM) |
| vllm_ascend | 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909,/vllm-workspace/vllm-ascend(.github/vllm-release-tag.commit = v0.27.1) | S |
| vllm-hust-ext | 0.2.0.dev0 | S |
| 环境变量 | 见下方「环境变量快照」;用户确认运行期间环境变量未改动 | S + U |
| torch / torch_npu / transformers | 2.13.0+cpu / 2.13.0rc1 / 5.14.1 | S |
| CANN | 9.1.0 | S |
| 硬件 | Ascend 910B2, 64GB HBM/卡 | S |

### 环境变量快照 (S,2026-10-08 ssh 登录 shell;用户确认运行期间未改动)
```
ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
ATB_CXX_ABI=1  ATB_MATMUL_SHUFFLE_K_ENABLE=1  ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5  ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
ATB_COMPARE_TILING_EVERY_KERNEL=0  ATB_STREAM_SYNC_EVERY_{KERNEL,OPERATION,RUNNER}_ENABLE=0
OMP_NUM_THREADS=1
TASK_QUEUE_ENABLE=1
```
未设置(快照中不存在): VLLM_*、HCCL_*、PYTORCH_NPU_ALLOC_CONF、HF_*_OFFLINE、TOKENIZERS_PARALLELISM、PYTHONHASHSEED。
`ASCEND_RT_VISIBLE_DEVICES=0,1` 由各文档 serve 命令前的 `export` 设置(D),不在登录 shell 环境中。
镜像/环境名称: 「vLLM-HUST 验证环境(CANN 9.1.0 + Python 3.12)」(U,用户提供;与快照中 CANN 9.1.0、Python 3.12.13 相符)。工作区模板 ascend-devbox,版本 npc-0930-1548(coder list)。镜像 digest: 仍无记录(X)。

## 3. 模型工件
路径 `/models/Qwen3.5-35B-A3B`, 架构 Qwen3_5MoeForConditionalGeneration, 14 个 safetensors 分片。
**权重文件哈希: 未计算 (X)**。下列小文件哈希为 S 级(2026-10-08 取得):

- `config.json`  `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`
- `generation_config.json`  `4f25002776b741773666203dcea8f54619f177ace3ae483d311102092a4658e0`
- `tokenizer.json`  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- `tokenizer_config.json`  `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8`
- `merges.txt`  `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d`
- `model.safetensors.index.json`  `d8d0b7ca4e61ae107e3e87a3ff21136b3ac7c789e64bb24267227ca804e04205`
- `chat_template.jinja`  `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`

## 4. 逐数据集绑定 (D 文档 + R 结果)

| 数据集 | 准备操作.md SHA-256 | materialized.jsonl SHA-256 | 结果 JSON SHA-256 | 运行时间 | 完成/失败/计划 |
|---|---|---|---|---|---|
| AI2-ARC | `c6d7e3464a5cf059b1519c74af13fb22b93dc46fd5ab862e5c9370f2101a65a4` | `59432cdb60b2cdc5c431e806e98ee25c00c6b414b4f64cedc63f43fe0262c736` | `08627fdab13503bb7d0c55bd41164b9f519847beac91cc641b135fe59e0cb7f5` | 20261004-084729 | 200/0/200 |
| AIME-2024 | `51695773434ac9496b78a302bd031a4eee1aab2520feee7667dfde176eb67cc0` | `4be566c15c0ebb09ef33673123a6731922529a403b91af309f626b1897264373` | `6a37e310adf395ba653121fa6a3b0fff7eb6bf7acbd02828135a7c482d16bf5e` | 20261004-074754 | 200/0/200 |
| BBH | `3342f1bc7e909f66937db242e9ab6986f1114e17d7e7a384f829b2bdf6c0bcc4` | `9876254d882dafe2323b5359c7d630be4fbdc154f324c0af9170d2c54a4289e7` | `MISSING` | — | 无结果 |
| BFCL-V3 | `bf4c0db16fefaf9f8a23f3e738eecd6a30e4785b7de7f4c4a874304fde37e490` | `a041b35af5ba6ce5a1bce6bd0cbfb8ba2d8f270061cac3df9b75f0b2868491af` | `39fa9c4ab7c7838b6049e86d0e98bad6921822284f1a2c985fcd0f02eddeb153` | 20261004-081233 | 200/0/200 |
| EVOSCIENTIST | `6e168c8ec5cebb417706bb5ff96574045b416ff9175d0bc71eb06376b05b9f13` | `5a045460769a3c06ce7ecec6c71fe288043d197fa67683b2479fc2fe503edf45` | `69ce5ccace242739b9e3a3fa1315b2160a8ab1c2f568b89c16e0a12fcb623012` | 20261004-083608 | 200/0/200 |
| GSM8K | `a3110b06ae61cd74be75e134b86f4d0d28f82cb9b69929c4821f4067ad9dba70` | `d72fc82a7b12032b50eace454549c58740dd7b30f98cf0de25c5d21fd55cb1a6` | `c7ae952389cb70d00f0352c162a260a29236dcd498fd330a0bdb7fe3a2dc98bb` | 20261004-090105 | 200/0/200 |
| HOTPOT-QA | `bd93d37e8d462f0441823d25529b75cb5ac96f3eb2771bbaec242b231b128b38` | `070cff6bf4f44977c434ac9aac114149160d529a82c274a3a6b410c054720332` | `1b954489bbfc8bd3d5b52a82a94f05598ff5d342a0347575ce3aa009410fb43c` | 20261004-091657 | 200/0/200 |
| HUMANEVAL | `c186d9deffd42c9ccfaf305b8453dbcc5f4992964b436013b687fc2bc3e3cd36` | `5fc958b1f97fa41f0fc9d3ef76875bffc6c4e30c3ff8867ec6f7d39add08f22e` | `1e82bb6e31de1d977be1e00facd06e4194321aafaffdc57e61b5a3cf2253c7b2` | 20261004-093557 | 164/0/164 |
| INSTRUCTCODER | `4780399f339af279aa3d1d1e9bf9e2a0e9bff38c97b8badf5e1549657c3b2f13` | `db0cd2ac0b704f080aec892dcd7a5b38c1a37a9aa2508fe91ebf3db0850d9781` | `8ec845d1d86d919ee9ea63c3c191caf96735f87503a0acab333779eaa9087c6d` | 20261004-120054 | 200/0/200 |
| JSONSCHEMABENCH | `99f562079ff9069ef72e5c6843682280098cbf3b6b0a34be5c771cee31b9c3d8` | `4538a7f28d107e640952f1a4dfaff4fabd95533853ea51e275d4913834396d37` | `167c202fdbeb85f41b4fe22efee087e4db75019443fe9bdc622bde42b3227597` | 20261004-122540 | 199/1/200 |
| LIVECODEBENCH-CODE-GENERATION | `6f831d21d12ec016725fc4389daa3c32fe2c3a31a85b5398903553d40b0522e8` | `6c3c62f44f10c3eb43cd9b16581213b74764b7308bc315b3036b84b3563686dd` | `e292e668137be47b2f2ce6692f3f81f2f42fa313ebecc96c292f86a48feeb745` | 20261004-123712 | 200/0/200 |
| LIVECODEBENCH-EXECUTION | `ee5bed4c953b590e213a6641af19bd50c2690cf28aaf4438b5e3ca13eb492fae` | `6dd0c6bda3320ef80242feb3067fcc5c3db1cb13a3532ed853161a010af0e196` | `618ae4820fb11b25474dd029df690e18ea93708fa16f96d3dedfccc3a47b5623` | 20261004-125008 | 200/0/200 |
| LONGBENCH | `41b9d3cafe789b70f12b0ac892edaad6332fae656d485325ebe3edff8ad36944` | `1fbf9d349e0840331b292f8714bb7bf8c25e813d941755177ad007f00653d315` | `d4adda42a97e0674ae6f2c88213b9245bd8c323134bcbbbb3a9a7ed04b948f58` | 20261004-132624 | 200/0/200 |
| LONGBENCH-V2 | `0dbcbb1d7ba9da3accdddda32e15a501340f3a1e3f1f38c84453f9dcd3a167d5` | `59211f85205e341ba2da61d19a55406b343959f8511276e4a3f7db26325fea19` | `a2bb22c980229c61682d49c2dfafcaa0d3336166c49315c326ce39cf842c1ddf` | 20261004-135515 | 198/2/200 |
| MATH-500 | `10a84909c3411a640f7a15471587ef027bdcc1fd2018644e73861c20fab28da4` | `e27545f3f596a4a3019cff582dcd630a63270a052c998f1078eac4688f4c2c94` | `027589223d69c6d8f8c2270bf4b86f92ac9e73919270cd47388078c44a42570e` | 20261005-045005 | 200/0/200 |
| MBPP | `e5afa51635f15dc5080f4165e34f5ac10564737d0cd7cddfdd2cf616ad5e901d` | `e138d2a1d52df656d4b5c25092efb8f74f1ea64841e76ab1e60faa3e68a3a5ef` | `c7f54a937b5e71a73668a7c8d70ea2cab583546b6b2ee0526557e6c502e055c2` | 20261005-051719 | 200/0/200 |
| MMLU-PRO | `68beb881ff45dcf2f3ca8803811256ffa34d48b655aa12ab1871c63a8c6ff77d` | `e9f4cc3db2a07ce2b8cb1abb0337113efd84f9d29a495a7839b13963166d3e4b` | `1ba137cf005f58b1cb018699d0b369b525d2d1deed104c03a330e44653160d1e` | 20261005-053217 | 200/0/200 |
| OASST1 | `59dba30e0e6ae2ab83082b6ac88b96a85338b521501bc29f65b1c8368cb5facc` | `28306c8ca13a012153716313060fbe71fc6503c0635c75f14722763c04602e55` | `ae7c2ebf8a731b77e801ab013d0ae6d868093859546de7e17ab6b83f49bb6a74` | 20261005-055121 | 200/0/200 |
| PAIO-CHAT-1000 | `a2612fb53cbe227933c7685dd474393b2d6ba24285ec6b0eadb3ac27bcf3a5f7` | `1a9682d0ebc402e79bca9beea2144a8c055655e1456b5f2fd66b40f2d8617ed1` | `bd8bd23f5020e4ded9c7ccff5c1ce54841e97356ccfc26af88222834095a387a` | 20261004-154303 | 200/0/200 |
| PAIO-CODE-EVAL-1000 | `1954f5b97238da9b3d9ca3b81516bbfdae1795952909bfaa17a60c12dc7e49db` | `389e373a2e4e51d43d89e388b5e1281b3df9a38ff597298d7db6cb242daf0d5c` | `3a891635d8767a6fa5572695cc4be96980a02c32605886909c5ffb709f9110c5` | 20261004-171456 | 200/0/200 |
| PAIO-JSON-500 | `69750cc02e627bfaf33448db025b673a44d613f491865c9f1a7b4e370c84d464` | `730b024c68db6199d354965f6e229f8f4a1fec39de05c2424924ab8f9baa7bdb` | `b3239554c8ae2c1778a09d9de7c97ad80b031ab54ed182937c5d504ab6d32d19` | 20261005-060742 | 200/0/200 |
| PAIO-LONG-PREFILL-5000 | `47c13850b5e2a005e94c96fd6959c799c9e021abea05fd44457efd8048764af0` | `028fd39071e98f8b2012cd80729f91d6f6c1271267e02829fc9aaac1d079dc05` | `b9af70f6201accd1f6cc6f0cce3f417ac3aa6e9be5c346bba9349fe322c1f0bf` | 20261005-062022 | 200/0/200 |

## 5. 已知问题
1. BBH 无结果 JSON(X),xlsx 中 BBH 列也为空。有结果的是 21 个数据集。INSTRUCTCODER 的 JSON 位于 `/root/dataset/instructcoder/instructcoder-baseline-vllm-hust/instructcoder.json`(不在 benchmark_results 子目录下,不在 remote_hashes.txt 内)。
2. HUMANEVAL 结果文件位于 `HUMANEVAL/benchmark_results/aime-2024-baseline-vllm-hust/HUMANEVAL.json`,目录名沿用 AIME,内容为 HUMANEVAL(164 条)。
3. JSONSCHEMABENCH 失败 1 条(199/200),LONGBENCH-V2 失败 2 条(198/200)。
4. 结果 JSON 不记录服务端参数,配置与结果之间的绑定仅靠文档与时间,不是密码学绑定。
5. 当前工作区约 4 小时 50 分钟前重建,晚于 10-04/05 的运行,S 级版本信息单凭文件不能证明是运行时版本;vLLM 版本和环境变量未改动这两点依据的是用户确认(U)。
6. 修改前 xlsx 的 JSONSCHEMABENCH、LONGBENCH 两列与 JSON 不一致,已按用户指示以 JSON 为准修正(见第 7 节)。

## 6. 绑定关系
- 结果表: `35B B0 baseline.xlsx` = `ddb9d8dab05dbccb2adb094b0598c812f2c2e69976128337ce23343fb8c16506`(原件 `82e81a8dbe574cf7593c3d6f74eb3a835351c3279dc12e9f3581d6f91157b743`)
- 远程哈希原始清单: `_evidence/remote_hashes.txt` = `a42a07f018d5005174d04b2c3fd86b037f075779de51a3fab3cc215e1ca8bff2`(含 20 份结果 JSON、materialized、模型小文件哈希)
- 第 4 节每行把「文档 → materialized → 结果 JSON」绑到同一数据集;xlsx 通过本文件(记录其哈希)与它们关联。

## 7. xlsx 与结果 JSON 核对 (2026-10-08)
方法: xlsx 第 2/4/5/6/7 行(请求吞吐、输出吞吐、总吞吐、TTFT mean、TPOT mean)与 JSON 的 request_throughput / output_throughput / total_token_throughput / mean_ttft_ms / mean_tpot_ms 比较,取 2 位小数。
xlsx 其余行全部为空;BBH 列整列为空;14B 数据集的列全部为空,无 14B 数据混入。

- 修改前: 90 个一致,10 个不一致(JSONSCHEMABENCH、LONGBENCH 各 5 项),INSTRUCTCODER 当时误判为无 JSON。
- 按用户指示以 JSON 为准,修改了下表 10 个单元格;修改后 21 个数据集 × 5 项 = 105 个单元格全部一致。
- 修改后的 xlsx 通过 zip 完整性检查。

| 数据集 | 单元格 | 指标 | 修改前 | 修改后 |
|---|---|---|---|---|
| JSONSCHEMABENCH | T2 | 请求吞吐 | 1.81 | 1.8 |
| JSONSCHEMABENCH | T4 | 输出吞吐 | 462.85 | 461.04 |
| JSONSCHEMABENCH | T5 | 总吞吐 | 4488.37 | 4470.8 |
| JSONSCHEMABENCH | T6 | TTFT | 44070.49 | 44233.91 |
| JSONSCHEMABENCH | T7 | TPOT | 30.16 | 30.23 |
| LONGBENCH | X2 | 请求吞吐 | 1.23 | 1.21 |
| LONGBENCH | X4 | 输出吞吐 | 313.96 | 308.72 |
| LONGBENCH | X5 | 总吞吐 | 8049.71 | 7915.46 |
| LONGBENCH | X6 | TTFT | 80210.74 | 81211.22 |
| LONGBENCH | X7 | TPOT | 45.71 | 46.31 |

JSONSCHEMABENCH JSON: 完成 199 / 失败 1;LONGBENCH-V2 JSON: 完成 198 / 失败 2。
