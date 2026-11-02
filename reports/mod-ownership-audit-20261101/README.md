# MOD ownership audit — 2026-11-01

Reviewed all 44 catalog components with named maintainers across 43 canonical repositories,
including hidden entries. Repository declarations, assigned project tasks, confirmed organization
identities and current user corrections were compared. Commit counts were used only as supporting
evidence, never as an ownership rule.

Corrections:

- MuxPD: restore 毛潮云 (Irisuko), advised by 罗瑞坤; the organization is infrastructure support.
  Confirmed by the project owner and issues #3, #11 and #23.
- KVCompress: retain 张家万 (Jiawan23), add 韦若皓 (kotoriqaq0) and project lead 刘思辰 (Seas0), as
  explicitly declared in the repository README. Advisor remains 万瑶.
- OpFuse: replace the organization placeholder in repository metadata with 张睿诚 (KimmoZAG); use his
  confirmed name on the website.
- Use confirmed names for 彭浩然 (Tkhkrnx), 毛言粲 (yancanmao), and 张书豪 (ShuhaoZhangTony) across the
  remaining affected cards. These are display-name corrections, not ownership transfers.

Organization identity reference:
[people.json](https://github.com/vLLM-HUST/.github/blob/main/profile/people.json). Only confirmed
public names and advisor relationships were used. Existing metrics, measurements, visibility and
historical authorship remain unchanged.

| Component | Current maintainers | Repository evidence | Decision | |---|---|---|---| |
state-scheduler | ShuhaoZhangTony |
[README.md](https://github.com/Qixin-Gaoke/state-scheduler-plugin/blob/a4188b1e0f10dbf5e11b052711e45335d79838a5/README.md);
[task #23](https://github.com/Qixin-Gaoke/state-scheduler-plugin/issues/23) | Retained; no
conflicting current owner found | | pegaflow | cybber695, zhangshengxiang682 |
[README.md](https://github.com/vLLM-HUST/pegaflow-hust/blob/839bae7b603199dcb8914963bd7fd4a8afa1246e/README.md);
[task #33](https://github.com/vLLM-HUST/pegaflow-hust/issues/33) | Retained; no conflicting current
owner found | | pegaflow-vllm-connectors | cybber695, zhangshengxiang682 |
[README.md](https://github.com/vLLM-HUST/pegaflow-hust/blob/839bae7b603199dcb8914963bd7fd4a8afa1246e/README.md);
[task #33](https://github.com/vLLM-HUST/pegaflow-hust/issues/33) | Retained; no conflicting current
owner found | | adm | sad-and-bad1231 |
[MOD_METADATA.json](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/3bc9e876ee862b20accbe98149c0ad6ff458e16d/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/ascend-distributed-metadata/blob/3bc9e876ee862b20accbe98149c0ad6ff458e16d/README.md);
[task #7](https://github.com/vLLM-HUST/ascend-distributed-metadata/issues/7) | Retained; no
conflicting current owner found | | bidkv | MingqiWang-coder, cybber695 |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-bidkv/blob/8394bdeb09353337199f27497e40f8202bcfefad/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-bidkv/blob/8394bdeb09353337199f27497e40f8202bcfefad/README.md);
[task #20](https://github.com/vLLM-HUST/vllm-hust-bidkv/issues/20) | Retained; no conflicting
current owner found | | diffspec | dzcixy |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/blob/1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/blob/1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254/README.md);
[task #8](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/issues/8) | Retained; no
conflicting current owner found | | vspec | Renty-0 |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/MAINTAINERS.md);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/README.md);
[task #18](https://github.com/vLLM-HUST/vllm-hust-vSpec/issues/18) | Retained; no conflicting
current owner found | | latchmoe | Li-changwu, pluviophile-chen |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/blob/f2cc19ca6d48a94d2da88378f7c1e2621c6f58b9/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/blob/f2cc19ca6d48a94d2da88378f7c1e2621c6f58b9/README.md);
[task #118](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/118) | Retained; no
conflicting current owner found | | opfuse | KimmoZAG |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-opfuse/blob/2ae4fd17d38879da5a3debb1ce014c2400b4e44a/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-opfuse/blob/2ae4fd17d38879da5a3debb1ce014c2400b4e44a/README.md);
[task #1](https://github.com/vLLM-HUST/vllm-hust-opfuse/issues/1) | Corrected | |
ascend-adaptive-quantized-kv | XilingGao |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/blob/9a6f45a7e06aa93196f0395c8bd5b2b4166d9292/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/blob/9a6f45a7e06aa93196f0395c8bd5b2b4166d9292/README.md);
[task #10](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/10) |
Retained; no conflicting current owner found | | ascend-quant-toolkit | aly16-k |
[README.md](https://github.com/vLLM-HUST/vllm-ascend-quant-hust/blob/db3f55632263997744a3125abc78324edeb0643b/README.md);
[task #2](https://github.com/vLLM-HUST/vllm-ascend-quant-hust/issues/2) | Retained; no conflicting
current owner found | | kvcompress-ascend | Jiawan23, kotoriqaq0, Seas0 |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/README.md);
[task #32](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/32) | Corrected | |
prefix-router-migration | Amber1qq, WMASTER123, Adr1anZheng |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/.github/CODEOWNERS);
[README.md](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/README.md);
[task #3](https://github.com/vLLM-HUST/vllm-hust-prefix-router/issues/3) | Retained; no conflicting
current owner found | | kv-tiering-migration | JieYang2001 |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/README.md);
[task #7](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/issues/7) | Retained; no conflicting
current owner found | | knorm-migration | kotoriqaq0, SuccinctPaul |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-knorm/blob/1efd0103a348ba63716341b9d25141780e4689de/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-knorm/blob/1efd0103a348ba63716341b9d25141780e4689de/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-knorm/blob/1efd0103a348ba63716341b9d25141780e4689de/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-knorm/blob/1efd0103a348ba63716341b9d25141780e4689de/README.md);
[task #7](https://github.com/vLLM-HUST/vllm-hust-knorm/issues/7) | Retained; no conflicting current
owner found | | pyramidkv-ascend-migration | Irisuko |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/3355b3567b855f4bc2188602b820c6682ce4d986/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/3355b3567b855f4bc2188602b820c6682ce4d986/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/3355b3567b855f4bc2188602b820c6682ce4d986/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/3355b3567b855f4bc2188602b820c6682ce4d986/README.md);
[task #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8) | Retained; no
conflicting current owner found | | slicegpt-migration | qingfengyuhuoda |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-slicegpt/blob/005a79127c5111a53a4797f1cb1db9845c87c73c/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-slicegpt/blob/005a79127c5111a53a4797f1cb1db9845c87c73c/.github/CODEOWNERS);
[README.md](https://github.com/vLLM-HUST/vllm-hust-slicegpt/blob/005a79127c5111a53a4797f1cb1db9845c87c73c/README.md)
| Retained; no conflicting current owner found | | quantized-kv-cache-migration | hustcui,
SuccinctPaul |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/README.md);
[task #8](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/issues/8) | Retained; no
conflicting current owner found | | simllm-migration | GuMorming |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/b649015ec385cab3a345daf26169155a21ad0548/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/b649015ec385cab3a345daf26169155a21ad0548/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/b649015ec385cab3a345daf26169155a21ad0548/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/b649015ec385cab3a345daf26169155a21ad0548/README.md);
[task #7](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/issues/7) | Retained; no conflicting
current owner found | | unified-communication-migration | machuanhu |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-unified-comm/blob/b4d6475498e94792e570e34277f5a4edf8b1c244/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-unified-comm/blob/b4d6475498e94792e570e34277f5a4edf8b1c244/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-unified-comm/blob/b4d6475498e94792e570e34277f5a4edf8b1c244/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-unified-comm/blob/b4d6475498e94792e570e34277f5a4edf8b1c244/README.md);
[task #2](https://github.com/vLLM-HUST/vllm-hust-unified-comm/issues/2) | Retained; no conflicting
current owner found | | split-batch-full-graph-migration | Raing5Days, ilnnfover |
[.github/MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/d9af5a1f4643f747b1313f9cffc17429db15a8fe/.github/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/d9af5a1f4643f747b1313f9cffc17429db15a8fe/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/d9af5a1f4643f747b1313f9cffc17429db15a8fe/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/d9af5a1f4643f747b1313f9cffc17429db15a8fe/README.md);
[task #8](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/issues/8) | Retained; no
conflicting current owner found | | kv-transfer-observability-migration | xiehanlong834-gif,
Remygred |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-kv-transfer-observability/blob/c4f41c3864ee3bf3415466ba7a8e3436ff1e0924/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-kv-transfer-observability/blob/c4f41c3864ee3bf3415466ba7a8e3436ff1e0924/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-kv-transfer-observability/blob/c4f41c3864ee3bf3415466ba7a8e3436ff1e0924/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-kv-transfer-observability/blob/c4f41c3864ee3bf3415466ba7a8e3436ff1e0924/README.md);
[task #2](https://github.com/vLLM-HUST/vllm-hust-kv-transfer-observability/issues/2) | Retained; no
conflicting current owner found | | layered-prefill-migration | Yushuo-star |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-ascend-layered-prefill-hust/blob/80aa107dbe4d6c5e659a380618b9c17599abf87f/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-ascend-layered-prefill-hust/blob/80aa107dbe4d6c5e659a380618b9c17599abf87f/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-ascend-layered-prefill-hust/blob/80aa107dbe4d6c5e659a380618b9c17599abf87f/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-ascend-layered-prefill-hust/blob/80aa107dbe4d6c5e659a380618b9c17599abf87f/README.md);
[task #2](https://github.com/vLLM-HUST/vllm-ascend-layered-prefill-hust/issues/2) | Retained; no
conflicting current owner found | | activation-sparsity-migration | llxler |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-activation-sparsity/blob/77f151d1f6851933230fad0da1289fc666be1f0f/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-activation-sparsity/blob/77f151d1f6851933230fad0da1289fc666be1f0f/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-activation-sparsity/blob/77f151d1f6851933230fad0da1289fc666be1f0f/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-activation-sparsity/blob/77f151d1f6851933230fad0da1289fc666be1f0f/README.md);
[task #2](https://github.com/vLLM-HUST/vllm-hust-activation-sparsity/issues/2) | Retained; no
conflicting current owner found | | pipeline-microbatch-migration | xsun2001 |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/README.md);
[task #7](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/issues/7) | Retained; no
conflicting current owner found | | qos-scheduler-migration | Yushuo-star |
[MAINTAINERS.md](https://github.com/vLLM-HUST/vllm-hust-qos-scheduler/blob/bedd62c5aae0dc6b93277da5aac2d743868a5d93/MAINTAINERS.md);
[.github/CODEOWNERS](https://github.com/vLLM-HUST/vllm-hust-qos-scheduler/blob/bedd62c5aae0dc6b93277da5aac2d743868a5d93/.github/CODEOWNERS);
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-qos-scheduler/blob/bedd62c5aae0dc6b93277da5aac2d743868a5d93/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-qos-scheduler/blob/bedd62c5aae0dc6b93277da5aac2d743868a5d93/README.md);
[task #2](https://github.com/vLLM-HUST/vllm-hust-qos-scheduler/issues/2) | Retained; no conflicting
current owner found | | request-lifecycle-profiler | Remygred |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-request-lifecycle-profiler/blob/9d43beb935210c7835c4124ab6845e005feb5518/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-request-lifecycle-profiler/blob/9d43beb935210c7835c4124ab6845e005feb5518/README.md)
| Retained; no conflicting current owner found | | tricard-clm-lifecycle | xiehanlong834-gif |
[README.md](https://github.com/vLLM-HUST/Tricard/blob/1d141da1c427a18859643b056b5514f5ebf511ce/README.md)
| Retained; no conflicting current owner found | | quality-bounded-inference | Tkhkrnx |
[README.md](https://github.com/vLLM-HUST/quality-bounded-inference-plugin/blob/b2ed0136f0a399fa919287fb334946f71d6a9ac7/README.md)
| Corrected | | llm-serving-cost-pricing-model | ShuhaoZhangTony |
[README.md](https://github.com/vLLM-HUST/llm-serving-cost-pricing-model/blob/1559df80e40172299231f7cee381f41738720ce2/README.md)
| Existing explicit catalog declaration retained; repository has no separate owner declaration | |
clm-lifecycle | xiehanlong834-gif |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-clm-lifecycle/blob/0ff972186a0c599de8cb2e36cf54f492d4905ed2/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-clm-lifecycle/blob/0ff972186a0c599de8cb2e36cf54f492d4905ed2/README.md)
| Retained; no conflicting current owner found | | kv-materialization-arrival-control |
healer-positive |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control/blob/cc85310916ffaa08aa52fe248e66ee777da89802/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control/blob/cc85310916ffaa08aa52fe248e66ee777da89802/README.md);
[task #30](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control/issues/30) |
Retained; no conflicting current owner found | | betterscale | CubeLander |
[MOD_METADATA.json](https://github.com/vLLM-HUST/BetterScale/blob/a83f3e2371672fa8f46fb3cdf96a1041c59e4604/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/BetterScale/blob/a83f3e2371672fa8f46fb3cdf96a1041c59e4604/README.md);
[task #23](https://github.com/vLLM-HUST/BetterScale/issues/23) | Retained; no conflicting current
owner found | | ascend-attention-boundary | xmdhb |
[README.md](https://github.com/xmdhb/vllm-hust-ascend-attention-boundary/blob/3572713e7944ea877b22f2a8a6bb053f348bfd25/README.md)
| Retained; no conflicting current owner found | | dla | xiehanlong834-gif |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-dla/blob/cc927e0d88c350babe9ce434abe04cf0ec2bb075/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-dla/blob/cc927e0d88c350babe9ce434abe04cf0ec2bb075/README.md);
[task #6](https://github.com/vLLM-HUST/vllm-hust-dla/issues/6) | Retained; no conflicting current
owner found | | stateaxis-automatic-prefix-cache | ShuhaoZhangTony |
[MOD_METADATA.json](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache/blob/92327b78ab2d8aa717544ede3621b0f109df31fd/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache/blob/92327b78ab2d8aa717544ede3621b0f109df31fd/README.md);
[task #8](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache/issues/8) | Corrected | |
stateaxis-async-kv-transfer | ShuhaoZhangTony |
[MOD_METADATA.json](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer/blob/03143f99a5e9718f93310af8228ea349e4f45897/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer/blob/03143f99a5e9718f93310af8228ea349e4f45897/README.md);
[task #7](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer/issues/7) | Corrected | |
stateaxis-pipelined-weight-loading | ShuhaoZhangTony |
[MOD_METADATA.json](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading/blob/c28890ec48ed69dad9e980be3e4e6d10422465d2/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading/blob/c28890ec48ed69dad9e980be3e4e6d10422465d2/README.md);
[task #7](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading/issues/7) | Corrected | |
gdn-state-codec | ShuhaoZhangTony |
[MOD_METADATA.json](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/README.md);
[task #20](https://github.com/vLLM-HUST/gdn-state-codec/issues/20) | Corrected | | vllm-hust-opset |
keridone |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-opset/blob/7d3cf948da548220fb2e0a32bf2aff80b5647dc4/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-opset/blob/7d3cf948da548220fb2e0a32bf2aff80b5647dc4/README.md);
[task #4](https://github.com/vLLM-HUST/vllm-hust-opset/issues/4) | Retained; no conflicting current
owner found | | traceloom | CubeLander |
[MOD_METADATA.json](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer/blob/ee763f467d100d043c740695e72c120c6e91ca1c/MOD_METADATA.json);
[README.md](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer/blob/ee763f467d100d043c740695e72c120c6e91ca1c/README.md);
[task #30](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer/issues/30) | Retained; no
conflicting current owner found | | request-throttling-controller | yancanmao |
[.github/CODEOWNERS](https://github.com/vLLM-HUST/request-throttling-controller/blob/b900dabf283f81b31560ab4a9f4deab38683e5ec/.github/CODEOWNERS);
[README.md](https://github.com/vLLM-HUST/request-throttling-controller/blob/b900dabf283f81b31560ab4a9f4deab38683e5ec/README.md)
| Corrected | | core-attention-boundary | xmdhb |
[README.md](https://github.com/xmdhb/vllm-hust-core-attention-boundary/blob/b5bc92a513a4ad043e39c7e2e3d6c4f4351114ff/README.md)
| Retained; no conflicting current owner found | | muxpd | Irisuko |
[README.md](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/blob/262beff5bb9116b5709774a41826a1e08a519a65/README.md);
[task #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23) | Corrected |
