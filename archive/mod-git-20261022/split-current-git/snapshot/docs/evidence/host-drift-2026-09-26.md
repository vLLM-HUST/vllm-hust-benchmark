# 宿主漂移对拍（2026-09-26）：已钉版本 vs 当前 main

> 工具：[`tools/seam_diff.py`](../../tools/seam_diff.py)（纯 AST，不 import 宿主，故只需一份已安装宿主即可对拍两棵树）。
> 方法：`git worktree add --detach` 取待升版本（不动主树），然后按仓分别对拍。
> 判据来源：`docs/release.md` §4 的六项清单。

| 侧 | 已钉（本机已安装） | 待升 | 提交差 |
|---|---|---|---|
| vllm-ascend-hust | `74f0c0a27`（2026-09-08） | `fbe4911bb`（2026-09-25, origin/main） | **+480** |
| vllm-hust（core） | `f18cf803c5`（2026-09-08） | `fb1fd64f93`（2026-09-25, origin/main） | **+991** |

## 1. 结论速览

| 项 | 结果 |
|---|---|
| 接缝**签名**（两侧共 12 个符号） | **全部一致**（含 `*args`/`**kwargs`/默认值个数） |
| 接缝**调用点**（runner / cudagraph / dispatch） | **一致**（接收者、位置参数个数、关键字参数名都相同） |
| rope-fix 反向漂移守卫 | 对**新树**源码跑 `tests/test_rope_fix_drift.py` → **11/11 passed** |
| 需人工过目的差异 | vllm-ascend 侧 2 处调用点消失；core 侧 `dispatch` 新增 2 处（新 v2 工人路径）；`triton_mrope` 新增 1 处调用（非 Ascend 路径） |
| **未覆盖** | 方法体语义（对象如何构造、描述符如何填充、图池键如何形成）→ 需真机冒烟 |

## 2. vllm-ascend 侧原始输出

```
<unknown>:326: SyntaxWarning: invalid escape sequence '\{'
<unknown>:326: SyntaxWarning: invalid escape sequence '\{'
旧树 /vllm-workspace/vllm-ascend-hust
新树 /tmp/vllm-ascend-fbe4911

== 1. 签名 ==
[一致] vllm_ascend/worker/model_runner_v1.py
        <未找到>
        <未找到>
        _determine_batch_execution_and_padding(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn, allow_microbatching, force_eager, force_uniform_decode, force_has_lora, force_num_active_loras, num_encoder_reqs)  [默认值 6]
        _model_forward(self, num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)  [默认值 4]
        _update_full_graph_params_if_needed(self, forward_context, num_tokens_padded)  [默认值 0]
[一致] vllm_ascend/attention/attention_v1.py
        <未找到>
        forward_fused_infer_attention(self, query, key, value, attn_metadata, output, kv_cache)  [默认值 1]
        class AscendAttentionMetadataBuilder
        class AscendAttentionBackendImpl
[一致] vllm_ascend/compilation/acl_graph.py
        get_graph_params()  [默认值 0]
        update_graph_params_workspaces(num_tokens, workspace)  [默认值 0]
        update_full_graph_params(attn_backend, update_stream, forward_context, num_tokens, vllm_config, speculative_config, draft_attn_metadatas)  [默认值 2]
        class ACLGraphWrapper
[一致] vllm_ascend/compilation/compiler_interface.py
        _disable_pytorch_aot_cache_for_npugraph_ex()  [默认值 0]
        class AscendCompiler
[一致] vllm_ascend/ops/rotary_embedding.py
        <未找到>

== 2. 调用点（接收者 + 位置参数个数 + 关键字参数） ==
[一致] _capture_cudagraphs: 0 个调用点
[一致] _determine_batch_execution_and_padding: 3 个调用点
[一致] _disable_pytorch_aot_cache_for_npugraph_ex: 1 个调用点
[一致] _model_forward: 1 个调用点
[一致] _update_full_graph_params_if_needed: 4 个调用点
[一致] _warmup_and_capture: 0 个调用点
[一致] add_cudagraph_key: 0 个调用点
[一致] dispatch: 4 个调用点
[一致] forward_fused_infer_attention: 1 个调用点
[变化] get_graph_params: 6 -> 5 点
        仅旧树: vllm_ascend/attention/attention_v1.py  <bare>.get_graph_params(pos=0, kw=[])
[一致] triton_mrope: 1 个调用点
[一致] update_full_graph_params: 5 个调用点
[变化] update_graph_params_workspaces: 4 -> 3 点
        仅旧树: vllm_ascend/attention/attention_v1.py  <bare>.update_graph_params_workspaces(pos=2, kw=[])

== 3. 结构面（新树独有目录） ==
    + csrc/attention/fused_lightning_indexer_manage
    + csrc/attention/fused_scatter_copy_sparse_flash_attention
    + csrc/attention/mla_prolog_v3_k3
    + csrc/attention/sparse_flash_mla
    + csrc/attention/sparse_flash_mla_metadata
    + tests/ut/kv_transfer
    + tests/ut/proxy
    + vllm_ascend/models/common
    + vllm_ascend/models/deepseek_v41

合计：签名变化 2 项、符号缺失 0 项
注意：本脚本只证明**静态面**一致。方法体语义（对象如何被构造、描述符如何被填充）必须靠真机冒烟，见 docs/release.md §4 后半段。
```

## 3. core 侧原始输出

```
<unknown>:369: SyntaxWarning: invalid escape sequence '\ '
<unknown>:367: SyntaxWarning: invalid escape sequence '\ '
旧树 /vllm-workspace/vllm-hust
新树 /tmp/vllm-core-fb1fd64f

== 1. 签名 ==
[一致] vllm/v1/cudagraph_dispatcher.py
        add_cudagraph_key(self, runtime_mode, batch_descriptor)  [默认值 0]
        dispatch(self, num_tokens, uniform_decode, has_lora, num_active_loras, valid_modes, invalid_modes)  [默认值 5]
[一致] vllm/v1/worker/gpu_model_runner.py
        _capture_cudagraphs(self, batch_descriptors, cudagraph_runtime_mode, profiler)  [默认值 1]
        _warmup_and_capture(self, desc, cudagraph_runtime_mode, profile_seq_lens, allow_microbatching, num_warmups, profiler)  [默认值 4]
        _determine_batch_execution_and_padding(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn, allow_microbatching, force_eager, force_uniform_decode, force_has_lora, force_num_active_loras, num_encoder_reqs)  [默认值 6]
        _model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)  [默认值 4]
        <未找到>

== 2. 调用点（接收者 + 位置参数个数 + 关键字参数） ==
[一致] _capture_cudagraphs: 1 个调用点
[一致] _determine_batch_execution_and_padding: 9 个调用点
[一致] _disable_pytorch_aot_cache_for_npugraph_ex: 0 个调用点
[一致] _model_forward: 1 个调用点
[一致] _update_full_graph_params_if_needed: 0 个调用点
[一致] _warmup_and_capture: 2 个调用点
[一致] add_cudagraph_key: 1 个调用点
[变化] dispatch: 5 -> 7 点
        仅新树: vllm/v1/worker/gpu/attn_utils.py  self.cudagraph_manager.dispatch(pos=0, kw=['num_active_loras', 'num_reqs', 'num_tokens', 'uniform_token_count'])
        仅新树: vllm/v1/worker/gpu/dp_utils.py  cudagraph_manager.dispatch(pos=3, kw=['num_active_loras', 'num_ubatches'])
[一致] forward_fused_infer_attention: 0 个调用点
[一致] get_graph_params: 0 个调用点
[变化] triton_mrope: 3 -> 4 点
        仅新树: vllm/models/minimax_m3/common/vision_tower.py  <bare>.triton_mrope(pos=7, kw=['is_neox_style', 'mrope_interleaved'])
[一致] update_full_graph_params: 0 个调用点
[一致] update_graph_params_workspaces: 0 个调用点

== 3. 结构面（新树独有目录） ==
    + examples/features/structured_diffusion
    + rust/proto/src
    + tests/compile/rocm
    + tests/distributed/aux_output_connector
    + tests/entrypoints/cli
    + tests/models/glm5next
    + tests/parser/cohere
    + tests/watermarking
    + tools/ci
    + vllm/distributed/aux_output_connector
    + vllm/models/deepseek_v41
    + vllm/multimodal/cache
    + vllm/snapshot
    + vllm/v1/hisparse
    + vllm/v1/kv_hints
    + vllm/v1/watermarking

合计：签名变化 3 项、符号缺失 0 项
注意：本脚本只证明**静态面**一致。方法体语义（对象如何被构造、描述符如何被填充）必须靠真机冒烟，见 docs/release.md §4 后半段。
```

## 4. 工具判别力自证

```
== seam_diff 判别力自证 ==
  [selftest] 未漂移 -> rc=0（期望 0）
  [selftest] 删掉 profiler 关键字 -> rc=1（期望 1）
```

自证的意义：工具报全一致时有两种可能 —— 真的没漂移，或它根本没在检查。
故 `--selftest` 用合成小树注入一次 D1 类漂移（删掉调用点的 `profiler` 关键字）并要求报 1；
该自证由 `tests/test_seam_diff_tool.py` 纳入 CPU 门槛。
