## 探针四腿对比（四变体 × 四腿）

### interleaved

| 腿 | 生产类 | 实际 fwd | kernel_reachable | 接线判据 | e2e 判定 | 对 host 公式偏差 | 公式分歧 |
|---|---|---|---|---|---|---|---|
| prefix | `AscendRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| control-off | `AscendRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| carrier-on | `AscendRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| forkfix | `AscendRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |

### llama3

| 腿 | 生产类 | 实际 fwd | kernel_reachable | 接线判据 | e2e 判定 | 对 host 公式偏差 | 公式分歧 |
|---|---|---|---|---|---|---|---|
| prefix | `Llama3RotaryEmbedding` | `forward_oot` | 断点(生产类未接线/报错) | **False** (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| control-off | `Llama3RotaryEmbedding` | `forward_oot` | 断点(生产类未接线/报错) | **False** (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| carrier-on | `AscendFixLlama3RotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| forkfix | `AscendLlama3RotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |

### yarn

| 腿 | 生产类 | 实际 fwd | kernel_reachable | 接线判据 | e2e 判定 | 对 host 公式偏差 | 公式分歧 |
|---|---|---|---|---|---|---|---|
| prefix | `AscendYaRNRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS*(链自洽) — 但生成端公式与 host/GPU 分歧，端到端偏差被它主导 | 5.10156 | 5.09375 |
| control-off | `AscendYaRNRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS*(链自洽) — 但生成端公式与 host/GPU 分歧，端到端偏差被它主导 | 5.10156 | 5.09375 |
| carrier-on | `AscendFixYaRNRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| forkfix | `AscendYaRNRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (O≡直接kernel) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |

### mrope

| 腿 | 生产类 | 实际 fwd | kernel_reachable | 接线判据 | e2e 判定 | 对 host 公式偏差 | 公式分歧 |
|---|---|---|---|---|---|---|---|
| prefix | `AscendMRotaryEmbedding` | `forward_oot` | 断点(生产类未接线/报错) | **False** (包装正常返回) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| control-off | `AscendMRotaryEmbedding` | `forward_oot` | 断点(生产类未接线/报错) | **False** (包装正常返回) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| carrier-on | `AscendFixMRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (包装正常返回) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |
| forkfix | `AscendMRotaryEmbedding` | `forward_oot` | PASS(生产类接线到 triton) | True (包装正常返回) | PASS(链自洽；总偏差=cache bf16 底噪) | 0.03125 | 0 |

## 判定字段（carrier-on vs forkfix / control-off vs prefix）

- carrier_vs_forkfix: PASS
- control_off_vs_prefix: PASS

## 附加：OFF 腿复现修前三断点

- llama3 生产类 `Llama3RotaryEmbedding` / fwd `CustomOp.forward_oot`（= host 类 + `CustomOp.forward_oot` 默认实现）
- mrope 包装调用：`TypeError: triton_mrope() missing 1 required positional argument: 'is_neox_style'`（位置 ['File "/vllm-workspace/vllm-ascend-hust/vllm_ascend/ops/rotary_embedding.py", line 562, in forward_oot', 'File "/vllm-workspace/vllm-ascend-hust/vllm_ascend/ops/rotary_embedding.py", line 541, in forward_triton']）
- yarn 公式分歧：5.09375（端到端偏差 5.10156）
- control-off JSON `oot_rope_keys` = ['ApplyRotaryEmb', 'DeepseekScalingRotaryEmbedding', 'MRotaryEmbedding', 'RotaryEmbedding', 'YaRNScalingRotaryEmbedding']
- carrier-on JSON `oot_rope_keys` = ['ApplyRotaryEmb', 'DeepseekScalingRotaryEmbedding', 'Llama3RotaryEmbedding', 'MRotaryEmbedding', 'RotaryEmbedding', 'YaRNScalingRotaryEmbedding']
