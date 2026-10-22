# GDN State Codec windowed GDN M2

## Scope correction

The standalone boundary codec in `GDN_STATE_CODEC_QWEN38_DEVICE_CODEC_M1.md` is a
layout and correctness screen only. Its encode/copy/decode timing is not a
reproduction of LeapQuant and must not be used to accept or reject the paper's
performance claims.

M2 follows the paper's decode representation for Qwen GDN:

- `p = 16` real-token records per window;
- `r = 4` FP16 compensator records;
- an INT8 residual with FP32 row smoothing and value-channel scales;
- no dense-state materialization between boundaries;
- direct readout from the fixed boundary plus buffered rank-one records;
- power-iteration refitting only at a window boundary;
- matched comparison against the complete FP32 GDN layer kernel, never against
  an isolated device-to-device copy.

For a read vector `x`, the implementation evaluates

```text
S_l^T x = dequant(R_0)^T C Gamma(1:l) x
          + U_tilde (K_tilde^T Gamma(1:l) x)
          + sum_j u_j (k_j^T Gamma(j+1:l) x)
```

The per-token correction is `u_i = v_i - S_(i-1)^T beta_i`. The output after
the update is evaluated without forming `S_i`:

```text
o_i = S_(i-1)^T (alpha_i * q_i) + u_i * (k_i^T q_i)
```

## Qwen mapping

For the provider recurrence

```text
R <- exp(g) R
delta <- beta * (v - R k)
R <- R + delta k^T
o <- scale * R q
```

and `S = R^T`, one buffered record is:

- scalar decay `alpha = exp(g)`;
- write key `beta * k`;
- read vector `exp(g) * k`;
- correction `v - S^T read`;
- output query `scale * q`.

## Admission gates

1. CPU algebra oracle: 256 tokens, all 16 boundaries, canonical output digest
   unchanged, state and output error gates unchanged.
2. Ascend single-layer component: 32 heads, `d_k = d_v = 128`, window
   positions 0 through 15, exact lifecycle and canary checks.
3. Matched performance: identical batch, inputs, graph mode, stream policy,
   warmups, and repetitions against the FP32 GDN kernel. Report batch 1, 16,
   64, 256, and the largest admitted batch.
4. Boundary cost includes power iteration, residual smoothing, quantization,
   and record reset, amortized over exactly 16 tokens.
5. Runtime integration is prohibited unless the combined non-boundary and
   amortized boundary cost passes no-harm and produces repeatable end-to-end
   gain.

## Evidence labels

Until the authors release their announced vLLM and TileLang implementation,
all M2 results are labeled `independent-paper-reimplementation`. Component
results are not online or end-to-end results.

## 910B2 checkpoint (2026-10-07)

The first non-boundary kernel gate used 32 Qwen3.5 GDN heads, deterministic
synthetic FP16 inputs, one stream, three warmups, and 11 measured launches on
one otherwise-idle Ascend 910B2. Host-to-device restoration was excluded from
timing. The FP32 comparator performs the same complete state read, rank-one
update, state write, and output read; it is not a copy baseline.

| Window position | Output relative RMS | State relative RMS | Windowed median | FP32 median | Gate |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 3.28108e-7 | 7.73236e-6 | 928.061 us | 3572.81 us | pass |
| 7 | 3.20788e-7 | 1.10618e-5 | 1070.23 us | 3510.27 us | pass |
| 15 | 2.70129e-7 | 1.94153e-5 | 1245.41 us | 3529.61 us | pass |

The mutable per-head position uses a 32-byte stride. Compact four-byte
positions caused nondeterministic lost writes when independent vector cores
updated values in the same cache line; padding removed that false sharing in
three consecutive position-7 runs.

This checkpoint passes only the synthetic, single-layer, non-boundary
component gate. It does not include the power-iteration/refit boundary cost,
model-derived tensors, AgentX, graph mode, or an end-to-end serving result, and
must not be cited as a Qwen3.5 throughput or latency improvement.

### Batch sweep

The same position-7 gate was repeated with 32 heads per request. Mutable
position metadata is written through a 32-byte UB buffer and MTE3 copy; direct
cross-core scalar GM stores lost writes at batch 16 and, even with an explicit
cache clean, lost 2 of 2048 writes at batch 64. Those failed variants are not
admitted.

| Batch | States | Output relative RMS | State relative RMS | Windowed median | FP32 median | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 32 | 3.20788e-7 | 1.10618e-5 | 1077.12 us | 3568.89 us | pass |
| 16 | 512 | 2.75708e-7 | 1.14121e-5 | 16932.1 us | 51617.6 us | pass |
| 64 | 2048 | 2.79422e-7 | 1.13893e-5 | 66245.9 us | 206794 us | pass |
| 256 | 8192 | 2.78709e-7 | 1.13830e-5 | 264091 us | 836743 us | pass |
| 512 | 16384 | 2.79196e-7 | 1.13791e-5 | 526868 us | 1672840 us | pass |

These scalar correctness kernels show that compressed-state traffic can be
reduced without changing the recurrence, but their absolute latency is not a
production target. Boundary refit cost remains excluded and therefore the
ratios in this table are not retained end-to-end speedups.

### Boundary correctness checkpoint

The first boundary kernel operates directly on the compressed boundary and 16
rank-one records; it does not materialize the old FP32 state. It fits four new
FP16 compensators by power iteration, requantizes the residual to INT8, and
resets the record lifecycle. On the same 32 deterministic synthetic heads:

| Power iterations/rank | Boundary relative RMS | No-compensator INT8 RMS | Boundary median | Lifecycle |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 0.00346330 | 0.00417742 | 24254.9 us | pass |
| 2 | 0.00320901 | 0.00417742 | 28978.7 us | pass |
| 4 | 0.00308235 | 0.00417742 | 38542.6 us | pass |
| 8 | 0.00305492 | 0.00417742 | 57726.1 us | pass |
| 16 | 0.00305883 | 0.00417742 | 96122.8 us | pass |
| 32 | 0.00305907 | 0.00417742 | 173024 us | pass |

Eight iterations converge for this synthetic input, but the paper does not
state its iteration count and model-derived tensors have not yet been checked.
At eight iterations the boundary adds about 3.61 ms/token when amortized over
16 tokens. Combined with the non-boundary kernel, this fails the matched FP32
no-harm gate. The boundary implementation is therefore a correctness baseline
only; end-to-end integration remains prohibited until its matrix operations
are moved off scalar execution and the combined gate passes.

## Qwen3.5-35B runtime qualification (2026-10-07)

The production-path implementation now consumes and emits the provider's BF16
decode tensors directly.  It caches the eager execution stream and stable slot
pointers, launches only the TP-local number of tasks, reads the INT8 residual
in eight-column DMA/cast tiles, and reduces the 16 read/output projections for
each tile as one vector batch.  The quant-only boundary uses the same p=16
lifecycle and does not attribute earlier standalone codec results to the paper.

On Qwen3.5-35B TP2 (16 value heads per rank), the matched single-layer profile
measures the v68 kernel at 11.22 us median versus 10.80 us for the production
`RecurrentGatedDeltaRule`.  Max output relative RMS is 0.008021.  A clean
release build independently passes correctness and no-harm, with steady and
batched ratios of 0.565 and 0.487 after host/runtime overhead is included.

The admitted end-to-end workload is one eager real-online AgentX request: the
first conversation/turn, 649 prompt tokens, 128 generated tokens, temperature
0, seed 0.  The comparison used two independently started v68 lifecycles around
a freshly started native lifecycle:

| Arm/lifecycle | Measurements | Total median | Run-level ITL median |
| --- | ---: | ---: | ---: |
| v68 candidate A | 6 | 28.4146 s | 219.9409 ms |
| native baseline B | 6 | 28.6807 s | 222.3608 ms |
| v68 candidate A2 | 6 | 28.4343 s | 219.8913 ms |
| combined v68 | 12 | 28.4164 s | 219.9409 ms |

Combined v68 is 0.922% faster in total time and 1.088% lower in run-level
median ITL than the contemporaneous native arm.  All 18 formal responses have
the same content SHA-256.  This is a workload-qualified result only: it does
not establish a benefit for other prompts, output lengths, concurrency,
speculative decode, graph mode, or dense-state capacity.  Earlier short-window
and v65 neutral/negative results remain part of the evidence record.
