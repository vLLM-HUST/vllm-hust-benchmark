# Rank-1 active-rank boundary specialization

Date: 2026-10-08
Evidence label: `model-derived-component-candidate`

## Claim boundary

This round tests only the Qwen3.5-35B-A3B TP2 layer-0 recurrent-state shape
with 16 local states and the first-turn AgentX-derived fixture. It is a matched
82-step component result, not an online serving or general workload speedup.

The candidate reduces the fitted compensator count from four to one. At the
first boundary it still materializes all four exact-SVD initializer ranks;
later boundaries track the one active old rank and skip three unused
old-factor read/Axpy paths. The ABI-v1 rank-four behavior remains available;
the bounded rank selection and active-rank lifecycle use ABI v2.

## Reproducibility boundary

- StateAxis candidate commit:
  `913853597069ca5684f565b1a0c5d3e49a93fd02`
- source parent: `1995614325bbc409219f024fc8efa8ea75446155`
- branch: `experiment/leapquant-rank-scan-20261008`
- model revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- AgentX conversation: `4b88101f5ed7507b73138a73dfbf55f972de`
- hardware/runtime: Ascend 910B2, logical NPU 0, CANN 9.1.0, eager component
- matched timeline: 82 steps, p=16, five refits, one power iteration
- raw local archive:
  `/root/stateaxis-evidence/leapquant/2026-10-08-rank-scan-20261008T151156Z/`

## Results

The frozen fixture rank scan passed the unchanged correctness gate for ranks
1 through 4. Rank 1 reduced the probe boundary from `4082.38 us` at rank 4 to
`2981.95 us`; reconstruction RMS increased from `7.23e-05` to `0.00119814`,
remaining below the 2% gate.

After active-old-rank specialization, three independent alternating-order
matched runs produced:

| run | compressed (us) | native (us) | ratio | max output RMS |
| --- | ---: | ---: | ---: | ---: |
| r1 | 9949.801 | 10043.067 | 0.990713x | 0.009449386 |
| r2 | 9989.891 | 10361.726 | 0.964115x | 0.009449386 |
| r3 | 9889.791 | 10125.873 | 0.976685x | 0.009449386 |

The median ratio is `0.976685x`, or about `2.33%` lower elapsed time. All
three repetitions passed correctness and batched no-harm. A final ABI-v2 smoke
run also passed (`0.932515x`, maximum RMS `0.009449386`). The original rank-4
row-write result was `1.5829x`, so this is a real component-level turnaround,
but its margin is small and native timing remains variable.

## Decision

Keep disabled and experimental. The result justifies a longer randomized
paired component run and a matched real-online Qwen3.5-35B/AgentX gate. It
does not supersede the narrowly qualified v68 online result until those gates
pass. The next optimization target is the one-time rank-4-to-rank-1 first
boundary, which remains roughly twice as slow as later rank-1 boundaries.

Bitmap or sparse-record encoding is not promoted by this result: the tested
p=16 boundaries contain all 16 records, so there is no demonstrated record
sparsity to exploit. Failed Cube-backend, mixed-library, and fitted-rank-only
runs remain in the raw archive and were not discarded.
