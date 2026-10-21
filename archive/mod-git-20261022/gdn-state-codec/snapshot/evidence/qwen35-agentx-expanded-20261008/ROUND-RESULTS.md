# Qwen3.5-35B / AgentX expanded qualification

Date: 2026-10-08  
Evidence label: `experimental-negative`

## Reproducibility boundary

- StateAxis base: `3fce1239c4c3f8fddf45a3f98259ccc79bc0d194`
- Mod base: `87b0b149065f97be17c2867a781ea5cad5c20e69`
- Model: `Qwen/Qwen3.5-35B-A3B`
- Hardware: 2 x Ascend 910B2, TP2
- Runtime: CANN 9.1.0, vLLM 0.23-derived StateAxis, eager mode,
  `max_model_len=2048`
- AgentX first-turn prompt sizes: 328, 457, 648, and 1033 tokens
- Full raw evidence archive:
  `/root/stateaxis-evidence/leapquant/2026-10-08-agentx-expanded/`

No unmodified three-turn AgentX segment fits the 2048-token limit, so the
round makes no multi-turn claim. The native concurrency-2 control completed;
the experimental path failed closed because it currently admits a single
decode request.

## Result

Exact SVD plus four boundary power iterations passed a 128-token real-shadow
request, but its component amortized ratio was 22.280x native and boundary
refits took about 32–33 ms. It fails the no-harm gate.

A cheap rank-four sketch whose boundary discarded the prior compensator failed
the 2% real-shadow relative-RMS gate at step 22. Reusing exact SVD initialization
with that boundary failed at step 37, identifying boundary information loss as
the dominant problem.

Preserving and scaling the old rank-four compensator passed the component gate:
maximum component output relative RMS was 0.0079148, amortized ratio 0.979686,
and batched ratio 0.468530. In the real Qwen3.5-35B shadow run it failed at
`language_model.model.layers.17.linear_attn step=65` with relative RMS
0.023863366. That is progress in failure distance, not a qualified speedup.
Replace-mode testing was not run after the exactness failure.

The earlier quant-only candidate was also rejected: it was 0.44% slower in
median request time and 0.37% slower in median ITL than the warmed native
control, while three of four output hashes differed.

### Periodic-refresh oracle

A follow-up probe used Qwen3.5's actual BF16 recurrent-state contract and one
exact p4 refresh every four cheap boundaries. Across 82 component steps it
held maximum RMS to 0.0097923, but cost 2.574x native in total (3.446x in the
batched loop); the exact refresh itself took 32.432 ms. A 128-token real-shadow
request completed with the exact-reference output hash, 33.5293 seconds total,
246.646 ms median ITL, and worst logged layer RMS 0.019905092. This confirms
the accumulated-boundary-error diagnosis, but it is too close to the 2% gate
and fails no-harm. It is an oracle, not a retained optimization.

## Probe correction

The server may emit HTTP 200, partial SSE output, and then an embedded engine
error. `tools/agentx_online_probe.py` now requires both a finish reason and
terminal usage. Two unit tests cover accepted completion and truncated-200
rejection.

Validation completed with 23 StateAxis parser tests, five Ascend Python
component tests, and two probe tests passing. A fresh AscendC build produced
the library used in the real runs; a later reuse of its generated tree hit a
CANN preprocess `unknown file type` failure and is classified as an
incremental-build infrastructure failure.

## Bound raw artifacts

The source archive's `SHA256SUMS` contains every raw measurement and log. Key
bindings are:

- `component-exact-svd-p4.json`:
  `3674882b7ea5035e038e2906f0fea9884e7c46324f9aa3ff788521f63b388d68`
- `component-sketch4-preserve-comp-fast-boundary-r4.json`:
  `64ebfcfec1ebd3fcf4cd458fbc53eeabb250217751028a041807c1893c80a56c`
- `exact-svd-p4-shadow-server.log`:
  `f01c7fa9947f86353483f795b7a4cec998d076c18912d3f6e4cc0cd02a53f2f7`
- `sketch4-preserve-shadow-server.log`:
  `7924b8cece8c37d10af8df6812a90b6310f8e86e28df55bb330d28d9a93ab72e`
- full round summary:
  `bd3aeb3aaaa13dbd6be7e742cf939479b0005bda228a1baab9b7426e93916d89`
- `component-periodic-refresh-i4-s82.json`:
  `33974aae39cddc97b281dcab573ec2eb793b0f0dcee9bf0d18afe546f3b914c7`
- `periodic-i4-shadow-first.jsonl`:
  `17a0a36348ecb17183997c287e2f7a8ea37d1b98ee4ec04dec4e32ec95ab81ec`
- `periodic-i4-shadow-server-v2.log`:
  `40e10e81e4d8a095360e68d8b172688639ccd828eec9bc70887cf75c783126a1`

## Decision

Do not merge or enable the experimental kernel. The next candidate should
retain the prior rank-four factors, sketch the newly accumulated residual,
and cheaply recompress the combined rank-eight update to rank four on Cube.
It must pass the unchanged 2% layer-by-layer real-shadow gate before any
replace-mode measurement.
