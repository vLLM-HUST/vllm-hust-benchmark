# Findings

## 2026-07-18 Initial State

- Current branch is `main`, `HEAD=f648c44f50fe3a0cda890a37b222dd55e967d4e4`,
  and `HEAD...origin/main` is `0 0`.
- Worktree is dirty before this session; existing modified files include
  NPU6 runtime hook analysis, diagnosis outputs, paper assets, and tests.
- Submodules:
  - `third_party/llm-serving-workloads` at
    `76e24c85bcab76ecfabb831c9444002b6efffd58`.
  - `third_party/vllm-hust` at
    `0daab7a300fb91efebd35b221b9a6f6c6d7486f8` on
    `faculty-twin-runtime-20260706-21-g0daab7a30`.
- Top-level AGENTS requires graph-mode formal vLLM-HUST evidence; eager is
  diagnostic only and must not be published as formal benchmark data.

## Evidence Audit

- Existing live-ish controlled artifacts are still `existing-server-probe`
  source runs plus `derived-artifact` diagnoses, not benchmark-owned
  repo-launched `real-online` runs.
- Positive controlled cases present:
  - slow-stream client fault: 4/4 measured success, client diagnosis
    `streaming`, runtime hook coverage 5/5 complete chains.
  - structured decode-heavy: 4/4 measured success, client diagnosis `decode`,
    runtime hook coverage 5/5 complete chains.
  - prompt-heavy low-output: 8/8 measured success, client diagnosis `prefill`,
    runtime hook coverage 9/9 complete chains.
  - concurrency sweep: 27/27 measured success, runtime hook coverage 30/30
    complete chains, concurrency 2 exposes a severe runtime prefill tail.
- Missing positive live matrix classes: queue, KV pressure, cleanup. Existing
  synthetic seven-fault accuracy cannot support live diagnosis accuracy.
- Existing 12.2s tail analysis localizes the anomaly to coarse runtime
  `scheduled`→`prefill_done`; observed metadata does not distinguish scheduler
  handoff, KV allocation/cache pressure, graph/batch transition, or prefill
  kernel execution.
- Existing overhead pair plan reports TTFT/latency deltas and trace bytes, but
  not TPOT, CPU, host memory, or NPU HBM overhead.

## Controlled Fault Matrix Audit

- Added `.benchmarks/results/npu6_controlled_fault_matrix/` as a
  `derived-artifact` audit over checked-in NPU6 probe/diagnosis directories.
- Available rows: slow streaming backpressure, structured decode-heavy,
  prompt-heavy low-output prefill, and the concurrency-2 prefill tail. These
  are still `existing-server-probe` plus `derived-artifact`, not repo-launched
  `real-online`.
- Missing real-online rows are explicit: queue pressure, KV pressure, cleanup.
  Raw-log baseline is missing, and manual time-to-root-cause is not measured.
- Available-row causal-rule agreement is 4/4 with missing-event-rate p95 max
  0.0, but this only audits checked-in artifacts and must not be used as
  internal-only diagnosis accuracy.

## 2026-07-26 Submission Readiness Audit

- The existing three-page PDF is not submission-ready: it uses eight small
  tables, directory-by-directory narration, internal readiness strings, and
  repeated claim disclaimers instead of a normal systems-paper argument.
- The strongest controlled evidence is limited to three existing-server-probe
  scenario correspondences: injected slow client reads map to streaming (4/4),
  decode-heavy output maps to decode (4/4), and prompt-heavy/low-output maps to
  prefill (8/8). Repeated requests within each scenario are not independent
  lifecycle experiments and cannot establish a population accuracy rate.
- The concurrency-2 tail is a separate observation, not controlled-fault
  ground truth. Two chains exceed one second and runtime prefill p95 reaches
  12,261.98 ms, but current fields cannot distinguish scheduler handoff, KV
  allocation/cache pressure, graph or batch transition, or device execution.
- Existing-server manifests do not establish graph mode or custody of the
  serving process. Formal evidence therefore requires a repository-controlled
  graph-mode protocol; the current artifacts cannot be promoted to formal
  online results.
- The small client pair measures probe event-construction overhead. The
  separate small runtime-hook pair does not support TPOT, CPU, host-memory,
  NPU-HBM, or broad-workload overhead claims.
- Paper restructuring now uses two main displays: a human-readable
  coverage/gap table and a compact concurrency-tail table. The source assets
  remain available for audit but are no longer each promoted to a main-text
  table.
- PDF visual review, pages 1--2: title, abstract, contributions, background,
  design, method, and result lead-in are readable at 144 dpi with balanced
  columns, embedded fonts, no clipping, and no visibly excessive word
  fragmentation. Page 1 states the not-submission-ready boundary in both the
  abstract and introduction without turning the paper into an internal status
  report.
- Final PDF visual review, page 3: the coverage table uses human-facing labels,
  combines the three unmeasured fault classes into a readable gap row, and
  explicitly labels the concurrency tail as an observation. The final-page
  columns are balanced; the disclosure and six references are legible with no
  overlap, clipping, or excessive orphaned whitespace.
- Tectonic alone retained volatile XMP dates. Reconstructing the PDF page by
  page with pypdf, clearing creation/modification metadata, and atomically
  replacing the file preserves embedded fonts and produces stable bytes.
  Two complete builds match at SHA-256
  `b4da67d59a466b107bd5e8a4d3e3236b39ed1b397415d7099ad5dcba6543cf71`.

## 2026-07-27 Offline Intervention Contract

- Added a matched control/intervention fixture and evaluator that reports
  dominant-span localization independently from causal support.
- A dominant prefill span without intervention linkage remains
  `localization_only`. A deterministic decode intervention is supported only
  when the target delta clears the threshold and every non-target span remains
  unchanged.
- The fixture is `simulation/model`; it validates evaluator semantics, not a
  server or Ascend causal claim. Controlled graph-mode live attribution remains
  `NOT_M0_PROVEN`.
