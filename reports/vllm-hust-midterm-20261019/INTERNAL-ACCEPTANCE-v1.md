# Internal Midterm Acceptance v1

Status: prospective internal protocol, frozen before new measurements. This is not third-party
certification or a claim about actual procurement expense.

## Scope And Gates

### Baseline Identity Clarification (2026-10-22)

The designated acceptance baseline is the official vLLM 0.18 / vLLM-Ascend 0.18 release line. Reuse
existing 0.18 evidence wherever its workload, metric and provenance support the comparison.
Compatibility backports or environment differences do not automatically require a rerun: assess
their actual effect, disclose the applied source identities and retain the original records. An
unmodified official binary is not an additional acceptance gate.

Existing 0.28.1 / Ascend 0.25.1rc2 development-stack comparisons remain useful independent
observations, but are not measurements against the designated 0.18 baseline. B1 may use its
explicitly identified optimized runtime. Historical plans and measured values remain unchanged. See
`BASELINE-IDENTITY-AUDIT-20261022.md` for the recovered evidence and its scope.

B1 means "currently best measured vLLM-HUST optimization solution" （当前已实测最优的vLLM-HUST优化方案）, selected
per workload and metric among eligible measured MODs or compatible combinations. Selection retains
exact configuration and provenance; different optima are not a jointly measured configuration.
Candidate screening and independent acceptance confirmation are separate stages. This clarification
does not change frozen run inputs or retrospectively promote a screening result to acceptance
evidence.

Use Qwen3.5-35B-A3B BF16 on the available Ascend 910B2 hardware. Default to one TP2 instance per
arm. Pin actual model revision, tokenizer, runtime versions, device IDs, workload manifest and
request generation parameters in the run plan. Select the candidate using existing engineering
evidence before the new run; never select the best candidate or concurrency from the measured
acceptance data. Project-level comparison is Native B0 versus vLLM-HUST B1. BetterScale and other
enabled modules are component provenance, not replacements for the project name. Independent
component runs retain their original identities and are not relabeled as completed integrated-system
measurements.

Native B0 and vLLM-HUST candidate B1 must use the same task inputs, precision, hardware allocation,
offered load and accounting boundaries. Runtime/MOD configuration differences must be explicitly
recorded, not silently prohibited or concealed.

Internal gates: workload-reference compute attainment >=50%; output throughput improvement >=10%;
mean request TTFT and TPOT improvements each >=5%; actual input context >=16384 tokens; unit
successful-output-token cost reduction >=10%. Report task correctness and all failures alongside
these gates; no performance pass can hide a task-quality regression. Preserve original dataset
scorers and denominators. Do not infer a new quality threshold after seeing the results.

## Minimum Supplemental Serving Run

Recover existing raw timing evidence first. For missing timing/utilization only, use a dedicated
endpoint, one fixed paired B0/B1 run, each with an excluded 60-second qualification followed by 300
seconds of measured workload and a separately recorded drain. Fixed client concurrency 8; this is a
prospective sampling choice, not a declaration of device saturation. Record running/queued requests
and utilization; do not call the run high-load solely from offered C8. Freeze the actual >=16K input
manifest before launch; use genuine accessible long-input requests when available. Synthetic
long-input supplements must be labelled synthetic and must not be relabelled primary-dataset quality
results. Freeze max output tokens at 256 and temperature at zero for the supplemental serving test;
record the template and thinking setting before execution.

Report all five consecutive 60-second windows, including empty/adverse windows. Window throughput
counts actual output token arrivals within \[start,end). Request latency cohorts are assigned by
request-start time and followed through drain. TTFT uses client start to first output; TPOT uses
first-to-last output divided by output_tokens-1 for requests with >=2 output tokens. Report excluded
undefined latency counts and every failed request. P95/P99 use empirical nearest rank, ceil(p\*n)-1
in zero-based sorted samples. Chunk-level timing must be named as such when individual token timings
are unavailable. Keep whole-run aggregates separate from per-window observations. One pair is an
initial internal test, not a statistically conclusive improvement estimate; diagnose borderline or
unstable results with a predeclared follow-up rather than selecting windows.

## Dataset-Aligned Supplement (2026-10-19)

The dataset supplement uses the existing positive-result configuration and original workload, as
requested before measurement. Its frozen `experiment-plan.json` is retained in
`aligned-mmlu-common.tar.gz`. The C8 synthetic protocol above remains a separate diagnostic; it is
not substituted for the dataset configuration.

MMLU-Pro uses the original 200 selected prompts, unchanged materialized JSONL SHA-256, infinite
arrival rate, 256 output-token cap, temperature 0 and request seed 0. B0 uses the original MMLU
baseline profile; B1 uses the school runtime, BF16/TP2, E16/R36, context 65536, batch-token budget
4096, FULL graphs, MTP2, 20 GiB State/rank and 8 GiB host cache/rank. The same two physical cards
execute sequential B0 then B1 services. All four cards are reserved to exclude shared load; both
two-card service and four-card reservation accounting are reported. Record the common Triton
distribution-name correction and the unchanged-source host rebuild.

Use the original vLLM bench streaming client and initial single-prompt test. Throughput uses
whole-burst output divided by whole-burst duration. Mean TTFT and TPOT retain the client's SSE-chunk
semantics; TPOT is `(latency - TTFT)/(output_tokens - 1)` for multi-token responses. Derived report
P95/P99 use nearest rank with explicit sample count and rank. Keep the client's original linearly
interpolated estimates in raw evidence and separate derived fields. Do not reinterpret this burst as
a fixed-duration window or an actual 16K input test. The existing numeric thresholds are unchanged.
Report request-interval and startup-to-release cost separately, including adverse results; do not
generalize a single burst lifecycle to a long-lived deployment. Instrumented counter runs are not
substituted for uninstrumented serving measurements. Preserve existing full-task quality evidence
separately from 256-token performance outputs.

## Internal Cost Accounting

The previously proposed CNY 3000/card/month is withdrawn before measurement: there is no evidence
supporting it as a representative comprehensive price. Do not put an unsupported currency amount in
the results table or invent a price range. This correction does not require delaying hardware
measurements.

Primary accounting reports allocation consumption q=k*T/N_success_output in card-seconds/token and
card-hours/million tokens (q*1000000/3600). For an identical per-card comprehensive rate in both
arms, report conditional cost reduction = 100\*(1-q_B1/q_B0), explicitly labelled equal-rate
accounting. This is a real resource-efficiency calculation, not a measured currency expense.
Absolute currency cost remains unreported until an approved internal cost sheet or documented
external rate exists. Record its date, hardware, currency, tax, included services, card count and
billing-period convention. If rates differ, use 100\*(1-(rate_B1*q_B1)/(rate_B0*q_B0)) with rates in
currency/card-second. Monthly currency cost/token =
monthly_fee*k*T/(billing_month_seconds\*N_success_output). No arbitrary utilization discounts,
margin factors or unrecorded month-length assumptions are allowed. Rate sensitivity may be shown
only as a labelled scenario, separate from acceptance results, with its range justified or
explicitly illustrative. For equal fees the percentage reduction is independent of the chosen
currency amount. Matched T, k and successful token counts are still mandatory. Do not substitute
historical unmatched runs as an accepted pair merely because the assumed price cancels. All
allocated cards belong in k, including idle allocated cards; isolate service lanes or account for
the entire shared allocation.

Collect timestamped per-card AI Core samples at 1-second intervals throughout qualification,
measured workload and drain. Compute time-weighted utilization on measured boundaries only and
report missing samples and coverage. No arbitrary utilization threshold is imposed as an extra
midterm gate.

## Effective Compute Reference

Use independent same-device/precision workload-shaped compute, HBM bandwidth and collective
communication calibrations. Freeze representative operation shapes, payload sizes and
effective-operation accounting before calibration. Preserve calibration raw durations, bytes,
collective topology and chip roster. With aggregate effective operations F, aggregate HBM traffic B,
collective traffic V, and calibrated aggregate capacities P, BW, CW, use the optimistic overlapping
resource lower bound T_ref=max(F/P,B/BW,V/CW), with zero communication term only when the workload
truly has no communication. Then per-chip reference rate is F/(k*T_ref), measured effective rate is
F/(k*T), attainment is 100\*T_ref/T. This is an explicitly chosen internal resource-bound model, not
vendor peak or the exact application critical path. State traffic-accounting limitations. Do not
describe sampled calibration capacities as guaranteed physical maxima. Report this as attainment
relative to an empirical workload reference. Match each calibration to the actual shapes, precision,
topology and concurrency; do not use one unrelated GEMM maximum for every operation. Define whether
communication bytes count sent, received or link traffic, without double-counting them as HBM
traffic. Retain calibration repetitions and dispersion. If attainment exceeds 100%, do not clip it:
investigate accounting or reference mismatch and withhold the acceptance verdict until the
discrepancy is explained. Do not substitute parameter-count token heuristics or AI Core percentage
for F. If profiler coverage cannot establish F/B/V, this gate stays unverified while the serving
gates continue; publish coverage gaps, not a fabricated attainment.

## Safety And Delivery

Before each launch recheck devices, processes, disk and existing reservations; use the existing
shared reservation mechanism, never kill or preempt other work. Do not modify the shared runtime.
Capture source/import identity, exact commands, request/timing records, server logs, device
sampling, interval/drain boundaries, own-process exit/resource release and SHA-256 manifests. Raw
evidence is immutable; derived calculations and limitations are stored separately. Internal
acceptance and external/customer acceptance must be labelled separately in every report.

## Reporting Credibility

Every headline number must include unit, B0/B1 run IDs, measured boundaries, sample size, failures,
source artifact and a reproducible formula/script. Label each field as measured, derived, assumed,
or unavailable. Unsupported fields are null, never zero. Round displayed numbers to justified
precision (normally three significant digits and one decimal place for percentages); preserve full
raw precision in machine-readable artifacts. Do not present precision as accuracy.

Report all observed windows, not only a peak. Do not count five adjacent windows as five independent
experimental repeats. A single pair is provisional; if a result is close to its threshold or drifts,
specify the repeat plan before collecting more data and retain every repetition. No post-hoc
concurrency search or exclusion of slow windows. Report order and warmup/cache treatment, and use
reversed B1/B0 order in a planned confirmation pair when checking order effects. Do not invent
confidence intervals without a defensible sampling model.

For tail statistics give the actual nearest-rank index and n. A P99 based on fewer than 100 requests
is effectively an extreme order statistic, not stable tail evidence; mark it low-sample and do not
imply precision from decimal places. Even n>=100 does not guarantee tail stability. Throughput and
cost use the same token/time accounting boundary: whole-request output counts must not be divided by
a shorter fixed window, and drained output must not leak into window numerator. Record chunk/token
counting method, successful-output definition and failed-request resource consumption. A
resource-efficiency result is not task-success cost.
