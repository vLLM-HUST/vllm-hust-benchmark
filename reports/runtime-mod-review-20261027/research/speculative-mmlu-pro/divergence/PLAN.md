# First-divergence diagnostic — 2026-10-27

Fletcher explicitly authorized locating the first divergence and comparing token/batch/prefill
configurations. No new full MMLU or DiffSpec campaign is authorized here.

1. Offline compare saved token sequences for all frozen requests, without score selection; preserve
   first mismatch and parity of prompt lengths.
1. On ascend87 physical0/1, using existing mapping/locks/placement guard, run three serial fresh
   engines: native, MTP2, adaptive. Preserve frozen engine settings, 14×16-token warmup and first16
   perf56 prompts in original order; cap diagnostic generation at256 tokens. This changes the
   workload and cannot reproduce the exact historical asynchronous schedule. Synchronous diagnostics
   are not performance evidence.
1. Record actual scheduler token allocation, per-request computed positions, target top4 logits,
   speculative drafts/target indices/bonus and actual sampler output. Compare strict greedy
   reference against actual sampler; no changes to acceptance/model math. Probe instrumentation can
   alter online-controller timing.
1. Each arm limited to900sec, total initial diagnostic budget45min. Stop on instrumentation/model
   failure and inspect rather than blindly repeat. No shared source/package edits.
1. Distinguish (a) accepted token inconsistent with target argmax, (b) differing target logits under
   common history, (c) position/history/state mismatch. If needed, use a small fixed-common-prefix
   replay after initial evidence; do not claim root cause from top-k difference alone.

## Norm lead (before leaf probe)

Fletcher pointed to GenNova's native Qwen norm precision correction. The app project label GenNova
maps to local `LiveInference`. Read its norm numerical guide and standalone probe before rerunning.
Host custom AddRmsNormBias five kernel headers byte-match GenNova's audited donor. Add one bounded
physical0-only leaf check, max900sec, same-input synthetic BF162048-wide
rows16/32/48/80/112/1024/8192, native loaded operator vs GenNova reference-semantic Triton kernel
and CPU oracle. Capture/replay each; no shared source edits, no new full-model load, no performance
or MMLU claim. Save actual extension/library paths. Source match does not independently establish
prior model-run binary execution.

## Corrected-norm counterfactual (authorized2026-10-27)

Fletcher explicitly requested installing the norm correction and checking improvement. Install only
via separate private `norm-fix-guard`: replace AscendGemmaRMSNorm.forward_oot in all three arms with
GenNova's existing reference-semantic Triton implementation, preserving residual API and offset
weights. Dedicated custom-op wrappers retain graph compilation and provide actual per-worker
kernel-call witnesses; no shared package changes. Repeat same first16 perf56,256-token cap,14×16
warmup and synchronous logits instrumentation. Same physical0/1, three serial fresh engines,900sec
per arm; stop on any failure. Compare output-prefix agreement and target ranking margins against
prior three arms; short256-token results cannot measure MMLU quality. No full280 re-evaluation
automatically authorized by this short-stage protocol.

## Bucket-only follow-up (user redirect)

Fletcher stopped further norm investigation and requested a bucket comparison. Native/MTP2
corrected-norm short runs completed; adaptive was interrupted during warmup/generation and must not
be reported as completed. Preserve original outputs and stop receipt.

Run one new native-MTP2 arm with original norm, fixedgamma2 and all frozen settings except requested
capture list copied from adaptive. Same first16 perf56,256 cap,warmup,diagnostic
probes,physical0/1,900sec bound. Host source shows `adjust_cudagraph_sizes_for_spec_decode` rounds
buckets to multiples of3: identical requested lists will NOT yield identical runtime graph
descriptors. Record resolved sizes and keys after runner initialization; this tests baseline bucket
expansion, not perfect cross-gamma graph identity. Other host cards have foreign workloads; no
timing claims. Do not patch baseline dispatcher or import vSpec just to force unsupported buckets.
