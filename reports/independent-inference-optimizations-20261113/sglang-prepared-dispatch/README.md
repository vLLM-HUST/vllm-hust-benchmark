# SGLang Prepared Dispatch

The original paired confirmation used AB/BA/BA/AB order, eight fresh unprofiled
processes, six measured cohorts and two warmups per process. All four pairs
were faster: throughput ratios 1.091281754,1.078443068,1.095640901,1.134347335;
median 1.093461328. Both arms retained the accepted batch-copy patch and normal
update stream. Logprob output was enabled.240 requests/15360 generated tokens
had zero token/top1-ID mismatch and zero selected/top1-logprob difference.

Independent sampled diagnostics measured 28 dispatch calls totalling median
2.8736695 vs1.9533055 ms and whole update worker4.624073 vs3.941599 ms.
Those sampled diagnostics are separate from the unprofiled throughput result.

Device qualification also covered batch2 and a heterogeneous active batch
3→2→1, with every live KV payload checked. The initial generated-header ABI
candidate failed before timing; the successful revision uses the registered
TensorList-out signature. No CANN executor cache or alternate stream is added.

The packaged C++ source has exactly the measured source SHA256. Packaging adds
strict runtime/source checks, explicit mount/unmount, isolated build identity,
and optional PID receipts. Its CPU tests cover live KV forwarding, unsupported
payload rejection, atomic plan preparation, inert defaults, kill switch, source
drift and exact rollback. **The packaged adapter has not yet had a new device
performance confirmation.** The historical gain refers to the measured core
under the pinned conditions in `evidence/protocol.json`.

To requalify this package, mount the same accepted copy baseline for both arms;
use the protocol's batch/input/output/cohort counts and AB/BA/BA/AB process
order. First run the three dynamic cases with the metadata guard enabled and
retain all token IDs and selected/top1 logprobs. Then run unprofiled confirmation
with metadata diagnostics off. Record exact commands, package/engine source
revisions and diffs, model identity, hardware/software identity, native build
receipt, per-worker installation/effect receipts, full outputs and every paired
timing in fresh directories. Do not multiply this incremental gain by the
earlier copy-baseline result.

Canonical code repository: https://github.com/vLLM-HUST/sglang-hust-prepared-dispatch
