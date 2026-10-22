# vLLM Ascend PyramidKV HUST

Owner-maintained extraction of the PyramidKV provider work preserved in the
archived vLLM-HUST and vLLM-Ascend-HUST repositories.

**Current status: active external method with an exact-stack functional
qualification and paired evaluation evidence. The explicit 1368/beta-2 profile
passes the recorded quality gates; performance promotion remains pending.**

The package is discoverable as `org.vllm-hust.ascend-pyramidkv` and publishes
the `pyramidkv` method through the shared adapter's
`vllm_ascend_kvcompress.methods` entry-point group. The manifest carrier is
active. Manifest 0.3 declares the shared adapter as an explicit Bundle
dependency and claims the PyramidKV method name exclusively, so ECPA rejects
missing/disabled/incompatible owners and duplicate registrations before host
startup. Importing the top-level package never patches or activates vLLM.

The release-validation model is **Qwen3.5-35B**, using the official model ID
`Qwen/Qwen3.5-35B-A3B`. The package also retains the earlier Qwen2.5-14B
grouped-GQA device oracle as historical provider-only evidence; that result is
not a substitute for Qwen3.5 serving validation.

## Host ownership and current interface gap

The former host-side Drafts in `vllm-hust` and `vllm-ascend-hust` were
withdrawn. The confirmed shared lifecycle owner is
[`vllm-ascend-kvcompress-hust`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust),
whose documented external method namespace is
`vllm_ascend_kvcompress.methods`. Its merged
[PR #9](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/9)
provides the accepted final-prefill query-window observation contract.

Merged [PR #10](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/10)
adds eager consumption of method-returned per-layer physical lengths. Merged
[PR #12](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/12)
now consumes `required_recompute_tokens` during prefix-cache admission and
fails closed when Query/APC contracts or host lookup seams are inconsistent.
Merged [PR #13](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/13)
adds predeclared, address-stable per-layer
metadata for `FULL_AND_PIECEWISE`, layer-scoped Query observation, and an MTP2
speculative-common metadata view with draft rollback safety. Follow-up fixes
cover synthetic FULL capture batches, graph-replay query observation, and
empty async execution steps.

The registered `PyramidKVMethod` implementation targets the public contracts
from merged shared-host PR #13 and requires shared-host method API v1,
published by the 0.9 release line. Its
CPU tests cover chunk-spanning query capture, transaction identity, paged
materialization, unequal per-layer results, auxiliary MTP cache materialization,
and graph-stable metadata. The combined APC, exact Qwen3.5 MTP2, async,
chunked-prefill, and `FULL_AND_PIECEWISE` profile completed a 5007-token
functional serving request on TP=2 Ascend 910B2 with CANN 9.1; the transaction
compressed the full-attention KV state to 2048 tokens and completed decode.

Longer task evaluation subsequently exposed a FULL-graph cache-write bug in
the shared adapter: capture retained common slot addresses while replay updated
per-layer buffers. The installation below pins the host-compatible fix from
[shared PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19).
The [paired evaluation receipt](evidence/current/2026-10-09-paired-evaluation/README.md)
preserves the initial failure, corrective evaluation, and limits of the result.
A short HTTP-success smoke alone did not detect this decoding failure.

The corrected 409-request-per-arm run passed separate 150-example development
and holdout QA-F1 gates (mean losses 0.59 and 0.26 points). Both arms completed
32K-context requests at concurrency 1 and 4. Compressed long-input throughput
was 2.6%–10.3% lower, and peak device HBM was about 0.4 GiB higher despite
lower active KV-pool usage. These results support the stated functional and
quality scope, not a general speedup or device-memory-saving claim. The original
512/beta-20 profile did not pass the development quality gate.

## Inspect and enable

```bash
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@98903e416bdb593186b8245fd95180dafde995b9"
python -m pip install --no-deps "vllm-ascend-kvcompress-hust @ git+https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust.git@19f322130e1b3841da953b1b421bcf3259e9b8dc"
python -m pip install --no-deps .
vllm-hust-ext extension inspect org.vllm-hust.ascend-pyramidkv
```

Inspection must report an active `vllm_ascend_kvcompress.methods:pyramidkv`
implementation. Enable both the shared lifecycle owner and this method bundle:

```bash
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension enable org.vllm-hust.ascend-pyramidkv
```

ECPA intentionally refuses the second command when the shared owner is not
installed, version-compatible, and already enabled. Enabling this Bundle
registers the method; selecting `pyramidkv` and its method configuration still
belongs to the shared adapter's saved configuration.

For standalone provider tests, install the test extra in an environment with a
host-compatible PyTorch build:

```bash
python -m pip install -e ".[test]"
pytest -q
```

On an Ascend test host, run the opt-in grouped-GQA device oracle with:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 PYRAMIDKV_RUN_NPU_TESTS=1 \
  pytest -q tests/npu/test_selection.py
```

Do not install a generic PyTorch wheel over an existing torch-npu environment.
Use the matched PyTorch supplied by the Ascend host and install this package
with `--no-deps` when appropriate.

## Current boundary

- The package registers only the method entry point; the shared adapter owns
  the `vllm.general_plugins` host integration.
- No import-time monkey patching is performed.
- Runtime compatibility fails closed outside the exact qualified model,
  dtype, TP, CANN, scheduling, MTP, cache, and Graph profile.
- Historical NPU, LongBench, and performance results are supporting evidence,
  not measurements of this repository's current head.
- The current provider head has a real-device Qwen 40-to-8 GQA selection
  oracle on Ascend 910B2/CANN 9.0; this does not replace serving validation and
  will not be rerun as a substitute.
- The qualified functional target is CANN 9.1, official model
  `Qwen/Qwen3.5-35B-A3B` (display name: Qwen3.5-35B), BF16, TP=2, APC, MTP=2,
  async scheduling, chunked prefill, `FULL_AND_PIECEWISE`, and
  `mamba_cache_mode=align`.
- Current-source quality, capacity, latency, throughput, and HBM measurements
  are available for the explicit 1368/beta-2 profile. Release promotion and
  shared graph-fix review remain pending; the measured long-input throughput
  regression needs optimization before a positive performance claim.

See:

- [provenance](PROVENANCE.md)
- [maintainers](MAINTAINERS.md)
- [support matrix](docs/support-matrix.md)
- [host contract](docs/host-contract.md)
- [install and rollback](docs/install-and-rollback.md)
- [Qwen3.5 source-install serving smoke and runbook](docs/qwen35-serving-smoke.md)
- [Paired evaluation protocol and reproduction](docs/issue-1-evaluation-protocol.md)
- [Current paired quality, capacity, performance, and HBM evidence](evidence/current/2026-10-09-paired-evaluation/README.md)
- [legacy evidence inventory](evidence/legacy/README.md)
- [public historical Ascend 910B2 result](evidence/legacy/2026-08-13-ascend910b2/README.md)
- [current provider-only Ascend 910B2 oracle](evidence/current/2026-09-03-provider-npu/README.md)

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
