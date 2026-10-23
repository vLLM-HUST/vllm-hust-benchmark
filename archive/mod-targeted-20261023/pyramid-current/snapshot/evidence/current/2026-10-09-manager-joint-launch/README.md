# Managed Qwen3.5 joint-launch receipt

This functional receipt advances issue #1. Manager commit
`d22088cf6a45aeb7c47e39101607a00d87bf2006` (merged through
[Manager PR #40](https://github.com/vLLM-HUST/extension-manager/pull/40)) fixes
provider-plan composition. The exact tested wheel SHA-256 is in
`wheel-identity.json`; its changed source files were compared with the committed
sources before launch. The merge tree is identical to that tested source tree.

## Result

- Real `vllm-hust-ext run -- python ... serve` supervision, with both Ascend
  KVCompress and PyramidKV bundle IDs in the child process environment.
- Configuration loaded from isolated Manager state: budget 1368, beta 2,
  threshold 4096; no `VLLM_ASCEND_KVCOMPRESS_CONFIG` override in the server.
- Three expected-text HTTP responses passed. Both long requests compressed
  5,312 semantic tokens to 2,048 physical tokens, with matched scheduler commits
  and acknowledgements from TP0 and TP1.
- Repeated prompt hit 2,048 cached tokens; MTP drafted 14 and accepted 10 tokens.
- Final health check succeeded and the final managed server was left running.

The host is Qwen3.5-35B-A3B BF16 TP2 on Ascend 910B2 physical devices 6/7,
CANN 9.1, the pinned v0.23 source stack and shared graph-slot backport. APC,
MTP2, async scheduling, chunked prefill, Mamba-align and FULL_AND_PIECEWISE graph
capture remain enabled. `source-revisions.json`, `processes.json`, the launch
wrapper and `manager-config.json` record the actual identities and arguments.
The earlier paired report records the model shard hashes and container identity;
these were not changed by the Manager fix.

## Lifecycle and measurement boundary

A preparatory managed launch using an earlier development wheel also passed the
smoke, then received SIGTERM. `between-launches-npu.txt` records no running
processes on either NPU before the final launch. This is a before-launch resource
release receipt, not a claim that the final wheel was stopped again. The exact
final wheel remains serving. The two development wheels differed in type
annotations/variable naming and formatting, not the unchanged supervisor module.

The final startup monitor issued two model-file-only cache-advice events above
the 48 GiB threshold in this 64 GiB pod. It ended at readiness, before the final
smoke. `ready.json` and `startup-cache-events.json` preserve this sequence. Local
readiness uses a proxy-free HTTP client. No cache advice runs during requests.

This is not a new paired experiment or quality/performance claim. The prior
LongBench tradeoffs and MMLU-Pro non-exercise result remain unchanged. The
provider algorithm, system Torch and model files were not changed.

## Reproduction and environment promotion

Use [the managed-launch runbook](../../../docs/qwen35-managed-launch.md) in the
same prepared host. `serve.sh.txt` and `env.sh.txt` record the isolated wheel and
state used for final validation. The raw smoke inputs, outputs, metrics and
matched log excerpt are in `smoke/`; the complete server log is also preserved.
`processes.json` records only a selected non-secret environment allowlist.

After success, the same wheel was installed into the existing isolated runtime
venv with `--no-deps`, and its default Manager state was configured to the same
validated selection. The old default configuration is backed up locally. The
promoted prepared-host entry point is `/root/workspace/pyramidkv-runtime/serve-managed.sh`
(`promoted-serve-managed.sh.txt`); its default-state dry-run is archived. Existing
direct-launch paired archives are untouched. Stop the current managed server
before invoking another server on the same port/devices.

Validation: Manager 617 tests, strict mypy, Ruff, build and Python 3.10/3.12 CI;
PyramidKV 68 passed / one opt-in NPU test skipped, plus this real-device smoke.
Package tests and the functional smoke do not extend the qualified model matrix.

Verify the immutable artifact set from this directory with `sha256sum -c SHA256SUMS`.
