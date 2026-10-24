# Historical 0.9 Packaging and Release Guide

English | [简体中文](../../development/packaging-and-release.zh.md)

> Historical 0.9.0 release recipe. Use the [current release guide](../../development/packaging-and-release.md).

This repository follows the BidKV packaging guide. The PyPI distribution is
`vllm-ascend-kvcompress-hust`, the import package is
`vllm_ascend_kvcompress`, and the stable Extension Manager ID is
`org.vllm-hust.ascend-kvcompress`.

## Release gates

Before publishing, keep the project version, `__version__`, and manifest
version identical; run CPU, package, NPU, lifecycle, and declared service
tests; freeze host/model/calibration/data provenance; and review the complete
diff. PyPI files are immutable, so any code change requires a new version.

Version 0.9.0 remains experimental. It publishes method API v1 for external
compression methods. Its validation supports bounded engineering performance
statements, not production readiness or an official website score.
The release gate is the current
[Frontier protocol](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.md): matched synchronized host runtime,
complete 262,144-context APC/MTP2/async/FULL_AND_PIECEWISE/align service checks,
auditable B0/B1 runs for both official 900-second SWE and AgentX workloads, and
at least one positive, protocol-valid Frontier workload result. Disclose any
regression in the other workload. Historical V4.6 records do not substitute
for those results. Do not upload to PyPI until the release owner confirms the
exact candidate. The post-merge AgentX 900-second
pair improved output throughput by 11.18% and decode P90 by 5.27%; the SWE
pair was protocol-valid but regressed by 2.95% and 1.34% respectively. Neither
pair reached a 256K actual prompt, and neither is an official website result.
The 2026-10-08 exact 0.9.0 candidate repeated both official 900-second pairs on
the pinned release stack. SWE decode P90 improved 1.05% while throughput and
TTFT regressed; AgentX decode P90 improved 1.14%, TTFT and ITL improved, and
throughput was flat within 0.0001%. Both pairs were protocol-valid with no
failed requests or invalidity reasons. These local smokes satisfy the bounded
release gate, but remain single engineering observations rather than formal
one-hour or official website results.

The exact 0.8.0 wheel passed Ascend kernel and plugin-owned GDN fallback
numerical smokes in the locked CANN 9.1/PyTorch 2.13 environment. A Qwen2.5-
14B-Instruct FP16/TP=1 service using that wheel, NPU 0, and `--enforce-eager`
returned `/health` 200 and answered `42` on a 10,843-token retrieval prompt.
The scheduler and worker acknowledged compression to 8,192 physical KV tokens.
The same host's default compiled start did **not** pass: its AOT path raised
`AssertionError: expected OutputCode, got GraphModuleImpl` during cache
initialization. This additional wheel smoke therefore qualifies the eager
path only; it does not turn the earlier Frontier results into a compiled-mode
Qwen2.5 qualification.

## Build and inspect

Prefer the BidKV `uv` workflow:

```bash
uv build --no-sources --out-dir dist
```

If `uv` is unavailable in the release environment, the equivalent validated
fallback is:

```bash
python -m build --no-isolation --outdir dist
```

Then inspect exactly the candidate files:

```bash
python -m zipfile -l \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
python -m twine check \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
sha256sum dist/vllm_ascend_kvcompress_hust-0.9.0*
```

The wheel must contain code, `LICENSE`, `NOTICE`, and
`manifests/vllm-hust-extension-v0.3.json`; entry-point metadata must contain
`vllm.general_plugins`, `vllm_hust.extension_bundles`, and the
`vllm-ascend-kvcompress-calibrate` console script. Neither archive
may contain `artifacts/*.pt`, raw data, service logs, or historical
`docs/dev/results`. The sdist includes the static HTML leaderboard and its
generated browser data bundle; `docs/evidence` may contain only aggregate
evidence summaries.

## Isolated lifecycle

Install the wheel into the same environment as the supported hosts, with no
source checkout on `PYTHONPATH`:

```bash
python -m pip install --no-deps \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension check org.vllm-hust.ascend-kvcompress
vllm-hust-ext run --dry-run -- python -c 'print("manager-run-ok")'
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall -y vllm-ascend-kvcompress-hust
```

The wheel must have no core `Requires-Dist` entries. In particular, do not add
`vllm`, `vllm-ascend`, Triton-Ascend, NumPy, or OpenCV to the plugin's core
dependencies. The accelerator host is provisioned from a separately validated
lock, and the extension manifest owns the compatible host range. Re-resolving
the host while installing a plugin can combine the NumPy-1-only
Triton-Ascend 3.2.2 line with the NumPy-2-only OpenCV requirement in current
vLLM releases.
The optional `calibration` extra may declare Accelerate; multi-device automatic
calibration must fail with an actionable error when it is absent.

Confirm that `extension list` no longer discovers the plugin. Reinstall,
configure, and enable the exact wheel that will be published.

## Publish and verify

Use a project-scoped token owned by the authorized `intellistream` PyPI
organization/project. Do not pass the token on the command line or commit it:

```bash
export UV_PUBLISH_TOKEN='<read from the secret store>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
unset UV_PUBLISH_TOKEN
```

If the protected publisher uses Twine instead, set `TWINE_USERNAME=__token__`
and `TWINE_PASSWORD` from the secret store, then upload the same two explicit
files. Finally install 0.9.0 from production PyPI without cache, verify hashes,
manager discovery, enablement, `/health`, and one correctness case.
