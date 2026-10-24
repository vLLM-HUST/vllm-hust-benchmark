# Native-current NPU qualification

Date: 2026-10-11

This record closes the native-current host/NPU release gate with a pinned,
clean-wheel lifecycle. It establishes ECPA activation and rollback semantics;
it is not a performance result and does not qualify every model, device, or
MOD combination.

## Pinned stack

- Extension Manager source: `00d1b940dc6ce8d60e2cd28708b80d80e9c33a09`
- Manager wheel: `vllm_hust_ext-0.3.0.dev0-py3-none-any.whl`, SHA-256
  `b4d691ce09f3588e17ed4b9a99b7b458b10ae7672f1810a0f8489978c7575078`
- vLLM-HUST host: `fa6f7d29db99d5350e1b2b9c2742c9b621b22b25`
- vLLM-Ascend-HUST qualification patch: PR #46, device-tested source commit
  `95334cc38a3938406fe24caf748030cf4ece5917`; review head
  `2ebe41b411b6190a205edaa6bbd928b485bee83e` additionally carries the
  verified-host import compatibility and lint-only package-index fallback.
  PR #46 is merged on main at
  `82f586f8dcb5f92f8f3d55727bc05b3e2887606d`.
- KV materialization plugin: `e343fc5ce6b4c64611a79df5d797b3db12cd6dac`
- Plugin wheel: `vllm_kv_materialization-0.1.0-py3-none-any.whl`, SHA-256
  `5d467c5dde1ad7d85aa4c4ea7ce119713521e115af573ee20cd589084791bee5`
- Python 3.12.13, CANN 9.1.0, `torch==2.13.0+cpu`,
  `torch-npu==2.13.0rc1`, `transformers==5.19.0`
- Official Triton-Ascend CI wheel built from
  `2be16a4a46e67bfac159529a69ec3c5ba689fd53`, SHA-256
  `f3a5396053794de707110bd8bbc440baed3ef353912ca373013cfa55d02c0f8b`
- Allocated device: Ascend 910B2, container-visible NPU 0
- Model: local Qwen3-0.6B, single device, eager mode

The Triton wheel is a development artifact rather than a public stable PyPI
release. This exact hash is part of the evidence boundary and does not create a
floating Triton support promise.

## Lifecycle evidence

The clean Manager and plugin wheels completed discovery, inspection, check,
enable, plan, render, and supervised `run`. The host reported both
`vllm.request-processing-hook` 1.0 and
`vllm.kv-materialization-runtime-control` 1.0, and retained the built-in
`ascend` platform plugin while adding `kv_materialization`.

The current Host and Ascend sources loaded Qwen3-0.6B, allocated 13.99 GiB of
KV cache, completed Triton kernel warmup, and served two HTTP completion
requests. Engine PID `1283080` emitted two bound `runtime_effective` receipts
for launch `8efd6555-a703-4e4d-9927-a1ebcdb71a31`, with occurrence IDs tied to
the two real request IDs. While that exact process identity was live,
`extension status` projected `runtime_effective`.

SIGINT was delivered to the ECPA supervisor. The API and engine processes
exited, port 8100 became bindable, NPU 0 returned to its 3436 MiB idle baseline,
and `npu-smi` reported no process on either container-visible device. A
post-stop status no longer projected `runtime_effective`; persisted enable
intent remained only `enabled`. Disable removed the bundle from the rendered
environment, forget removed saved intent, and package uninstall removed
discovery. External services were not started, stopped, or claimed by ECPA.

## Boundaries

- Installation, enable intent, live runtime effect, and performance remain
  separate states.
- The observer result is process-owned evidence, not an inference from an
  environment variable, successful import, or HTTP success alone.
- The result qualifies this pinned native lifecycle and release gate. It does
  not assert throughput improvement, production readiness of the plugin, or
  universal support for the wider experimental catalog.
