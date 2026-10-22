# ECPA Manifest 0.3 activation smoke

Date: 2026-10-09

Evidence label: `real-online-activation-smoke`

Performance claim: none

## Result

`org.vllm-hust.gdn-state-codec@0.2.0.dev1` was discovered, configured,
enabled, checked, rendered, and launched by `vllm-hust-ext` using its static
`0.3-experimental` Bundle manifest. The Manager projected
`VLLM_PLUGINS=ascend,gdn_state_codec`, the declared environment, and the
bounded `gdn_state_codec` vLLM additional configuration.

The real Qwen3.5-35B-A3B TP2 eager server reached healthy state on two Ascend
910B2 devices. A 19-token chat prompt completed 16 decode tokens with HTTP
200. Both TP workers emitted one `runtime_effective` event, proving that the
codec path executed in the processes that owned GDN decode. The server was
then stopped by operator SIGINT; `npu-smi` showed no remaining process.

This is activation and lifecycle evidence only. It is not latency,
throughput, quality, or performance-qualification evidence.

## Command

```bash
vllm-hust-ext extension configure org.vllm-hust.gdn-state-codec \
  --file /tmp/gdn-manager03-config-20261009.json
vllm-hust-ext extension enable org.vllm-hust.gdn-state-codec
vllm-hust-ext extension check org.vllm-hust.gdn-state-codec
vllm-hust-ext run -- vllm serve /models/Qwen3.5-35B-A3B \
  --host 127.0.0.1 --port 18085 \
  --served-model-name Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 --max-model-len 2048 \
  --gpu-memory-utilization 0.9 --enforce-eager
```

The Manager configuration enabled the otherwise inert plugin and selected the
release library directory:

```json
{
  "environment": {
    "GDN_STATE_CODEC_LIBRARY_DIR": "/root/stateaxis-builds/gdn-batched-quant-init-20261009/lib",
    "VLLM_HUST_GDN_STATE_CODEC_ENABLE": "1"
  }
}
```

## Preserved setup failure and correction

The first real launch reached both workers but failed while constructing the
model because `libstatecentric_gdn_state_codec_boundary_runtime.so` could not
resolve a same-directory kernel dependency. This exposed an accidental
dependency on the former StateAxis launch environment. The codec runtime now
preloads all six local DT_NEEDED kernel libraries with `RTLD_GLOBAL`; a direct
loader smoke passed and the identical ECPA launch then reached healthy state
and served the request.

## Reproducibility boundary

- codec commit: `0de7f0c30910421c78b082efe84d4316871b11c3`, clean
- Extension Manager commit:
  `075dbcbee641e22376300069aab1f7e9f275aa5d`, clean
- vLLM commit: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, clean
- vLLM Ascend commit: `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`, clean
- wheel SHA-256:
  `40cb8a4fdcfcc986052e411beb751e58fe8578e55d2cafb1d4f9b126ec5b905a`
- native-library-set SHA-256:
  `cd766fcd8ffa315defd887ef9571693126924cfaa8c6fab64f7964510e9a2e1a`
- model config SHA-256:
  `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`
- runtime: vLLM 0.23.0, vLLM Ascend 0.23.0.post1, CANN 9.1.0
- hardware: two Ascend 910B2 devices, TP2
- graph mode: eager

## Validation

- repository tests: 10 passed
- focused Ruff checks: passed
- clean wheel build: passed
- ECPA discovery/configure/enable/check/dry-run: passed
- real ECPA server start/health/request: passed
- worker `runtime_effective` evidence: two events, one per TP worker

Decision: retain the ECPA-managed integration as the sole supported launch
path. Keep it disabled and unqualified until matched quality and performance
tests pass.
