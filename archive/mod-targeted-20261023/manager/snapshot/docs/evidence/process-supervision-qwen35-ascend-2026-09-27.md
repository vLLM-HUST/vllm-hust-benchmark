# ECPA process-supervision acceptance on Ascend — 2026-09-27

Result: **pass** for the Manager-owned launch-group shutdown boundary.

## Bound identity

- Extension Manager PR head:
  `a9c2e7a369e52acf0c13e1ac29a074e757de212b`.
- Loaded module:
  `/accept/source/src/vllm_hust_ext/process_supervisor.py`.
- Loaded module SHA-256:
  `b59004c1244fd369b54c3cca632c25a59480cd833dbbf4cc52f25c79073a52f5`.
- Runtime image: `vllm-hust-verification:latest`, image ID
  `sha256:c037d56b32ebbfaab0deb00fdc632222266ff9d31e536648b1a6daebb7a40efa`.
- vLLM:
  `0.28.1.post1.dev143+gf18cf803c.empty`.
- vLLM Ascend:
  `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`.
- Model: Qwen3.5-35B-A3B, 66.97 GiB checkpoint, TP2 on Ascend
  910B2 devices 2 and 3.

The Manager was invoked with a 30-second shutdown grace period. vLLM used the
multiprocessing executor, eager mode, an 8,192-token maximum model length, and
90% device-memory utilization.

## Anti-confounding setup

The container did not use the Manager as PID 1. An outer shell launched the
Manager, waited for it, recorded the remaining process table, and then stayed
alive for 300 seconds. Therefore Docker container teardown could not remove
workers on the Manager's behalf.

Before termination, the observed host process relationship was:

```text
manager 32533
└── launch group/session 32543
    ├── API server 32543
    ├── EngineCore 44612
    ├── Worker_TP0 48034
    └── Worker_TP1 48035
```

Both workers were visible in host-owned `npu-smi`, each using 56,397 MiB of
process memory. Total HBM usage was 59,769 MiB on each selected device. An
OpenAI-compatible completion request returned `finish_reason=stop` and text
containing `ECPA_READY` before shutdown.

## Intervention and observation

At `2026-09-27T04:20:24Z`, the observer sent `SIGTERM` only to the Manager.
The Manager exited one second later with shell-compatible status 143. The
outer wrapper remained alive. While it was still alive:

- manager PID 32533 no longer existed;
- API PID 32543 no longer existed;
- EngineCore PID 44612 no longer existed;
- worker PIDs 48034 and 48035 no longer existed;
- process group 32543 was empty;
- `npu-smi` reported no running process on devices 2 or 3.

HBM fell from 59,769 MiB per device to 7,547/9,595 MiB immediately after the
Manager exited, then returned within five seconds to the exact measured
pre-run baseline of 3,424 MiB on both devices.

The first preflight attempt, which omitted the host-driver mount, failed before
model loading with `rtGetSocVersion` and is excluded from the acceptance run.

## Claim boundary

This run validates process-group ownership, signal forwarding, descendant
cleanup, and observed accelerator-memory release for a real vLLM-HUST
Qwen3.5 TP2 serving process. It does not validate Mooncake data-path
effectiveness, general plugin runtime effectiveness, or a performance gain.
The machine-readable companion record contains the same bounded result.
