# Device reallocation and managed-service recovery

This receipt separates the failed pre-rebind health observation from the later recovery attempt.
All machine timestamps in the raw artifacts are UTC. This is functional evidence, not a new
matched performance comparison or a five-dataset task-quality result.

## Before rebinding

`receipt.json`, `npu-smi.txt`, and `health.txt` preserve the original observation:

- The previously recorded Manager PID 710605 was absent and localhost port 8000 refused the health
  request. This does not establish when or why the previous service exited.
- Only physical NPU 1/6 device nodes were exposed. The archived NPU 1 sample reported 23,896 MiB HBM
  and 11% AICore utilization despite no visible process in this container. A container process list
  cannot establish that a physical device is free; the unknown workload was not terminated.
- The observed container memory limit was 64 GiB. Its zero OOM counters do not prove that a prior
  container never experienced OOM.
- `runtime_restarted=false` and `model_calls=0` describe this original observation only. The files
  have not been rewritten to describe the later recovery.

## Recovery after rebinding

The new allocation exposes physical NPU 0–7. The initial recovery sample found them idle; the
service selected physical 6/7, corresponding to logical TP workers 0/1. The allocation now has
32 CPU quota and 256 GiB memory, whereas the earlier benchmark had 8 CPU quota and 64 GiB memory.
These differences prohibit treating recovery timings as a matched comparison with that benchmark.

The task-local `env.sh` now accepts `PYRAMIDKV_VISIBLE_DEVICES`, defaulting to `6,7` for this
allocation. `serve-managed.sh`, the startup monitor, and the saved Manager method configuration
retain the previous recipe. Source checkout `533c18937bb8280dcac66f82daf0a20bcd1128d1` includes the
accepted selected-page implementation from PR #13. The core, Ascend, shared adapter, worker and
model pins are recorded in `recovery-provenance.json`.

The profile remains Qwen3.5-35B-A3B BF16/TP2, 32K context, concurrency limit 4, 8 GiB KV pool per
worker, FULL_AND_PIECEWISE graphs, MTP2, APC, async scheduling, chunked prefill and the explicit
1368/beta-2/4096-threshold PyramidKV configuration. Multimodal input remains disabled.

The strict smoke passed all three real chat requests with HTTP 200 and exact expected outputs.
Both long requests had 5,312 prompt tokens and committed compression to 2,048 physical tokens;
each scheduler commit matched both TP0 and TP1 acknowledgements by request ID. The repeated
long request hit 2,048 cached prompt tokens. MTP counters increased by 14 draft and 10 accepted
tokens. `recovery-summary.json` records these checks; `recovery-receipt.json` records the final
health and process identities. This establishes functional recovery of this profile only.

`recovery-raw.tar.gz` contains 29 files (29,864 compressed bytes). Every archive member was
independently rehashed against `recovery-members.json`.

Startup logged an AOT compile-cache load warning on both ranks and automatically recompiled
the model. The complete warning and subsequent startup outcome are preserved in the server log;
no cache-load success is inferred from eventual service readiness.

The raw archive includes the exact launch and environment scripts, saved Manager config, startup
cache-advice events, frozen server log, before/after NPU snapshots, original smoke script and every
smoke request/response/metric. The startup-only cache monitor retains its 48 GiB threshold and
stops at readiness. The recovered server is left running; the published server log is a snapshot.

## Independent remaining gates

Shared [PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19) merged as
`023170af0213cb23576bdc64d5606c60744effb7`. The public PyPI 0.9.0 wheel still has SHA-256
`5f809dd8a96653bc4ce983af3a1b704b6422d5d284bf8b8b11931c517a333119`, matching the
[earlier static audit](../2026-10-09-public-package-audit/README.md). Merging source does not replace
that artifact or qualify the newer shared branch against this older host. Continue using the
host-qualified `19f322130e1b3841da953b1b421bcf3259e9b8dc` pin. The release owner must choose/publish
a corrected immutable artifact before artifact-specific installation and runtime acceptance.

`dataset-backend-recheck.json` records that this ARM64 container still has no Docker/Podman
executable, Docker socket or known remote-backend environment variables. No fresh AMD64 sandbox
was executed. Terminal/SWE need an execution backend and reviewed task contracts; FrontierScience
needs a fixed judge and actual calibration; HLE still needs scope, rights and its evaluator/image
path. The previous candidate Harbor local transport receipt remains diagnostic only. Hardware
recovery does not satisfy these independent #8/#254 gates. No dataset scores or official readiness
are changed. The dataset owner supplies the judge/contracts; the infrastructure owner supplies
the sandbox backend and its model connection.

PR #13 was accepted for local operator latency only. Its approximately 36% operator-time reduction,
negative/near-zero service comparisons and existing raw evidence remain unchanged.

## Verify

```bash
sha256sum -c SHA256SUMS
python - <<'PY'
import hashlib
import json
import tarfile
from pathlib import Path

expected = json.loads(Path("recovery-members.json").read_text())
with tarfile.open("recovery-raw.tar.gz", "r:gz") as archive:
    actual = {
        member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        for member in archive.getmembers()
    }
assert actual == expected
PY
```
