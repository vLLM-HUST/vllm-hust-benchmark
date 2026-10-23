# Shared 0.10.0 artifact audit and legacy-host candidate

The public release is available, but **the published 0.10.0 wheel is not accepted on this
PyramidKV host**. Its model-load helper imports a newer Ascend router package absent from the
explicitly supported legacy host. Shared [PR #23](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/23)
adds an ABI guard. Candidate validation below applies to the locally repaired wheel and a temporary
PyramidKV dependency descriptor, not to the public wheel.

## Artifact identities and reproduced failure

- Public shared 0.10.0 was uploaded to PyPI on 2026-10-18. Downloaded wheel SHA-256:
  `456a2bb800dc2cd67a8604495c9235748a9260b0da36b3c84ef1e5f5c7046710`.
  All 23 Python/manifest members match shared source
  `0dbc8c8137caa1d33fdacbc8d1763662fd73f479` byte for byte. This is a content comparison,
  not a build attestation. The wheel includes Manifest 0.3 and the FULL-graph cache-slot fix.
- Released Manager 0.3.0 came from its GitHub release asset, SHA-256
  `10ba92df1675fa8355bdcba659e2a558b815a85f5b40a22ef56b6efd0c36e8d5`,
  tag source `4450214cfea81dd3a6356aaaf9598178b40f44cb`. The public PyPI project query returned
  HTTP 404; the GitHub release provides the actual downloaded artifact.
- The public shared wheel's `install_grouped_topk_router_compat()` raised
  `ModuleNotFoundError: No module named 'vllm_ascend.ops.fused_moe.router'` on core
  `d0f22d2bda562156e4dbf433ce645e1769b4f804` / Ascend
  `03766ac696fde5ab1980d80ca0b8543d3580c989`. The model-load hook calls this helper
  before loading weights. This was a direct helper probe, not a full failed server launch.
- The regression tests against the unchanged public wheel produced one expected failure and
  two passes. The stable rerun and first diagnostic attempt are both retained.

The legacy core already accepts `routed_scaling_factor`; it needs neither the shim nor the
newer router import. The proposed fix checks that core signature first. Modern-core scaling
behavior, wrapper idempotence and import-error propagation remain covered by tests. No host
source is patched and missing dependencies on a core requiring the shim are not suppressed.

## Candidate isolation and validation

A new venv inherits system Torch/torch-npu and clones prepared host helpers/metadata from the
old venv, excluding shared adapter, PyramidKV and Manager. Those three packages are installed
from wheel files into the new venv. This is an isolated plugin installation on the prepared
host, not a clean installation of the whole Ascend stack. The old venv and saved Manager state
are unchanged. `cloned-host-packages.json` records what was copied.

The shared candidate is built from the PR #23 source recorded in `artifact-provenance.json`.
Its version string remains 0.10.0, but its hash differs from the public wheel; the candidate is
not a published release. The PyramidKV candidate uses base
`50a09697ddf6b510369588f3fd2d54b0dd214cee` with only the Python/manifest dependency range
locally widened to `<0.11` for this test. Algorithm bytes are unchanged. That widening is
included in the raw patch and candidate wheel; it is **not merged into the repository**.

Validation completed before model requests:

- Shared focused suite: 33 passed / 1 skipped; full suite: 212 passed / 1 skipped.
  The existing worker-seam probe skipped on a caught `DeviceOperator` circular import;
  this is not a passed worker import check.
- PyramidKV candidate suite: 78 passed / 1 opt-in NPU test skipped.
- Ruff check/format and wheel builds passed. Candidate shared package files were compared
  with the tested source; artifact hashes are recorded.
- Released Manager 0.3.0 accepted the joint plan. Separate lifecycle state verifies dependency
  enable/disable rejection and ordered disable/forget without changing the running server state.

The installed candidate wheels completed the real managed BF16/TP2 graph smoke: three HTTP 200
responses matched the expected outputs. Both 5,312-token long requests compressed to 2,048
physical tokens, with each scheduler transaction matched to both TP0 and TP1 acknowledgements.
The repeated prompt hit 2,048 prefix-cache tokens; MTP deltas were 14 draft and 10 accepted tokens.
`candidate-smoke-summary.json` records the request IDs and counters. NPU process identities map
TP0/TP1 to physical devices 1/2. The candidate service is left running on localhost port 8000.
The archive contains 51 files (175,725 compressed bytes); all member hashes were independently
verified. It includes the local candidate wheels as distinct artifacts, not a replacement public release.

The profile keeps BF16/TP2, 32K context, max four sequences, 8 GiB KV pool per worker,
FULL_AND_PIECEWISE graphs, APC, MTP2, async/chunked prefill, seed 17, and the 1368/beta-2
PyramidKV configuration. This allocation exposes physical NPU 1/2 with 8 CPU quota and 64 GiB
container memory; the isolated launcher selects runtime ordinals 0/1. Startup-only cache
advice retains the 48 GiB threshold. This is functional validation, not a paired performance,
full quality, capacity, or published-package acceptance result.

## Remaining work

The shared maintainer must review the repair and provide a corrected immutable artifact.
That exact artifact needs its own install/runtime/rollback acceptance. Neither a source PR
nor the repaired candidate changes the public 0.10.0 wheel's failed result. Main keeps the
original `<0.10` dependency restriction and qualified shared source pin.

For #8, this work supplies no AMD64 sandbox, remote model connection, judge/calibration or
reviewed task/evaluator rights and execution contracts. The previous #254 requests still need
those deliveries; no new PyramidKV task scores or official readiness are reported.

There are now public development scoring precedents: [FrontierScience PR #278](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/278)
and [HLE PR #280](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/280), plus BetterScale reruns.
They use Qwen as both solver and substitute judge. The Frontier receipt explicitly retains failed
strict-final calibration and a separately authorized first-complete-VERDICT policy; HLE retains
unresolved strict-JSON judgments. These are useful protocol references, not a delivered calibrated
judge or authorization to silently change PyramidKV's existing evaluation contract. Adopting that
development-only scoring policy requires a scope decision; other MOD scores are not PyramidKV results.
The user explicitly chose on 2026-10-19 (Asia/Shanghai) to retain the existing calibration
requirement and wait for a qualified judge. No same-model development scoring run is authorized
by this receipt; that judge/calibration gate remains open.

## Evidence verification

Machine timestamps use UTC. `raw-members.json` binds every member of `raw.tar.gz`, including
published-wheel failure, tests, candidate patches/wheels, release metadata, launch/config
scripts, startup logs and smoke artifacts. The archive excludes the model and credentials.

```bash
sha256sum -c SHA256SUMS
python - <<'PY'
import hashlib
import json
import tarfile
from pathlib import Path

expected = json.loads(Path("raw-members.json").read_text())
with tarfile.open("raw.tar.gz", "r:gz") as archive:
    actual = {
        member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        for member in archive.getmembers()
    }
assert actual == expected
PY
```
