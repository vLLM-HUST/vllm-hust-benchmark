# Strict host repeat orchestrator

`scripts/run-strict-host-repeat.sh` is the host-side admission and evidence
boundary for exactly one side of one official target. It must be invoked on the
real Ascend host. It never relies on container-side `npu-smi`.

The orchestrator:

- takes non-blocking `flock` leases for each requested physical NPU, the service
  port, and a unique temporary container;
- creates its own labeled `--rm --network none` container from an exact
  `sha256:<OCI-config-digest>` image;
- maps only the requested `/dev/davinciN` devices (to logical devices starting
  at zero), the three management devices, and read-only driver/model/data
  mounts;
- records two complete host snapshots at least 15 seconds apart and refuses
  compute PIDs, device FD holders, unstable/high HBM, port/session/lease
  conflicts, or an unprovable Docker configuration;
- starts only that owned container, samples host `npu-smi`, and proves every
  worker PID through `/proc/<pid>/cgroup` against the owned container ID and
  physical card;
- preserves a hashed raw record for every owned compute PID. When a card has
  multiple owned PIDs, the canonical PID is selected deterministically by
  preferring `VLLMWorker`/`Worker_TP`, then `EngineCore`, then other processes,
  with the lowest host PID as the tie-breaker; the validator recomputes this
  rule from the raw records;
- accepts a repeat only after the command exits zero, the owned container is
  stopped or removed, its PIDs and port are absent, host HBM matches the
  submission metric, and the immutable-input attestation exists;
- atomically writes the validator-compatible `strict_execution_evidence.json`
  last. A rejected attempt writes `strict_execution_failure.json` and never a
  canonical strict evidence file.

Dry-run performs admission and creates/removes a stopped owned container, but
does not start the command and never writes canonical strict evidence.

Owned runtime containers are nonprivileged by default. A privileged owned
runtime requires both `--owned-runtime-privileged` and an immutable
`--privileged-authorization-source` reference. The create argv, Docker inspect,
owned-container identity, and strict execution evidence must all agree on that
choice; the official validator rejects missing, extra, or contradictory fields.
This opt-in applies only to the uniquely named, labeled container created by the
orchestrator and does not authorize changes to any existing container. The
contract is versioned by `schemas/owned_runtime_security_v1.schema.json`.

```bash
scripts/run-strict-host-repeat.sh \
  --repeat-dir /data/home/vllm-hust-codex-21rc/home/vllm-hust-benchmark/RESULT_DIR/repeat-01 \
  --target-id OFFICIAL_TARGET_ID \
  --side upstream \
  --physical-npu 0 \
  --service-port 18080 \
  --runtime-image-digest sha256:OCI_CONFIG_DIGEST \
  -- \
  env RESULT_DIR=/workspace/vllm-hust-benchmark/RESULT_DIR/repeat-01 \
  bash /workspace/vllm-hust-benchmark/scripts/run-official-ascend-goal-baseline.sh \
  /workspace/vllm-hust-benchmark/docs/official-baselines/TARGET_SPEC.json
```

The command after `--` is the command stored in the temporary container. It is
not evaluated by a host shell. The caller remains responsible for supplying
the exact runner environment and matching target spec. Real mode re-executes
itself through `sudo -n` when necessary; no password prompt or fallback is
allowed. Host `npu-smi` subprocesses receive only the fixed absolute library
directories declared by `--host-ld-library-path`; they do not inherit an
arbitrary caller `LD_LIBRARY_PATH` through sudo.
