# Issue #136 Capacity Port-Reuse Fix Audit

Date: 2026-10-07 UTC

## Scope

CPU-only independent review of
`run-issue136-capacity-pilot-v3.sh` after the port-reuse fix. The active
capacity campaign, NPU processes, campaign lock, fixed ports, and active
benchmark worktree were not modified or exercised.

## Verdict

**PASS for the reported TIME_WAIT failure mode.**

The new probe uses an IPv4 TCP socket with `SO_REUSEADDR` and binds the exact
server address and configured port. The cleanup path requires both all four
physical NPUs to be idle and that probe to succeed before advancing to the
next rate. The same probe is repeated immediately before each server launch.

## Checks performed

- `bash -n` passed for the runner.
- The current source digest is
  `f8b32e765f3ef2b8acba0ec09029b40653a4a02ca6e3e99a640ca082c1912c0b`.
- `run-issue136-capacity-pilot-v3.sh.sha256` passed.
- Every entry in `issue136-next-stage-sources.sha256` passed, including the
  updated capacity runner.
- The failed artifact's `SHA256SUMS` passed in full. Its recorded pre-fix
  source digest remains
  `2d6c96e354421b8f921163a10fd1017cdde65df14c0c7f817de310d02130b1f8`;
  it was not rewritten to claim the fixed source.
- The failed artifact chronology is coherent: rate 0.5 completed and produced
  a result, cleanup completed, an empty rate-1 directory was created, then the
  run finalized as failed before a new server log or result was produced.
- An isolated, kernel-selected ephemeral-port test confirmed that the probe
  rejects a live listener both with and without `SO_REUSEADDR` on the listener.
- An isolated close/TIME_WAIT test confirmed that the probe accepts immediate
  reuse after a `SO_REUSEADDR` listener closes.

## Exit and cleanup review

- `server_pid`, `finalized`, `out`, and the original CLI `port` are initialized
  before the EXIT trap is installed; no unbound-variable path was found under
  `set -u`.
- `finalize` disables its traps before cleanup, prevents recursion, preserves
  the triggering status, upgrades a cleanup timeout to failure, records final
  NPU state, and writes `STATUS` before generating `SHA256SUMS`.
- `stop_server` terminates the server process group, waits for/reaps the leader,
  escalates after 30 seconds, clears `server_pid`, and then gates continuation
  on both NPU idleness and port reusability.
- A remaining child that still owns NPU memory is caught by `all_npus_idle`; a
  remaining listener is caught by `port_available`.

## Residual risks (non-blocking for this fix)

1. The availability probe and subsequent server bind are necessarily separate
   operations. A new unrelated process could claim the port in that interval.
   Normal bind/readiness failure remains fail-closed, but an adversarial or
   coincident look-alike service exposing the same health and model identity is
   not cryptographically tied to `server_pid`. The exclusive campaign lock and
   node-idle contract make this an operational residual, not a TIME_WAIT
   regression.
2. `SHA256SUMS` authenticates every file present when finalization runs, and the
   failed receipt currently verifies. Plain `sha256sum -c` does not reject a
   file added later (for example inside the empty `rate-1` directory). Archive
   or publication validation must continue to enforce an exact allowed file
   set; the checksum file alone is not an immutability boundary.
3. An input/preflight failure after the trap is installed still calls the full
   240-second resource-release gate. Invalid nonnumeric ports can therefore
   delay finalization, although they do not trigger an unbound-variable error
   or permit a false successful run. Lock contention occurs before the trap and
   may leave the newly-created output directory without `STATUS`; this behavior
   predates and is independent of the port-reuse change.

No blocking defect was found in the new port-reuse logic.
