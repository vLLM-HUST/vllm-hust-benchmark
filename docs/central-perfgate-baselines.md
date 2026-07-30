# Central Perfgate Baselines

Perfgate baselines are owned by `vLLM-HUST/vllm-hust-benchmark`, not by the target repositories that
execute benchmarks. The initial protocol supports the required `random-online` scenario for these
targets:

- `vLLM-HUST/vllm-hust`
- `vLLM-HUST/vllm-ascend-hust`

## Storage Key

An exact baseline is stored under the complete comparison identity:

```text
baselines/
  <target-owner>/
    <target-repository>/
      <target-sha>/
        <scenario>/
          <spec-id>/
            <spec-hash>/
              baseline-metadata.json
              run_leaderboard.json
```

`baseline-metadata.json` embeds a `perfgate-measurement/v2` record. Publication requires at least
one discarded warmup run and an odd measured-run count of at least three; the current producers use
exactly three. The measured run whose `throughput_tps` is in the middle after sorting by
`(throughput_tps, run_index)` is selected. The selected run's complete throughput, TTFT, TBT, and
error-rate tuple is published. The record contains ordered SHA-256 checksums for every warmup and
measured raw result. The trusted writer must recompute those checksums from the producer artifact
before publication.

The latest-main pointer is namespaced by target, scenario, and spec:

```text
pointers/<target-owner>/<target-repository>/<scenario>/<spec-id>/latest-main.json
```

Consumers of a required gate must use the exact path. They must not silently fall back to the
latest-main pointer.

## Validation

`perfgate-baseline store` requires:

- a full target commit that is part of the declared main history;
- matching target repository and commit metadata in the artifact;
- matching scenario, spec ID, and resolved spec hash;
- finite throughput, TTFT, and TBT metrics;
- at least one warmup and three measured runs with ordered raw-result checksums;
- selection metadata recomputed from all measured runs;
- a published client-metric tuple that exactly matches the selected real run;
- exact core, plugin, and benchmark runner revisions;
- hardware, CANN, PyTorch, and torch-npu provenance.

An existing baseline key is immutable. Repeating the same write is idempotent, while different
artifact bytes or provenance fail without overwriting data. The latest-main pointer can be updated
only when the target SHA is the current main tip.

`perfgate-baseline publish` applies the same validation before updating the central Git branch. It
creates the branch on the first write, treats a repeated identical write as success, and retries
from a fresh checkout after a concurrent non-fast-forward push. Credentials remain the caller's
responsibility and are never accepted as command-line arguments.

## Trust Boundary

This protocol does not grant or consume cross-repository credentials. A trusted producer or central
writer supplies the checked-out target main history and the central baseline worktree. Pull-request
workflows remain read-only consumers. Credential management, serialized cross-repository pushes, and
hardware backfill orchestration are separate rollout steps.

## Quarantine And Withdrawal

An invalid published baseline is retained for audit and blocked by an immutable record under the
same complete identity:

```text
revoked/<target-owner>/<target-repository>/<target-sha>/<scenario>/<spec-id>/<spec-hash>/
  quarantined.json
  withdrawn.json
```

Use `python -m vllm_hust_benchmark.perfgate_baselines revoke` in a clean `benchmark-baselines`
checkout, then commit and push the new `revoked/` record with the scoped central writer. Both states
make exact fetch and validation fail closed. Valid transitions are
`active -> quarantined -> withdrawn` or `active -> withdrawn`; `withdrawn` is terminal. Baseline
files are never deleted or overwritten, and consumers must not fall back to pointers or legacy data.
