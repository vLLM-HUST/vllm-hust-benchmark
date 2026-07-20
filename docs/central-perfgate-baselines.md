# Central Perfgate Baselines

Perfgate baselines are owned by `vLLM-HUST/vllm-hust-benchmark`, not by the
target repositories that execute benchmarks. The initial protocol supports the
required `random-online` scenario for these targets:

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

The latest-main pointer is namespaced by target, scenario, and spec:

```text
pointers/<target-owner>/<target-repository>/<scenario>/<spec-id>/latest-main.json
```

Consumers of a required gate must use the exact path. They must not silently
fall back to the latest-main pointer.

## Validation

`perfgate-baseline store` requires:

- a full target commit that is part of the declared main history;
- matching target repository and commit metadata in the artifact;
- matching scenario, spec ID, and resolved spec hash;
- finite throughput, TTFT, and TBT metrics;
- exact core, plugin, and benchmark runner revisions;
- hardware, CANN, PyTorch, and torch-npu provenance.

An existing baseline key is immutable. Repeating the same write is idempotent,
while different artifact bytes or provenance fail without overwriting data.
The latest-main pointer can be updated only when the target SHA is the current
main tip.

## Trust Boundary

This protocol does not grant or consume cross-repository credentials. A trusted
producer or central writer supplies the checked-out target main history and the
central baseline worktree. Pull-request workflows remain read-only consumers.
Credential management, serialized cross-repository pushes, and hardware
backfill orchestration are separate rollout steps.
