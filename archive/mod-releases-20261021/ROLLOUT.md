# Issue 326 incremental custody rollout

This is a point-in-time receipt, not closure of
[issue 326](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/326).

`repository-protection.json` records the public repositories newly protected in this rollout. Each
entry was read back from GitHub after its PR checks passed and the PR was merged, followed by a
successful trusted-base custody workflow run. Required status checks are strict, apply to
administrators, and bind `Benchmark-only data (organization)` to GitHub Actions application 15368.
Other existing required checks were preserved. Previously protected repositories are not counted
again in this incremental list.

The trusted workflow checks fully paginated PR metadata without executing PR code. Its central
policy is pinned to `0cfe5109551e9b1c30ff3a75a2d93ee6ba8c9aa4`. The shared policy's positive and
negative unit cases run in each consumer repository. The
[fresh public organization drift audit](https://github.com/vLLM-HUST/.github/actions/runs/39598460911)
also passed. That audit compares an existing debt baseline: it does **not** mean all historic
evidence has moved. Arbitrary filenames and embedded measurements still require human review.
Private repository metadata is intentionally absent.

## Original-data links and cleanup

- [Benchmark PR 363](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/363) published
  section-level original-evidence locators, the 172-file long-context archive checked against
  existing source SHA256 seals, and the KV/MuxPD cleanup receipts. No experiment was rerun and no
  reported metric was changed.
- [Website PR 500](https://github.com/vLLM-HUST/vllm-hust-website/pull/500) adds immutable benchmark
  links to the midterm result tables and headline metrics. Publication-source links on other
  supported pages are labeled as published data/provenance, not falsely described as complete raw
  archives.
- `kv-arrival-cleanup.json` records 17 removed source Release assets. The benchmark copies were
  freshly downloaded and checked before deletion; the retained source tags, Release pages and
  historical attestations resolve through the canonical source map. The post-cleanup
  [consumer rebuild](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control/actions/runs/39585937483)
  passed on its second attempt.
- `muxpd-cleanup.json` records the earlier four-asset cleanup. Benchmark archives remain available
  for recovery; source Release pages and tags were retained.

## Still open

Remaining MOD Git-embedded evidence and consumers need case-by-case migration; unchanged inventory
entries are debt, not approved permanent copies. Not every historical website metric yet has a
complete original archive. Remaining policy PRs with dependency failures, unavailable
hardware/review runners, or repository-specific review requirements were not bypassed or
administratively merged. No organization-wide ruleset or GitHub Team upgrade is assumed.
