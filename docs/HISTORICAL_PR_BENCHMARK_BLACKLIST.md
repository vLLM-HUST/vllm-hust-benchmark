# Historical PR Benchmark Blacklist

This ledger records historical runtime or plugin revisions that must not be benchmarked for the
public leaderboard. The machine-enforced source of the same decisions is
`docs/leaderboard-exclusions.json`.

## Rules

- Match exclusions by the full runtime or plugin commit, never by a run name, branch name,
  abbreviated SHA, or workload directory.
- Do not backfill, rerun, or publish any workload for a blacklisted commit.
- Remove already-published rows and public raw HF submissions in a recoverable commit, while
  retaining this ledger and the remote commit history as the audit trail.
- A follow-up fix is evaluated as a new full commit. It does not rehabilitate or overwrite the
  blacklisted commit.
- Removing an entry requires a reviewed repository change that supplies new fixed-revision
  correctness evidence and explains why the original exclusion was wrong. Operational convenience or
  missing coverage is not sufficient.

## Entries

| Status      | Repository / PR                 | Full commit                                | Scope                                                          | Reason                                                                                                                                                                                             | Operator action                                                                                                                                                       |
| ----------- | ------------------------------- | ------------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Blacklisted | `vLLM-HUST/vllm-ascend-hust#53` | `bf2984e34a8923ac254251c6e265dffbad4aa70d` | All public leaderboard workloads and all paired core revisions | The merge changed the source-documented unsafe Qwen2 ACL graph path from explicit opt-in to default-on, disabled the native safe RoPE fallback by default, and supplied no correctness validation. | Do not rerun the missing `sharegpt-throughput` cell or any other backfill. Delete all matching public rows/raw HF submissions and keep the dual issue ledger updated. |

Evidence:

- PR: <https://github.com/vLLM-HUST/vllm-ascend-hust/pull/53>
- Merge:
  <https://github.com/vLLM-HUST/vllm-ascend-hust/commit/bf2984e34a8923ac254251c6e265dffbad4aa70d>
- The parent implementation warned that the Qwen2 ACL graph path could produce incorrect outputs and
  required `VLLM_ASCEND_ALLOW_UNSAFE_QWEN2_ACLGRAPH=1`. PR #53 changed that default to enabled and
  changed the native fallback default to disabled.
- The PR description states that no additional tests were run.
