# Agent Operating Constraints

This repository is the source of truth for vllm-hust benchmark specs, raw
submissions, leaderboard snapshots, and CI performance-gate inputs. Agents must
preserve the distinction between public benchmark data and private/CI smoke
tests. When in doubt, stop and ask for clarification instead of making a
compatibility shortcut.

## Hardware Truth

- The current active Ascend benchmark machines are **910B2**. Do not introduce
  new active/default `910B3` specs, workflow defaults, submissions, or
  leaderboard records unless a real run artifact explicitly proves the hardware
  was 910B3.
- Never infer `910B2` or `910B3` from a filename, directory name, branch name,
  PR title, old archive path, or screenshot. Hardware labels must come from
  actual run artifacts or a verified machine inventory.
- Historical `910B3` labels under `archive/pre-v0.18.0/` are legacy records.
  Treat them as archive-only unless the user explicitly asks to analyze old
  v0.11 data.
- If a workflow or script defaults to `ascend910b3` for current runs, fix the
  default to the verified current hardware instead of adding compatibility files
  to hide the problem.

## Spec Directory Boundaries

- `docs/official-baselines/` is for public, comparable official leaderboard
  specs. Active public specs should use the pinned public baseline family:
  vLLM/vLLM-Ascend `v0.18.0`, 910B2, FP16, and the actual workload model.
- Do not place perfgate, smoke-test, tuning, temporary, BF16, 3B, or legacy
  v0.11 specs in `docs/official-baselines/`.
- PR/CI performance-gate specs belong in a separate directory such as
  `docs/perfgate-specs/`. If that directory does not exist yet, create it and
  update the scripts/tests/workflows that consume those specs.
- Do not solve missing-spec CI failures by restoring stale files into
  `docs/official-baselines/`. Restore or migrate the spec to the correct
  directory and update the caller.

## Benchmark Data Integrity

- Do not fabricate, replay, simulate, or manually edit benchmark metrics to make
  leaderboard lines look better. Submissions must come from real runs.
- All real submissions must preserve full provenance: engine, package versions,
  git commit or PR number, workload, model identity, precision, chip model,
  chip count, node count, same-spec id/hash, server parameters, client
  parameters, raw result path, and timestamp.
- `vllm-hust` comparisons must be against the relevant `vllm` baseline for the
  same spec. Do not treat two `vllm-hust` historical versions as the preferred
  baseline/current pair.
- For trend lines, keep workload, model canonical id, chip model, chip count,
  node count, precision, and same-spec id/hash stable. Changing any of these
  creates a different series and may produce isolated points.

## PR Review And Merge Rules

- Before merging a PR that restores a missing spec, verify both the path and the
  hardware label. A PR is not ready if it keeps active `910B3` defaults for the
  current 910B2 machines.
- A PR that adds both `910B2` and `910B3` active perfgate specs is suspect.
  Prefer one verified current spec and archive legacy variants explicitly.
- CI passing is necessary but not sufficient. Confirm the semantic boundary:
  public official baseline specs, perfgate smoke specs, archived legacy specs,
  and leaderboard submissions must not be mixed.
- Do not revert or overwrite unrelated local changes. This repo often contains
  in-progress experiment outputs; stage or commit only the files required for
  the current task.

## When Updating Defaults

- Search for all active references before changing benchmark defaults:
  workflows, scripts, docs, tests, spec files, and generated submissions.
- If you change a spec path, update every caller and add or update a regression
  test that checks the intended path.
- If you change a hardware label or precision, update documentation explaining
  why the new value is verified and where the evidence came from.

