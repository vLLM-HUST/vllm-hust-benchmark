# Agent Operating Constraints

Read [AGENTS.md](AGENTS.md) and the shared [release/evaluation/website workflow](AGENT_WORKFLOW.md)
with this file. The workflow unifies Engine, benchmark and website responsibilities; this file
retains the official leaderboard, hardware and data-integrity boundaries below.

## Data navigation

Before reading, adding, or editing data, use [DATA_INDEX.md](DATA_INDEX.md) and the generated
[collection index](docs/data-index.json). Distinguish immutable originals, versioned analysis,
contracts, publication sources, and synthetic fixtures. After adding/removing tracked data paths,
regenerate the index with `python scripts/data_index.py --write` and validate with `--check`. An
index entry is not experimental acceptance or permission to overwrite evidence or run devices. For
website references and Release assets, use [linked entry points](docs/data-entrypoints.json) and
`python scripts/data_entrypoints.py --query <term>`. Regenerate with `--write` after publication
data changes; refresh remote snapshots explicitly as documented in DATA_INDEX.md. Offline CI does
not prove that remote inventory or deployment is still current.

This repository is the source of truth for vllm-hust benchmark specs, raw submissions, leaderboard
snapshots, and CI performance-gate inputs. Agents must preserve the distinction between public
benchmark data and private/CI smoke tests. Resolve routine choices from the current frozen contracts
and the user's confirmed instructions. Ask a specific question only when missing information blocks
the next necessary action, and continue independent authorized work. Never invent evidence or use a
compatibility shortcut to hide a failed contract.

## Hardware Truth

- The current active Ascend benchmark machines are **910B2**. Do not introduce new active/default
  `910B3` specs, workflow defaults, submissions, or leaderboard records unless a real run artifact
  explicitly proves the hardware was 910B3.
- Never infer `910B2` or `910B3` from a filename, directory name, branch name, PR title, old archive
  path, or screenshot. Hardware labels must come from actual run artifacts or a verified machine
  inventory.
- Historical `910B3` labels under `archive/pre-v0.18.0/` are legacy records. Treat them as
  archive-only unless the user explicitly asks to analyze old v0.11 data.
- If a workflow or script defaults to `ascend910b3` for current runs, fix the default to the
  verified current hardware instead of adding compatibility files to hide the problem.

## Spec Directory Boundaries

- `docs/official-baselines/` is for public, comparable official leaderboard specs. Active public
  specs should use the pinned public baseline family: vLLM/vLLM-Ascend `v0.18.0`, 910B2, FP16, and
  the actual workload model.
- Do not place perfgate, smoke-test, tuning, temporary, BF16, 3B, or legacy v0.11 specs in
  `docs/official-baselines/`.
- PR/CI performance-gate specs belong in a separate directory such as `docs/perfgate-specs/`. If
  that directory does not exist yet, create it and update the scripts/tests/workflows that consume
  those specs.
- Do not solve missing-spec CI failures by restoring stale files into `docs/official-baselines/`.
  Restore or migrate the spec to the correct directory and update the caller.

## Benchmark Data Integrity

- Do not fabricate metrics, publish simulated/reused timings as a new hardware run, or manually
  improve leaderboard values. Submissions must come from real runs. CPU reconstruction from sealed
  original streams is an audit, not a new measurement. Frozen public SWE/Terminal trace requests may
  be submitted to a real engine under the trace inference contract; this does not measure full agent
  task success. Preserve the original streams, planned denominator and each run identity.
- Do not use `--enforce-eager` for formal leaderboard data, official baselines, historical PR
  backfills, or website-facing snapshots. Do not enable eager mode through wrapper flags,
  environment variables, empty same-spec fields, retry fallbacks, or generated argument lists. If
  graph-mode serving cannot run, mark the experiment failed or blocked instead of publishing an
  eager fallback. Eager mode is allowed only for separately labelled diagnostic runs that never mix
  into public leaderboard snapshots.
- All real submissions must preserve full provenance: engine, package versions, git commit or PR
  number, workload, model identity, precision, chip model, chip count, node count, same-spec
  id/hash, server parameters, client parameters, raw result path, and timestamp.
- `vllm-hust` comparisons must be against the relevant `vllm` baseline for the same spec. Do not
  treat two `vllm-hust` historical versions as the preferred baseline/current pair.
- For trend lines, keep workload, model canonical id, chip model, chip count, node count, precision,
  and same-spec id/hash stable. Changing any of these creates a different series and may produce
  isolated points.

## PR Review And Merge Rules

- Before merging a PR that restores a missing spec, verify both the path and the hardware label. A
  PR is not ready if it keeps active `910B3` defaults for the current 910B2 machines.
- A PR that adds both `910B2` and `910B3` active perfgate specs is suspect. Prefer one verified
  current spec and archive legacy variants explicitly.
- CI passing is necessary but not sufficient. Confirm the semantic boundary: public official
  baseline specs, perfgate smoke specs, archived legacy specs, and leaderboard submissions must not
  be mixed.
- Do not revert or overwrite unrelated local changes. This repo often contains in-progress
  experiment outputs; stage or commit only the files required for the current task.

## When Updating Defaults

- Search for all active references before changing benchmark defaults: workflows, scripts, docs,
  tests, spec files, and generated submissions.
- If you change a spec path, update every caller and add or update a regression test that checks the
  intended path.
- If you change a hardware label or precision, update documentation explaining why the new value is
  verified and where the evidence came from.

## Release evaluation and publication

Follow [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) for the fixed five-profile full test, MMLU200 quick
test, explicit sealed-stage reuse, qualified reference denominator, release/MOD identity and
publication acceptance. Project B0/B1 evaluation has its own confirmed model/precision scope; it is
not automatically an official leaderboard submission. Do not apply the official FP16 leaderboard
restriction above to relabel an already confirmed BF16/35B project experiment.

All website versions retain the same profile and metric sections, including failures and missing
data. Publish source data here, then update fixed-source consumers and verify the canonical site.
Keep package metadata sync, release catalog updates, measurement publication and website deployment
distinct. A catalog entry, a successful CI run or a deployed page is not evidence that inference or
all four acceptance goals completed. Preserve raw failures and version corrections separately.

## Archive ownership for new work

Each agent that generates or collects new research data is responsible for archiving that data
as part of its own task. The central historical-container recovery task does not continuously
back up newly created files. Complete archiving at a stable stage boundary or task handoff;
do not interrupt another chat to delegate this responsibility.

- Archive actual file bytes, including raw records, failures and negative results, together with
  the source scripts, configurations and inputs needed to reconstruct them. A reference, path list,
  URL or index alone is not a data archive. Preserve original versions; corrections are new versions.
- Route vLLM, ANISE, LUMI, LLMWorkloadSuite and PrefixCache Routing materials to
  `vLLM-HUST/vllm-hust-benchmark`; route GenNova materials to
  `GenNova-HUST/gennova-benchmark`. Follow the destination's existing privacy and publication rules;
  this reminder does not authorize a new public release. Keep private keys and live credentials out
  of uploads, and retain the existing protected-original handling where applicable.
- Include source/version identities, original-to-restored path mappings, file counts, byte counts,
  SHA256 manifests and a runnable recovery command. Check script and data dependencies, permissions,
  links and required directories. List intentionally excluded runtimes or model weights and how to
  provision them separately; do not describe unverified dependencies as recovered.
- Download the uploaded artifacts again, verify archive and original-file hashes, and test recovery
  into a fresh directory. Report the repository, immutable commit or Release asset, checksums,
  recovery entry point, actual verification result and any remaining gaps in the task handoff.
  Successful upload, an index entry or a passing CI job alone does not prove recoverability.
- Only clean redundant local copies within the user's authorized scope after the remote bytes and
  fresh recovery have been verified. Preserve unique originals, frozen historical evidence and
  required local decryption identities. Archiving does not require rerunning experiments or taking
  additional devices.
