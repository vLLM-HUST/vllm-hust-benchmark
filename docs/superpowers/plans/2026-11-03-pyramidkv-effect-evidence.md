# PyramidKV current-release evidence implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish reproducible evidence for PyramidKV issue #31, then close the issue only after its
full acceptance is proved and link the result to #1.

**Architecture:** Keep frozen measurement originals unchanged. Export a public capsule with an
explicit private-input inventory, validate requests and runtime evidence independently, and
regenerate a deterministic analysis from the originals. Dataset-backed verification additionally
requires the pinned local LongBench inputs.

**Tech Stack:** Python 3.10+ standard library, pytest, existing benchmark navigation and CI.

**Spec:** `docs/plans/pyramidkv-standalone-qualification-20261103.md`; PyramidKV issue #31; frozen
`contract-v1.json`, `contract-v2.json`, pressure amendment and timeout clarification in
`/root/.cache/pyramidkv-issue31-20261103`.

## Global Constraints

- Published `0.4.0rc1` wheel SHA256 is
  `46a0a9b2dd02b7523fe61d5032a78abd7dceb653b660d0304f2dc52d9ae5ae81`; tool source is independent of
  runtime package source.
- Preserve the preregistered 12 lifecycles, 1834 requests, 210112 maximum output tokens and stop
  policy. Do not restart a live or completed lifecycle to improve evidence.
- Quality uses 100 rows each from narrativeqa/qasper/2wikimqa; development 0–49 and holdout 50–99
  separately require mean F1 drop \<=3 and every task drop \<=5 points.
- Service uses OFF/ON/ON/OFF at each of 8 GiB and 2 GiB; preserve every cell, warmup, probe, failure
  and cohort. The primary pressure cell is 24576/C4.
- The v2 amendment restores the original pressure cache policy before any 2 GiB request. Eight-GiB
  synthetic cached tokens must be zero; pressure counters require explicit preemption/recomputation
  review. Preserve both contract versions and their actual per-process use.
- Six-hundred seconds is the socket timeout, not a whole streaming-request deadline. Retain the
  pre-request clarification and actual lifecycle limits.
- Capacity C1/C2/C4 proves tested operating points only. KV ownership release is distinct from HBM
  savings and maximum resident capacity.
- The first quality-OFF lifecycle has frozen launch source and parsed startup configuration but no
  captured `/proc` snapshot. Report this evidence limitation explicitly.
- Dataset questions, references and quality request payloads remain local. Publish row IDs, hashes,
  predictions and scores; support independent recomputation with pinned official files.
- No local tests, lint, tokenization or data audits during measured requests. Use startup intervals
  or run after the series finishes.
- Changes are isolated in a temporary benchmark branch outside the cleaned workspace. Do not alter
  historical evidence or public leaderboard defaults.

## Review Focus

- An archive with duplicate/unsafe members or a rewritten manifest must not silently pass: Task 1
  tests reject duplicate entries, path traversal, nonregular members, missing inventory and
  mismatched bytes.
- A self-consistent but altered derived view must fail semantic recomputation, including under
  optimized Python: Task 2 tests change scores, transaction acknowledgements, counters, usage and
  analysis independently of outer hashes.
- Pressure recomputation must preserve every transaction and cached-token count: Task 2 covers
  repeated commits, missing/extra TP acknowledgement sequences and unexplained cached use.
- Missing private inputs must never be described as fully verified QA payloads or F1 recomputation:
  Task 2 exposes explicit public versus dataset-backed coverage and tests missing/wrong dataset
  identity.
- Partial execution or a changed contract version cannot silently become a complete experiment: Task
  2 checks all expected lifecycle identities, ordering and request membership; Task 3 reconciles
  every issue checklist item to evidence.

### Task 1: Public capsule integrity

**Files:**

- Create: `reports/pyramidkv-standalone-effects-20261103/evidence.py`
- Test: `tests/test_pyramidkv_effect_evidence.py`

**Interfaces:**

- Consumes: report directory containing `SHA256SUMS`, `members.json`, and one or more raw archive
  parts.

- Produces: `load_capsule(root: Path) -> dict[str, bytes]`;
  `require(condition: bool, message: str) -> None`, raising `ValueError` without relying on
  assertions.

- [ ] Write synthetic capsule tests: valid bytes returned; duplicate members, unsafe names,
  symlinks, missing members, wrong length/hash, duplicate checksum lines and changed outer file
  rejected. Exercise the loader under `python -O`.

- [ ] Run `python -m pytest -q tests/test_pyramidkv_effect_evidence.py`; expect failure because the
  loader does not exist.

- [ ] Implement strict read-only archive loading and complete file/member inventories; never extract
  archive paths or execute bundled code.

- [ ] Run the same test command; expect all tests passing.

- [ ] Commit the loader, tests and this plan.

### Task 2: Request, runtime and analysis verification

**Files:**

- Extend: `reports/pyramidkv-standalone-effects-20261103/evidence.py`
- Create: `reports/pyramidkv-standalone-effects-20261103/analyze.py`
- Create: `reports/pyramidkv-standalone-effects-20261103/verify.py`
- Create: `reports/pyramidkv-standalone-effects-20261103/runtime.py`
- Create: `reports/pyramidkv-standalone-effects-20261103/private_inputs.py`
- Extend: `tests/test_pyramidkv_effect_evidence.py`

**Interfaces:**

- Consumes: Task 1's validated capsule bytes; frozen public case indexes and optional pinned dataset
  files/reconstructed private plans.

- Produces: `verify_evidence(raw: dict[str, bytes], private_root: Path | None = None) -> dict`;
  `analyze(raw: dict[str, bytes]) -> dict`; CLI emits coverage, lifecycle counts, quality gates and
  analysis equality.

- [ ] Write failing tests for exact selected membership/order, raw SSE text/ID/usage/DONE agreement,
  fixed generation parameters, TP lifecycle sequence conservation, OFF absence and 8 GiB zero-cache
  enforcement.

- [ ] Run the new tests; expect the missing verification functions to fail.

- [ ] Implement complete request/transaction/runtime validation against each lifecycle's actual
  contract version. Reject orphan scheduler/worker entries and preserve pressure repeats for review.

- [ ] Add failing tests for hand-calculated throughput pooling, percentile/latency summaries,
  separate quality gates, missing/altered private references, telemetry/counter inconsistencies and
  changed outer analysis with refreshed checksums.

- [ ] Implement deterministic quality/service/capacity analysis, explicit public-only versus
  dataset-backed verification, and CLI comparison of the recorded view with recomputation. No
  fabricated token timestamps: distinguish recomputed aggregates from client-observed timing
  originals.

- [ ] Run the full new test file, actual completed evidence checks and optimized-Python negative
  controls; expect all required checks to pass or retain a concrete failure that prevents
  qualification.

- [ ] Commit the verifier and analysis with tests.

### Task 3: Immutable delivery and issue reconciliation

**Files:**

- Create: `reports/pyramidkv-standalone-effects-20261103/export.py`

- Create: `reports/pyramidkv-standalone-effects-20261103/README.md`, `analysis.json`,
  `members.json`, `SHA256SUMS`, capsule parts and verification receipt.
- Modify: `docs/data-index.json` through the generator only.

**Interfaces:**

- Consumes: complete frozen run originals, Task 2 verifier and analysis.

- Produces: reviewed benchmark PR, immutable canonical commit, anonymous download/recomputation
  receipt, issue #31 closure and parent #1 evidence link if every requirement is satisfied.

- [ ] Export originals using an explicit allowlist; include scripts, contract versions/amendments,
  launch identities, failures, raw SSE, metrics and telemetry. Exclude all private prompts/answers
  and record exact excluded-file hashes. Preserve local private custody.

- [ ] Produce per-process/per-cohort and pooled service tables, development/holdout quality tables,
  capacity operating points, MTP/cache/preemption/KV/HBM evidence and limitations. Retain all
  positive, zero, negative and failed outcomes.

- [ ] Run public verification, dataset-backed local verification and negative controls; record
  commands, coverage and results. Do not describe timing snapshots or hashes as stronger evidence
  than they are.

- [ ] Stage new paths; run `python scripts/data_index.py --write` and `--check`. Run appropriate
  formatting and the project test suite; keep any unrelated baseline failures explicit and resolve
  required CI checks before merge.

- [ ] Request one fresh whole-branch review per executing-plans. Resolve substantive findings with
  failing-then-passing regressions and rerun required checks.

- [ ] Merge only after review and CI pass. Download published bytes anonymously at the immutable
  commit, verify hashes and recompute using the delivered verifier.

- [ ] Audit issue #31 against exact evidence; close only if complete, then write immutable results
  and remaining limitations to #1. Stop owned services and remove temporary copies after custody is
  established.
