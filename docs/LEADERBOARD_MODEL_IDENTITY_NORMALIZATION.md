# Leaderboard Model Identity Normalization RFC

Status: Implemented and enforced on checked-in benchmark submissions and website snapshots.

This document records the approved Phase 0 decisions for model identity normalization in the vllm-hust leaderboard pipeline.

Current status:

- checked-in benchmark submissions are backfilled to the normalized identity contract
- website aggregation no longer uses heuristic fallback to infer repo ids from short names or cache paths
- website schema/examples require `model.canonical_id`, `model.repo_id`, `model.short_name`, and `model.display_name`

The problem statement below describes the pre-migration state that motivated this RFC.

See also:

- `src/vllm_hust_benchmark/data/model_identity_registry.json` for the initial seed registry
- `docs/LEADERBOARD_MODEL_IDENTITY_MIGRATION_CHECKLIST.md` for the implementation checklist

## Problem Statement

The current leaderboard pipeline uses `model.name` for multiple meanings at once:

- canonical upstream model coordinate, for example `Qwen/Qwen2.5-14B-Instruct`
- short model alias, for example `Qwen2.5-14B-Instruct`
- UI display label in some downstream discussions

Because the exporter writes `--model-name` into the artifact unchanged, different submission sources can publish different strings for the same model. The website filter then de-duplicates on the raw `model.name` value and exposes multiple entries for one logical model.

## Observed Current State

Representative examples already exist in checked-in snapshots:

- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen2.5-14B-Instruct`

The current behavior is internally inconsistent:

- main leaderboard snapshots preserve the raw submitted `model.name`
- compare goal grouping already strips the namespace for some matching paths
- official same-spec and baseline specs use the full upstream coordinate

This means the system already treats these values as partially equivalent, but that equivalence is not expressed in the data contract.

## Industry Convention

For machine-readable model identity, the industry-standard format is a registry coordinate or repository coordinate, not a short alias.

Examples:

- Hugging Face: `Qwen/Qwen2.5-14B-Instruct`
- ModelScope or internal registries: typically `registry + namespace/model`

Short names such as `Qwen2.5-14B-Instruct` are useful as aliases or display values, but they are not a strong canonical identifier because they lose the namespace and can collide across forks, mirrors, or vendor-specific repacks.

## Design Goal

Split model identity into separate fields so each field has one meaning only:

- canonical identity for matching, deduplication, and contracts
- upstream source coordinate for traceability
- short alias for convenience
- display label for UI rendering

The design should preserve merge safety for benchmark artifacts and avoid ad-hoc website-side normalization.

## Frozen Decisions

This section completes Phase 0. The following decisions are fixed unless a later RFC explicitly supersedes them.

### Decision 1: Canonical Identifier Format

`model.canonical_id` uses the format `<registry>:<repo_id>`.

Examples:

- `hf:Qwen/Qwen2.5-14B-Instruct`
- `hf:deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

Rules:

- `registry` is a short lowercase token such as `hf`
- `repo_id` is the upstream namespace-qualified coordinate without registry prefix
- `canonical_id` is the only field allowed to act as the machine identity for deduplication, compare grouping, filtering, and validation
- when the source model comes from Hugging Face, `canonical_id` must be `hf:<repo_id>`
- local cache paths, snapshot paths, and bare short aliases are never valid canonical ids

Reasoning:

- this preserves namespace information
- this leaves room for future multi-registry ingestion without another schema break
- this avoids treating short aliases as globally unique identifiers

### Decision 2: Registry Source Of Truth Location

The source-of-truth registry file will live at:

- `src/vllm_hust_benchmark/data/model_identity_registry.json`

Why this location is fixed:

- the benchmark package already ships JSON data files from `src/vllm_hust_benchmark/data`
- both exporter and aggregation logic can consume it through `importlib.resources`
- the registry is part of the benchmark contract, not a website-only concern

The registry will store one canonical record per logical model, plus explicit aliases.

Recommended shape:

```json
{
  "version": 1,
  "models": [
    {
      "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
      "registry": "hf",
      "repo_id": "Qwen/Qwen2.5-14B-Instruct",
      "short_name": "Qwen2.5-14B-Instruct",
      "display_name": "Qwen2.5-14B-Instruct",
      "aliases": [
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen2.5-14B-Instruct"
      ]
    }
  ]
}
```

Operational rules:

- one canonical record per logical model
- aliases must resolve to exactly one canonical record
- short-name collisions must be handled by adding an explicit non-ambiguous alias, not by inference
- official baseline specs and same-spec docs should reference the same canonical record

### Decision 3: Compatibility Policy For `model.name`

`model.name` remains in schema v1 during the migration window, but its semantics are frozen as a compatibility mirror of `model.repo_id`.

Rules:

- new writers must populate `model.name` with the same value as `model.repo_id`
- new writers must not emit a short alias in `model.name`
- readers must treat `model.name` as legacy compatibility input only
- new matching, grouping, and filtering logic must not use `model.name` when `canonical_id` is available
- schema v1 compatibility ends only after all active writers and checked-in snapshots have migrated

This freezes the transition policy and avoids a mixed state where some new submissions keep using short aliases in `model.name`.

### Decision 4: Minimal Field Set For Phase 1 Writers

All new normalized writers must produce at least:

- `model.canonical_id`
- `model.repo_id`
- `model.short_name`
- `model.display_name`
- `model.name` as a compatibility mirror of `model.repo_id`

No new writer may publish only a short alias.

## Proposed Data Contract

Add explicit model identity fields to leaderboard artifacts.

Recommended model payload:

```json
{
  "model": {
    "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
    "repo_id": "Qwen/Qwen2.5-14B-Instruct",
    "short_name": "Qwen2.5-14B-Instruct",
    "display_name": "Qwen2.5-14B-Instruct",
    "name": "Qwen/Qwen2.5-14B-Instruct",
    "parameters": "14B",
    "precision": "FP16",
    "quantization": null
  }
}
```

Field semantics:

- `model.canonical_id`: authoritative machine identifier used by aggregation, compare grouping, and filters
- `model.repo_id`: upstream repository coordinate when the model comes from a registry such as Hugging Face
- `model.short_name`: stable human-typed alias without namespace
- `model.display_name`: final UI label shown in dropdowns and tables; for the current contract it mirrors the industry-standard public model release string and must not be used as a machine identifier
- `model.name`: compatibility field kept during migration only; it must mirror `model.repo_id`

The Phase 0 decision is to use prefixed canonical ids from day one, for example `hf:Qwen/Qwen2.5-14B-Instruct`.

## Normalization Rules

Normalization should happen at the benchmark export or aggregation boundary, not in the website render layer.

Required rules:

1. Raw input may arrive as a repo coordinate, a short alias, or a local snapshot path.
1. The exporter or aggregator resolves the raw input into one canonical model identity record.
1. Local cache paths must never become published canonical identifiers.
1. Unknown aliases must fail fast or be surfaced as validation errors instead of silently creating new logical models.
1. `display_name` is derived from `short_name`, not from `canonical_id`, and never includes the registry prefix or namespace.
1. The default `display_name` is the industry-standard release string carried by `short_name`, for example `Qwen2.5-14B-Instruct`.
1. Writers may curate `display_name` only through the central registry; code matches on `canonical_id`, not on `display_name`.

Required resolution order:

1. explicit model registry metadata from the submission
1. exact alias-map match
1. exact repo-id match
1. explicit rejection with remediation guidance

## Alias Registry

To support historical data and operator convenience, maintain a small explicit alias registry.

Example:

```json
{
  "Qwen2.5-14B-Instruct": {
    "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
    "repo_id": "Qwen/Qwen2.5-14B-Instruct",
    "short_name": "Qwen2.5-14B-Instruct",
    "display_name": "Qwen2.5-14B-Instruct"
  }
}
```

This registry should be versioned with the benchmark repository, reviewed like a contract file, and reused by:

- exporter validation
- snapshot aggregation
- compare scope generation
- website filter labeling

## Website Rendering Rules

The website should stop treating `model.name` as both label and identity.

Recommended UI behavior:

- filter option value: `model.canonical_id`
- filter option label: `model.display_name`
- table primary text: `model.display_name`
- tooltip or details panel: `model.repo_id`

This keeps the user-facing UI clean while preserving source traceability.

## Migration Plan

### Phase 0: Proposal And Schema Preparation

- document the new field semantics
- freeze the canonical id format as `<registry>:<repo_id>`
- freeze the registry location as `src/vllm_hust_benchmark/data/model_identity_registry.json`
- freeze `model.name` as a compatibility mirror of `model.repo_id`
- extend the schema to allow the new fields

### Phase 1: Writer Normalization

- update benchmark export paths to resolve raw model inputs into canonical identity fields
- update same-spec-related exports so their canonical model remains stable even if runtime paths differ
- reject unresolved aliases during publish

### Phase 2: Reader Compatibility

- update aggregation to prefer `canonical_id`, then fall back to legacy `name`
- normalize historical snapshots through the same alias registry
- keep backward compatibility for old artifacts during a transition window

### Phase 3: Website Consumption

- switch filter, grouping, and display logic to the new fields
- stop deriving logical identity from raw `model.name`

### Phase 4: Historical Backfill

- regenerate `leaderboard_single.json`, `leaderboard_multi.json`, and `leaderboard_compare.json`
- ensure each logical model appears once per canonical identity
- verify compare scopes and hard-constraint scope keys remain stable where intended

### Phase 5: Contract Tightening

- deprecate new writes to `model.name` as an identity field
- keep `model.name` as a mirror of `repo_id` for one compatibility release after all active writers migrate
- remove legacy fallback logic after all active producers are migrated

## Why Frontend-Only Normalization Is Not Enough

A website-only fix can hide duplicate filter entries, but it does not solve the underlying contract problem.

It leaves the following issues in place:

- compare grouping and validation still see mixed identifiers
- historical snapshots remain semantically inconsistent
- new submissions can continue to publish ambiguous names
- downstream analytics cannot rely on one stable model key

Therefore the recommended fix point is the benchmark data boundary, with the website consuming normalized fields.

## Compatibility And Risk Notes

Main risks:

- accidental merge of genuinely different models that share a short alias
- drift between alias registry and official baseline specs
- compare scope churn if identity normalization changes grouping keys without a controlled migration

Mitigations:

- require explicit alias mapping for short names
- prefer repo coordinates over inferred aliases
- keep compare grouping on canonical identity only after backfill is ready
- add snapshot validation that rejects multiple display labels for one canonical id

## Acceptance Criteria

The normalization is complete when all of the following hold:

- one logical model produces one filter option in the website
- aggregation and compare logic use a stable canonical model identifier
- official baselines, same-spec exports, and ad-hoc benchmark submissions converge on the same identity record
- legacy snapshots can still be read during the migration window
- new submissions cannot introduce unregistered ambiguous short names

## Phase 0 Outcome

Phase 0 is complete when future implementation work treats the following as already decided:

- canonical ids use `<registry>:<repo_id>` and Hugging Face models use the `hf:` prefix
- the source-of-truth identity registry lives at `src/vllm_hust_benchmark/data/model_identity_registry.json`
- `model.name` remains in schema v1 only as a compatibility mirror of `model.repo_id`

The next implementation step is to update schema and writer paths to emit the frozen field set without changing website behavior yet.
