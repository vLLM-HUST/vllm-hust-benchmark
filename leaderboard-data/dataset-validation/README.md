# Dataset validation publication

This directory is the canonical, website-neutral publication for the Dataset Matrix. The benchmark
repository owns measured facts and comparison contracts; consumer repositories render synchronized
mirrors.

## Files

- `dataset_validation_index_v1.json` lists every independently selectable model/configuration.
- Each `data_file` is one `dataset-validation-v1` artifact. Different models or materially different
  hardware and serving configurations must use different artifacts.
- `SHA256SUMS` seals the JSON publication set.
- Large raw logs and archives stay in their producing repositories or object storage. Result cells
  retain immutable URLs, revisions, and SHA256 values instead of copying those binaries here.

The index deliberately uses `data_file`, not a website URL. A website mirror may render it as
`./data/<data_file>`, but that path is a consumer concern.

Every dataset declares `applicable_metric_ids`. Every dataset/metric cell is present in the
artifact: inapplicable cells use `not_applicable`, while applicable cells without an admitted
measurement use `not_tested`, `queued`, or `running` with a reason. This prevents a blank cell from
silently meaning either unsupported or unfinished. A populated B1 is valid only when the same cell
also has a matched B0; unmatched candidate evidence stays under `candidate_search` and is not
rendered as a comparison.

## Validation

Run from the repository root:

```bash
python -m vllm_hust_benchmark.dataset_validation \
  --root leaderboard-data/dataset-validation
```

The checked-in coverage expansion is reproducible and idempotent:

```bash
python scripts/build_dataset_matrix_coverage.py
```

The validator checks scenario identity, unique dimensions and cells, result references, provenance
for populated B1 values, and every checksum. Tests additionally enforce both JSON schemas, and CI
runs the validator and test suite.

## Publishing changes

1. Add or update a model-specific artifact from real evidence.
1. Retain raw artifact links, source revisions, workload identity, and hashes.
1. Update the index and regenerate `SHA256SUMS`.
1. Pass the central validator and tests.
1. Synchronize the website mirror with its dataset-validation sync command.

Never edit the website mirror first and treat it as the source of truth.
