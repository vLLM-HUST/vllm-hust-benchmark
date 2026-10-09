# Dataset validation publication

This directory is the canonical, website-neutral publication for the Dataset Matrix. The benchmark
repository owns measured facts and comparison contracts; consumer repositories render synchronized
mirrors.

## Files

- `dataset_validation_index_v1.json` lists every independently selectable model/configuration.
- `dataset_program_v1.json` defines the evaluation-program taxonomy independently of measured
  scenarios. The five primary datasets are the **Pujiang-specified dataset scope**: MMLU-Pro,
  HLE-Verified, SWE-bench-Pro, FrontierScience, and Terminal-Bench 2.1. This designation freezes
  names only, not dataset revisions, splits, task manifests, sampling rules, scorers, execution
  images, licenses, or hashes. Every other registered or planned dataset is supplementary material
  by default.
- Planning-only artifacts remain in the index for direct evidence links but set
  `selector_visible: false`; consumer selectors must omit them.
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

Program registration is not measurement. A primary dataset with a pending contract or run must
appear in the program view, not as invented rows or empty cells in a measured scenario. Existing
supplementary measurements remain first-class evidence and keep their configuration-specific
scenario; the taxonomy changes their presentation priority, not their facts.

The historical 21-dataset B0 workbook is a separate serving-telemetry artifact. Among the five
Pujiang-specified datasets it contains only MMLU-Pro, and that row is not a task-accuracy result. It
must not be used to imply coverage of HLE-Verified, SWE-bench-Pro, FrontierScience, or
Terminal-Bench 2.1.

Measured B0 values may be published with an explicit evidence grade even when later metadata repair
cannot recreate a same-run cryptographic binding. Such a scenario must preserve the measured
configuration as its own selectable setting, identify the repaired fields and their sources, and
must not be pooled with a materially different configuration or used for an unmatched B1 gain.

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
