# Dev-hub evidence custody transfer

This index preserves the committed evidence in dev-hub's `results/`, `docs/evidence/`,
`Ascend-Machine/results/` and `Ascend-Machine/910B2/results/`.

`manifest.json` records the immutable source commit, original path, byte count and SHA256 for every
file. Fifteen files reuse identical benchmark Git blobs at pinned commits. The remaining 343 files
are stored without modification in the
[benchmark release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-devhub-20261020-v1).
No unmatched file duplicates a member SHA256 from the three `evidence-frontier-20260927-v1` campaign
archives.

This is evidence custody, not a new experiment or an approval of experimental pairing. Failed
attempts and original qualifications remain unchanged. No curves or gains are recomputed by this
migration.

## Verification

Download `devhub-evidence-20261020.tar.gz` from the release. Check the archive SHA256 against
`manifest.json`, then use `verify` in `scripts/migrate_devhub_evidence.py` to verify every member
and every reused Git blob. Verification reads members without extracting archive paths.

## Scope and source cleanup

The `outside_scope_review` list identifies additional potential evidence paths, including test code
that must not be treated as measurements automatically. Source removal requires verified remote
custody and migrated consumers. Until those checks are completed, source files remain in dev-hub;
their presence is not evidence that cleanup has finished.

Future datasets and experimental results belong in this repository, including Release assets. Other
organization repositories should contain implementation, methodology and pinned benchmark
references, not independent experimental data copies.
