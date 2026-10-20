# Frontier release evidence index

[Existing release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-frontier-20260927-v1)
contains the original Native, candidate-retry and follow-up archives. This directory indexes those
assets without uploading duplicate archive bytes.

`manifest.json` preserves the original archive names, release URLs, archive SHA256, 493 member paths
and hashes, and 10 original run-status receipts. Both passed and failed attempts are retained.
Member paths include their original leading `./` where present.

The index is a custody record, not an approval of a curve or a new measurement. Pairing still
requires the original configuration, workload, baseline, timing and correctness evidence. Never
combine the best points of different treatments into a single curve.

Reproduce the index using `scripts/index_frontier_release.py --source DIRECTORY --output NEW_FILE`.
The source directory must contain the three original filenames listed in the script; each digest
must match the existing release. Original archives are not modified.

The separately migrated dev-hub evidence is indexed in `../dev-hub-20261020/manifest.json`.
