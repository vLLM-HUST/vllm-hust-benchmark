# Historical evidence custody

Raw measurements, reports, failed runs and their original context are now held
by [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git).
The exact source and per-file digests are pinned in [EVIDENCE_SOURCES.json](EVIDENCE_SOURCES.json).
This is a storage/reference migration, not new performance or correctness
qualification. Historical eager/graph cohorts and negative results remain
separate. Runtime code, default-off behavior and host integration are unchanged.

## Retrieve originals

Download the frozen source tree to a new directory outside this checkout:

```bash
python tools/benchmark_evidence.py --all --destination /tmp/split-batch-evidence
```

The downloader checks the immutable manifest and every file's length/SHA256,
rejects unsafe paths, and refuses to overwrite a corrupt cache. Empty/invalid
historical JSON is preserved, not repaired. It never executes downloaded code.
Use a separate external tree when investigating historical scripts: their
original relative paths refer to the full archived tree, not this trimmed MOD.

The remaining source scripts in historical evidence directories are read-only
provenance, not local result stores. Their exact hashes are covered by the
custody test; the frozen fi-sampling source/hash test remains unchanged.

## Future run output

The active `bench/bench_*.py` entry points require an absolute external output
directory, including when launched through the shell wrappers:

```bash
export SPLIT_BATCH_RESULTS_DIR=/tmp/split-batch-new-run
```

Do not reuse historical output directories or commit generated results here.
This setting does not authorize a device run, alter cache policy, or remove any
existing experiment approval requirement. Publish reviewed raw evidence to
benchmark and link it from the MOD. CI verifies archive retrieval and rejects
new tracked data in the cleaned directories; the trusted organization policy
continues to check pull requests.

## Installation assets are retained

Historical source-package evidence is archived under
[split-history-release-*](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022).
Published distributions are not rewritten: the existing Release assets match
PyPI and remain referenced by [release instructions](docs/release.md).
Archiving their evidence is not proof that their installation consumers have
disappeared. The sdist allowlist and distribution verifier prevent evidence from
entering future packages. No published installation asset is deleted here.
