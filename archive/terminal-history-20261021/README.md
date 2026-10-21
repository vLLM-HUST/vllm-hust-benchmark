# Historical Experiment File Custody

Original selected files are stored in the benchmark
[Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-terminal-history-20261021-v1).
`release.json` pins the release and attachment hashes. Large files are Release assets, not Git
blobs.

The selection covers historical directories under `/root/gennova-evidence` dated before 2026-10-19
whose receipts report `passed`, `failed`, `completed`, `native_controller_passed`, or
`offline_export_passed`. Files larger than 1 MiB and their run receipts are archived. Both
successful and failed experiments retain their original bytes and status.

This is a selected-file archive, not a full machine backup. Current midterm evidence, model weights,
runtime dependencies, recovery candidates, directories without receipts and nonterminal runs are
excluded. Small local files remain in place. Archival custody does not establish experiment validity
or admit any performance result.

- `terminal-history.tar.gz`: original selected file bytes, with paths relative to
  `/root/gennova-evidence`.
- `manifest.json`: source paths, terminal receipt hashes, member sizes and SHA256, and cleanup
  scope.
- `verification.json`: initial complete member validation and bounded credential-pattern scan.
- `cleanup-receipt.json`: downloaded-archive verification and the exact local files removed.

Cleanup requires an independent download, verification of every member, unchanged local source
hashes and identities, and no open file descriptors or mapped candidate files in other processes.
Original local run receipts are retained. The credential scan covers specified credential patterns;
it is not a universal privacy guarantee.

Download the archive and manifest for restoration:

```bash
gh release download evidence-terminal-history-20261021-v1 \
  --repo vLLM-HUST/vllm-hust-benchmark \
  --pattern terminal-history.tar.gz --pattern manifest.json --dir /path/to/new-directory
```

Verify the archive hash against `release.json` and each requested member against `manifest.json`
before restoring into a new directory. Do not overwrite newer experiment files or reuse archived
runtime caches in a new measurement.
