# Recovered server receipt custody

Canonical evidence is held in the benchmark
[Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-frontier-recovered-20261021-v1).
`release.json` pins asset hashes and the stable Release API ID. Large indexes remain Release
attachments rather than duplicated Git blobs.

The snapshot covers 5,847 selected file paths from
`/data/users/admin/frontier-mods-qwen35-20260925`. The new archive contains 4,630 unique files;
byte-identical files already archived in `evidence-frontier-20260927-v1` are referenced by archive
URL, archive SHA256 and member path. Each original path retains its own identity, including paths
whose contents are duplicates.

- `manifest.json`: authoritative path-to-custody map, member hashes and selection exclusions.
- `inventory.json`: original acquisition receipt. Its local intermediate archive names describe
  acquisition only; use `manifest.json` for remote retrieval.
- `frontier-recovered-receipts-20261021.tar.gz`: additional original file bytes.

All three attachments were downloaded back. Their hashes matched the local snapshot and all 4,630
new archive members passed length and SHA256 verification. Original server files were not changed or
removed. This is not a full server backup: model weights, runtime binaries, KV storage and
undeclared suffixes are excluded.

The inventory includes 71 status records. One is invalid JSON and remains archived unchanged with
`passed: null` and `parse_error: invalid-json` in the index. Neither parseable status nor archival
custody independently establishes performance validity. Failed attempts are not discarded.

This migration does not produce new measurements, combine experiment configurations, choose
baseline/candidate pairs or admit curve points. Consumers must verify each run's complete contract
and boundaries before using it. Source deletion and consumer migration remain tracked in issue #326.

Before upload, a credential-pattern scan and an offline detect-secrets scan covered all 5,847
selected paths without findings. Hex/base64 entropy plugins were disabled to avoid treating
experimental hashes as credentials. This is a bounded screening result, not a universal privacy
guarantee.

To reconstruct the canonical archive from the original source and downloaded acquisition receipt:

```bash
python scripts/prepare_recovered_receipt_custody.py \
  --inventory /path/to/inventory.json \
  --destination /path/to/new-output
```

The command refuses changed source bytes or an existing output directory. Its existing-archive index
is `archive/frontier-20260927-v1/manifest.json`.
