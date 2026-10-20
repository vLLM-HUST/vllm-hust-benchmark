# Recovered calibration and source capsules

This
[benchmark Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-frontier-capsules-20261021-v1)
supplements the text-receipt archive with 25 original paths and 18 distinct files: calibration `.pt`
files, source `.tar`/`.gz` capsules and patches. `release.json` pins the three attachment hashes and
stable Release ID.

All three attachments were downloaded back. Every new tar member passed SHA256 and length
verification. The original server files remain unchanged. Repeated calibration contents retain their
separate original paths while sharing one archived byte sequence.

No pickle was loaded or executed. The inventory script performs bounded raw/ZIP/gzip byte-pattern
screening without extracting files or running code. This is not a comprehensive credential or
privacy audit. The selection excludes model/runtime trees, KV storage and other undeclared suffixes;
it is not a full filesystem backup.

`manifest.json` is the canonical retrieval map. `capsules-inventory.json` is the acquisition
receipt. No performance result, matched pair, or curve point is created by this archive.

Reproduce from the original source directory:

```bash
python scripts/inventory_recovered_capsules.py \
  --source /data/users/admin/frontier-mods-qwen35-20260925 \
  --output /path/to/new-capsules-inventory.json
python scripts/prepare_recovered_receipt_custody.py \
  --inventory /path/to/new-capsules-inventory.json \
  --destination /path/to/new-output \
  --release-tag evidence-frontier-capsules-20261021-v1 \
  --asset-name frontier-recovered-capsules-20261021.tar.gz
```

Source cleanup and consumer migrations remain tracked in benchmark issue #326.
