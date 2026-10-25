# Evidence, catalog and website publication order

This is an operator procedure, not evidence that an upload, merge or deployment has happened. The
proposed benchmark evidence tag is `evidence-muxpd-ecpa-20261025`; its read-only lookup returned 404
during preparation. Recheck before use and never replace existing assets with `--clobber`. The
software release `v0.2.0a1` belongs to the plugin repository and is separate from this benchmark
raw-evidence release.

## Freeze the evidence-only benchmark change

1. The Manager local full suite and final inventory are terminal. The final v4 archive is locally
   verified (347 members); remote CI success, local failure, preserved failed attempts, and targeted
   follow-up outcomes remain separate. The packer rejects live inputs or missing digests.

1. The v4 output is `/tmp/muxpd-qualification-archive-v4`; v1/v2 plans and the v3 draft were not
   replaced. Both Manager wheels, runtime-byte bridge, qualification/numerical originals and
   terminal CPU/CI receipts are included. Every decompressed member matched size and SHA256. Keep
   the tarball outside Git.

1. Copy only compact final archive/index/verification and CI records into this report directory,
   then explicitly stage the report, analysis tools and their integrity tests. Do not stage catalog
   drafts from `/tmp/muxpd-catalog-draft`, website data, runtime repositories or the compressed
   archive.

1. Regenerate indexes **after** staging new report paths because the generators inspect
   `git ls-files`:

   ```sh
   git add reports/muxpd-ecpa-20261025 \
     tests/test_muxpd_qualification_custody.py
   python scripts/data_index.py --write
   python scripts/data_entrypoints.py --write
   python scripts/build_website_evidence_links.py
   python scripts/data_index.py --check
   python scripts/data_entrypoints.py --check
   python scripts/build_website_evidence_links.py --check
   git diff --check
   ```

   Review and stage only the resulting index changes. Run the repository CI-parity checks; targeted
   custody/publication tests alone are not the complete benchmark CI suite. Push and merge the
   **evidence-only** PR. Record the actual merged commit, never a predicted or example SHA.

## Upload and download back the evidence assets

Only the root operator performs GitHub mutations. After the evidence PR is merged, bind the new
release tag to that actual commit. Below, each variable must refer to a reviewed existing value:

```sh
EVIDENCE_COMMIT=ACTUAL_MERGED_BENCHMARK_COMMIT
EVIDENCE_ASSET_DIR=/path/to/final-v4-assets
EVIDENCE_TAG=evidence-muxpd-ecpa-20261025
EVIDENCE_REPO=vLLM-HUST/vllm-hust-benchmark

cd "$EVIDENCE_ASSET_DIR"
sha256sum muxpd-ecpa-qualification-20261025.tar.gz archive-index.json \
  archive-verification.json > SHA256SUMS
sha256sum -c SHA256SUMS
gh release create "$EVIDENCE_TAG" \
  --repo "$EVIDENCE_REPO" --target "$EVIDENCE_COMMIT" --prerelease \
  --title 'MuxPD ECPA installed-wheel qualification — restricted TP4 evidence' \
  --notes-file /path/to/reviewed/RELEASE-NOTES.md \
  muxpd-ecpa-qualification-20261025.tar.gz archive-index.json \
  archive-verification.json SHA256SUMS
```

Use a **new** download directory, inspect every asset's `state=uploaded`, and record the API
metadata:

```sh
gh api "repos/$EVIDENCE_REPO/releases/tags/$EVIDENCE_TAG" > release-api.json
gh release download "$EVIDENCE_TAG" --repo "$EVIDENCE_REPO" \
  --dir /new/path/download-back
cd /new/path/download-back
sha256sum -c SHA256SUMS
python /path/to/committed/archive_qualification.py \
  --verify muxpd-ecpa-qualification-20261025.tar.gz --index archive-index.json
```

Compare downloaded bytes/digests with the local final index as well; a checksum file downloaded from
the same source is not an independent expected digest. Store the public tag, asset IDs/URLs,
observed states, sizes, download hashes and verification command outcomes in a new receipt. An API
inventory alone is not a download-back check. Retain failed uploads/downloads rather than changing
old receipts.

## Catalog PR, then code-only website pin

The second benchmark PR applies the reviewed four-table MuxPD draft and adds the new upload/download
receipt. Point `public_effect_url` and the qualification link at the **first PR's immutable merged
report SHA**. Keep Issue #23 as the numerical follow-up, the organization as maintainer, and the
historical Goodput URL explicitly historical. Do not add a current `output_tps` observation.

Validate all four schemas and joins; regenerate the indexes in the same order and merge this second
PR. Only then update the website benchmark pins to the second PR's actual merged SHA:

- `.github/workflows/ci.yml`: **six** literal occurrences, covering both jobs' checkout, publish and
  check steps.
- `.github/workflows/deploy-benchmark-pages.yml`: **one** checkout occurrence; publish/check use the
  checked-out commit variable.

Replace all seven occurrences consistently and verify with a full-file search, not remembered counts
alone. Keep website data generated from the benchmark commit, never edited or committed in the
website code-only PR. Assemble locally, run the presentation/publication tests, merge and check the
Pages run.

Completion additionally requires live `benchmark-source-manifest.json` identity and file digests,
the MuxPD records in all four joined catalogs, and the rendered card, filters, install instructions,
numerical limitation and rollback text. A merged PR or uploaded tarball is not itself website
delivery.
