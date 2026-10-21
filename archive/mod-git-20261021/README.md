# Public MOD Git evidence custody

The four StateAxis snapshots contain 264 original files: 243 evidence-directory files plus 21
context documents (including embedded historical measurements, source provenance, and license
notices). Every file retains its exact original bytes, source Git blob, SHA256 and length in its
repository manifest. Git itself deduplicates identical blobs; source path identities remain
distinct.

This is custody, not a new experiment or qualification. Historical positive, negative, failed,
diagnostic, and non-performance process checks retain their original boundaries. Fourteen empty
`health.json` files are deliberately retained and enumerated in `invalid_json_paths`; no success
state is inferred from them. Frozen snapshot files are excluded from code/Markdown rewriting and raw
JSON parsing. The custody verifier instead checks their complete file sets, original hashes, lengths
and declared JSON parse failures. Active migration code, tests and manifests remain linted and
validated.

Sources were explicitly public at capture. The migration script checks common credential signatures
before writing, and the snapshots were also reviewed for credential fields. Source code and
accompanying notices inside the snapshots are historical reproducibility context; they are never
executed by custody verification. No additional license grant is inferred.

The archive must merge and be downloaded back for byte verification before source copies are
removed. Consumer migration and source cleanup require their own passing CI and receipts; this
archive alone does not close issue 326.

## Source cleanup

[source-cleanup.json](source-cleanup.json) records the four merged consumer PRs, their complete
successful CI checks, default-branch trusted verification runs, and read-back strict/admin branch
protection bound to GitHub Actions. After downloading all originals from the merged benchmark commit
and verifying every byte length and SHA256, the source repositories removed 240 data/report copies.
Three historical Python source files remain unchanged as code. README and metadata measurements now
link to the immutable original reports; configuration and the negative-control regression test read
verified originals from caches outside the MOD repository. No historical commits were rewritten, and
no Release assets were deleted in this batch.

Hash-bound research contracts remain unchanged; changing their historical qualification wording
requires coordinated host identity updates. No new NPU experiment was run. Local Python tests and
builds passed in all four MODs; native tests passed for both Rust components. APC's optional strict
Clippy check still reports existing API-shape lints; these were not suppressed or presented as
passing. This batch does not complete the organization migration.
