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
