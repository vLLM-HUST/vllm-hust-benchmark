# Dev-hub Release custody

## Visibility correction

Two source Releases were drafts: attempt 2, and the original checkpoint release (also a prerelease).
The initial migration incorrectly published their destination copies; both were restored to their
original draft/prerelease states. Attempt 3 remains public. `visibility-verification.json` records
the corrected states and rechecked asset digests. Original asset bytes and the initial custody
snapshot were not rewritten.

The migration tool now fails closed on missing or mismatched visibility metadata, creates drafts as
drafts, and verifies draft Releases by API ID instead of the public tag endpoint. Its `--publish`
flag means copying assets to the destination, not promoting a draft to public.

Draft HTML/download links can use GitHub-generated `untagged-*` paths. Use the stable release and
asset API IDs in [the final verification](final-verification.json) with authenticated access, not
invented tag URLs. The final verification checks the original declared tag names as well as
visibility and bytes. Release metadata edits must explicitly preserve the tag name and
draft/prerelease flags. Draft assets require an owner-only local cache directory.

The three SZYN evidence Releases originally hosted in dev-hub are copied into vllm-hust-benchmark
without changing asset bytes. `manifest.json` records original release IDs, titles, notes, asset
URLs, destination URLs, byte counts and SHA256 digests.

The attempt 2 release remains **INVALIDATED**. Copying it is preservation of an invalidated attempt,
not admission as a valid result. Original and adjudicated results must not be interchanged or pooled
across attempts.

`scripts/migrate_devhub_releases.py` verifies every downloaded source asset and checks the target
GitHub asset digest and size after upload. It never overwrites conflicting target assets and never
deletes source releases or assets.

Source assets remain temporarily available until current consumers have migrated. Historical Git
commits are not rewritten. See issue #326 for the remaining consumer changes and source cleanup;
this archive does not claim those are complete.
