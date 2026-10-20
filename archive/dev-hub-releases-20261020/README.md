# Dev-hub Release custody

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
