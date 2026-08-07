# Legacy Submission Archive

This directory retains original benchmark submission bytes that are not part of the active
publication set. Workflows, resolvers, and aggregators must read only `submissions/`; they must not
use this archive as an implicit fallback.

## 2026-08-07 isolation

The isolation was planned from benchmark commit `d90affbe47c2b7d9ffa1762eb8d304580bf384d6` and
contains:

- 224 submissions without evidence required by current admission, indexed in
  `incomplete-evidence/2026-08-07/index.json`;
- one incomplete duplicate involved in a superseded coexistence conflict; the retained copy has
  checksum-covered measurement and raw-result evidence;
- six submissions without `metadata.target_id`.

The coexistence and missing-target records are indexed in
`supplemental-isolation/2026-08-07/index.json`. No submission payload, manifest, environment
evidence, checksum file, or result metadata was edited during isolation.

Verify both indexes from the repository root:

```bash
python3 scripts/plan_legacy_submission_isolation.py \
  --archive-date 2026-08-07 \
  --verify-index archive/legacy/incomplete-evidence/2026-08-07/index.json
python3 scripts/verify_legacy_submission_archive.py \
  --index archive/legacy/supplemental-isolation/2026-08-07/index.json
```

Restoration requires a reviewed commit, matching the archived file inventory, followed by the full
admission, checksum, coexistence, snapshot, and trend validation suite. Historical provenance or
checksums must not be synthesized after the original run.
