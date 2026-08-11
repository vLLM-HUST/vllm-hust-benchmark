# Superseded coexistence migration (2026-08-11)

This archive contains two validated historical backfill runs that were published alongside a later
run for the same effective configuration and code combination. The retained submissions now record
`metadata.supersedes`; the original payload files in this archive are preserved byte-for-byte.

| Archived entry                         | Retained entry                         | Reason                                                                 |
| -------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| `455f6d1f-1da0-4c8b-8419-0969b58ad5e7` | `5055df08-dc36-4269-8230-e28d6cddf273` | PR77 same effective agent-research configuration; later foreground run |
| `6fdb3e91-00b1-49c4-893a-144c765e67b6` | `bf7cae25-d018-4c01-902c-9a43dac48459` | PR69 same effective agent-research configuration; later r5 run         |

The archive is audit-only. Production aggregation and resolvers must read only `submissions/` and
must not use this directory as a fallback.

The machine-readable mapping is in `index.json`.

The migration preserves the archived directories' original files and checksum manifests. The added
`README.md` files are explanatory metadata and are not part of the original manifests.
