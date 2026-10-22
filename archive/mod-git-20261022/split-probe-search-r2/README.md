# Remaining probe search: bounded negative receipt

The [search receipt](search-receipt.json) records the expanded search for the 33 unresolved
reference occurrences (30 distinct paths) in the previous
[recovery audit](../split-probe-recovery/local-reference-audit.json). No additional original was
recovered. All 33 remain unresolved; previous recoveries, original documents, and missing-reference
statuses are unchanged.

The search checked historical Git objects, including locally present unreachable objects; old
workspaces; and actual members of eight public Release archives. Archive digests matched GitHub
asset metadata. All 8,020 regular archive members were hashed, not merely searched through a
manifest. Eight missing logs have previously declared SHA256 digests; none matched. Other missing
paths lack a declared digest, and no exact original-path/provenance identity was established.

The initial size exclusions (26 local logs and one dev-hub blob) were subsequently hashed in full,
without a match. Archive member content was also searched for original paths in UTF-8 and
JSON-escaped forms, without a match. Historical JSON provenance mappings did not resolve to matching
original bytes.

Remaining exclusions include absent promisor objects, offline storage, excluded dependency/cache
trees, and non-recursive handling of nested archives. A negative search does not prove permanent
loss. Generic coverage logs and assessment reports from other experiments are not substitutes for
the missing registration probes.

No raw data, references, runtime code, Release assets, installation consumers, hardware
authorization, or reviewer gates were changed. Further recovery needs the original probe/profiler
directories, a backup, or another provenance-backed copy; a new run or a reconstructed summary is
not the historical original.
