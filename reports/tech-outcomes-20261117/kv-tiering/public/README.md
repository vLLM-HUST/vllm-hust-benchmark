# Public numeric evidence

These are **allowlisted derivatives of existing frozen measurements**, not new device runs,
byte-identical copies of complete request logs, official leaderboard entries or task-quality scores.

- `*-requests.jsonl.gz`: request/task/source identities and hashes; input/output token counts; HTTP,
  DONE and finish status; TTFT, request wall and TPOT. Only the first nonempty chunk time and total
  nonempty chunk count are retained from `chunk_times_s`; later chunk timings and all output text,
  prompt file paths, prompt token arrays and source conversations are omitted.
- `*-phases.json.gz`: all round counts and wall times, original client-process wall, and allowlisted
  original Prometheus counter values before/after each round. No rounding or tail trimming is
  applied.
- `full*-samples.jsonl.gz`: the original monitoring epochs, arm/stage, error flag and allowlisted
  pressure/queue/working-set gauges. These are sampled observations, not continuous device maxima.
- `manifest.json`: compressed and decoded hashes, plus original archive/member paths, sizes and
  SHA-256. It links each derivative to its source; hashes alone do not prove semantic correctness.

Coverage is fixed: three complete 128-task arms, each with 2,215 requests across 26 rounds, and six
complete 32-task arms, each with 540 requests across 22 rounds. Half-length retests, interrupted
campaigns and preflight-only failures are not silently substituted for completed arms. Negative
results remain in the public set.

From the repository root, with Python 3.10+ and no third-party libraries:

```bash
python reports/tech-outcomes-20261117/kv-tiering/public_evidence.py --check
python reports/tech-outcomes-20261117/kv-tiering/test_public_evidence.py
```

The helper recomputes throughput from output-token totals and all phase wall times, mean TTFT/TPOT
from individual requests, per-round H2D/D2H and other counter deltas, matched input identities and
sampled pressure peaks. It checks the historical analyses without overwriting them. An optional
`--output /tmp/new-numeric-audit.json` writes a separate analysis; it is not an original
measurement. Native does not expose plugin load/store counters; missing plugin counters in Native
are treated as zero, while core token/prefill metrics are required. Counters in these runs measure
HBM↔CPU payloads; **no filesystem read is claimed**.

An authorized reviewer with controlled originals can check the projection itself:

```bash
python reports/tech-outcomes-20261117/kv-tiering/public_evidence.py \
  --check-originals /path/to/original-archives
```

This verifies archive sizes/SHA-256, re-reads the selected original request/phase/Prometheus/monitor
members, and compares decoded public bytes and original-member custody. Gzip headers/compression may
differ across Python/zlib versions; both stored and decoded hashes are recorded. The export mode
`--export-from` refuses existing output files and never executes archived scripts or extracts paths.

This public check does not verify complete prompt or SSE bytes, output semantics, solve rate, SLO
qualification or statistical repeatability. The existing full-SSE audit and
[controlled original access index](../controlled-location.json) cover a separate verification route.
