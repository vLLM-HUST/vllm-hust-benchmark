# Legacy evidence inventory

The archived work produced real Ascend 910B2 correctness, LongBench, async,
prefix-caching, chunked-prefill, graph, abort/recovery, and online-performance
artifacts. Those results were generated from the legacy Core/Ascend pair, not
from this repository.

A reviewed subset of the final matched evidence is now public:

- [2026-08-13 Ascend 910B2 result](2026-08-13-ascend910b2/README.md)
- [provenance](2026-08-13-ascend910b2/provenance.json)
- [performance aggregate](2026-08-13-ascend910b2/performance-eager-prefix-aggregate.json)
- [quality aggregate](2026-08-13-ascend910b2/quality-summary.json)

Evidence classification for every imported result is therefore:

```text
legacy supporting evidence — not current-head acceptance
```

Before copying any additional raw artifact into this repository, record:

- immutable source path or archive URL;
- SHA-256 checksum;
- Core, Ascend, provider, model, and dataset revisions;
- Ascend device and CANN/torch/torch-npu versions;
- complete command and environment allowlist;
- workload shape, request count, concurrency, warmup, and repeat count;
- pass/fail rule and known anomalies; and
- whether the artifact was produced locally or through an organization runner.

The first active alpha must regenerate the required results from the pushed
provider and host exact heads. Historical summaries alone cannot satisfy that
gate.
