# Issue 214 effective-cache audit

This internal audit separates runtime cache behavior from the historical v0.18 target identity.
Historical target JSON and accepted historical results are not rewritten.

P2 comparisons use contract `p2-explicit-cache/v1`:

- every non-prefix workload records `no_enable_prefix_caching=true`;
- prefix repetition records `enable_prefix_caching=true` and `VLLM_KNORM_ENABLED=0`;
- both compared revisions must resolve the same remaining workload, model, hardware, request, rate,
  length, precision, quantization, CANN, and driver fields;
- three independent successful invocations are retained per cell and the predeclared primary metric
  uses their median, never the best value.

The original upstream baseline and official v0.23.0 pair are unavailable on the current CANN 8.5.1
host because `aclnnAddRmsNormBias` is absent. Fusion was not disabled and those failures remain
diagnostic evidence.

The completed A1-A6 audit and P2 9-by-2 matrix found no actionable regression. Prefix mean TTFT had
a high-variance, below-threshold shift: same-card six-run medians were 373.461 ms (C3) and
392.090 ms (current), +4.988%; output throughput was flat (-0.073%). ShareGPT online TTFT shifted
by +1.606%, also below the 5% latency threshold. The remaining seven primary metrics were flat or
improved.

The frozen aggregates, presentation-safe table, evidence suite locator, and checksums are in
`reports/issue-214-p2-closure-20260817/`.
