# KV Controller 0.1.6 replica routing on 9455

The replay client called the installed RIDE Mod and sent each request to the selected live vLLM
replica. Both arms used Qwen2.5-0.5B-Instruct on two RTX A6000 cards, 512 MiB device KV and 1 GiB
native CPU KV per replica, on-demand reload and zero speculative warmups. The baseline accounts for
the full prompt of in-flight work. The candidate subtracts the estimated reusable portion and keeps
the same decode weight. Routing latency is included.

Each workload size has three matched pairs across two SWE-smith replay cohorts, with the order
reversed on the repeated cohort. Model outputs were generated on hardware and tool waits came from
the replay. The public data preserves both arms and SHA256 identifiers of the original derived
summaries.

- 48 programs (965, 965 and 1,021 requests per arm) passed the replay gate. Median batch time fell
  14.71%, median throughput rose 17.25% and overall request P95 fell 37.52% to 56.55%.
- 24 programs (479, 479 and 497 requests per arm) had a median batch time reduction of 4.35%, below
  the 5% gate. Overall request P95 fell 67.19% to 71.91%.

The test covers replay routing. Production Request Throttling Controller integration, larger models
and online traffic remain separate acceptance work. Prefix coverage is estimated from recent
successful requests. These measurements do not establish a cross-replica KV transfer result or a
full Agent solve rate.

[Public data and per-pair timings](../../website-publication/data/kv-controller-routing-v1.json)
