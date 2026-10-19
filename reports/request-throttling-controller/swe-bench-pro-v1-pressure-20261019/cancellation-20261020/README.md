# Non-streaming client-disconnect cancellation diagnostic

Goal: repair cross-arm contamination risk without changing the pinned upstream scheduler. This is a
transport diagnostic, not an Agent or engine-performance result.

Baseline: one request per direct/Native/Controller route. The direct route was idle at the first
post-timeout sample; both proxy routes still executed after release=true and drained about 8.2
seconds later.

Fix: cancel the awaited official forwarding operation when its HTTP caller disconnects. Both
isolated fixed routes drained in about 0.14 seconds, producing 5/8 tokens below the 512-token
diagnostic limit, and then returned normal 16-token responses. The engine abort counter stayed zero;
no exact abort-counter witness is claimed.

The first packaging failure and the second assertion failure remain documented. Frozen upstream
files remain unchanged. Current production proxies were not replaced. No successful 256-Agent
comparison or MOD benefit is established. See report.json for hashes and limits.

Run `python verify.py` for public integrity and reported-evidence consistency. It does not re-open
the private raw bundle.
