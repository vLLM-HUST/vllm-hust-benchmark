# Production Stack Router real-model acceptance — 2026-09-01

This acceptance connected an isolated official Production Stack Router build to
an already-running vLLM service on `180-ascend-bench`. It did not restart,
reconfigure, or enter the production vLLM container, and it did not apply any
Kubernetes resource.

## Fixed inputs

- host architecture: `arm64`;
- official Production Stack source commit:
  `1b87c11a24c144f6b63a64dbae4fc8c875059731`;
- Router package version reported by that source build:
  `0.1.dev1+g1b87c11a2.d20260831`;
- test-only Router image ID:
  `sha256:9e4f01c60fe4a7478de9620afd6aeb8944a8a61f905191cfc9724028f7c85cb8`;
- existing vLLM model: `zai-org/GLM-4-32B-0414`, tensor parallelism four,
  system fingerprint `vllm-dev-tp4-1d1b60ee`;
- isolated Router address: `127.0.0.1:18085`;
- existing vLLM address: `127.0.0.1:8001`.

The published official
`ghcr.io/vllm-project/production-stack/router:v0.1.12` image could not be used:
its manifest has no `linux/arm64/v8` variant. The acceptance therefore built
the exact upstream source commit on arm64. The host Docker installation lacked
BuildKit/buildx, so the Dockerfile's two cache-mount annotations were removed;
the commands, source, entry point, and installed Router package were otherwise
unchanged. Optional semantic-cache dependencies were deliberately
excluded because this test covers the control-plane Router, not a KV extension.
The Python 3.12 arm64 base digest was
`sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217`.

## Failure and recovery

1. The Router first used the intentionally absent backend
   `http://127.0.0.1:65534`. Router `/health` returned 200, but a valid
   `POST /v1/chat/completions` returned HTTP 500. Structured logs recorded a
   connection refusal and the selected backend. This proves that process
   liveness alone is insufficient health evidence.
2. The external operator removed only the isolated Router container and
   recreated it with `http://127.0.0.1:8001`. The model service was not touched.
3. The same request then returned HTTP 200, model
   `zai-org/GLM-4-32B-0414`, content `ROUTER_OK`, and the vLLM system
   fingerprint. Router logs independently recorded routing to port 8001 and a
   200 response.
4. The direct model endpoint reported the same model ID and model root before
   and after the test. Its production container remained `running` with the
   original start timestamp `2026-08-25T02:29:03.080670219Z`.
5. The isolated Router container, source checkout, response files, test image,
   and newly pulled base-image tags were removed. The production model endpoint
   still returned its model after cleanup.

## Manager projection

`rollout_healthy=true` now also requires structured
`router_data_plane_evidence`. A mock backend is rejected as smoke evidence; a
real model, a 5xx failure, a subsequent 2xx recovery, Router version,
architecture, and release-image support must all be stated. The successful
source-built arm64 run is therefore `healthy` but remains `degraded` because
the official v0.1.12 release image does not support this architecture.

The post-change Manager wheel SHA-256 is
`48172a83869e364e48c5163760f9394d9618da271b3414b7820ac5bcfbc0d931`; the
Production Stack profile wheel SHA-256 is
`9d7269a64155e7f3edfd89ecfb39b5226185ef2a1943206082f3e24dae1dba77`.
A fresh Windows virtual environment installed both wheels and passed static
list, inspect, and validate. The installed manifest exposed both required
services—`kubernetes-api` through `kube_context` and `vllm-openai-backend`
through `router_backend_endpoint`—without importing cluster code.

This closes the real-model traffic gate. It does not close the release image
matrix gate and does not grant the Manager authority to start model services,
restart Routers, or mutate clusters.
