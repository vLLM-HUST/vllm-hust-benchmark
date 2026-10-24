# Qwen3.5 serving smoke on a prepared Ascend host

The [2026-10-09 UTC receipt](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/result.json)
records real Qwen3.5-35B-A3B inference on two Ascend 910B2 devices with
PyramidKV, APC, MTP2, async scheduling, chunked prefill, and
`FULL_AND_PIECEWISE`. Three requests returned the expected text. Both long
requests committed 5312-to-2048 full-attention KV compression on both TP ranks;
the repeated request reported 2048 cached prompt tokens. The initial run
drafted 14 tokens and accepted 10.

These are functional observations, not quality, capacity, or performance
qualification. The maximum configured context was 32768, but the longest
tested prompt was 5312 tokens. The 2048 physical-token result reflects block
alignment; it is not a measurement of whole-model HBM savings. Request timings
include compilation/cache effects and must not be compared as speedup results.

The subsequent [paired evaluation](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9b81690f767aca56524ec6651535a9b8d484a4c3/archive/mod-git-20261022/pyramidkv-website-source/snapshot/evidence/current/2026-10-09-paired-evaluation/README.md)
retains a graph cache-write failure discovered beyond this short smoke, its
shared-adapter fix, and a larger quality/capacity/performance comparison. Use
its adapter revision and explicit method configuration when reproducing that
evaluation. The revisions and observations below describe the original smoke.

## Prepared environment

Use an isolated environment inheriting the server's matched PyTorch/torch-npu
installation. Do not replace it with a generic PyTorch wheel. This run used:

| Component | Source revision / version |
| --- | --- |
| [vllm-hust](https://github.com/vLLM-HUST/vllm-hust) | `d0f22d2bda562156e4dbf433ce645e1769b4f804` |
| [vllm-ascend-hust](https://github.com/vLLM-HUST/vllm-ascend-hust) | `03766ac696fde5ab1980d80ca0b8543d3580c989` |
| Shared adapter | `a82e08798c10cee8bc0f0468fb0b05def064ae78` + manifest-only backport below |
| PyramidKV | `b0cd3fd73635b97b7c7372219e393e366df7c8d0` |
| [dev-hub](https://github.com/vLLM-HUST/vllm-hust-dev-hub) worker helpers | `1265f52680a9530cfcf77a5de13a6bf7ef30dc2c` |
| Extension Manager | `98903e416bdb593186b8245fd95180dafde995b9` |
| CANN / torch / torch-npu | `9.1.0` / `2.10.0+cpu` / `2.10.0.post4` |
| Transformers / triton-ascend | `5.14.1` / `3.2.2` |
| FastAPI / Starlette | `0.136.3` / `1.7.0` |

The local model revision was `59d61f3ce65a6d9863b86d2e96597125219dc754`.
All 14 weight shards matched the download metadata SHA256 values; see
[model hashes](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/model-hashes.json).
This is a different model revision from the September candidate qualification.

The installed Ascend metadata was `0.23.0.post1`. `VLLM_VERSION=0.25.1` in the
launcher selects the pinned host's ABI branch; it is not the installed vLLM
package version. Exact source revisions are more informative than version
strings for these experimental hosts.

The host's native PyTorch bindings and the Qwen-required CANN operators were
built from source. The upstream `csrc/build.sh --ops` selector limited the
operator build to this deployment. The evidence directory retains the
[operator build command](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/build-qwen.sh.txt)
and [native build command](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/build-native.py.txt).
These are snapshots from an already prepared checkout, not a fresh-server
installer. On the original interrupted build, unused binary output directories
had to be moved out of `csrc/build/binary/ascend910b/bin` before rerunning
`cmake --build csrc/build -j8 --target package`; the recorded build scripts do
not automate recovery of arbitrary stale build trees.

The pinned shared adapter has a Manifest 0.2 carrier. For Manager inspection,
this run applied the existing upstream manifest migration to an isolated
checkout using the retained
[manifest-only patch](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/manager-manifest-backport.patch).
Its runtime Python was unchanged. Editable plugin installations should be
discovered through their installed distribution metadata; adding their `src`
directories to `PYTHONPATH` can expose stale egg-info and break discovery.

## Start and verify

The scripts expect CANN, the built custom operator vendor environment, and the
pinned Python packages to be loaded already. `pipeline_worker` and
`frontier_worker` must be importable from the dev-hub worker helper directories.
The [environment snapshot](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/environment.sh.txt)
shows the exact original setup. In that prepared workspace:

```bash
source /root/workspace/pyramidkv-runtime/env.sh
export MODEL_PATH=/root/workspace/Qwen3.5-35B-A3B
bash scripts/qwen35/serve.sh > /tmp/qwen35-pyramidkv.log 2>&1 &
server_pid=$!
```

Run from the repository root. Adapt environment/model paths for another host.
Stop any existing deployment on the same devices and port before starting.
The default listener is `127.0.0.1:8000`, with model name `qwen35-pyramidkv`;
`SERVE_HOST` and `SERVE_PORT` can override the listener. This profile enables
text only, reserves 8 GiB of KV capacity per worker, and allows four sequences.
Startup includes model loading, compilation, and graph capture; wait for
`Application startup complete` and a successful health check:

```bash
curl --noproxy '*' -f http://127.0.0.1:8000/health
python scripts/qwen35/smoke.py \
  --model-path "$MODEL_PATH" \
  --server-log /tmp/qwen35-pyramidkv.log \
  --output /tmp/qwen35-smoke-result
```

Use an otherwise idle server: MTP counters are server-wide. The output directory
must not already exist. The client ignores HTTP proxy variables for its local
connection, stores raw requests/responses and metrics, and returns nonzero if
answers, request-matched compression commits, TP acknowledgements, APC hits, or
new MTP draft/acceptance counters are missing. `failure.json` records validation
errors. Both the public completion ID and the engine's suffixed request ID are
retained so stale successful transactions cannot satisfy a new run.

Stop the process when finished and wait for device memory to be released:

```bash
kill -TERM "$server_pid"
wait "$server_pid"
```

The environment activation is process-local. Set `PYRAMIDKV_ENABLED=0` for a
baseline launch through the shared adapter's disabled path; the default is
`1`. `PYRAMIDKV_CONFIG` selects a configuration file, with the original
512/beta-20 smoke profile as the default. Manager disablement alone does not
undo explicit environment variables.

## Remaining work

- The pinned Manager's joint launch plan reports a `vllm.environment` conflict
  between the shared adapter and method. This run used direct environment
  activation. Historical Manager lifecycle evidence remains separate; it does
  not establish that this source-install combination supports Manager launch.
- Inherited package metadata is not dependency-clean. The retained
  [dependency check](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/93090eda628b9c6038d1d7aa87ee115808e50ced/archive/mod-targeted-20261023/pyramid-current/snapshot/evidence/current/2026-10-09-qwen35-serving/dependency-check.txt)
  includes host/API version constraints and unrelated profiler dependencies.
  Successful inference does not imply all optional integrations work.
- This smoke receipt does not qualify quality, capacity, or comparative
  performance. The subsequent paired evaluation linked above records those
  measurements and their limits; release promotion remains a separate review.
