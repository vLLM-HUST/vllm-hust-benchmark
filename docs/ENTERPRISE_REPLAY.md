# Enterprise replay

Enterprise request assets are external to Git. The package stores only logical relative paths, record counts, SHA-256 digests, source format, aggregate workload-family mapping, and replay case definitions.

## Mandatory gates

1. Pass `--data-root` or set `VLLM_HUST_ENTERPRISE_DATA_ROOT`; no repository-adjacent path discovery is performed.
2. Provide a dataset-scoped authorization file, either explicitly or as `.vllm-hust-enterprise-authorized.json` under the data root.
3. The loader verifies the entire file checksum and non-empty JSONL record count before parsing any request.
4. Sampling is deterministic over dataset ID, case ID, seed, sampler version, and source index.
5. The original source model is retained as provenance; the execution model is applied only when creating the OpenAI payload.
6. Redacted result records contain hashes and request-shape fields, never messages, prompts, tools, or complete request bodies.

Authorization file example:

```json
{
  "schema_version": "enterprise-data-authorization/v1",
  "authorized_dataset_ids": ["<dataset-id-from-registry>"]
}
```

List cases:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.enterprise_replay --list-cases
```

Validate, sample, and emit redacted records:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.enterprise_replay \
  --case-id <case-id> \
  --data-root <authorized-data-root> \
  --authorization-file <authorization.json> \
  --served-model <model-repo-id> \
  --limit 100 --seed 0 --output redacted-records.json
```

The Python API returns request bodies in memory only to the authorized runner. Call `EnterpriseReplayRequest.to_openai_payload(served_model=...)` immediately before submission and persist only `redacted_record(...)` plus runtime metrics.
